"""Build the documented team-level ML contract and PostgreSQL feature table."""
from __future__ import annotations

import pandas as pd

from db import engine, query
from features import build_team_prematch_features


IDENTIFIERS = ["season", "matchday", "match_id", "team", "opponent", "venue"]
NUMERIC_FEATURES = [
    "form_points", "form_goal_diff", "elo_rating", "opponent_elo", "elo_delta",
    "matchup_points", "home_advantage", "season_progress",
]
MODEL_FEATURES = NUMERIC_FEATURES
TARGET = "on_pace"


def final_season_labels(matches: pd.DataFrame) -> pd.DataFrame:
    """Label top-half finishers using final points, goal difference, then goals scored."""
    matches = matches.copy()
    matches["home_goals"] = pd.to_numeric(matches.home_goals, errors="raise")
    matches["away_goals"] = pd.to_numeric(matches.away_goals, errors="raise")
    home = matches.assign(
        team=matches.home_team, goals_for=matches.home_goals, goals_against=matches.away_goals
    )
    away = matches.assign(
        team=matches.away_team, goals_for=matches.away_goals, goals_against=matches.home_goals
    )
    results = pd.concat([home, away], ignore_index=True)
    results["points"] = (results.goals_for > results.goals_against).astype(int) * 3
    results.loc[results.goals_for == results.goals_against, "points"] = 1
    table = results.groupby(["season", "team"], as_index=False).agg(
        final_points=("points", "sum"),
        final_goals_for=("goals_for", "sum"),
        final_goals_against=("goals_against", "sum"),
        final_played=("match_id", "nunique"),
    )
    table["final_goal_difference"] = table.final_goals_for - table.final_goals_against
    table = table.sort_values(
        ["season", "final_points", "final_goal_difference", "final_goals_for", "team"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    table["final_rank"] = table.groupby("season").cumcount() + 1
    team_counts = table.groupby("season").team.transform("count")
    table[TARGET] = (table.final_rank <= ((team_counts + 1) // 2)).astype(int)
    return table


def build_ml_dataset(matches: pd.DataFrame) -> pd.DataFrame:
    """Attach final-season labels to pre-match features without using labels as features."""
    features = build_team_prematch_features(matches)
    labels = final_season_labels(matches)
    result = features.merge(labels, on=["season", "team"], validate="many_to_one")
    required = [*IDENTIFIERS, *MODEL_FEATURES, TARGET, "final_rank", "final_points"]
    return result[required].sort_values(["season", "matchday", "match_id", "team"], kind="stable")


def first_half_observations(dataset: pd.DataFrame) -> pd.DataFrame:
    maximum = dataset.groupby("season").matchday.transform("max")
    return dataset[dataset.matchday <= (maximum / 2).apply(int)].copy()


def chronological_split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_half = first_half_observations(dataset)
    seasons = sorted(first_half.season.astype(str).unique())
    if len(seasons) < 2:
        raise ValueError("Chronological evaluation requires at least two seasons")
    test_season = seasons[-1]
    return first_half[first_half.season.astype(str) != test_season], first_half[first_half.season.astype(str) == test_season]


def read_serving_matches() -> pd.DataFrame:
    return query(
        """SELECT season, matchday, match_id, kickoff_ts, home_team, away_team,
                  home_goals, away_goals
           FROM analytics.matches ORDER BY season, matchday, kickoff_ts, match_id"""
    )


def build_and_store_ml_features(**_context) -> int:
    feature_rows = build_team_prematch_features(read_serving_matches())
    with engine().begin() as connection:
        feature_rows.to_sql("team_prematch_features", connection, schema="analytics", if_exists="replace", index=False)
    return len(feature_rows)
