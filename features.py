"""Leak-free pre-match feature generation from completed fixture history."""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_MATCH_COLUMNS = {
    "season", "matchday", "match_id", "kickoff_ts", "home_team", "away_team", "home_goals", "away_goals"
}


def _validated_matches(matches: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_MATCH_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")
    data = matches.copy()
    for column in ("matchday", "home_goals", "away_goals"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data.match_id.duplicated().any():
        duplicate = data.loc[data.match_id.duplicated(), "match_id"].iloc[0]
        raise ValueError(f"Duplicate match_id: {duplicate}")
    return data.sort_values(["season", "matchday", "kickoff_ts", "match_id"], kind="stable")


def _long_results(matches: pd.DataFrame) -> pd.DataFrame:
    home = matches.assign(
        team=matches.home_team,
        opponent=matches.away_team,
        venue="home",
        goals_for=matches.home_goals,
        goals_against=matches.away_goals,
    )
    away = matches.assign(
        team=matches.away_team,
        opponent=matches.home_team,
        venue="away",
        goals_for=matches.away_goals,
        goals_against=matches.home_goals,
    )
    long = pd.concat([home, away], ignore_index=True)
    long["venue_order"] = long.venue.map({"home": 0, "away": 1})
    long = long.sort_values(
        ["season", "matchday", "kickoff_ts", "match_id", "venue_order"], kind="stable"
    )
    long["points"] = np.select(
        [long.goals_for > long.goals_against, long.goals_for == long.goals_against],
        [3, 1],
        default=0,
    )
    long["goal_difference"] = long.goals_for - long.goals_against
    return long


def _pre_match_elo(long: pd.DataFrame, initial: float = 1500.0, k: float = 20.0) -> pd.Series:
    ratings: dict[tuple[str, str], float] = {}
    output = pd.Series(index=long.index, dtype=float)
    fixture_order = long[["season", "match_id"]].drop_duplicates().itertuples(index=False, name=None)
    for season, match_id in fixture_order:
        fixture = long[(long.season == season) & (long.match_id == match_id)]
        if len(fixture) != 2:
            raise ValueError(f"Fixture {season}/{match_id} does not have exactly two team rows")
        home = fixture.loc[fixture.venue == "home"].iloc[0]
        away = fixture.loc[fixture.venue == "away"].iloc[0]
        home_key, away_key = (str(season), home.team), (str(season), away.team)
        home_rating, away_rating = ratings.get(home_key, initial), ratings.get(away_key, initial)
        output.loc[home.name], output.loc[away.name] = home_rating, away_rating
        expected_home = 1 / (1 + 10 ** ((away_rating - home_rating) / 400))
        actual_home = 1.0 if home.goals_for > home.goals_against else 0.5 if home.goals_for == home.goals_against else 0.0
        ratings[home_key] = home_rating + k * (actual_home - expected_home)
        ratings[away_key] = away_rating + k * ((1 - actual_home) - (1 - expected_home))
    return output


def build_team_prematch_features(matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Return one row per team and fixture using information available before kickoff."""
    data = _validated_matches(matches)
    long = _long_results(data)
    team_history = long.groupby(["season", "team"], sort=False)
    long["form_points"] = team_history.points.transform(
        lambda values: values.shift().rolling(form_window, min_periods=1).mean()
    )
    long["form_goal_diff"] = team_history.goal_difference.transform(
        lambda values: values.shift().rolling(form_window, min_periods=1).mean()
    )
    long["elo_rating"] = _pre_match_elo(long)
    matchup_history = long.groupby(["season", "team", "opponent"], sort=False).points
    long["matchup_points"] = matchup_history.transform(lambda values: values.shift().expanding().mean())
    season_last_day = long.groupby("season").matchday.transform("max")
    long["season_progress"] = (long.matchday - 1) / season_last_day.clip(lower=1)
    long["home_advantage"] = (long.venue == "home").astype(int)

    opponent_elo = long[["season", "match_id", "team", "elo_rating"]].rename(
        columns={"team": "opponent", "elo_rating": "opponent_elo"}
    )
    long = long.merge(opponent_elo, on=["season", "match_id", "opponent"], how="left", validate="many_to_one")
    long["elo_delta"] = long.elo_rating - long.opponent_elo
    columns = [
        "season", "matchday", "match_id", "kickoff_ts", "team", "opponent", "venue",
        "form_points", "form_goal_diff", "elo_rating", "opponent_elo", "elo_delta",
        "matchup_points", "home_advantage", "season_progress",
    ]
    return long[columns].sort_values(["season", "matchday", "match_id", "team"], kind="stable").reset_index(drop=True)


def build_prematch_features(matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Backward-compatible fixture-level view derived from the team-level contract."""
    data = _validated_matches(matches)
    team_features = build_team_prematch_features(data, form_window=form_window)
    feature_columns = ["form_points", "form_goal_diff", "elo_rating", "matchup_points"]
    home = team_features[team_features.venue == "home"][["match_id", *feature_columns]].rename(
        columns={column: f"home_{column.replace('elo_rating', 'elo_pre')}" for column in feature_columns}
    )
    away = team_features[team_features.venue == "away"][["match_id", *feature_columns]].rename(
        columns={column: f"away_{column.replace('elo_rating', 'elo_pre')}" for column in feature_columns}
    )
    return data.merge(home, on="match_id", validate="one_to_one").merge(
        away, on="match_id", validate="one_to_one"
    ).assign(
        elo_delta=lambda frame: frame.home_elo_pre - frame.away_elo_pre,
        home_advantage=1,
    )
