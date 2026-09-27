"""Leak-free, pre-match team features for training and scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_prematch_features(matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Use only matches completed before each fixture to build both teams' features."""
    required = {"season", "matchday", "match_id", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required.difference(matches.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")

    data = matches.copy().sort_values(["season", "matchday", "match_id"])
    home = data.assign(
        team=data.home_team, opponent=data.away_team, venue="home",
        goals_for=data.home_goals, goals_against=data.away_goals,
    )
    away = data.assign(
        team=data.away_team, opponent=data.home_team, venue="away",
        goals_for=data.away_goals, goals_against=data.home_goals,
    )
    long = pd.concat([home, away], ignore_index=True).sort_values(["season", "matchday", "match_id"])
    long["points"] = np.select(
        [long.goals_for > long.goals_against, long.goals_for == long.goals_against], [3, 1], default=0
    )
    groups = long.groupby(["season", "team"], sort=False)
    long["form_points"] = groups.points.transform(lambda s: s.shift().rolling(form_window, min_periods=1).mean())
    long["goal_difference"] = long.goals_for - long.goals_against
    long["form_goal_diff"] = groups.goal_difference.transform(
        lambda s: s.shift().rolling(form_window, min_periods=1).mean()
    )
    long["elo_pre"] = _pre_match_elo(long)
    matchup = long.groupby(["season", "team", "opponent"], sort=False).points
    long["matchup_points"] = matchup.transform(lambda s: s.shift().expanding().mean())

    selected = long[["match_id", "team", "form_points", "form_goal_diff", "elo_pre", "matchup_points"]]
    home_features = selected.rename(columns={c: f"home_{c}" for c in selected.columns if c != "match_id"})
    away_features = selected.rename(columns={c: f"away_{c}" for c in selected.columns if c != "match_id"})
    result = data.merge(home_features, left_on=["match_id", "home_team"], right_on=["match_id", "home_team"])
    result = result.merge(away_features, left_on=["match_id", "away_team"], right_on=["match_id", "away_team"])
    result["elo_delta"] = result.home_elo_pre - result.away_elo_pre
    result["home_advantage"] = 1
    return result


def _pre_match_elo(long: pd.DataFrame, initial: float = 1500.0, k: float = 20.0) -> pd.Series:
    ratings: dict[tuple[str, str], float] = {}
    output = pd.Series(index=long.index, dtype=float)
    for (_season, _match_id), fixture in long.groupby(["season", "match_id"], sort=False):
        if len(fixture) != 2:
            continue
        first, second = fixture.iloc[0], fixture.iloc[1]
        key_a, key_b = (str(first.season), first.team), (str(second.season), second.team)
        rating_a, rating_b = ratings.get(key_a, initial), ratings.get(key_b, initial)
        output.loc[first.name], output.loc[second.name] = rating_a, rating_b
        expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        actual_a = 1.0 if first.goals_for > first.goals_against else 0.5 if first.goals_for == first.goals_against else 0.0
        ratings[key_a] = rating_a + k * (actual_a - expected_a)
        ratings[key_b] = rating_b + k * ((1 - actual_a) - (1 - expected_a))
    return output
