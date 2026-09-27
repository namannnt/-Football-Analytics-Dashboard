"""Canonical column contracts shared by ingestion, ML, and validation tools."""
from __future__ import annotations

MATCH_COLUMNS = ("season", "matchday", "match_id", "kickoff_ts", "home_team", "away_team", "home_goals", "away_goals")
PLAYER_PROFILE_COLUMNS = ("player_id", "full_name", "positions", "nationality", "overall_rating")
PLAYER_MATCH_COLUMNS = (
    "season", "matchday", "match_id", "player_id", "player_name", "team",
    "minutes", "goals", "assists", "shots", "tackles", "passes_completed",
)
HIVE_STANDINGS_COLUMNS = (
    "season", "matchday", "team", "played", "wins", "draws", "losses",
    "goals_for", "goals_against", "goal_difference", "points",
)
HIVE_PLAYER_ROLLING_COLUMNS = (
    "season", "matchday", "match_id", "player_id", "player_name", "team",
    "minutes_last_5", "goals_per90_last5", "assists_per90_last5",
    "shots_per90_last5", "tackles_per90_last5", "passes_completed_per90_last5",
)
CURATED_COLUMNS = (
    "season", "matchday", "match_id", "player_id", "player_name", "team",
    "team_played", "team_wins", "team_draws", "team_losses", "team_goals_for",
    "team_goals_against", "team_goal_difference", "team_points", "minutes_last_5",
    "goals_per90_last5", "assists_per90_last5", "shots_per90_last5",
    "tackles_per90_last5", "passes_completed_per90_last5",
)
