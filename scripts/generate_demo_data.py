"""Regenerate deterministic synthetic CSVs used only for architecture validation."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEAMS = ("Atlas FC", "Boreal United", "Comet City", "Dynamo Rovers")
PLAYERS = {
    team: [(f"{team[:2].upper()}{number}", f"{team.split()[0]} Player {number}") for number in range(1, 4)]
    for team in TEAMS
}
SCHEDULE = (
    (("Atlas FC", "Boreal United"), ("Comet City", "Dynamo Rovers")),
    (("Atlas FC", "Comet City"), ("Boreal United", "Dynamo Rovers")),
    (("Atlas FC", "Dynamo Rovers"), ("Boreal United", "Comet City")),
    (("Boreal United", "Atlas FC"), ("Dynamo Rovers", "Comet City")),
    (("Comet City", "Atlas FC"), ("Dynamo Rovers", "Boreal United")),
    (("Dynamo Rovers", "Atlas FC"), ("Comet City", "Boreal United")),
)


def score(season_index: int, matchday: int, match_index: int) -> tuple[int, int]:
    """Return varied, deterministic results without claiming historical truth."""
    home = (season_index + matchday + match_index * 2) % 4
    away = (season_index * 2 + matchday * 3 + match_index) % 3
    return home, away


def allocation(total: int) -> tuple[int, int, int]:
    return (min(total, 1), min(max(total - 1, 0), 1), max(total - 2, 0))


def generate() -> tuple[int, int]:
    DATA.mkdir(exist_ok=True)
    matches_path = DATA / "demo_matches.csv"
    events_path = DATA / "demo_player_match_stats.csv"
    match_count = event_count = 0
    with matches_path.open("w", encoding="utf-8", newline="") as matches_file, events_path.open(
        "w", encoding="utf-8", newline=""
    ) as events_file:
        matches = csv.writer(matches_file)
        events = csv.writer(events_file)
        matches.writerow(("season", "matchday", "match_id", "kickoff_ts", "home_team", "away_team", "home_goals", "away_goals"))
        events.writerow(("season", "matchday", "match_id", "player_id", "player_name", "team", "minutes", "goals", "assists", "shots", "tackles", "passes_completed"))
        for season_index, season in enumerate(("2023-demo", "2024-demo", "2025-demo")):
            season_start = datetime(2023 + season_index, 8, 5, 15, 0, tzinfo=timezone.utc)
            for matchday, fixtures in enumerate(SCHEDULE, 1):
                for match_index, (home_team, away_team) in enumerate(fixtures, 1):
                    match_id = f"{season[:4]}-D{matchday:02d}-M{match_index}"
                    kickoff = season_start + timedelta(days=(matchday - 1) * 7, hours=(match_index - 1) * 3)
                    home_goals, away_goals = score(season_index, matchday, match_index)
                    matches.writerow((season, matchday, match_id, kickoff.isoformat(), home_team, away_team, home_goals, away_goals))
                    match_count += 1
                    for team, team_goals, opponent_goals in (
                        (home_team, home_goals, away_goals), (away_team, away_goals, home_goals)
                    ):
                        goals = allocation(team_goals)
                        for player_index, (player_id, player_name) in enumerate(PLAYERS[team]):
                            player_goals = goals[player_index]
                            assists = 1 if player_index == 2 and team_goals > 0 else 0
                            shots = player_goals + 1 + ((matchday + player_index) % 3)
                            tackles = 1 + ((matchday + player_index + opponent_goals) % 5)
                            passes = 24 + matchday * 2 + player_index * 7 + (3 if team == home_team else 0)
                            events.writerow((
                                season, matchday, match_id, player_id, player_name, team, 90,
                                player_goals, assists, shots, tackles, passes,
                            ))
                            event_count += 1
    return match_count, event_count


if __name__ == "__main__":
    matches, events = generate()
    print({"demo_match_rows": matches, "demo_player_event_rows": events})
