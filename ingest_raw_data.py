"""Normalize source CSV files and land immutable/current snapshots in HDFS."""
from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _input(env_name: str, *defaults: str) -> Path:
    configured = os.getenv(env_name)
    candidates = [Path(configured)] if configured else [ROOT / name for name in defaults]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Set {env_name}; checked: {', '.join(map(str, candidates))}")


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _get(row: dict[str, str], *aliases: str) -> str:
    normalized = {_key(k): (v or "").strip() for k, v in row.items() if k}
    return next((normalized[_key(a)] for a in aliases if normalized.get(_key(a))), "")


def _read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _id(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def _matches(source: Path, target: Path) -> int:
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for index, row in enumerate(_read(source), 1):
            home = _get(row, "home_team", "team1", "team_1")
            away = _get(row, "away_team", "team2", "team_2", "team 2")
            hg = _get(row, "home_goals", "team1_goals", "team_1_goals")
            ag = _get(row, "away_goals", "team2_goals", "team_2_goals", "team _2goals")
            if not all((home, away, hg, ag)):
                continue
            season = _get(row, "season", "league_season") or os.getenv("FOOTBALL_DEFAULT_SEASON", "unknown")
            day = _get(row, "matchday", "match_week", "matchweek", "round")
            kickoff = _get(row, "kickoff_ts", "kickoff", "datetime", "date")
            match_id = _get(row, "match_id", "fixture_id", "id") or _id(
                season, day, kickoff, home, away, str(index)
            )
            writer.writerow((season, day, match_id, kickoff, home, away, hg, ag))
            count += 1
    return count


def _players(source: Path, target: Path) -> int:
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in _read(source):
            name = _get(row, "full_name", "player_name", "name")
            if not name:
                continue
            writer.writerow((
                _get(row, "player_id", "sofifa_id", "id") or _id(name), name,
                _get(row, "positions", "position"), _get(row, "nationality", "country"),
                _get(row, "overall_rating", "overall", "rating"),
            ))
            count += 1
    return count


def _player_events(source: Path, target: Path) -> int:
    """Extract event rows when the source is match-grained; profile-only input yields zero rows."""
    count = 0
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in _read(source):
            name = _get(row, "player_name", "full_name", "name")
            day, match_id = _get(row, "matchday", "matchweek", "round"), _get(row, "match_id", "fixture_id")
            team, minutes = _get(row, "team", "club", "team_name"), _get(row, "minutes", "minutes_played")
            if not all((name, day, match_id, team, minutes)):
                continue
            writer.writerow((
                _get(row, "season", "league_season") or os.getenv("FOOTBALL_DEFAULT_SEASON", "unknown"),
                day, match_id, _get(row, "player_id", "sofifa_id", "id") or _id(name), name, team, minutes,
                _get(row, "goals", "goals_scored") or "0", _get(row, "assists") or "0",
                _get(row, "shots", "total_shots") or "0", _get(row, "tackles", "total_tackles") or "0",
                _get(row, "passes_completed", "completed_passes") or "0",
            ))
            count += 1
    return count


def _hdfs(*args: str) -> None:
    subprocess.run([os.getenv("HADOOP_FS_BIN", "hdfs"), "dfs", *args], check=True)


def _land(local: Path, remote: str) -> None:
    parent = remote.rsplit("/", 1)[0]
    _hdfs("-mkdir", "-p", parent)
    _hdfs("-put", "-f", str(local), remote)


def ingest_raw_data(**_context) -> dict[str, object]:
    match_source = _input("FOOTBALL_MATCH_CSV", "combined.csv", "combined (1).csv")
    player_source = _input("FOOTBALL_PLAYER_CSV", "playerdata.csv")
    root = os.getenv("FOOTBALL_HDFS_RAW_ROOT", "hdfs:///football/raw").rstrip("/")
    run_id = os.getenv("FOOTBALL_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with tempfile.TemporaryDirectory(prefix="football-ingest-") as directory:
        directory = Path(directory)
        matches, players, events = directory / "matches.csv", directory / "players.csv", directory / "player_match_stats.csv"
        counts = {
            "matches": _matches(match_source, matches),
            "players": _players(player_source, players),
            "player_match_stats": _player_events(player_source, events),
        }
        _land(match_source, f"{root}/archive/{run_id}/matches/{match_source.name}")
        _land(player_source, f"{root}/archive/{run_id}/players/{player_source.name}")
        _land(matches, f"{root}/current/matches/matches.csv")
        _land(players, f"{root}/current/players/players.csv")
        _land(events, f"{root}/current/player_match_stats/player_match_stats.csv")

    return {**counts, "hdfs_root": root, "run_id": run_id}


if __name__ == "__main__":
    print(ingest_raw_data())
