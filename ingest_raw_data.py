"""Validate, normalize, and land football source files for Hive external tables."""
from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from contracts import MATCH_COLUMNS, PLAYER_MATCH_COLUMNS

ROOT = Path(__file__).resolve().parent

MATCH_ALIASES = {
    "season": ("season", "league_season"),
    "matchday": ("matchday", "match_week", "matchweek", "round"),
    "match_id": ("match_id", "fixture_id", "id"),
    "kickoff_ts": ("kickoff_ts", "kickoff", "datetime", "date"),
    "home_team": ("home_team", "team1", "team_1"),
    "away_team": ("away_team", "team2", "team_2", "team 2"),
    "home_goals": ("home_goals", "team1_goals", "team_1_goals"),
    "away_goals": ("away_goals", "team2_goals", "team_2_goals", "team _2goals"),
}
EVENT_ALIASES = {
    "season": ("season", "league_season"),
    "matchday": ("matchday", "match_week", "matchweek", "round"),
    "match_id": ("match_id", "fixture_id"),
    "player_id": ("player_id", "sofifa_id", "id"),
    "player_name": ("player_name", "full_name", "name"),
    "team": ("team", "club", "team_name"),
    "minutes": ("minutes", "minutes_played"),
    "goals": ("goals", "goals_scored"),
    "assists": ("assists",),
    "shots": ("shots", "total_shots"),
    "tackles": ("tackles", "total_tackles"),
    "passes_completed": ("passes_completed", "completed_passes"),
}
PROFILE_ALIASES = {
    "player_id": ("player_id", "sofifa_id", "id"),
    "full_name": ("full_name", "player_name", "name"),
    "positions": ("positions", "position"),
    "nationality": ("nationality", "country"),
    "overall_rating": ("overall_rating", "overall", "rating"),
}


@dataclass(frozen=True)
class InputSources:
    matches: Path
    player_events: Path | None
    player_profiles: Path | None
    strict: bool
    dataset: str


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def _headers(path: Path, aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        source_headers = next(csv.reader(handle), [])
    by_key = {_key(header): header for header in source_headers}
    return {
        name: next((by_key[_key(alias)] for alias in choices if _key(alias) in by_key), "")
        for name, choices in aliases.items()
    }


def _validate_headers(path: Path, aliases, required, label: str) -> dict[str, str]:
    resolved = _headers(path, aliases)
    missing = [name for name in required if not resolved.get(name)]
    if missing:
        raise ValueError(f"{label} file {path} is missing required columns: {', '.join(missing)}")
    return resolved


def _read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _value(row: dict[str, str], column: str) -> str:
    return (row.get(column) or "").strip() if column else ""


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def resolve_sources(dataset: str | None = None) -> InputSources:
    dataset = (dataset or os.getenv("FOOTBALL_DATASET", "legacy")).lower()
    if dataset == "demo":
        return InputSources(
            ROOT / "data" / "demo_matches.csv",
            ROOT / "data" / "demo_player_match_stats.csv",
            None,
            True,
            dataset,
        )
    if dataset == "legacy":
        event_value = os.getenv("FOOTBALL_PLAYER_EVENT_CSV")
        return InputSources(
            Path(os.getenv("FOOTBALL_MATCH_CSV", ROOT / "combined (1).csv")),
            Path(event_value) if event_value else None,
            Path(os.getenv("FOOTBALL_PLAYER_PROFILE_CSV", ROOT / "playerdata.csv")),
            False,
            dataset,
        )
    if dataset == "custom":
        match_value, event_value = os.getenv("FOOTBALL_MATCH_CSV"), os.getenv("FOOTBALL_PLAYER_EVENT_CSV")
        if not match_value or not event_value:
            raise ValueError("custom input requires FOOTBALL_MATCH_CSV and FOOTBALL_PLAYER_EVENT_CSV")
        profile_value = os.getenv("FOOTBALL_PLAYER_PROFILE_CSV")
        return InputSources(
            Path(match_value), Path(event_value), Path(profile_value) if profile_value else None, True, dataset
        )
    raise ValueError("FOOTBALL_DATASET must be one of: legacy, demo, custom")


def _require_files(sources: InputSources) -> None:
    for label, path in (("match", sources.matches), ("player event", sources.player_events), ("player profile", sources.player_profiles)):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Configured {label} file does not exist: {path}")


def normalize_matches(source: Path, target: Path, strict: bool) -> dict[str, int]:
    required = MATCH_COLUMNS if strict else ("home_team", "away_team", "home_goals", "away_goals")
    columns = _validate_headers(source, MATCH_ALIASES, required, "match")
    written = rejected = 0
    seen_ids: set[tuple[str, str]] = set()
    generated_occurrences: dict[str, int] = {}
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row_number, row in enumerate(_read(source), 2):
            values = {name: _value(row, columns.get(name, "")) for name in MATCH_COLUMNS}
            values["season"] = values["season"] or os.getenv("FOOTBALL_DEFAULT_SEASON", "unknown")
            generated_id = not values["match_id"]
            if generated_id:
                base_id = _stable_id(
                    values["season"], values["matchday"], values["kickoff_ts"],
                    values["home_team"], values["away_team"], values["home_goals"], values["away_goals"],
                )
                occurrence = generated_occurrences.get(base_id, 0) + 1
                generated_occurrences[base_id] = occurrence
                values["match_id"] = base_id if occurrence == 1 else f"{base_id}-{occurrence}"
            missing = [name for name in required if not values[name]]
            numeric_invalid = any(
                values[name] and not values[name].lstrip("-").isdigit()
                for name in ("matchday", "home_goals", "away_goals")
            )
            if missing or numeric_invalid or values["home_team"] == values["away_team"]:
                if strict:
                    detail = ", ".join(missing) if missing else "invalid numeric/team values"
                    raise ValueError(f"Invalid match row {row_number} in {source}: {detail}")
                rejected += 1
                continue
            identity = values["season"], values["match_id"]
            if identity in seen_ids:
                raise ValueError(f"Duplicate season/match_id {identity} in {source}")
            seen_ids.add(identity)
            writer.writerow([values[name] for name in MATCH_COLUMNS])
            written += 1
    if not written:
        raise ValueError(f"No valid match rows found in {source}")
    return {"written": written, "rejected": rejected}


def normalize_player_events(source: Path | None, target: Path, strict: bool) -> dict[str, int]:
    if source is None:
        target.touch()
        return {"written": 0, "rejected": 0}
    required = PLAYER_MATCH_COLUMNS if strict else ("season", "matchday", "match_id", "player_name", "team", "minutes")
    columns = _validate_headers(source, EVENT_ALIASES, required, "player event")
    written = rejected = 0
    seen: set[tuple[str, str, str]] = set()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row_number, row in enumerate(_read(source), 2):
            values = {name: _value(row, columns.get(name, "")) for name in PLAYER_MATCH_COLUMNS}
            values["player_id"] = values["player_id"] or (_stable_id(values["player_name"]) if values["player_name"] else "")
            for metric in ("goals", "assists", "shots", "tackles", "passes_completed"):
                values[metric] = values[metric] or "0"
            missing = [name for name in required if not values[name]]
            numeric_invalid = any(
                values[name] and not values[name].lstrip("-").isdigit()
                for name in ("matchday", "minutes", "goals", "assists", "shots", "tackles", "passes_completed")
            )
            if missing or numeric_invalid:
                if strict:
                    detail = ", ".join(missing) if missing else "invalid numeric values"
                    raise ValueError(f"Invalid player event row {row_number} in {source}: {detail}")
                rejected += 1
                continue
            identity = values["season"], values["match_id"], values["player_id"]
            if identity in seen:
                raise ValueError(f"Duplicate player appearance {identity} in {source}")
            seen.add(identity)
            writer.writerow([values[name] for name in PLAYER_MATCH_COLUMNS])
            written += 1
    if strict and not written:
        raise ValueError(f"No valid player event rows found in {source}")
    return {"written": written, "rejected": rejected}


def normalize_profiles(source: Path | None, event_source: Path | None, target: Path) -> int:
    if source:
        aliases, required, name_key = PROFILE_ALIASES, ("full_name",), "full_name"
    elif event_source:
        aliases, required, name_key = EVENT_ALIASES, ("player_id", "player_name"), "player_name"
        source = event_source
    else:
        target.touch()
        return 0
    columns = _validate_headers(source, aliases, required, "player profile")
    count, seen = 0, set()
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in _read(source):
            name = _value(row, columns.get(name_key, ""))
            player_id = _value(row, columns.get("player_id", "")) or (_stable_id(name) if name else "")
            if not name or player_id in seen:
                continue
            seen.add(player_id)
            writer.writerow((
                player_id, name, _value(row, columns.get("positions", "")),
                _value(row, columns.get("nationality", "")), _value(row, columns.get("overall_rating", "")),
            ))
            count += 1
    return count


def validate_event_relationships(matches: Path, events: Path) -> None:
    """Ensure every player appearance refers to one of that fixture's two teams."""
    with matches.open(encoding="utf-8", newline="") as handle:
        fixtures = {
            (row[0], row[2]): {row[4], row[5]}
            for row in csv.reader(handle)
            if row
        }
    with events.open(encoding="utf-8", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle), 1):
            if not row:
                continue
            identity = row[0], row[2]
            if identity not in fixtures:
                raise ValueError(f"Player event row {row_number} references unknown fixture {identity}")
            if row[5] not in fixtures[identity]:
                raise ValueError(
                    f"Player event row {row_number} team {row[5]} is not in fixture {identity}"
                )


def _hdfs_land(local: Path, remote: str) -> None:
    executable = os.getenv("HADOOP_FS_BIN", "hdfs")
    subprocess.run([executable, "dfs", "-mkdir", "-p", remote.rsplit("/", 1)[0]], check=True)
    subprocess.run([executable, "dfs", "-put", "-f", str(local), remote], check=True)


def _local_land(local: Path, remote: str) -> None:
    destination = Path(remote)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local, destination)


def ingest_raw_data(dataset: str | None = None, landing_mode: str | None = None, **_context) -> dict[str, object]:
    sources = resolve_sources(dataset)
    _require_files(sources)
    landing_mode = (landing_mode or os.getenv("FOOTBALL_LANDING_MODE", "hdfs")).lower()
    if landing_mode not in {"hdfs", "local"}:
        raise ValueError("FOOTBALL_LANDING_MODE must be hdfs or local")
    variable = "FOOTBALL_HDFS_RAW_ROOT" if landing_mode == "hdfs" else "FOOTBALL_LOCAL_RAW_ROOT"
    default_root = "hdfs:///football/raw" if landing_mode == "hdfs" else str(ROOT / "build" / "raw")
    root = os.getenv(variable, default_root).rstrip("/\\")
    run_id = os.getenv("FOOTBALL_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    land = _hdfs_land if landing_mode == "hdfs" else _local_land

    with tempfile.TemporaryDirectory(prefix="football-ingest-") as directory_name:
        directory = Path(directory_name)
        matches, players, events = directory / "matches.csv", directory / "players.csv", directory / "player_match_stats.csv"
        match_counts = normalize_matches(sources.matches, matches, sources.strict)
        event_counts = normalize_player_events(sources.player_events, events, sources.strict)
        profile_count = normalize_profiles(sources.player_profiles, sources.player_events, players)
        if event_counts["written"]:
            validate_event_relationships(matches, events)
        separator = "/" if landing_mode == "hdfs" else os.sep

        def target(*parts: str) -> str:
            return root + separator + separator.join(parts)

        land(sources.matches, target("archive", run_id, "matches", sources.matches.name))
        if sources.player_events:
            land(sources.player_events, target("archive", run_id, "player_events", sources.player_events.name))
        if sources.player_profiles:
            land(sources.player_profiles, target("archive", run_id, "player_profiles", sources.player_profiles.name))
        land(matches, target("current", "matches", "matches.csv"))
        land(players, target("current", "players", "players.csv"))
        land(events, target("current", "player_match_stats", "player_match_stats.csv"))

    return {
        "dataset": sources.dataset, "landing_mode": landing_mode,
        "match_rows": match_counts["written"], "match_rows_rejected": match_counts["rejected"],
        "player_rows": profile_count, "player_match_rows": event_counts["written"],
        "player_match_rows_rejected": event_counts["rejected"], "raw_root": root, "run_id": run_id,
    }


if __name__ == "__main__":
    print(ingest_raw_data())
