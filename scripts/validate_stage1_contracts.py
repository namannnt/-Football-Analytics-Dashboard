"""Check the file-to-Hive-to-Spark column contract without cluster services."""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contracts import (  # noqa: E402
    CURATED_COLUMNS,
    HIVE_PLAYER_ROLLING_COLUMNS,
    HIVE_STANDINGS_COLUMNS,
    MATCH_COLUMNS,
    PLAYER_MATCH_COLUMNS,
)
from spark_layer import CURATED_QUERY  # noqa: E402


def header(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(next(csv.reader(handle)))


def main() -> None:
    match_path = ROOT / "data" / "demo_matches.csv"
    event_path = ROOT / "data" / "demo_player_match_stats.csv"
    assert header(match_path) == MATCH_COLUMNS
    assert header(event_path) == PLAYER_MATCH_COLUMNS
    with match_path.open(encoding="utf-8", newline="") as handle:
        matches = list(csv.DictReader(handle))
    with event_path.open(encoding="utf-8", newline="") as handle:
        events = list(csv.DictReader(handle))
    fixtures = {row["match_id"]: {row["home_team"], row["away_team"]} for row in matches}
    assert len(matches) == 60 and len(events) == 360
    assert all(row["match_id"] in fixtures and row["team"] in fixtures[row["match_id"]] for row in events)
    appearances = Counter((row["season"], row["player_id"]) for row in events)
    assert min(appearances.values()) >= 5

    standings_hql = (ROOT / "hive_queries" / "season_standings_raw.hql").read_text(encoding="utf-8").lower()
    rolling_hql = (ROOT / "hive_queries" / "player_rolling_stats_raw.hql").read_text(encoding="utf-8").lower()
    assert all(column in standings_hql for column in HIVE_STANDINGS_COLUMNS)
    assert all(column in rolling_hql for column in HIVE_PLAYER_ROLLING_COLUMNS)
    query = CURATED_QUERY.lower()
    missing = {column for column in CURATED_COLUMNS if column.lower() not in query}
    if missing:
        raise AssertionError(f"Spark query does not emit contract columns: {sorted(missing)}")
    print("Stage 1 contracts align: source CSV -> Hive raw/aggregate -> Spark curated columns")


if __name__ == "__main__":
    main()
