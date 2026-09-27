from __future__ import annotations

import csv
from pathlib import Path

import pytest

from contracts import MATCH_COLUMNS, PLAYER_MATCH_COLUMNS
from ingest_raw_data import normalize_matches, normalize_player_events


def write_csv(path: Path, headers, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def read_rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_match_aliases_and_deterministic_id(tmp_path):
    source = tmp_path / "legacy.csv"
    write_csv(source, ["team1", "team 2", "team 1_goals", "team _2goals"], [["A", "B", 2, 1]])
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    assert normalize_matches(source, first, strict=False) == {"written": 1, "rejected": 0}
    normalize_matches(source, second, strict=False)
    assert read_rows(first) == read_rows(second)
    assert len(read_rows(first)[0]) == len(MATCH_COLUMNS)


def test_strict_match_contract_rejects_missing_column(tmp_path):
    source, target = tmp_path / "matches.csv", tmp_path / "out.csv"
    write_csv(source, MATCH_COLUMNS[:-1], [["demo", 1, "m1", "2026-01-01", "A", "B", 1]])
    with pytest.raises(ValueError, match="away_goals"):
        normalize_matches(source, target, strict=True)


def test_strict_match_contract_rejects_invalid_row(tmp_path):
    source, target = tmp_path / "matches.csv", tmp_path / "out.csv"
    write_csv(source, MATCH_COLUMNS, [["demo", 1, "m1", "2026-01-01", "A", "B", "bad", 1]])
    with pytest.raises(ValueError, match="row 2"):
        normalize_matches(source, target, strict=True)


def test_non_strict_invalid_rows_are_counted(tmp_path):
    source, target = tmp_path / "matches.csv", tmp_path / "out.csv"
    headers = ["team1", "team 2", "team 1_goals", "team _2goals"]
    write_csv(source, headers, [["A", "B", 2, 1], ["A", "", 1, 0]])
    assert normalize_matches(source, target, strict=False) == {"written": 1, "rejected": 1}


def test_player_event_contract_and_duplicate_guard(tmp_path):
    source, target = tmp_path / "events.csv", tmp_path / "out.csv"
    row = ["demo", 1, "m1", "p1", "Player", "A", 90, 1, 0, 2, 3, 40]
    write_csv(source, PLAYER_MATCH_COLUMNS, [row, row])
    with pytest.raises(ValueError, match="Duplicate player appearance"):
        normalize_player_events(source, target, strict=True)
