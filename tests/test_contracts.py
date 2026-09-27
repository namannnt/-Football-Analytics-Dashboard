from __future__ import annotations

from pathlib import Path

from contracts import CURATED_COLUMNS, MATCH_COLUMNS, PLAYER_MATCH_COLUMNS
from spark_layer import CURATED_QUERY, MATCH_QUERY


ROOT = Path(__file__).resolve().parents[1]


def test_hive_raw_contract_columns_are_declared():
    hql = (ROOT / "hive_queries" / "external_raw_tables.hql").read_text(encoding="utf-8").lower()
    assert all(column in hql for column in MATCH_COLUMNS)
    assert all(column in hql for column in PLAYER_MATCH_COLUMNS)


def test_spark_queries_emit_serving_contracts():
    curated = CURATED_QUERY.lower()
    matches = MATCH_QUERY.lower()
    assert all(column in curated for column in CURATED_COLUMNS)
    assert all(column in matches for column in MATCH_COLUMNS)
    assert "hive_db.season_standings" in curated
    assert "hive_db.player_rolling_stats" in curated
    assert "hive_db.match_history" in matches


def test_airflow_task_order_is_explicit():
    dag = (ROOT / "ingest_dag.py").read_text(encoding="utf-8")
    expected = "ingest >> hive_aggregate >> spark_hive_read >> load_postgres >> refresh >> anomalies >> ml_features >> pace_predictions"
    assert expected in dag
