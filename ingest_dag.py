"""Airflow orchestration for the raw-to-serving football analytics pipeline."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from anomaly_flagging import run_anomaly_flags
from db import refresh_stored_procedures
from ingest_raw_data import ingest_raw_data


ROOT = Path(__file__).resolve().parent
HDFS_ROOT = os.getenv("FOOTBALL_HDFS_RAW_ROOT", "hdfs:///football/raw").rstrip("/")
SPARK_PACKAGES = os.getenv("FOOTBALL_SPARK_PACKAGES", "org.postgresql:postgresql:42.7.4")


default_args = {
    "owner": "analytics-engineering",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="football_raw_to_serving",
    description="Hive batch warehouse to Spark curation to PostgreSQL serving",
    start_date=datetime(2024, 1, 1),
    schedule=os.getenv("FOOTBALL_DAG_SCHEDULE", "0 4 * * 1"),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["football", "hive", "spark", "postgres"],
) as dag:
    ingest = PythonOperator(
        task_id="ingest_raw_data",
        python_callable=ingest_raw_data,
    )

    hive_aggregate = BashOperator(
        task_id="hive_batch_aggregate",
        bash_command=(
            f"cd '{ROOT}' && "
            "beeline -u \"${HIVE_JDBC_URL}\" "
            f"--hivevar football_matches_path='{HDFS_ROOT}/current/matches' "
            f"--hivevar football_players_path='{HDFS_ROOT}/current/players' "
            f"--hivevar football_player_stats_path='{HDFS_ROOT}/current/player_match_stats' "
            "-f hive_queries.hql"
        ),
        env={"HIVE_JDBC_URL": os.getenv("HIVE_JDBC_URL", "jdbc:hive2://hive-server:10000/default")},
        append_env=True,
    )

    spark_hive_read = BashOperator(
        task_id="spark_hive_read",
        bash_command=(
            f"spark-submit --packages '{SPARK_PACKAGES}' '{ROOT / 'spark_layer.py'}' "
            "--mode transform"
        ),
    )

    load_postgres = BashOperator(
        task_id="load_to_postgres",
        bash_command=(
            f"spark-submit --packages '{SPARK_PACKAGES}' '{ROOT / 'spark_layer.py'}' "
            "--mode load-postgres"
        ),
    )

    refresh = PythonOperator(
        task_id="refresh_stored_procedures",
        python_callable=refresh_stored_procedures,
    )

    anomalies = PythonOperator(
        task_id="run_anomaly_flags",
        python_callable=run_anomaly_flags,
    )

    ingest >> hive_aggregate >> spark_hive_read >> load_postgres >> refresh >> anomalies
