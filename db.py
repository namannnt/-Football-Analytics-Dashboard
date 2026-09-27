"""PostgreSQL serving-layer helpers used by the pipeline and Streamlit app."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parent


def database_url() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://football:football@localhost:5432/football",
    )


def engine():
    return create_engine(database_url(), pool_pre_ping=True)


@contextmanager
def transaction():
    with engine().begin() as connection:
        yield connection


def initialize_schema() -> None:
    with transaction() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS analytics"))


def run_sql_file(path: str | Path) -> None:
    sql = Path(path).read_text(encoding="utf-8")
    raw = engine().raw_connection()
    try:
        with raw.cursor() as cursor:
            cursor.execute(sql)
        raw.commit()
    finally:
        raw.close()


def refresh_stored_procedures(**_context) -> None:
    initialize_schema()
    run_sql_file(ROOT / "stored_procedures.sql")
    with transaction() as connection:
        connection.execute(text("CALL analytics.refresh_football_serving()"))


def query(sql: str, parameters: dict | None = None) -> pd.DataFrame:
    with engine().connect() as connection:
        return pd.read_sql_query(text(sql), connection, params=parameters or {})


def read_table_if_exists(qualified_name: str) -> pd.DataFrame:
    if not qualified_name.startswith("analytics.") or not qualified_name.replace("analytics.", "").isidentifier():
        raise ValueError("Only analytics schema tables may be read")
    with engine().connect() as connection:
        exists = connection.execute(text("SELECT to_regclass(:name)"), {"name": qualified_name}).scalar()
        if exists is None:
            return pd.DataFrame()
        return pd.read_sql_query(text(f"SELECT * FROM {qualified_name}"), connection)
