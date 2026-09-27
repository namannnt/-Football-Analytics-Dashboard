"""Score latest available team observations and publish PostgreSQL predictions."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from db import engine, query
from ml_dataset import MODEL_FEATURES, normalize_season
from model import load_model, score_on_pace


PREDICTION_COLUMNS = [
    "season", "matchday", "team", "on_pace_probability", "pace_status", "model_version", "generated_at"
]
ROOT = Path(__file__).resolve().parent


def latest_first_half_observations(features: pd.DataFrame) -> pd.DataFrame:
    """Select the latest season's first-half rows using normalized season keys."""
    keyed = features.assign(_season_key=features.season.map(normalize_season))
    season_dates = keyed.groupby("_season_key").kickoff_ts.min().sort_values()
    if season_dates.empty:
        return keyed.drop(columns="_season_key")
    latest_season = season_dates.index[-1]
    eligible = keyed[keyed._season_key == latest_season]
    eligible = eligible[eligible.season_progress < 0.5]
    return eligible.drop(columns="_season_key")


def _ensure_table() -> None:
    with engine().begin() as connection:
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS analytics.team_pace_predictions (
                season TEXT NOT NULL,
                matchday INTEGER NOT NULL,
                team TEXT NOT NULL,
                on_pace_probability DOUBLE PRECISION NOT NULL,
                pace_status TEXT NOT NULL,
                model_version TEXT NOT NULL,
                generated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (season, matchday, team, model_version)
            )
        """))


def score_pace_model(**_context) -> dict[str, object]:
    model_path = Path(os.getenv("FOOTBALL_MODEL_PATH", ROOT / "artifacts" / "on_pace.joblib"))
    metadata_path = Path(
        os.getenv("FOOTBALL_MODEL_METADATA_PATH", ROOT / "artifacts" / "on_pace.metadata.json")
    )
    _ensure_table()
    if not model_path.is_file() or not metadata_path.is_file():
        return {"status": "skipped", "reason": "trained model artifact is not available"}
    features = query("SELECT * FROM analytics.team_prematch_features")
    if features.empty:
        return {"status": "skipped", "reason": "no feature rows are available"}
    eligible = latest_first_half_observations(features)
    latest = eligible.sort_values(["matchday", "kickoff_ts", "match_id"]).groupby("team", as_index=False).tail(1)
    if latest.empty:
        return {"status": "skipped", "reason": "no first-half observations are available"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("features") != MODEL_FEATURES:
        raise ValueError("Model metadata feature contract does not match the current application")
    scored = score_on_pace(load_model(str(model_path)), latest)
    predictions = scored[["season", "matchday", "team", "on_pace_probability", "pace_status"]].copy()
    predictions["model_version"] = metadata["model_version"]
    predictions["generated_at"] = pd.Timestamp.now(tz="UTC")
    with engine().begin() as connection:
        connection.execute(text("TRUNCATE TABLE analytics.team_pace_predictions"))
        predictions.to_sql("team_pace_predictions", connection, schema="analytics", if_exists="append", index=False)
    return {"status": "scored", "rows": len(predictions), "model_version": metadata["model_version"]}


if __name__ == "__main__":
    print(score_pace_model())
