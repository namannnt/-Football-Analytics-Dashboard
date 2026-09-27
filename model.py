"""Training and inference for the team pace classifier."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_dataset import MODEL_FEATURES, TARGET


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])


def train_on_pace_model(training: pd.DataFrame, model_path: str = "artifacts/on_pace.joblib") -> Pipeline:
    if training.empty or training[TARGET].nunique() < 2:
        raise ValueError("Training needs observations from both On-Pace and At-Risk classes")
    pipeline = create_pipeline()
    pipeline.fit(training[MODEL_FEATURES], training[TARGET].astype(int))
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return pipeline


def score_on_pace(model: Pipeline, observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["on_pace_probability"] = model.predict_proba(result[MODEL_FEATURES])[:, 1]
    result["pace_status"] = result.on_pace_probability.map(lambda value: "On-Pace" if value >= 0.5 else "At-Risk")
    return result


def load_model(model_path: str = "artifacts/on_pace.joblib") -> Pipeline:
    return joblib.load(model_path)


def save_metadata(metadata: dict, path: str = "artifacts/on_pace.metadata.json") -> None:
    payload = {**metadata, "created_at": datetime.now(timezone.utc).isoformat()}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
