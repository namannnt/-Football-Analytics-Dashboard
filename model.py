"""On-Pace versus At-Risk classifier trained from first-half signals."""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ["form_points", "form_goal_diff", "elo_delta", "matchup_points", "home_advantage"]
CATEGORICAL = ["team", "season"]


def train_on_pace_model(team_matchdays: pd.DataFrame, model_path: str = "artifacts/on_pace.joblib"):
    data = team_matchdays.copy()
    midpoint = data.groupby("season").matchday.transform("max") / 2
    training = data[data.matchday <= midpoint].dropna(subset=["on_pace"])
    if training.empty or training.on_pace.nunique() < 2:
        raise ValueError("Training needs first-half observations from both On-Pace and At-Risk classes")
    transform = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    pipeline = Pipeline([("features", transform), ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    pipeline.fit(training[NUMERIC + CATEGORICAL], training.on_pace.astype(int))
    from pathlib import Path
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return pipeline


def score_on_pace(model, observations: pd.DataFrame) -> pd.DataFrame:
    result = observations.copy()
    result["on_pace_probability"] = model.predict_proba(result[NUMERIC + CATEGORICAL])[:, 1]
    result["pace_status"] = result.on_pace_probability.map(lambda p: "On-Pace" if p >= 0.5 else "At-Risk")
    return result
