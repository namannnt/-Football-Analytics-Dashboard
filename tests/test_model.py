from __future__ import annotations

from ml_dataset import MODEL_FEATURES, build_ml_dataset, chronological_split
from model import load_model, score_on_pace, train_on_pace_model


def test_training_imputes_missing_values_and_round_trips(demo_matches, tmp_path):
    training, testing = chronological_split(build_ml_dataset(demo_matches))
    assert training[MODEL_FEATURES].isna().any().any()
    path = tmp_path / "pace.joblib"
    train_on_pace_model(training, str(path))
    scored = score_on_pace(load_model(str(path)), testing)
    assert {"on_pace_probability", "pace_status"}.issubset(scored.columns)
    assert scored.on_pace_probability.between(0, 1).all()
    assert set(scored.pace_status).issubset({"On-Pace", "At-Risk"})
