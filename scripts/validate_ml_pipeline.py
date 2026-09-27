"""Focused Stage 2 validation without PostgreSQL or distributed services."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features import build_team_prematch_features  # noqa: E402
from ml_dataset import MODEL_FEATURES, build_ml_dataset, chronological_split  # noqa: E402
from model import load_model, score_on_pace, train_on_pace_model  # noqa: E402


def main() -> None:
    matches = pd.read_csv(ROOT / "data" / "demo_matches.csv")
    features = build_team_prematch_features(matches)
    assert len(features) == len(matches) * 2
    assert set(MODEL_FEATURES).issubset(features.columns)
    assert (features.loc[features.matchday == 1, "elo_rating"] == 1500).all()
    assert features.loc[features.matchday == 1, "form_points"].isna().all()

    changed = matches.copy()
    changed.loc[(changed.season == "2023-demo") & (changed.matchday == 3), "home_goals"] += 20
    changed_features = build_team_prematch_features(changed)
    before_or_at_result = features[(features.season == "2023-demo") & (features.matchday <= 3)][MODEL_FEATURES]
    changed_before_or_at = changed_features[(changed_features.season == "2023-demo") & (changed_features.matchday <= 3)][MODEL_FEATURES]
    pd.testing.assert_frame_equal(before_or_at_result.reset_index(drop=True), changed_before_or_at.reset_index(drop=True))

    dataset = build_ml_dataset(matches)
    training, testing = chronological_split(dataset)
    assert max(training.season.astype(str)) < min(testing.season.astype(str))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.joblib"
        train_on_pace_model(training, str(path))
        scored = score_on_pace(load_model(str(path)), testing)
        assert {"on_pace_probability", "pace_status"}.issubset(scored.columns)
        assert scored.on_pace_probability.between(0, 1).all()
    print("Stage 2 checks pass: pre-match leakage guard, chronological split, artifact round-trip, inference schema")


if __name__ == "__main__":
    main()
