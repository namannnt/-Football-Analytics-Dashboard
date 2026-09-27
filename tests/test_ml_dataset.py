from __future__ import annotations

from ml_dataset import (
    IDENTIFIERS,
    MODEL_FEATURES,
    TARGET,
    build_ml_dataset,
    chronological_split,
    final_season_labels,
)
from pace_inference import latest_first_half_observations


def test_team_level_contract_has_two_rows_per_fixture(demo_matches):
    dataset = build_ml_dataset(demo_matches)
    assert len(dataset) == len(demo_matches) * 2
    assert set([*IDENTIFIERS, *MODEL_FEATURES, TARGET]).issubset(dataset.columns)
    assert dataset.groupby(["season", "match_id"]).size().eq(2).all()


def test_label_is_top_half_of_final_table(demo_matches):
    labels = final_season_labels(demo_matches)
    assert labels.groupby("season")[TARGET].sum().eq(2).all()
    assert labels.loc[labels[TARGET] == 1, "final_rank"].le(2).all()
    assert labels.loc[labels[TARGET] == 0, "final_rank"].gt(2).all()


def test_chronological_split_holds_out_newest_season(demo_matches):
    training, testing = chronological_split(build_ml_dataset(demo_matches))
    assert set(training.season) == {"2023-demo", "2024-demo"}
    assert set(testing.season) == {"2025-demo"}
    assert training.matchday.max() == 5
    assert testing.matchday.max() == 5


def test_chronological_split_normalizes_numeric_seasons(demo_matches):
    numeric = demo_matches.copy()
    numeric["season"] = numeric["season"].str[:4].astype(int)
    training, testing = chronological_split(build_ml_dataset(numeric))
    assert set(training.season) == {2023, 2024}
    assert set(testing.season) == {2025}
    assert not set(training.season).intersection(testing.season)


def test_latest_inference_normalizes_numeric_seasons(demo_matches):
    numeric = demo_matches.copy()
    numeric["season"] = numeric["season"].str[:4].astype(int)
    features = build_ml_dataset(numeric)
    latest = latest_first_half_observations(features)
    assert set(latest.season) == {2025}
    assert latest.matchday.max() == 5


def test_latest_inference_preserves_descriptive_and_demo_seasons(demo_matches):
    features = build_ml_dataset(demo_matches)
    latest = latest_first_half_observations(features)
    assert set(latest.season) == {"2025-demo"}
