from __future__ import annotations

import pandas as pd
import pytest

from features import build_team_prematch_features


def test_first_fixture_has_no_history_and_initial_elo(demo_matches):
    features = build_team_prematch_features(demo_matches)
    first = features[features.matchday == 1]
    assert first.form_points.isna().all()
    assert first.form_goal_diff.isna().all()
    assert (first.elo_rating == 1500).all()
    assert first.matchup_points.isna().all()


def test_future_result_cannot_change_earlier_features(demo_matches):
    baseline = build_team_prematch_features(demo_matches)
    changed = demo_matches.copy()
    changed.loc[(changed.season == "2023-demo") & (changed.matchday == 6), "home_goals"] += 50
    candidate = build_team_prematch_features(changed)
    columns = ["form_points", "form_goal_diff", "elo_rating", "elo_delta", "matchup_points"]
    earlier = (baseline.season == "2023-demo") & (baseline.matchday <= 6)
    pd.testing.assert_frame_equal(
        baseline.loc[earlier, columns].reset_index(drop=True),
        candidate.loc[earlier, columns].reset_index(drop=True),
    )


def test_form_uses_prior_match_only(demo_matches):
    features = build_team_prematch_features(demo_matches)
    atlas = features[(features.season == "2023-demo") & (features.team == "Atlas FC")]
    second = atlas.sort_values("matchday").iloc[1]
    first_match = demo_matches[(demo_matches.season == "2023-demo") & (demo_matches.matchday == 1)]
    atlas_match = first_match[(first_match.home_team == "Atlas FC") | (first_match.away_team == "Atlas FC")].iloc[0]
    expected = 3 if atlas_match.home_goals > atlas_match.away_goals else 1 if atlas_match.home_goals == atlas_match.away_goals else 0
    assert second.form_points == expected


def test_matchup_history_excludes_current_fixture(demo_matches):
    features = build_team_prematch_features(demo_matches)
    pair = features[(features.season == "2023-demo") & (features.team == "Atlas FC") & (features.opponent == "Boreal United")]
    pair = pair.sort_values("matchday")
    assert pd.isna(pair.iloc[0].matchup_points)
    assert not pd.isna(pair.iloc[1].matchup_points)


def test_expected_season_length_controls_live_progress(demo_matches):
    features = build_team_prematch_features(demo_matches, expected_matchdays=20)
    assert features.loc[features.matchday == 5, "season_progress"].eq(0.2).all()
    with pytest.raises(ValueError, match="expected_matchdays"):
        build_team_prematch_features(demo_matches, expected_matchdays=9)
