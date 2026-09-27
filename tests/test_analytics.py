from __future__ import annotations

import pandas as pd

from ab_testing import compare_groups
from insights import generate_takeaways


def test_comparison_and_insight_helpers():
    pace = pd.DataFrame({
        "matchday": [1, 2, 3, 1, 2, 3],
        "team": ["A", "A", "A", "B", "B", "B"],
        "points_per_match": [1.0, 1.5, 2.0, 0.5, 0.8, 1.0],
        "projected_38_game_points": [38, 57, 76, 19, 30.4, 38],
    })
    result = compare_groups(pace, "points_per_match", "team", "A", "B")
    assert result["n_a"] == result["n_b"] == 3
    standings = pd.DataFrame({
        "matchday": [1, 1, 2, 2], "team": ["B", "A", "A", "B"],
        "standing": [1, 2, 1, 2], "points": [3, 0, 3, 3],
    })
    notes = generate_takeaways(standings, pace)
    assert any("rising" in note for note in notes)
    assert any("No model predictions" in note for note in notes)
