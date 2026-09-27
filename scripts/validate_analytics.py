"""Focused Stage 3 validation for analytics that do not need PostgreSQL."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ab_testing import compare_groups  # noqa: E402
from charts import correlation_heatmap, distribution, pacing_trend, radar_comparison  # noqa: E402
from insights import generate_takeaways  # noqa: E402


def main() -> None:
    pace = pd.DataFrame({
        "season": ["demo"] * 6,
        "matchday": [1, 2, 3, 1, 2, 3],
        "team": ["Atlas"] * 3 + ["Boreal"] * 3,
        "points": [1, 4, 7, 0, 1, 2],
        "played": [1, 2, 3, 1, 2, 3],
        "points_per_match": [1.0, 2.0, 7 / 3, 0.0, 0.5, 2 / 3],
        "projected_38_game_points": [38.0, 76.0, 88.7, 0.0, 19.0, 25.3],
    })
    standings = pd.DataFrame({
        "season": ["demo"] * 4,
        "matchday": [2, 2, 3, 3],
        "team": ["Boreal", "Atlas", "Atlas", "Boreal"],
        "standing": [1, 2, 1, 2],
        "wins": [1, 1, 2, 0],
        "goals_for": [2, 3, 6, 2],
        "goal_difference": [1, 1, 4, -2],
        "points": [4, 4, 7, 2],
    })
    anomalies = pd.DataFrame({"metric": ["goals", "goals"]})
    predictions = pd.DataFrame({
        "generated_at": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "pace_status": ["On-Pace", "At-Risk"],
        "model_version": ["demo123", "demo123"],
    })
    comparison = compare_groups(pace, "points_per_match", "team", "Atlas", "Boreal")
    assert comparison["n_a"] == 3 and comparison["n_b"] == 3
    assert {"mean_difference", "welch_t", "p_value", "cohens_d"}.issubset(comparison)
    notes = generate_takeaways(standings, pace, anomalies, predictions)
    assert any("rising" in note for note in notes)
    assert any("Model version" in note for note in notes)
    figures = (
        pacing_trend(pace),
        distribution(pace, "points_per_match", "team"),
        correlation_heatmap(pace, ["points", "points_per_match"]),
        radar_comparison(standings[standings.matchday == 3], "team", ["wins", "goals_for", "points"]),
    )
    assert all(figure.data for figure in figures)
    print("Stage 3 checks pass: comparison statistics, deterministic insights, and four chart types")


if __name__ == "__main__":
    main()
