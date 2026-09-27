"""Statistical comparisons for team, segment, and player metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compare_groups(data: pd.DataFrame, metric: str, group: str, a: str, b: str) -> dict[str, float | str | int]:
    left = data.loc[data[group] == a, metric].dropna().astype(float)
    right = data.loc[data[group] == b, metric].dropna().astype(float)
    if min(len(left), len(right)) < 2:
        raise ValueError("Each comparison group needs at least two observations")
    statistic, p_value = stats.ttest_ind(left, right, equal_var=False)
    pooled = np.sqrt(((left.var(ddof=1) + right.var(ddof=1)) / 2))
    effect = (left.mean() - right.mean()) / pooled if pooled else 0.0
    return {
        "group_a": a, "group_b": b, "n_a": len(left), "n_b": len(right),
        "mean_a": left.mean(), "mean_b": right.mean(), "mean_difference": left.mean() - right.mean(),
        "welch_t": statistic, "p_value": p_value, "cohens_d": effect,
    }
