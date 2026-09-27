from __future__ import annotations

import pandas as pd

from anomaly_flagging import _robust_score


def test_extreme_value_is_flagged_by_robust_score():
    scores = _robust_score(pd.Series([9.0, 10.0, 10.0, 11.0, 12.0, 100.0]))
    assert (scores.iloc[:-1].abs() < 3.5).all()
    assert scores.iloc[-1] > 3.5


def test_zero_mad_returns_finite_zero_scores():
    scores = _robust_score(pd.Series([4.0, 4.0, 4.0, 4.0]))
    assert scores.eq(0).all()
