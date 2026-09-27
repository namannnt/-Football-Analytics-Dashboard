from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def demo_matches() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "demo_matches.csv")
