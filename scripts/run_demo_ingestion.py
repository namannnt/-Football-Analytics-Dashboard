"""Validate and land the bundled synthetic demo data without requiring HDFS."""
from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest_raw_data import ingest_raw_data  # noqa: E402


if __name__ == "__main__":
    os.environ.setdefault("FOOTBALL_RUN_ID", "demo-validation")
    result = ingest_raw_data(dataset="demo", landing_mode="local")
    print(result)
    print(f"Canonical files written under: {result['raw_root']}")
