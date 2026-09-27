"""Export serving views as Power BI-ready CSV or Parquet datasets."""
from __future__ import annotations

from pathlib import Path

from db import query


TABLES = ("weekly_standings", "rolling_pace", "anomaly_flags")


def export_powerbi_dataset(output_dir: str = "powerbi", file_format: str = "parquet") -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for table in TABLES:
        frame = query(f"SELECT * FROM analytics.{table}")
        path = destination / f"{table}.{file_format}"
        if file_format == "csv":
            frame.to_csv(path, index=False)
        elif file_format == "parquet":
            frame.to_parquet(path, index=False)
        else:
            raise ValueError("file_format must be csv or parquet")
        outputs.append(path)
    return outputs
