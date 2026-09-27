"""Export serving views as Power BI-ready CSV or Parquet datasets."""
from __future__ import annotations

import argparse
from pathlib import Path

from db import read_table_if_exists


TABLES = ("weekly_standings", "rolling_pace", "anomaly_flags", "team_pace_predictions")


def export_powerbi_dataset(output_dir: str = "powerbi", file_format: str = "parquet") -> list[Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = []
    for table in TABLES:
        frame = read_table_if_exists(f"analytics.{table}")
        if frame.empty:
            continue
        path = destination / f"{table}.{file_format}"
        if file_format == "csv":
            frame.to_csv(path, index=False)
        elif file_format == "parquet":
            frame.to_parquet(path, index=False)
        else:
            raise ValueError("file_format must be csv or parquet")
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="powerbi")
    parser.add_argument("--format", choices=("csv", "parquet"), default="parquet")
    args = parser.parse_args()
    outputs = export_powerbi_dataset(args.output, args.format)
    if not outputs:
        raise SystemExit("No serving tables were available to export")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
