"""Static validation for the local Compose topology when Docker is unavailable."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "postgres", "hdfs", "hive-metastore", "hive-server", "spark-master",
    "spark-worker", "airflow", "dashboard",
}


def main() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose.get("services", {})
    missing = EXPECTED.difference(services)
    if missing:
        raise AssertionError(f"Compose services missing: {sorted(missing)}")
    airflow_environment = services["airflow"]["environment"]
    assert airflow_environment["FOOTBALL_DATASET"] == "${FOOTBALL_DATASET:-demo}"
    assert airflow_environment["SPARK_MASTER_URL"] == "${SPARK_MASTER_URL:-spark://spark-master:7077}"
    assert "change-me" not in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    for path in (
        "infra/hadoop/Dockerfile", "infra/hadoop/entrypoint.sh",
        "infra/hive/conf/hive-site.xml", "infra/spark/conf/spark-defaults.conf",
        "infra/spark/Dockerfile", "infra/airflow/Dockerfile",
    ):
        assert (ROOT / path).is_file(), path
    print("Stage 4 static checks pass: Compose services, wiring, configs, and sample-secret policy")


if __name__ == "__main__":
    main()
