from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from sqlalchemy import create_engine, text


@pytest.mark.integration
def test_demo_pipeline_outputs_exist_after_dag_run():
    if os.getenv("RUN_DISTRIBUTED_E2E") != "1":
        pytest.skip("set RUN_DISTRIBUTED_E2E=1 after starting Compose and running the DAG")
    if not shutil.which("docker"):
        pytest.skip("Docker CLI is unavailable")
    subprocess.run(["docker", "compose", "ps", "--status", "running"], check=True)
    url = os.getenv("DATABASE_URL", "postgresql+psycopg2://football:football_dev_only@localhost:5432/football")
    with create_engine(url).connect() as connection:
        match_count = connection.execute(text("SELECT COUNT(*) FROM analytics.matches")).scalar_one()
        standings_count = connection.execute(text("SELECT COUNT(*) FROM analytics.weekly_standings")).scalar_one()
        player_count = connection.execute(text("SELECT COUNT(*) FROM analytics.team_player_matchday WHERE player_id IS NOT NULL")).scalar_one()
    assert match_count > 0
    assert standings_count > 0
    assert player_count > 0
