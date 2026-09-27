"""Spark transform between Hive warehouse tables and PostgreSQL serving tables."""
from __future__ import annotations

import argparse
import os


CURATED_QUERY = """
SELECT
    s.season,
    s.matchday,
    p.match_id,
    p.player_id,
    p.player_name,
    s.team,
    s.played AS team_played,
    s.wins AS team_wins,
    s.draws AS team_draws,
    s.losses AS team_losses,
    s.goals_for AS team_goals_for,
    s.goals_against AS team_goals_against,
    s.goal_difference AS team_goal_difference,
    s.points AS team_points,
    p.minutes_last_5,
    p.goals_per90_last5,
    p.assists_per90_last5,
    p.shots_per90_last5,
    p.tackles_per90_last5,
    p.passes_completed_per90_last5
FROM hive_db.season_standings s
LEFT JOIN hive_db.player_rolling_stats p
  ON p.season = s.season
 AND p.matchday = s.matchday
 AND p.team = s.team
"""

MATCH_QUERY = """
SELECT
    season,
    CAST(matchday AS INT) AS matchday,
    match_id,
    kickoff_ts,
    home_team,
    away_team,
    CAST(home_goals AS INT) AS home_goals,
    CAST(away_goals AS INT) AS away_goals
FROM hive_db.match_history
"""


def spark_session():
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName("football-hive-curation")
        .enableHiveSupport()
        .getOrCreate()
    )


def build_curated(output_path: str | None = None) -> str:
    """Read Hive aggregates, perform the cross-grain join, and stage Parquet."""
    output_path = output_path or os.getenv(
        "FOOTBALL_CURATED_PATH", "hdfs:///football/curated"
    )
    spark = spark_session()
    try:
        curated = spark.sql(CURATED_QUERY).dropDuplicates(
            ["season", "matchday", "team", "match_id", "player_id"]
        )
        curated.write.mode("overwrite").partitionBy("season").parquet(
            f"{output_path.rstrip('/')}/team_player_matchday"
        )
        spark.sql(MATCH_QUERY).dropDuplicates(["season", "match_id"]).write.mode(
            "overwrite"
        ).partitionBy("season").parquet(f"{output_path.rstrip('/')}/matches")
    finally:
        spark.stop()
    return output_path


def load_to_postgres(input_path: str | None = None) -> None:
    """Publish the staged result through Spark JDBC into the serving layer."""
    input_path = input_path or os.getenv(
        "FOOTBALL_CURATED_PATH", "hdfs:///football/curated"
    )
    from db import initialize_schema

    initialize_schema()
    jdbc_url = os.environ["POSTGRES_JDBC_URL"]
    spark = spark_session()
    try:
        tables = {
            "team_player_matchday": spark.read.parquet(f"{input_path.rstrip('/')}/team_player_matchday"),
            "matches": spark.read.parquet(f"{input_path.rstrip('/')}/matches"),
        }
        for table, frame in tables.items():
            (
                frame.write.format("jdbc")
                .option("url", jdbc_url)
                .option("dbtable", f"analytics.{table}")
                .option("user", os.environ["POSTGRES_USER"])
                .option("password", os.environ["POSTGRES_PASSWORD"])
                .option("driver", "org.postgresql.Driver")
                .mode("overwrite")
                .save()
            )
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("transform", "load-postgres"), required=True)
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    if args.mode == "transform":
        build_curated(args.path)
    else:
        load_to_postgres(args.path)


if __name__ == "__main__":
    main()
