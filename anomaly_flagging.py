"""Flag unusual team pace and player production in the PostgreSQL serving layer."""
from __future__ import annotations

import numpy as np

from db import query, transaction


def _robust_score(values):
    clean = values.dropna()
    median = clean.median()
    mad = np.median(np.abs(clean - median))
    if not mad or np.isnan(mad):
        return values * 0.0
    return 0.6745 * (values - median) / mad


def run_anomaly_flags(**_context) -> int:
    frame = query(
        """
        SELECT season, matchday, match_id, player_id, team,
               team_points, goals_per90_last5, assists_per90_last5
        FROM analytics.team_player_matchday
        """
    )
    if frame.empty:
        return 0

    metrics = ("team_points", "goals_per90_last5", "assists_per90_last5")
    records = []
    for metric in metrics:
        sample = (
            frame.drop_duplicates(["season", "matchday", "team"])
            if metric == "team_points"
            else frame.dropna(subset=["player_id"])
        ).copy()
        scores = sample.groupby(["season", "matchday"])[metric].transform(_robust_score)
        unusual = scores.abs() >= 3.5
        for index in sample.index[unusual]:
            row = sample.loc[index]
            records.append({
                "season": row["season"], "matchday": int(row["matchday"]),
                "match_id": row["match_id"],
                "player_id": None if metric == "team_points" else row["player_id"],
                "team": row["team"], "metric": metric,
                "observed_value": float(row[metric]), "robust_z_score": float(scores[index]),
            })

    with transaction() as connection:
        connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS analytics")
        connection.exec_driver_sql(
            """CREATE TABLE IF NOT EXISTS analytics.anomaly_flags (
                season TEXT, matchday INTEGER, match_id TEXT, player_id TEXT, team TEXT,
                metric TEXT, observed_value DOUBLE PRECISION, robust_z_score DOUBLE PRECISION,
                flagged_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"""
        )
        connection.exec_driver_sql("TRUNCATE TABLE analytics.anomaly_flags")
        if records:
            connection.exec_driver_sql(
                """INSERT INTO analytics.anomaly_flags
                   (season, matchday, match_id, player_id, team, metric, observed_value, robust_z_score)
                   VALUES (%(season)s, %(matchday)s, %(match_id)s, %(player_id)s, %(team)s,
                           %(metric)s, %(observed_value)s, %(robust_z_score)s)""",
                records,
            )
    return len(records)


if __name__ == "__main__":
    print({"flags_written": run_anomaly_flags()})
