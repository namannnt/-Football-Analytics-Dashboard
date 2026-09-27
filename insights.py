"""Deterministic stakeholder takeaways from serving-layer outputs."""
from __future__ import annotations

import pandas as pd


def _standings_movement(standings: pd.DataFrame) -> str | None:
    days = sorted(standings.matchday.dropna().unique())
    if len(days) < 2:
        return None
    previous = standings[standings.matchday == days[-2]][["team", "standing"]].rename(columns={"standing": "previous"})
    current = standings[standings.matchday == days[-1]][["team", "standing"]].rename(columns={"standing": "current"})
    movement = current.merge(previous, on="team")
    movement["places"] = movement.previous - movement.current
    best = movement.sort_values(["places", "team"], ascending=[False, True]).iloc[0]
    if best.places > 0:
        return f"{best.team} made the largest table gain, rising {int(best.places)} place(s) on the latest matchday."
    return "No team moved up the table on the latest matchday."


def _pace_trend(pace: pd.DataFrame) -> str | None:
    changes = []
    for team, history in pace.sort_values("matchday").groupby("team"):
        if len(history) >= 2:
            last_two = history.tail(2).projected_38_game_points.astype(float).tolist()
            changes.append((team, last_two[-1] - last_two[-2]))
    if not changes:
        return None
    team, change = sorted(changes, key=lambda item: (-item[1], item[0]))[0]
    direction = "increased" if change > 0 else "decreased" if change < 0 else "held steady"
    amount = f" by {abs(change):.1f} projected points" if change else ""
    return f"{team}'s pace {direction}{amount} on the latest matchday."


def generate_takeaways(
    standings: pd.DataFrame,
    pace: pd.DataFrame,
    anomalies: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
) -> list[str]:
    notes: list[str] = []
    if not standings.empty:
        latest_day = standings.matchday.max()
        leader = standings[standings.matchday == latest_day].nsmallest(1, "standing").iloc[0]
        notes.append(f"{leader.team} leads after matchday {int(latest_day)} with {int(leader.points)} points.")
        movement = _standings_movement(standings)
        if movement:
            notes.append(movement)
    if not pace.empty:
        trend = _pace_trend(pace)
        if trend:
            notes.append(trend)
    if anomalies is not None and not anomalies.empty:
        counts = anomalies.groupby("metric").size().sort_values(ascending=False)
        notes.append(f"Anomaly review contains {len(anomalies)} flag(s); {counts.index[0]} is the most frequent metric ({int(counts.iloc[0])}).")
    else:
        notes.append("No anomaly flags are currently present.")
    if predictions is not None and not predictions.empty:
        latest = predictions[predictions.generated_at == predictions.generated_at.max()]
        on_pace = int((latest.pace_status == "On-Pace").sum())
        notes.append(f"Model version {latest.model_version.iloc[0]} classifies {on_pace} of {len(latest)} team(s) as On-Pace.")
    else:
        notes.append("No model predictions are available for the selected season.")
    return notes
