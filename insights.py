"""Deterministic stakeholder takeaways from curated outputs."""
from __future__ import annotations


def generate_takeaways(standings, pace, anomalies=None) -> list[str]:
    notes = []
    if not standings.empty:
        latest_matchday = standings.matchday.max()
        leader = standings[standings.matchday == latest_matchday].nsmallest(1, "standing").iloc[0]
        notes.append(f"{leader.team} leads the latest observed table with {int(leader.points)} points.")
    if not pace.empty:
        latest = pace.sort_values("matchday").groupby("team").tail(1)
        fastest = latest.nlargest(1, "projected_38_game_points").iloc[0]
        notes.append(f"{fastest.team} has the strongest current pace at {fastest.projected_38_game_points:.1f} projected points.")
    if anomalies is not None and not anomalies.empty:
        notes.append(f"The latest pipeline run flagged {len(anomalies)} unusual observations for review.")
    return notes or ["No curated observations are available yet."]
