"""Reusable Plotly charts for the Streamlit and export layers."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go


def pacing_trend(data):
    return px.line(data, x="matchday", y="projected_38_game_points", color="team", markers=True)


def distribution(data, metric: str, segment: str):
    return px.histogram(data, x=metric, color=segment, marginal="box", barmode="overlay")


def correlation_heatmap(data, metrics: list[str]):
    return px.imshow(data[metrics].corr(), text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)


def radar_comparison(data, entity: str, metrics: list[str]):
    normalized = data.copy()
    for metric in metrics:
        low, high = normalized[metric].min(), normalized[metric].max()
        normalized[metric] = 0.5 if high == low else (normalized[metric] - low) / (high - low)
    figure = go.Figure()
    for _, row in normalized.iterrows():
        values = [row[metric] for metric in metrics]
        figure.add_trace(go.Scatterpolar(
            r=[*values, values[0]], theta=[*metrics, metrics[0]], fill="toself", name=str(row[entity])
        ))
    figure.update_layout(polar={"radialaxis": {"visible": True, "range": [0, 1]}})
    return figure
