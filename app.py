"""Streamlit serving layer for the curated football pipeline."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ab_testing import compare_groups
from charts import correlation_heatmap, distribution, pacing_trend, radar_comparison
from db import query, read_table_if_exists
from excel_export import build_excel
from export import build_pdf
from insights import generate_takeaways
from pptx_export import build_pptx


st.set_page_config(page_title="Football Analytics", layout="wide")
st.title("Football Analytics Dashboard")
st.caption("Hive batch warehouse → Spark curation → PostgreSQL serving")


@st.cache_data(ttl=300)
def load_data():
    standings = query("SELECT * FROM analytics.weekly_standings ORDER BY season, matchday, standing")
    pace = query("SELECT * FROM analytics.rolling_pace ORDER BY season, matchday, team")
    anomalies = read_table_if_exists("analytics.anomaly_flags")
    predictions = read_table_if_exists("analytics.team_pace_predictions")
    return standings, pace, anomalies, predictions


try:
    standings, pace, anomalies, predictions = load_data()
except Exception as error:
    st.error("The PostgreSQL serving layer is not ready. Run the Airflow pipeline and check DATABASE_URL.")
    st.exception(error)
    st.stop()

if standings.empty:
    st.info("No curated matchday data is available yet.")
    st.stop()

season = st.sidebar.selectbox("Season", sorted(standings.season.unique(), reverse=True))
season_standings = standings[standings.season == season]
season_pace = pace[pace.season == season]
season_anomalies = anomalies[anomalies.season == season] if not anomalies.empty else anomalies
season_predictions = predictions[predictions.season == season] if not predictions.empty else predictions
latest_day = int(season_standings.matchday.max())
latest_table = season_standings[season_standings.matchday == latest_day].sort_values("standing")
takeaways = generate_takeaways(season_standings, season_pace, season_anomalies, season_predictions)

overview, trends, statistics, exports = st.tabs(("Overview", "Visual analysis", "Statistical comparison", "Exports"))

with overview:
    st.subheader(f"Standings after matchday {latest_day}")
    st.dataframe(latest_table, use_container_width=True, hide_index=True)
    st.subheader("First-half pace predictions")
    if season_predictions.empty:
        st.info("No trained model predictions are available. Train a model manually, then rerun the scoring task.")
    else:
        st.dataframe(
            season_predictions.sort_values("on_pace_probability", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    st.subheader("Stakeholder takeaways")
    for takeaway in takeaways:
        st.write(f"• {takeaway}")

with trends:
    teams = sorted(season_pace.team.unique())
    selected_teams = st.multiselect("Teams in pacing trend", teams, default=teams[: min(4, len(teams))])
    if selected_teams:
        st.plotly_chart(pacing_trend(season_pace[season_pace.team.isin(selected_teams)]), use_container_width=True)
    latest_teams = sorted(latest_table.team.unique())
    radar_teams = st.multiselect(
        "Teams in latest radar", latest_teams, default=latest_teams[:2], max_selections=4
    )
    radar_metrics = ["wins", "goals_for", "goal_difference", "points"]
    if len(radar_teams) >= 2:
        st.plotly_chart(radar_comparison(latest_table[latest_table.team.isin(radar_teams)], "team", radar_metrics), use_container_width=True)
    numeric = ["points", "points_per_match", "projected_38_game_points"]
    distribution_metric = st.selectbox("Distribution metric", numeric)
    st.plotly_chart(distribution(season_pace, distribution_metric, "team"), use_container_width=True)
    correlation_metrics = ["matchday", "points", "played", "points_per_match", "projected_38_game_points"]
    st.plotly_chart(correlation_heatmap(season_pace, correlation_metrics), use_container_width=True)

with statistics:
    st.write("Welch's t-test compares repeated matchday observations. It does not assume equal variance.")
    comparison_dimension = st.selectbox("Comparison dimension", ["team", "season"])
    comparison_data = season_pace if comparison_dimension == "team" else pace
    comparison_metric = st.selectbox("Comparison metric", ["points_per_match", "projected_38_game_points"])
    groups = sorted(comparison_data[comparison_dimension].dropna().unique())
    first = st.selectbox("First group", groups, index=0)
    second_options = [group for group in groups if group != first]
    second = st.selectbox("Second group", second_options, index=0) if second_options else None
    if second is not None:
        try:
            result = compare_groups(comparison_data, comparison_metric, comparison_dimension, first, second)
            st.dataframe(pd.DataFrame([result]), use_container_width=True, hide_index=True)
        except ValueError as error:
            st.info(str(error))
    st.caption("Statistical significance and practical significance answer different questions; interpret the p-value with sample sizes and Cohen's d.")

with exports:
    columns = st.columns(3)
    columns[0].download_button("Download PDF", build_pdf(f"Football Analytics — {season}", takeaways), "football_brief.pdf")
    columns[1].download_button(
        "Download Excel",
        build_excel({"Standings": season_standings, "Pace": season_pace, "Anomalies": season_anomalies, "Predictions": season_predictions}),
        "football_analytics.xlsx",
    )
    columns[2].download_button("Download PowerPoint", build_pptx(f"Football Analytics — {season}", takeaways), "football_analytics.pptx")
