"""Streamlit serving layer for the curated football pipeline."""
from __future__ import annotations

import streamlit as st

from charts import pacing_trend
from db import query
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
    anomalies = query("SELECT * FROM analytics.anomaly_flags ORDER BY flagged_at DESC")
    return standings, pace, anomalies


try:
    standings, pace, anomalies = load_data()
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
latest_day = int(season_standings.matchday.max())

st.subheader(f"Standings after matchday {latest_day}")
st.dataframe(
    season_standings[season_standings.matchday == latest_day].sort_values("standing"),
    use_container_width=True,
    hide_index=True,
)
st.plotly_chart(pacing_trend(season_pace), use_container_width=True)

takeaways = generate_takeaways(season_standings, season_pace, anomalies)
st.subheader("Stakeholder takeaways")
for takeaway in takeaways:
    st.write(f"• {takeaway}")

columns = st.columns(3)
columns[0].download_button("Download PDF", build_pdf(f"Football Analytics — {season}", takeaways), "football_brief.pdf")
columns[1].download_button(
    "Download Excel", build_excel({"Standings": season_standings, "Pace": season_pace, "Anomalies": anomalies}),
    "football_analytics.xlsx",
)
columns[2].download_button("Download PowerPoint", build_pptx(f"Football Analytics — {season}", takeaways), "football_analytics.pptx")
