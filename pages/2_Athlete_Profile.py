"""Page 2 — Athlete Profile."""

import sys
import os
import datetime as dt

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.events import EVENTS, DEFAULT_EVENT_KEY, GENDER_CODES
from src.modelling.predict import search_athletes, load_event_performances
from src.modelling.expected_time import compute_expected_time, fit_aqua_points_time_relationship

st.set_page_config(page_title="Athlete Profile", page_icon="👤", layout="wide")
st.title("👤 Athlete Profile")

event_key = DEFAULT_EVENT_KEY
event = EVENTS[event_key]

col1, col2 = st.columns([1, 2])
with col1:
    gender_label = st.radio("Gender", options=list(GENDER_CODES.keys()), horizontal=True)
    gender_code = GENDER_CODES[gender_label]

athletes_df = search_athletes(event_key, gender=gender_code)
if athletes_df.empty:
    st.error("No data loaded yet. Run `python update_data.py`.")
    st.stop()

query = st.text_input("Search athlete")
filtered = search_athletes(event_key, gender=gender_code, query=query) if query else athletes_df
filtered = filtered.head(50)
label_map = {
    row.athlete_id: f"{row.athlete_name} ({row.country}) — PB {row.personal_best:.2f}s"
    for row in filtered.itertuples()
}
athlete_id = st.selectbox("Athlete", options=filtered["athlete_id"].tolist(), format_func=lambda a: label_map.get(a, a))

if not athlete_id:
    st.stop()

perf_df = load_event_performances(event_key, gender=gender_code)
ath_perf = perf_df[perf_df["athlete_id"] == athlete_id].sort_values("competition_date", ascending=False)
row0 = ath_perf.iloc[0]

st.subheader(f"{row0['athlete_name']} — {row0['country']}")

aqua_fit = fit_aqua_points_time_relationship(perf_df)
today = dt.date.today() + dt.timedelta(days=1)
feats_result = compute_expected_time(perf_df, athlete_id, today, aqua_fit=aqua_fit)
feats = feats_result["features"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Personal Best", f"{feats['personal_best']:.2f}s" if feats['personal_best'] else "n/a")
season_label = f"Season Best ({feats['season_best_label']})" if feats.get("season_best_label") else "Season Best"
if not feats.get("season_best_is_current", True) and feats["season_best"]:
    season_label += " *"
m2.metric(season_label, f"{feats['season_best']:.2f}s" if feats['season_best'] else "n/a")
if feats["season_best"] and not feats.get("season_best_is_current", True):
    st.caption("* No races yet this season — showing the most recently completed season's best instead.")
m3.metric("Recent Weighted Time", f"{feats['recent_weighted_time']:.2f}s" if feats['recent_weighted_time'] else "n/a")
m4.metric("Latest AQUA Points (avg recent)", f"{feats['aqua_points_avg']:.0f}" if feats['aqua_points_avg'] else "n/a")

m5, m6, m7 = st.columns(3)
consistency = feats["consistency_std"]
m5.metric("Consistency (std dev, last 5)", f"{consistency:.2f}s" if consistency is not None else "n/a")
trend = feats["trend_slope"]
trend_str = "Improving" if trend and trend < -0.03 else ("Slowing" if trend and trend > 0.03 else "Stable") if trend is not None else "n/a"
m6.metric("Recent Trend", trend_str)
m7.metric("Races on file", feats["n_races"])

st.subheader("Recent Performances")
show_cols = ["competition_date", "competition", "competition_level", "time_seconds", "aqua_points", "rank_at_meet", "source_url"]
st.dataframe(
    ath_perf[show_cols].rename(columns={
        "competition_date": "Date", "competition": "Competition", "competition_level": "Level",
        "time_seconds": "Time (s)", "aqua_points": "AQUA Points", "rank_at_meet": "Rank at Meet",
        "source_url": "Source",
    }),
    hide_index=True, use_container_width=True,
)

st.subheader("Time Progression")
chart_df = ath_perf.sort_values("competition_date")[["competition_date", "time_seconds"]].set_index("competition_date")
st.line_chart(chart_df)

st.caption(f"Source: World Aquatics official rankings export. {len(ath_perf)} performance(s) on file for this event.")
