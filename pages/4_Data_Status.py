"""Page 4 — Data Status."""

import sys
import os
import glob

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RAW_DATA_DIR, DATABASE_PATH
from config.events import DEFAULT_EVENT_KEY
from src.utilities.db import get_connection

st.set_page_config(page_title="Data Status", page_icon="🗄️", layout="wide")
st.title("🗄️ Data Status")

if not os.path.exists(DATABASE_PATH):
    st.error("Database not found. Run `python update_data.py` first.")
    st.stop()

conn = get_connection()

n_athletes = conn.execute("SELECT COUNT(*) FROM athletes").fetchone()[0]
n_perf = conn.execute("SELECT COUNT(*) FROM performances").fetchone()[0]
n_flagged = conn.execute("SELECT COUNT(*) FROM performances WHERE is_flagged=1").fetchone()[0]
n_comps = conn.execute("SELECT COUNT(*) FROM competitions").fetchone()[0]
date_range = conn.execute("SELECT MIN(competition_date), MAX(competition_date) FROM performances").fetchone()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Athletes", n_athletes)
c2.metric("Performances", n_perf)
c3.metric("Competitions", n_comps)
c4.metric("Flagged records", n_flagged)

st.write(f"**Date range covered:** {date_range[0]} → {date_range[1]}")

st.subheader("Performances by Event & Gender")
by_event = pd.read_sql_query(
    """SELECT p.event_key, a.gender, COUNT(*) as n_performances, COUNT(DISTINCT p.athlete_id) as n_athletes
       FROM performances p JOIN athletes a ON a.athlete_id = p.athlete_id
       GROUP BY p.event_key, a.gender""",
    conn,
)
st.dataframe(by_event, hide_index=True, use_container_width=True)

st.subheader("Recent Ingestion Log")
log_df = pd.read_sql_query(
    "SELECT run_timestamp, event_key, status, rows_fetched, rows_inserted, message FROM ingestion_log ORDER BY run_timestamp DESC LIMIT 20",
    conn,
)
if log_df.empty:
    st.info("No ingestion runs logged yet.")
else:
    st.dataframe(log_df, hide_index=True, use_container_width=True)
    n_failed_recent = 0

st.subheader("Flagged Records (kept, not deleted — see Section 5.8 of spec)")
flagged_df = pd.read_sql_query(
    """SELECT p.performance_id, a.athlete_name, p.competition, p.competition_date, p.time_seconds, p.flag_reason
       FROM performances p JOIN athletes a ON a.athlete_id = p.athlete_id
       WHERE p.is_flagged = 1 LIMIT 100""",
    conn,
)
if flagged_df.empty:
    st.success("No flagged records.")
else:
    st.dataframe(flagged_df, hide_index=True, use_container_width=True)

st.subheader("Raw Data Files on Disk")
raw_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.csv")))
st.write(f"{len(raw_files)} raw CSV pull(s) stored in `data/raw/` (never overwritten).")
with st.expander("Show file list"):
    for f in raw_files:
        st.text(os.path.basename(f))

st.subheader("Data Source")
st.markdown(
    """
    **Primary source:** [World Aquatics](https://www.worldaquatics.com/swimming/rankings) official rankings export API
    (`api.worldaquatics.com/fina/rankings/swimming/report/csv`) — publicly accessible, no authentication required.

    **Fallback:** if live ingestion fails, place a World Aquatics CSV export into `data/raw/` following the
    `worldaquatics_<event_key>_<gender>_<year>_<timestamp>.csv` naming convention and re-run `python update_data.py`.
    """
)

conn.close()
