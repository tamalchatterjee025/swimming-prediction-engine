"""
Swimming Prediction Engine -- Page 1: Race Predictor

Run with: streamlit run app.py
"""

import sys
import os
import datetime as dt

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.events import EVENTS, DEFAULT_EVENT_KEY, GENDER_CODES
from src.modelling.predict import search_athletes, predict_race, save_prediction_run
from config.settings import N_SIMULATIONS, MODEL_VERSION

st.set_page_config(page_title="Swimming Prediction Engine", page_icon="🏊", layout="wide")

st.title("🏊 Swimming Prediction Engine")
st.caption("Statistical race prediction from official World Aquatics performance data — MVP: 100m Freestyle, Long Course")

with st.sidebar:
    st.header("Race Setup")
    event_key = st.selectbox(
        "Event",
        options=list(EVENTS.keys()),
        format_func=lambda k: f"{EVENTS[k].display_name} ({EVENTS[k].pool_type})",
        index=list(EVENTS.keys()).index(DEFAULT_EVENT_KEY),
    )
    event = EVENTS[event_key]
    gender_label = st.radio("Gender", options=list(GENDER_CODES.keys()), horizontal=True)
    gender_code = GENDER_CODES[gender_label]
    competition_name = st.text_input("Competition (label only, e.g. 'Asian Games 2026 Final')", value="")
    race_date = st.date_input(
        "Race date",
        value=dt.date.today(),
        help=(
            "The model only uses performances strictly BEFORE this date — exactly like a real "
            "pre-race prediction. Defaults to today. Set this to a past date to reconstruct a "
            "historical final (e.g. 2024-07-31 for the Paris 2024 100m Free final) using only the "
            "data that would have been available before that race."
        ),
    )
    st.divider()
    st.caption(f"Model version: `{MODEL_VERSION}`  ·  Simulations per run: {N_SIMULATIONS:,}")

if event.pool_type != "LCM" or event.key != DEFAULT_EVENT_KEY:
    st.warning(
        f"Data for **{event.display_name} ({event.pool_type})** has not been ingested yet in this MVP. "
        f"The event configuration exists (see `config/events.py`) but only "
        f"**{EVENTS[DEFAULT_EVENT_KEY].display_name} (LCM)** has data loaded. "
        f"Run `python update_data.py` after extending ingestion to add more events."
    )

st.subheader("1. Select Finalists")
st.write("Search the athlete database (built from official World Aquatics results) and add up to 8 finalists.")

athletes_df = search_athletes(event_key, gender=gender_code)

if athletes_df.empty:
    st.error("No athlete data found for this event/gender. Run `python update_data.py` first.")
    st.stop()

if "finalists" not in st.session_state:
    st.session_state.finalists = []

col_search, col_current = st.columns([2, 1])

with col_search:
    query = st.text_input("Search athlete by name (optional — narrows the list below)", value="")
    filtered = search_athletes(event_key, gender=gender_code, query=query) if query else athletes_df
    display = filtered.head(500).copy()  # effectively unlimited for current dataset sizes (~150/gender)
    display["label"] = display.apply(
        lambda r: f"{r['athlete_name']} ({r['country']}) — PB {r['personal_best']:.2f}s", axis=1
    )
    options = display["athlete_id"].tolist()
    picked = st.multiselect(
        f"Matching athletes ({len(display)} shown, sorted by PB — you can also type in this box to filter)",
        options=options,
        format_func=lambda aid: display.set_index("athlete_id").loc[aid, "label"],
    )
    if st.button("➕ Add selected to finalists"):
        for aid in picked:
            if aid not in st.session_state.finalists and len(st.session_state.finalists) < 8:
                st.session_state.finalists.append(aid)
        st.rerun()

with col_current:
    st.write("**Current finalists**")
    if not st.session_state.finalists:
        st.info("No finalists added yet.")
    else:
        name_lookup = athletes_df.set_index("athlete_id")["athlete_name"].to_dict()
        for aid in list(st.session_state.finalists):
            c1, c2 = st.columns([4, 1])
            c1.write(name_lookup.get(aid, aid))
            if c2.button("✕", key=f"rm_{aid}"):
                st.session_state.finalists.remove(aid)
                st.rerun()
        if st.button("Clear all"):
            st.session_state.finalists = []
            st.rerun()

st.divider()

if st.button("🏁 Run Prediction", type="primary", disabled=len(st.session_state.finalists) < 2):
    with st.spinner(f"Running {N_SIMULATIONS:,} race simulations..."):
        result = predict_race(event_key, st.session_state.finalists, as_of_date=race_date, competition_name=competition_name)
        st.session_state.last_result = result
        try:
            save_prediction_run(result, event_key, competition_name)
        except Exception as e:
            st.warning(f"Prediction computed but could not be saved to history: {e}")

if len(st.session_state.finalists) < 2:
    st.caption("Add at least 2 finalists to run a prediction.")

if "last_result" in st.session_state:
    result = st.session_state.last_result
    preds = result["predictions"]

    if result["skipped_athletes"]:
        name_lookup = athletes_df.set_index("athlete_id")["athlete_name"].to_dict()
        skipped_names = [name_lookup.get(a, a) for a in result["skipped_athletes"]]
        st.warning(f"Skipped (no usable pre-race history): {', '.join(skipped_names)}")

    st.subheader("2. Race Prediction")
    st.caption(f"Predictions as of {result['as_of_date']} · model {result['model_version']} · {N_SIMULATIONS:,} Monte Carlo simulations")

    table = preds.copy()
    table.insert(0, "Rank", range(1, len(table) + 1))
    display_table = table.rename(columns={
        "athlete_name": "Athlete", "country": "Country", "expected_time": "Expected Time",
        "win_probability": "Gold %", "silver_probability": "Silver %", "bronze_probability": "Bronze %",
        "top3_probability": "Top 3 %", "expected_position": "Expected Position",
    })
    for col in ["Gold %", "Silver %", "Bronze %", "Top 3 %"]:
        display_table[col] = (display_table[col] * 100).round(1)
    display_table["Expected Time"] = display_table["Expected Time"].round(2)
    display_table["Expected Position"] = display_table["Expected Position"].round(2)

    st.dataframe(
        display_table[["Rank", "Athlete", "Country", "Expected Time", "Gold %", "Silver %", "Bronze %", "Top 3 %", "Expected Position"]],
        hide_index=True, use_container_width=True,
    )

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    col_sum1.metric("Sum of Gold %", f"{display_table['Gold %'].sum():.1f}%")
    col_sum2.metric("Sum of Top-3 % / 3", f"{display_table['Top 3 %'].sum()/3:.1f}%")
    col_sum3.metric("Athletes simulated", len(display_table))

    st.subheader("3. Why the model sees each athlete at that level")
    for _, r in preds.iterrows():
        with st.expander(f"{r['athlete_name']} ({r['country']}) — Expected {r['expected_time']:.2f}s"):
            c1, c2 = st.columns(2)
            c1.write(f"**Recent form range:** {r['recent_form_range']}")
            if pd.notna(r['season_best']):
                season_tag = "" if r['season_best_is_current'] else f" (most recent completed season, {r['season_best_label']} — no races yet this season)"
                c1.write(f"**Season best:** {r['season_best']:.2f}s{season_tag}")
            else:
                c1.write("**Season best:** n/a (no races on file in this or the prior season)")
            c2.write(f"**Recent consistency:** {r['consistency_label']}")
            c2.write(f"**Trend:** {r['trend_label']}")
            st.caption(f"Based on {r['n_races_used']} race(s) before the as-of date. Uncertainty (std dev): {r['time_std']:.2f}s")

    st.caption(
        "Probabilities come from a Monte Carlo simulation of the final, not an arbitrary score. "
        "See the Model Validation page for backtested accuracy."
    )
