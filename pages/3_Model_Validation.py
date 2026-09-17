"""Page 3 — Model Validation (backtesting, calibration, error metrics)."""

import sys
import os

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.events import EVENTS, DEFAULT_EVENT_KEY, GENDER_CODES
from src.validation.backtest import run_backtest, calibration_curve
from src.validation.calibrate import fit_isotonic_calibrator, MIN_SAMPLES_FOR_CALIBRATION

st.set_page_config(page_title="Model Validation", page_icon="📊", layout="wide")
st.title("📊 Model Validation")
st.caption(
    "Backtested against historical major-championship finals already in the database. "
    "Each prediction uses ONLY performance data available before that final's date — no data leakage."
)

event_key = DEFAULT_EVENT_KEY
event = EVENTS[event_key]
gender_label = st.radio("Gender", options=list(GENDER_CODES.keys()), horizontal=True)
gender_code = GENDER_CODES[gender_label]

n_sims = st.select_slider("Simulations per backtested final (lower = faster)", options=[2000, 5000, 10000, 20000, 50000], value=10000)

if st.button("Run Backtest", type="primary"):
    with st.spinner("Re-running historical finals with pre-race-only data..."):
        result = run_backtest(event_key, gender_code, n_simulations=n_sims)
        st.session_state[f"backtest_{gender_code}"] = result

key = f"backtest_{gender_code}"
if key not in st.session_state:
    st.info("Click **Run Backtest** to evaluate the model against historical finals.")
    st.stop()

result = st.session_state[key]
metrics = result["metrics"]

if metrics.get("n_predictions", 0) == 0:
    st.warning(metrics.get("message", "No backtestable finals found."))
    st.stop()

st.subheader("Aggregate Metrics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Finals evaluated", metrics["n_finals"])
c2.metric("Top-1 hit rate", f"{metrics['top1_hit_rate']*100:.1f}%")
c3.metric("Top-3 hit rate", f"{metrics['top3_hit_rate']*100:.1f}%")
c4.metric("Brier score (win)", metrics["brier_score_win"])
c5.metric("Log loss (win)", metrics["log_loss_win"])
st.metric("Mean absolute error, expected time", f"{metrics['mae_expected_time_seconds']:.3f}s")

st.caption(
    "Brier score and log loss are for the **gold-medal win probability**; lower is better "
    "(0 = perfect, 0.25 = no-skill baseline for a 50/50 coin flip, but for an 8-way field the "
    "no-skill baseline is closer to 0.11 since P(win)≈1/8 for everyone)."
)

st.subheader("Calibration: Predicted Probability vs Actual Outcome Frequency")
calib_df = calibration_curve(result["win_probs"], result["actual_wins"])
if calib_df.empty:
    st.info("Not enough data to build a calibration curve yet.")
else:
    chart_df = calib_df.set_index("bin_mid")[["predicted_mean", "actual_freq"]]
    chart_df.columns = ["Predicted", "Actual"]
    st.line_chart(chart_df)
    st.dataframe(calib_df.rename(columns={
        "bin_mid": "Probability Bin", "predicted_mean": "Mean Predicted", "actual_freq": "Actual Win Frequency", "n": "N",
    }), hide_index=True, use_container_width=True)

if len(result["win_probs"]) >= MIN_SAMPLES_FOR_CALIBRATION:
    iso = fit_isotonic_calibrator(result["win_probs"], result["actual_wins"])
    st.success(f"Sample size ({len(result['win_probs'])}) is large enough — an isotonic calibrator was fit and is available for use.")
else:
    st.info(
        f"Sample size ({len(result['win_probs'])}) is below the {MIN_SAMPLES_FOR_CALIBRATION}-prediction threshold "
        "for fitting a calibration correction. Raw model probabilities are used as-is."
    )

st.subheader("Per-Final Results")
st.dataframe(result["per_final"], hide_index=True, use_container_width=True)
