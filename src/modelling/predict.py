"""
End-to-end prediction pipeline: given an event + a list of finalist
athlete_ids + an as_of_date, produce the full race prediction table.

This is the single function both the Streamlit app and the backtester call,
so live predictions and backtests always run through identical logic.
"""

import sys
import os
import datetime as dt
import uuid

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import MODEL_VERSION
from src.modelling.expected_time import compute_expected_time, fit_aqua_points_time_relationship
from src.simulation.monte_carlo import simulate_race
from src.utilities.db import get_connection
from src.cleaning.clean import classify_meet_level


def load_event_performances(event_key: str, gender: str = None) -> pd.DataFrame:
    conn = get_connection()
    q = """
        SELECT p.*, a.athlete_name, a.country, a.gender
        FROM performances p
        JOIN athletes a ON a.athlete_id = p.athlete_id
        WHERE p.event_key = ?
    """
    params = [event_key]
    if gender:
        q += " AND a.gender = ?"
        params.append(gender)
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    df["competition_date"] = pd.to_datetime(df["competition_date"]).dt.date
    df["competition_level"] = df["competition"].apply(classify_meet_level)
    return df


def search_athletes(event_key: str, gender: str = None, query: str = "") -> pd.DataFrame:
    conn = get_connection()
    q = """
        SELECT DISTINCT a.athlete_id, a.athlete_name, a.country, a.gender,
               MIN(p.time_seconds) as personal_best
        FROM performances p JOIN athletes a ON a.athlete_id = p.athlete_id
        WHERE p.event_key = ?
    """
    params = [event_key]
    if gender:
        q += " AND a.gender = ?"
        params.append(gender)
    if query:
        q += " AND a.athlete_name LIKE ?"
        params.append(f"%{query}%")
    q += " GROUP BY a.athlete_id ORDER BY personal_best ASC"
    df = pd.read_sql_query(q, conn, params=params)
    conn.close()
    return df


def predict_race(event_key: str, athlete_ids: list, as_of_date: dt.date = None,
                  competition_name: str = "", n_simulations: int = None) -> dict:
    """
    Returns {"predictions": DataFrame, "as_of_date": date, "run_id": str}
    predictions df columns include everything needed by the UI plus the
    per-athlete "why" feature explanation fields.
    """
    if as_of_date is None:
        as_of_date = dt.date.today() + dt.timedelta(days=1)  # "as of today" -> include today's races

    perf_df = load_event_performances(event_key)
    aqua_fit = fit_aqua_points_time_relationship(perf_df)

    athlete_predictions = []
    explanations = {}
    for aid in athlete_ids:
        res = compute_expected_time(perf_df, aid, as_of_date, aqua_fit=aqua_fit)
        name_rows = perf_df[perf_df["athlete_id"] == aid]
        name = name_rows["athlete_name"].iloc[0] if not name_rows.empty else aid
        country = name_rows["country"].iloc[0] if not name_rows.empty else ""

        if res["expected_time"] is None:
            # No usable history before as_of_date -- cannot predict this athlete.
            continue

        athlete_predictions.append({
            "athlete_id": aid, "athlete_name": name, "country": country,
            "expected_time": res["expected_time"], "time_std": res["time_std"],
        })
        explanations[aid] = res

    sim_kwargs = {}
    if n_simulations:
        sim_kwargs["n_simulations"] = n_simulations
    sim_df = simulate_race(athlete_predictions, **sim_kwargs)

    if not sim_df.empty:
        sim_df["recent_form_range"] = sim_df["athlete_id"].apply(
            lambda aid: _format_recent_range(explanations[aid]["features"])
        )
        sim_df["season_best"] = sim_df["athlete_id"].apply(
            lambda aid: explanations[aid]["features"]["season_best"]
        )
        sim_df["season_best_label"] = sim_df["athlete_id"].apply(
            lambda aid: explanations[aid]["features"]["season_best_label"]
        )
        sim_df["season_best_is_current"] = sim_df["athlete_id"].apply(
            lambda aid: explanations[aid]["features"]["season_best_is_current"]
        )
        sim_df["consistency_label"] = sim_df["athlete_id"].apply(
            lambda aid: _consistency_label(explanations[aid]["features"]["consistency_std"])
        )
        sim_df["trend_label"] = sim_df["athlete_id"].apply(
            lambda aid: _trend_label(explanations[aid]["features"]["trend_slope"])
        )
        sim_df["n_races_used"] = sim_df["athlete_id"].apply(
            lambda aid: explanations[aid]["features"]["n_races"]
        )

    run_id = str(uuid.uuid4())
    return {
        "predictions": sim_df,
        "as_of_date": as_of_date,
        "run_id": run_id,
        "model_version": MODEL_VERSION,
        "explanations": explanations,
        "skipped_athletes": [aid for aid in athlete_ids if aid not in explanations or explanations[aid]["expected_time"] is None],
    }


def _format_recent_range(feats):
    times = feats.get("recent_times_sample") or []
    if not times:
        return "No recent data"
    return f"{min(times):.2f}-{max(times):.2f}"


def _consistency_label(std):
    if std is None:
        return "Unknown (limited data)"
    if std < 0.20:
        return "High"
    if std < 0.40:
        return "Moderate"
    return "Low"


def _trend_label(slope):
    if slope is None:
        return "Unknown (limited data)"
    if slope < -0.03:
        return "Improving"
    if slope > 0.03:
        return "Slowing"
    return "Stable"


def save_prediction_run(result: dict, event_key: str, competition: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO prediction_runs (prediction_run_id, event_key, competition, prediction_timestamp, model_version) VALUES (?,?,?,?,?)",
        (result["run_id"], event_key, competition, dt.datetime.now().isoformat(), result["model_version"]),
    )
    for _, r in result["predictions"].iterrows():
        cur.execute(
            """INSERT INTO predictions (prediction_id, prediction_run_id, athlete_id, expected_time, time_std,
                   win_probability, silver_probability, bronze_probability, top3_probability, expected_position)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (str(uuid.uuid4()), result["run_id"], r["athlete_id"], r["expected_time"], r["time_std"],
             r["win_probability"], r["silver_probability"], r["bronze_probability"], r["top3_probability"],
             r["expected_position"]),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    from config.events import EVENTS, DEFAULT_EVENT_KEY
    ev = EVENTS[DEFAULT_EVENT_KEY]
    athletes = search_athletes(ev.key, gender="M").head(8)
    print(athletes)
    res = predict_race(ev.key, athletes["athlete_id"].tolist(), n_simulations=20000)
    cols = ["athlete_name", "country", "expected_time", "time_std", "win_probability",
            "silver_probability", "bronze_probability", "top3_probability", "expected_position"]
    print(res["predictions"][cols])
