"""
Backtesting framework (spec Section 11-12).

Strategy: for each major competition present in the database, take the actual
field of athletes who swam the final-event's race at that meet/date (their
rank_at_meet / time at that specific competition is the "ground truth"
finishing order), then rebuild predictions using ONLY performances strictly
before that competition's date (as_of_date = competition_date), exactly as
predict_race() does live. This guarantees no data leakage: the model never
sees the outcome it is being asked to predict.

We restrict backtest finals to meets classified as "Major" (Olympics, World
Championships, Asian Games, etc.) with >=4 swimmers, since those are true
finals-like fields (small heats/relat-only meets would be noisy ground truth).
"""

import sys
import os
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.modelling.predict import load_event_performances, predict_race
from src.cleaning.clean import classify_meet_level


def find_backtestable_finals(perf_df: pd.DataFrame, min_field_size: int = 4, max_field_size: int = 10) -> pd.DataFrame:
    """
    Identify (competition, competition_date, gender) groups that look like a
    real final: a Major meet with a plausible number of distinct athletes
    swimming the event on the same date.
    """
    df = perf_df[perf_df["is_flagged"] == 0].copy()
    df = df[df["competition_level"] == "Major"]
    grouped = (
        df.groupby(["competition", "competition_date", "gender"])
        .agg(n_athletes=("athlete_id", "nunique"))
        .reset_index()
    )
    finals = grouped[(grouped["n_athletes"] >= min_field_size) & (grouped["n_athletes"] <= max_field_size)]
    return finals.sort_values("competition_date")


def get_final_field(perf_df: pd.DataFrame, competition: str, competition_date, gender: str, max_field_size: int = 8) -> pd.DataFrame:
    """The actual result of that specific race: one row per athlete, ranked by time swum that day."""
    df = perf_df[
        (perf_df["competition"] == competition)
        & (perf_df["competition_date"] == competition_date)
        & (perf_df["gender"] == gender)
        & (perf_df["is_flagged"] == 0)
    ].copy()
    # Keep each athlete's best (fastest) swim that day, in case of prelim+final same date edge cases
    df = df.sort_values("time_seconds").drop_duplicates(subset=["athlete_id"])
    df = df.sort_values("time_seconds").head(max_field_size).reset_index(drop=True)
    df["actual_position"] = np.arange(1, len(df) + 1)
    return df


def run_backtest(event_key: str, gender: str, min_field_size: int = 4, max_field_size: int = 8,
                  n_simulations: int = 20000) -> dict:
    """
    Runs the full backtest for one event+gender. Returns per-final results and
    aggregate metrics (Section 11: accuracy, log loss, brier score, MAE, top-1/top-3 hit rate).
    """
    perf_df = load_event_performances(event_key, gender=gender)
    finals = find_backtestable_finals(perf_df, min_field_size, max_field_size)

    per_final_rows = []
    all_predicted_win_probs = []
    all_actual_wins = []
    all_predicted_top3_probs = []
    all_actual_top3 = []
    time_errors = []

    for _, f in finals.iterrows():
        field = get_final_field(perf_df, f["competition"], f["competition_date"], gender, max_field_size)
        if len(field) < min_field_size:
            continue
        athlete_ids = field["athlete_id"].tolist()
        as_of = pd.to_datetime(f["competition_date"]).date()

        result = predict_race(event_key, athlete_ids, as_of_date=as_of, n_simulations=n_simulations)
        preds = result["predictions"]
        if preds.empty or len(preds) < min_field_size:
            continue  # not enough athletes had usable pre-race history

        merged = preds.merge(field[["athlete_id", "actual_position", "time_seconds"]], on="athlete_id", how="inner")
        if len(merged) < min_field_size:
            continue

        predicted_winner = merged.loc[merged["win_probability"].idxmax(), "athlete_id"]
        actual_winner = field.loc[field["actual_position"] == 1, "athlete_id"].iloc[0]

        for _, r in merged.iterrows():
            actual_win = 1 if r["actual_position"] == 1 else 0
            actual_top3 = 1 if r["actual_position"] <= 3 else 0
            all_predicted_win_probs.append(r["win_probability"])
            all_actual_wins.append(actual_win)
            all_predicted_top3_probs.append(r["top3_probability"])
            all_actual_top3.append(actual_top3)
            time_errors.append(abs(r["expected_time"] - r["time_seconds"]))

        per_final_rows.append({
            "competition": f["competition"],
            "competition_date": str(f["competition_date"]),
            "gender": gender,
            "n_athletes": len(merged),
            "predicted_winner_correct": int(predicted_winner == actual_winner),
            "top3_hit_rate": float(
                len(set(merged.nsmallest(3, "expected_position")["athlete_id"]) &
                    set(field[field["actual_position"] <= 3]["athlete_id"]))
            ) / 3.0,
        })

    metrics = compute_metrics(all_predicted_win_probs, all_actual_wins, all_predicted_top3_probs,
                               all_actual_top3, time_errors, per_final_rows)

    return {
        "per_final": pd.DataFrame(per_final_rows),
        "metrics": metrics,
        "win_probs": all_predicted_win_probs,
        "actual_wins": all_actual_wins,
    }


def compute_metrics(win_probs, actual_wins, top3_probs, actual_top3, time_errors, per_final_rows):
    n = len(win_probs)
    if n == 0:
        return {"n_predictions": 0, "n_finals": 0, "message": "No backtestable finals found with current data."}

    win_probs = np.clip(np.array(win_probs), 1e-6, 1 - 1e-6)
    actual_wins = np.array(actual_wins)
    top3_probs = np.clip(np.array(top3_probs), 1e-6, 1 - 1e-6)
    actual_top3 = np.array(actual_top3)

    brier_win = float(np.mean((win_probs - actual_wins) ** 2))
    log_loss_win = float(-np.mean(actual_wins * np.log(win_probs) + (1 - actual_wins) * np.log(1 - win_probs)))

    top1_hit_rate = float(np.mean([r["predicted_winner_correct"] for r in per_final_rows])) if per_final_rows else None
    top3_hit_rate = float(np.mean([r["top3_hit_rate"] for r in per_final_rows])) if per_final_rows else None

    mae_time = float(np.mean(time_errors)) if time_errors else None

    return {
        "n_predictions": n,
        "n_finals": len(per_final_rows),
        "brier_score_win": round(brier_win, 4),
        "log_loss_win": round(log_loss_win, 4),
        "top1_hit_rate": round(top1_hit_rate, 3) if top1_hit_rate is not None else None,
        "top3_hit_rate": round(top3_hit_rate, 3) if top3_hit_rate is not None else None,
        "mae_expected_time_seconds": round(mae_time, 3) if mae_time is not None else None,
    }


def calibration_curve(win_probs, actual_wins, n_bins=5):
    """Predicted-probability vs actual-outcome-frequency bins for the calibration chart."""
    win_probs = np.array(win_probs)
    actual_wins = np.array(actual_wins)
    if len(win_probs) == 0:
        return pd.DataFrame(columns=["bin_mid", "predicted_mean", "actual_freq", "n"])
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (win_probs >= lo) & (win_probs < hi if i < n_bins - 1 else win_probs <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_mid": (lo + hi) / 2,
            "predicted_mean": float(win_probs[mask].mean()),
            "actual_freq": float(actual_wins[mask].mean()),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from config.events import EVENTS, DEFAULT_EVENT_KEY
    ev = EVENTS[DEFAULT_EVENT_KEY]
    for gender in ["M", "F"]:
        print(f"\n=== Backtest: {ev.display_name} ({gender}) ===")
        res = run_backtest(ev.key, gender, n_simulations=10000)
        print(res["metrics"])
        print(res["per_final"])
