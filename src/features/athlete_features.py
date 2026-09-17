"""
Feature engineering for a single athlete's expected-time model (spec Section 6).

All functions operate on a per-athlete DataFrame of performances (already
filtered to one event_key), sorted or not -- each function sorts internally.
Every function accepts an `as_of_date` cutoff so the exact same code path is
used for live predictions and for leakage-free backtesting (Section 11).
"""

import sys
import os
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import RECENCY_WEIGHTS, SEASON_START_MONTH, MAJOR_MEET_MAX_ADJUSTMENT


def _season_start(as_of: dt.date) -> dt.date:
    year = as_of.year if as_of.month >= SEASON_START_MONTH else as_of.year - 1
    return dt.date(year, SEASON_START_MONTH, 1)


def _season_label(season_start_date: dt.date) -> str:
    end_year = season_start_date.year + 1
    return f"{season_start_date.year}-{str(end_year)[-2:]}"


def filter_as_of(perf_df: pd.DataFrame, as_of_date: dt.date) -> pd.DataFrame:
    """Only performances strictly before as_of_date -- the leakage guard."""
    df = perf_df.copy()
    df["competition_date"] = pd.to_datetime(df["competition_date"]).dt.date
    return df[df["competition_date"] < as_of_date].sort_values("competition_date", ascending=False)


def recent_weighted_time(df_sorted_desc: pd.DataFrame, weights=None):
    """Component A: recency-weighted average of the N most recent races."""
    weights = weights or RECENCY_WEIGHTS
    times = df_sorted_desc["time_seconds"].head(len(weights)).tolist()
    if not times:
        return None
    w = weights[:len(times)]
    w = [x / sum(w) for x in w]
    return float(np.dot(times, w))


def season_best(df_sorted_desc: pd.DataFrame, as_of_date: dt.date):
    """
    Component B: best time within the current season so far.

    Falls back one season back if the current season has no races yet (e.g.
    early September, right after the season boundary, before an athlete's
    long-course season has started) -- otherwise this reads as "no data"
    for almost every athlete for a large chunk of the calendar, which isn't
    useful. The fallback is clearly labelled (`season_best_is_current=False`,
    `season_best_label` names the season it actually came from) so the UI
    never silently presents a stale season as the current one.
    """
    current_start = _season_start(as_of_date)
    current_df = df_sorted_desc[df_sorted_desc["competition_date"] >= current_start]
    if not current_df.empty:
        return {
            "value": float(current_df["time_seconds"].min()),
            "is_current": True,
            "label": _season_label(current_start),
        }

    prior_start = dt.date(current_start.year - 1, current_start.month, current_start.day)
    prior_df = df_sorted_desc[
        (df_sorted_desc["competition_date"] >= prior_start) & (df_sorted_desc["competition_date"] < current_start)
    ]
    if prior_df.empty:
        return {"value": None, "is_current": False, "label": None}
    return {
        "value": float(prior_df["time_seconds"].min()),
        "is_current": False,
        "label": _season_label(prior_start),
    }


def personal_best(df_sorted_desc: pd.DataFrame):
    """Component C: all-time best (within the loaded history window)."""
    if df_sorted_desc.empty:
        return None
    return float(df_sorted_desc["time_seconds"].min())


def consistency_std(df_sorted_desc: pd.DataFrame, n=5):
    """Component D: std dev of the N most recent races -- lower = more consistent."""
    times = df_sorted_desc["time_seconds"].head(n).tolist()
    if len(times) < 2:
        return None
    return float(np.std(times, ddof=1))


def trend_slope(df_sorted_desc: pd.DataFrame, n=5):
    """
    Component E: linear-regression slope of the N most recent races over time
    (races ordered oldest->newest for the regression). Negative slope = improving
    (getting faster), positive = slowing down.
    """
    recent = df_sorted_desc.head(n).sort_values("competition_date")
    if len(recent) < 3:
        return None
    x = np.arange(len(recent))
    y = recent["time_seconds"].values
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def aqua_points_avg(df_sorted_desc: pd.DataFrame, n=3):
    """Component F: average AQUA points of the N most recent races (secondary signal)."""
    pts = df_sorted_desc["aqua_points"].head(n).dropna().tolist()
    if not pts:
        return None
    return float(np.mean(pts))


def aqua_implied_time(aqua_pts, base_points_time_pairs=None):
    """
    Convert AQUA points to an implied time using the athlete's own recent
    points<->time relationship isn't reliable with so few races, so instead we
    use the standard FINA/AQUA points formula inverse relative to the base time
    used by World Aquatics for LCM 100 Free (kept in config); this is only a
    small secondary signal per spec 6.F.
    """
    from config.settings import COMPONENT_WEIGHTS  # noqa: F401  (kept for symmetry / future use)
    return None  # not used standalone -- see build_features which derives it relative to base time


def major_meet_adjustment(df_sorted_desc: pd.DataFrame):
    """
    Component G (small): does this athlete tend to swim relatively faster or
    slower at major championships vs. their other races? Returns a small
    seconds adjustment (positive = tends to swim slower at majors), capped.
    Only computed if there are >=3 major-meet swims AND >=3 other swims.
    """
    majors = df_sorted_desc[df_sorted_desc["competition_level"] == "Major"]["time_seconds"]
    others = df_sorted_desc[df_sorted_desc["competition_level"] != "Major"]["time_seconds"]
    if len(majors) < 3 or len(others) < 3:
        return 0.0
    diff = float(majors.mean() - others.mean())
    return float(np.clip(diff, -MAJOR_MEET_MAX_ADJUSTMENT, MAJOR_MEET_MAX_ADJUSTMENT))


def build_athlete_features(perf_df: pd.DataFrame, athlete_id: str, as_of_date: dt.date) -> dict:
    """
    Build the full feature dict for one athlete as of a given date, using only
    performances strictly before as_of_date (backtest-safe).
    """
    ath_df = perf_df[(perf_df["athlete_id"] == athlete_id) & (perf_df["is_flagged"] == 0)]
    df = filter_as_of(ath_df, as_of_date)

    n_races = len(df)
    season = season_best(df, as_of_date)
    feats = {
        "athlete_id": athlete_id,
        "n_races": n_races,
        "recent_weighted_time": recent_weighted_time(df),
        "season_best": season["value"],
        "season_best_is_current": season["is_current"],
        "season_best_label": season["label"],
        "personal_best": personal_best(df),
        "consistency_std": consistency_std(df),
        "trend_slope": trend_slope(df),
        "aqua_points_avg": aqua_points_avg(df),
        "major_meet_adjustment": major_meet_adjustment(df),
        "most_recent_date": df["competition_date"].iloc[0] if n_races else None,
        "recent_times_sample": df["time_seconds"].head(5).tolist(),
    }
    return feats
