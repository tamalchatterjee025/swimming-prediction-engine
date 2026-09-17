"""
Expected time + uncertainty model (spec Section 6-7).

Interpretable, statistical -- no deep learning. Expected time is a weighted
blend of: recency-weighted recent performance, season best, personal best,
and an AQUA-points-implied time (small secondary signal), plus a small
major-meet adjustment. Uncertainty (time_std) comes from the athlete's own
recent-race consistency, inflated for athletes with little data.
"""

import sys
import os
import datetime as dt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import COMPONENT_WEIGHTS
from src.features.athlete_features import build_athlete_features

# Fallback uncertainty (seconds) when an athlete has too few races to compute
# their own consistency_std. Derived from typical elite 100m free race-to-race
# variability (~0.3-0.5s); used only as a floor/fallback, not a primary driver.
DEFAULT_STD = 0.35
MIN_STD = 0.12
FEW_RACES_STD_INFLATION = 1.4  # multiplier applied when n_races < 3


def fit_aqua_points_time_relationship(perf_df: pd.DataFrame):
    """
    Data-driven points->time conversion (spec 6.F: "do not hard-code
    permanently if a data-driven alternative is practical"). Fits a simple
    linear regression of time_seconds on aqua_points across all unflagged
    performances for the event, separately isn't needed per gender because
    AQUA points are already gender-normalised by design.
    """
    df = perf_df[(perf_df["is_flagged"] == 0) & perf_df["aqua_points"].notna()]
    if len(df) < 20:
        return None
    slope, intercept = np.polyfit(df["aqua_points"], df["time_seconds"], 1)
    return {"slope": float(slope), "intercept": float(intercept)}


def aqua_points_to_time(aqua_points, fit):
    if aqua_points is None or fit is None:
        return None
    return fit["slope"] * aqua_points + fit["intercept"]


def compute_expected_time(perf_df: pd.DataFrame, athlete_id: str, as_of_date: dt.date, aqua_fit=None) -> dict:
    """
    Returns dict with expected_time, time_std, and the underlying feature
    values (for transparent "why the model sees each athlete at that level").
    """
    feats = build_athlete_features(perf_df, athlete_id, as_of_date)

    components = {}
    weights_used = {}

    if feats["recent_weighted_time"] is not None:
        components["recent"] = feats["recent_weighted_time"]
        weights_used["recent"] = COMPONENT_WEIGHTS["recent"]
    if feats["season_best"] is not None:
        components["season_best"] = feats["season_best"]
        weights_used["season_best"] = COMPONENT_WEIGHTS["season_best"]
    if feats["personal_best"] is not None:
        components["pb"] = feats["personal_best"]
        weights_used["pb"] = COMPONENT_WEIGHTS["pb"]
    aqua_time = aqua_points_to_time(feats["aqua_points_avg"], aqua_fit)
    if aqua_time is not None:
        components["aqua"] = aqua_time
        weights_used["aqua"] = COMPONENT_WEIGHTS["aqua"]

    if not components:
        expected_time = None
    else:
        total_w = sum(weights_used.values())
        norm_w = {k: v / total_w for k, v in weights_used.items()}
        expected_time = sum(components[k] * norm_w[k] for k in components)
        # trend: nudge expected time by a fraction of the recent slope (improving
        # athletes are expected to be slightly faster than their raw average).
        if feats["trend_slope"] is not None:
            expected_time += 0.3 * feats["trend_slope"]
        expected_time += feats["major_meet_adjustment"]

    # --- Uncertainty ------------------------------------------------------
    if feats["consistency_std"] is not None:
        std = max(MIN_STD, feats["consistency_std"])
    else:
        std = DEFAULT_STD
    if feats["n_races"] < 3:
        std *= FEW_RACES_STD_INFLATION

    return {
        "athlete_id": athlete_id,
        "expected_time": round(expected_time, 3) if expected_time is not None else None,
        "time_std": round(std, 3),
        "components": components,
        "weights_used": weights_used,
        "features": feats,
    }
