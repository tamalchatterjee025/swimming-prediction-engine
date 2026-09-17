"""
Calibration (spec Section 12): check whether predicted win probabilities
behave like true probabilities over the backtest sample, and fit an isotonic
calibrator if the sample is large enough and the model looks systematically
over/under-confident.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

MIN_SAMPLES_FOR_CALIBRATION = 60


def fit_isotonic_calibrator(win_probs, actual_wins):
    """
    Only fit if we have enough samples (spec: "only implement this if the
    historical sample is large enough. Keep the system simple.").
    Returns a fitted IsotonicRegression, or None if sample too small.
    """
    win_probs = np.array(win_probs)
    actual_wins = np.array(actual_wins)
    if len(win_probs) < MIN_SAMPLES_FOR_CALIBRATION:
        return None
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(win_probs, actual_wins)
    return iso


def apply_calibrator(iso, probs):
    if iso is None:
        return probs
    return iso.predict(np.array(probs))
