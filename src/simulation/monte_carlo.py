"""
Monte Carlo race simulation (spec Section 8) -- the core of the engine.

For every finalist, draw N simulated race times from Normal(expected_time,
time_std), rank each simulated race, and tally finishing-position frequencies
into gold/silver/bronze/top3 probabilities and expected finishing position.
"""

import numpy as np
import pandas as pd

from config.settings import N_SIMULATIONS, RANDOM_SEED


def simulate_race(athlete_predictions: list, n_simulations: int = N_SIMULATIONS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    athlete_predictions: list of dicts with at least
        {athlete_id, athlete_name, expected_time, time_std}

    Returns a DataFrame, one row per athlete, with:
        expected_time, time_std, win_probability, silver_probability,
        bronze_probability, top3_probability, expected_position
    """
    rng = np.random.default_rng(seed)
    n_athletes = len(athlete_predictions)
    if n_athletes == 0:
        return pd.DataFrame()

    means = np.array([p["expected_time"] for p in athlete_predictions])
    stds = np.array([max(p["time_std"], 0.01) for p in athlete_predictions])

    # (n_simulations, n_athletes) matrix of simulated times.
    sims = rng.normal(loc=means, scale=stds, size=(n_simulations, n_athletes))
    sims = np.clip(sims, 1.0, None)  # times can't be non-positive

    # rank[i, j] = finishing position (1 = fastest) of athlete j in simulation i
    ranks = sims.argsort(axis=1).argsort(axis=1) + 1

    results = []
    for j, pred in enumerate(athlete_predictions):
        athlete_ranks = ranks[:, j]
        win_p = float(np.mean(athlete_ranks == 1))
        silver_p = float(np.mean(athlete_ranks == 2))
        bronze_p = float(np.mean(athlete_ranks == 3))
        top3_p = float(np.mean(athlete_ranks <= 3))
        exp_pos = float(np.mean(athlete_ranks))
        results.append({
            "athlete_id": pred["athlete_id"],
            "athlete_name": pred.get("athlete_name", pred["athlete_id"]),
            "country": pred.get("country", ""),
            "expected_time": pred["expected_time"],
            "time_std": pred["time_std"],
            "win_probability": win_p,
            "silver_probability": silver_p,
            "bronze_probability": bronze_p,
            "top3_probability": top3_p,
            "expected_position": exp_pos,
        })

    df = pd.DataFrame(results).sort_values("win_probability", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    demo = [
        {"athlete_id": "A", "athlete_name": "Athlete A", "expected_time": 47.91, "time_std": 0.25},
        {"athlete_id": "B", "athlete_name": "Athlete B", "expected_time": 48.00, "time_std": 0.30},
        {"athlete_id": "C", "athlete_name": "Athlete C", "expected_time": 48.10, "time_std": 0.20},
    ]
    out = simulate_race(demo, n_simulations=100_000)
    print(out[["athlete_name", "expected_time", "win_probability", "top3_probability", "expected_position"]])
    print("Gold sums to:", out["win_probability"].sum())
