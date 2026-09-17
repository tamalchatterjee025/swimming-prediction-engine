"""Global settings for the Swimming Prediction Engine."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
DATABASE_DIR = os.path.join(DATA_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "swimming.db")

LOG_DIR = os.path.join(BASE_DIR, "logs")

# World Aquatics public ranking export endpoint (discovered by inspecting the
# "Download" links on https://www.worldaquatics.com/swimming/rankings).
# No API key required. Confirmed working 2026-09-17.
WORLD_AQUATICS_API_BASE = "https://api.worldaquatics.com/fina/rankings/swimming/report/csv"

# Years of history to pull per ingestion run (most recent N years, per spec section 3).
HISTORY_YEARS = 5

MODEL_VERSION = "v0.1-mvp"

# Monte Carlo simulation
N_SIMULATIONS = 100_000
RANDOM_SEED = 42

# Recency weighting for the "recent performance" feature component (Section 6.A).
# Applied to an athlete's most-recent-first sorted races. If an athlete has fewer
# races than weights, the remaining weight is renormalised across what exists.
RECENCY_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.10]

# Component blend weights for expected_time (Section 6). Must sum to 1.0.
# recent: weighted recent performance (A)
# season_best: current-season best (B)
# pb: all-time personal best (C)
# aqua: AQUA-points-implied time, small secondary signal (F)
COMPONENT_WEIGHTS = {
    "recent": 0.55,
    "season_best": 0.25,
    "pb": 0.15,
    "aqua": 0.05,
}

# Major-meet adjustment (Section 6.G) is capped to a small effect.
MAJOR_MEET_MAX_ADJUSTMENT = 0.15  # seconds, at most

# Season definition: a "season" runs Sept 1 - Aug 31 to roughly track the
# long-course championship calendar (majors are typically June-Sept).
SEASON_START_MONTH = 9
