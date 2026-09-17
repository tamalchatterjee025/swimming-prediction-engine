"""SQLite schema creation and connection helper."""

import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import DATABASE_PATH, DATABASE_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS athletes (
    athlete_id      TEXT PRIMARY KEY,
    athlete_name    TEXT NOT NULL,
    country         TEXT,
    gender          TEXT CHECK (gender IN ('M', 'F')),
    birth_date      TEXT
);

CREATE TABLE IF NOT EXISTS competitions (
    competition_id      TEXT PRIMARY KEY,
    competition_name    TEXT NOT NULL,
    date                 TEXT,
    location             TEXT,
    country              TEXT,
    level                TEXT,
    source_url           TEXT
);

CREATE TABLE IF NOT EXISTS performances (
    performance_id      TEXT PRIMARY KEY,
    athlete_id           TEXT NOT NULL REFERENCES athletes(athlete_id),
    event_key             TEXT NOT NULL,
    event                 TEXT,
    distance               INTEGER,
    stroke                 TEXT,
    pool_type              TEXT,
    time_seconds           REAL NOT NULL,
    aqua_points             REAL,
    competition_id          TEXT REFERENCES competitions(competition_id),
    competition             TEXT,
    competition_date        TEXT,
    competition_country      TEXT,
    rank_at_meet              INTEGER,
    source                     TEXT,
    source_url                  TEXT,
    is_flagged                   INTEGER DEFAULT 0,
    flag_reason                   TEXT,
    ingested_at                    TEXT,
    UNIQUE(athlete_id, event_key, time_seconds, competition_date, competition)
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    prediction_run_id   TEXT PRIMARY KEY,
    event_key             TEXT NOT NULL,
    competition             TEXT,
    prediction_timestamp     TEXT,
    model_version              TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id        TEXT PRIMARY KEY,
    prediction_run_id      TEXT NOT NULL REFERENCES prediction_runs(prediction_run_id),
    athlete_id                TEXT NOT NULL REFERENCES athletes(athlete_id),
    expected_time                REAL,
    time_std                       REAL,
    win_probability                  REAL,
    silver_probability                 REAL,
    bronze_probability                   REAL,
    top3_probability                       REAL,
    expected_position                        REAL
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp    TEXT,
    event_key           TEXT,
    gender                TEXT,
    year                    TEXT,
    status                    TEXT,
    rows_fetched                INTEGER,
    rows_inserted                  INTEGER,
    message                          TEXT
);

CREATE INDEX IF NOT EXISTS idx_perf_athlete ON performances(athlete_id);
CREATE INDEX IF NOT EXISTS idx_perf_event ON performances(event_key);
CREATE INDEX IF NOT EXISTS idx_perf_date ON performances(competition_date);
"""


def get_connection():
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialised at {DATABASE_PATH}")
