"""Load cleaned performances into the SQLite database (idempotent upsert)."""

import os
import sys
import hashlib
import datetime as dt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utilities.db import get_connection, init_db
from src.cleaning.clean import load_raw_files, clean_performances
from config.events import EventConfig


def upsert_athletes(conn, df: pd.DataFrame):
    athletes = (
        df[["athlete_id", "athlete_name", "country", "gender", "birth_date"]]
        .drop_duplicates(subset=["athlete_id"])
    )
    cur = conn.cursor()
    for _, r in athletes.iterrows():
        cur.execute(
            """INSERT INTO athletes (athlete_id, athlete_name, country, gender, birth_date)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(athlete_id) DO UPDATE SET
                   athlete_name=excluded.athlete_name,
                   country=excluded.country,
                   gender=excluded.gender,
                   birth_date=excluded.birth_date""",
            (r.athlete_id, r.athlete_name, r.country, r.gender, r.birth_date),
        )
    conn.commit()


def _stable_competition_id(competition: str, country: str) -> str:
    """
    Deterministic id, independent of process hash-randomisation.
    (Python's built-in hash() is salted per-process via PYTHONHASHSEED, so it
    must never be used for anything that needs to stay stable across runs --
    using it here previously caused every re-run of update_data.py to insert
    fresh duplicate competition rows instead of matching existing ones.)
    """
    key = f"{competition}|{country}".upper()
    return "COMP_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def upsert_competitions(conn, df: pd.DataFrame):
    comps = df[["competition", "competition_date", "competition_country", "competition_level", "source_url"]].copy()
    comps["competition_id"] = comps.apply(
        lambda r: _stable_competition_id(r.competition, r.competition_country), axis=1
    )
    comps = comps.drop_duplicates(subset=["competition_id"])
    cur = conn.cursor()
    for _, r in comps.iterrows():
        cur.execute(
            """INSERT INTO competitions (competition_id, competition_name, date, location, country, level, source_url)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(competition_id) DO NOTHING""",
            (r.competition_id, r.competition, r.competition_date, r.competition_country,
             r.competition_country, r.competition_level, r.source_url),
        )
    conn.commit()
    return comps.set_index(["competition", "competition_country"])["competition_id"].to_dict()


def upsert_performances(conn, df: pd.DataFrame, comp_lookup: dict) -> int:
    cur = conn.cursor()
    inserted = 0
    now = dt.datetime.now().isoformat()
    for _, r in df.iterrows():
        comp_id = comp_lookup.get((r.competition, r.competition_country))
        cur.execute(
            """INSERT INTO performances (
                    performance_id, athlete_id, event_key, event, distance, stroke, pool_type,
                    time_seconds, aqua_points, competition_id, competition, competition_date,
                    competition_country, rank_at_meet, source, source_url, is_flagged, flag_reason, ingested_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(athlete_id, event_key, time_seconds, competition_date, competition) DO NOTHING""",
            (r.performance_id, r.athlete_id, r.event_key, r.event, int(r.distance), r.stroke, r.pool_type,
             float(r.time_seconds), None if pd.isna(r.aqua_points) else float(r.aqua_points),
             comp_id, r.competition, r.competition_date, r.competition_country,
             None if pd.isna(r.rank_at_meet) else int(r.rank_at_meet),
             r.source, r.source_url, int(r.is_flagged), r.flag_reason, now),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    return inserted


def log_ingestion(conn, event_key, status, rows_fetched, rows_inserted, message=""):
    conn.execute(
        """INSERT INTO ingestion_log (run_timestamp, event_key, gender, year, status, rows_fetched, rows_inserted, message)
           VALUES (?,?,?,?,?,?,?,?)""",
        (dt.datetime.now().isoformat(), event_key, "ALL", "ALL", status, rows_fetched, rows_inserted, message),
    )
    conn.commit()


def load_event_to_db(event: EventConfig):
    init_db()
    raw = load_raw_files(event.key)
    clean = clean_performances(raw, event)
    conn = get_connection()
    try:
        if clean.empty:
            log_ingestion(conn, event.key, "no_data", 0, 0, "No raw files found / all rows unusable")
            return {"rows_fetched": 0, "rows_inserted": 0}
        upsert_athletes(conn, clean)
        comp_lookup = upsert_competitions(conn, clean)
        inserted = upsert_performances(conn, clean, comp_lookup)
        log_ingestion(conn, event.key, "ok", len(clean), inserted)
        return {"rows_fetched": len(clean), "rows_inserted": inserted}
    finally:
        conn.close()


if __name__ == "__main__":
    from config.events import EVENTS, DEFAULT_EVENT_KEY
    ev = EVENTS[DEFAULT_EVENT_KEY]
    result = load_event_to_db(ev)
    print(result)
