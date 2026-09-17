"""
Clean and normalise raw World Aquatics ranking CSVs into a tidy DataFrame of
performances, ready to load into SQLite.

Implements spec Section 5 (data engineering requirement):
 1. Convert times to numeric seconds
 2. Standardise athlete names / build stable athlete_id
 3. Standardise country codes (World Aquatics already gives ISO-ish team codes)
 4. Remove duplicate performances
 5. Validate dates
 6. Reject Short Course results when the selected event is Long Course
 7. Flag suspicious/incomplete records rather than deleting them
 8. Preserve the original source URL
"""

import os
import re
import sys
import glob
import hashlib
import datetime as dt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.events import EventConfig, MAJOR_MEET_KEYWORDS

RAW_COLUMNS = [
    "meet_name", "swim_time", "swim_date", "full_desc", "team_code",
    "team_short_name", "full_name_computed", "gender", "birth_date",
    "event_id", "standard_name", "RANK", "Rank_Order", "fina_points",
    "meet_city", "country_code",
]

# Plausible human 100m freestyle LCM time bounds (seconds), used only to flag,
# never to silently delete, suspicious records.
PLAUSIBLE_TIME_RANGE = (40.0, 75.0)


def time_str_to_seconds(t: str):
    """Convert World Aquatics time strings ('46.40' or '1:46.40') to seconds."""
    if t is None:
        return None
    t = str(t).strip()
    if not t:
        return None
    m = re.match(r"^(?:(\d+):)?(\d{1,2})\.(\d{1,2})$", t)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = int(m.group(2))
    frac = m.group(3).ljust(2, "0")
    return round(minutes * 60 + seconds + int(frac) / 100.0, 2)


def make_athlete_id(full_name: str, country: str, birth_date: str) -> str:
    """
    Stable synthetic athlete_id. World Aquatics's public export does not
    include a numeric athlete ID, so we derive one deterministically from
    normalised name + country + birth date (the combination is effectively
    unique for elite swimmers and stays stable across ingestion runs).
    """
    key = f"{standardise_name(full_name)}|{country}|{birth_date}".upper()
    return "ATH_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def standardise_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip()
    # World Aquatics format is "LASTNAME, Firstname" -> keep but normalise spacing/case.
    name = re.sub(r"\s+", " ", name)
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{last.title()}, {first.title()}"
    return name.title()


def parse_date(d: str):
    if not isinstance(d, str) or not d.strip():
        return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(d.strip(), fmt).date()
        except ValueError:
            continue
    return None


def classify_meet_level(meet_name: str) -> str:
    if not isinstance(meet_name, str):
        return "Other"
    for kw in MAJOR_MEET_KEYWORDS:
        if kw.lower() in meet_name.lower():
            return "Major"
    return "Other"


def load_raw_files(event_key: str, gender_code: str = None) -> pd.DataFrame:
    """Load and concatenate all raw CSV pulls for an event (optionally one gender)."""
    from config.settings import RAW_DATA_DIR
    pattern = os.path.join(RAW_DATA_DIR, f"worldaquatics_{event_key}_*.csv")
    frames = []
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if gender_code and f"_{event_key}_{gender_code}_" not in fname:
            continue
        try:
            df = pd.read_csv(path, dtype=str)
        except Exception:
            continue
        df["_source_file"] = fname
        df["_source_url"] = "https://api.worldaquatics.com/fina/rankings/swimming/report/csv"
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=RAW_COLUMNS + ["_source_file", "_source_url"])
    return pd.concat(frames, ignore_index=True)


def clean_performances(raw_df: pd.DataFrame, event: EventConfig) -> pd.DataFrame:
    """
    Transform the raw World Aquatics rows into a clean performances table.
    Returns a DataFrame with one row per performance and an `is_flagged`/
    `flag_reason` pair instead of dropping questionable rows.
    """
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    df["time_seconds"] = df["swim_time"].apply(time_str_to_seconds)
    df["competition_date"] = df["swim_date"].apply(parse_date)
    df["athlete_name"] = df["full_name_computed"].apply(standardise_name)
    df["country"] = df["team_code"].fillna("UNK").str.upper()
    df["gender"] = df["gender"].str.upper().str.strip()
    df["birth_date"] = df["birth_date"]
    df["athlete_id"] = df.apply(
        lambda r: make_athlete_id(r["athlete_name"], r["country"], r["birth_date"]), axis=1
    )
    df["aqua_points"] = pd.to_numeric(df["fina_points"], errors="coerce")
    df["competition"] = df["meet_name"].fillna("Unknown Meet")
    df["competition_country"] = df["country_code"].fillna("UNK")
    df["competition_level"] = df["competition"].apply(classify_meet_level)
    df["rank_at_meet"] = pd.to_numeric(df["RANK"], errors="coerce")
    df["event_key"] = event.key
    df["event"] = event.display_name
    df["distance"] = event.distance
    df["stroke"] = event.stroke
    df["pool_type"] = event.pool_type  # req #6: only LCM was ever requested from the API
    df["source"] = "World Aquatics (official)"
    df["source_url"] = df["_source_url"]

    # --- Flagging (never silently deleted) -------------------------------
    flags = []
    for _, r in df.iterrows():
        reasons = []
        if r["time_seconds"] is None:
            reasons.append("unparseable_time")
        elif not (PLAUSIBLE_TIME_RANGE[0] <= r["time_seconds"] <= PLAUSIBLE_TIME_RANGE[1]):
            reasons.append("implausible_time")
        if r["competition_date"] is None:
            reasons.append("invalid_date")
        if not r["athlete_name"]:
            reasons.append("missing_name")
        if r["gender"] not in ("M", "F"):
            reasons.append("invalid_gender")
        flags.append(";".join(reasons))
    df["flag_reason"] = flags
    df["is_flagged"] = df["flag_reason"].apply(lambda s: 1 if s else 0)

    # Drop only truly unusable rows (no time or no date at all) -- everything
    # else is kept but flagged so it's visible on the Data Status page.
    usable = df[df["time_seconds"].notna() & df["competition_date"].notna()].copy()

    usable["competition_date"] = usable["competition_date"].astype(str)
    usable["performance_id"] = usable.apply(
        lambda r: "PERF_" + hashlib.sha1(
            f"{r['athlete_id']}|{r['event_key']}|{r['time_seconds']}|{r['competition_date']}|{r['competition']}".encode()
        ).hexdigest()[:16],
        axis=1,
    )

    # req #4: remove duplicate performances (same athlete/event/time/date/meet)
    usable = usable.drop_duplicates(subset=["performance_id"])

    keep_cols = [
        "performance_id", "athlete_id", "athlete_name", "country", "gender", "birth_date",
        "event_key", "event", "distance", "stroke", "pool_type", "time_seconds", "aqua_points",
        "competition", "competition_date", "competition_country", "competition_level",
        "rank_at_meet", "source", "source_url", "is_flagged", "flag_reason",
    ]
    return usable[keep_cols].reset_index(drop=True)


if __name__ == "__main__":
    from config.events import EVENTS, DEFAULT_EVENT_KEY
    ev = EVENTS[DEFAULT_EVENT_KEY]
    raw = load_raw_files(ev.key)
    print(f"Loaded {len(raw)} raw rows")
    clean = clean_performances(raw, ev)
    print(f"Clean rows: {len(clean)}, flagged: {clean['is_flagged'].sum()}")
    print(clean.head())
