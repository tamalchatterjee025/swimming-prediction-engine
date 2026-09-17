"""
Ingestion from the World Aquatics public rankings export API.

Discovered by inspecting https://www.worldaquatics.com/swimming/rankings :
the rankings widget's "Download CSV/XLSX/PDF" links point at a public,
unauthenticated JSON/CSV export endpoint:

    https://api.worldaquatics.com/fina/rankings/swimming/report/csv
        ?gender=M|F
        &distance=<metres>
        &stroke=FREESTYLE|BACKSTROKE|BREASTSTROKE|BUTTERFLY|MEDLEY
        &poolConfiguration=LCM|SCM
        &year=<yyyy>            (blank = all years)
        &startDate=&endDate=    (alternative to year)
        &timesMode=ALL_TIMES
        &regionId=&countryId=
        &pageSize=<n>            (server appears to cap around 200 rows)

This returns one row per swim (not one row per athlete), which is exactly
what is needed to build an athlete's recent-performance history. Fields:
meet_name, swim_time, swim_date, full_desc, team_code, team_short_name,
full_name_computed, gender, birth_date, event_id, standard_name, RANK,
Rank_Order, fina_points (= AQUA points), meet_city, country_code.

No official API key or auth is required. This is treated as the PRIMARY
data source per the project spec. If this endpoint changes or becomes
unreachable, `fetch_rankings_csv` raises IngestionError and the caller
(update_data.py) falls back to any CSV files manually dropped in data/raw/.
"""

import os
import sys
import time
import shutil
import subprocess
import datetime as dt
from urllib.parse import urlencode

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import WORLD_AQUATICS_API_BASE, RAW_DATA_DIR, HISTORY_YEARS
from config.events import EventConfig, GENDER_CODES

USER_AGENT = "Mozilla/5.0 (SwimmingPredictionEngine/0.1; +data ingestion)"
TIMEOUT_SECONDS = 30


class IngestionError(Exception):
    pass


def _build_url(event: EventConfig, gender_code: str, year: str, page_size: int, region_id: str = "") -> str:
    params = {
        "gender": gender_code,
        "distance": event.distance,
        "stroke": event.stroke,
        "poolConfiguration": event.pool_type,
        "year": year,
        "startDate": "",
        "endDate": "",
        "timesMode": "ALL_TIMES",
        "regionId": region_id,
        "countryId": "",
        "pageSize": page_size,
    }
    return f"{WORLD_AQUATICS_API_BASE}?{urlencode(params)}"


def _fetch_via_curl(url: str) -> str:
    """
    Fetch via the `curl` binary, using the OS certificate store.

    Some sandboxed/corporate network environments TLS-intercept with a root CA
    that is present in the OS trust store (which curl on Windows uses) but not
    in Python's bundled `certifi` store, causing `requests` to fail SSL
    verification even though the endpoint is perfectly reachable. Shelling out
    to curl sidesteps that mismatch. Falls back to `requests` if curl is absent.
    """
    if shutil.which("curl") is None:
        raise IngestionError("curl not available")
    result = subprocess.run(
        ["curl", "-sL", "-A", USER_AGENT, "--max-time", str(TIMEOUT_SECONDS), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise IngestionError(f"curl exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def fetch_rankings_csv(event: EventConfig, gender_code: str, year: str = "", page_size: int = 200,
                        max_retries: int = 3, region_id: str = "") -> str:
    """
    Fetch raw CSV text for one (event, gender, year) slice. Returns CSV text.

    `region_id` (e.g. "ASIA") narrows the ranking to swims by athletes from
    that continent only. The server caps each response at ~200 rows
    regardless of pageSize, so a global (unfiltered) pull only captures the
    world top-200 for the year -- continentally-competitive swimmers who
    aren't globally elite (e.g. many Asian Games-level swimmers) fall outside
    that window entirely. Pulling additional region-filtered slices goes
    deeper into each continent's own ranking instead.
    """
    url = _build_url(event, gender_code, year, page_size, region_id=region_id)
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            try:
                text = _fetch_via_curl(url)
            except IngestionError:
                resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
                if resp.status_code != 200:
                    raise IngestionError(f"HTTP {resp.status_code} for {resp.url}")
                text = resp.text
            if not text.strip().startswith("meet_name"):
                raise IngestionError(f"Unexpected response body (not CSV) for {url}")
            return text
        except (requests.RequestException, IngestionError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * attempt)
    raise IngestionError(f"Failed to fetch after {max_retries} attempts: {last_err}")


def save_raw(text: str, event: EventConfig, gender_code: str, year: str, region_id: str = "") -> str:
    """Persist the raw CSV response, never overwriting previous pulls."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    year_tag = year if year else "ALL"
    if region_id:
        year_tag = f"{year_tag}_{region_id}"
    fname = f"worldaquatics_{event.key}_{gender_code}_{year_tag}_{stamp}.csv"
    path = os.path.join(RAW_DATA_DIR, fname)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return path


def fetch_event_history(event: EventConfig, years_back: int = HISTORY_YEARS, page_size: int = 200,
                         extra_regions: list = None):
    """
    Fetch recent-years history for both genders for the given event.

    In addition to the global (world top-200) pull, also fetches a
    region-filtered slice for each region in `extra_regions` (e.g. ["ASIA"])
    so continentally-competitive swimmers who don't crack the global top-200
    -- but would still contend at a continental championship like the Asian
    Games -- are captured too. Defaults to config.settings.EXTRA_REGIONS.

    Returns list of dicts: {gender_code, year, region, path, rows, status, message}
    """
    from config.settings import EXTRA_REGIONS
    extra_regions = extra_regions if extra_regions is not None else EXTRA_REGIONS

    current_year = dt.datetime.now().year
    years = [str(y) for y in range(current_year - years_back + 1, current_year + 1)]
    region_slices = [""] + list(extra_regions)
    results = []
    for gender_code in GENDER_CODES.values():
        for year in years:
            for region_id in region_slices:
                try:
                    text = fetch_rankings_csv(event, gender_code, year=year, page_size=page_size, region_id=region_id)
                    path = save_raw(text, event, gender_code, year, region_id=region_id)
                    n_rows = max(0, text.count("\n") - 1)
                    results.append({
                        "gender": gender_code, "year": year, "region": region_id or "GLOBAL", "path": path,
                        "rows": n_rows, "status": "ok", "message": "",
                    })
                except IngestionError as e:
                    results.append({
                        "gender": gender_code, "year": year, "region": region_id or "GLOBAL", "path": None,
                        "rows": 0, "status": "failed", "message": str(e),
                    })
    return results


if __name__ == "__main__":
    from config.events import EVENTS, DEFAULT_EVENT_KEY
    ev = EVENTS[DEFAULT_EVENT_KEY]
    res = fetch_event_history(ev)
    for r in res:
        print(r["status"], r["gender"], r["year"], r["rows"], r["path"] or r["message"])
