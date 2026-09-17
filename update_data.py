"""
Entry point: `python update_data.py`

1. Fetch latest World Aquatics ranking data for all configured events.
2. Clean + validate.
3. Load into SQLite (idempotent -- duplicates are skipped via UNIQUE constraint).
4. Print a short update log.

If live ingestion fails for a slice, that failure is logged and the pipeline
continues with whatever raw CSVs already exist in data/raw/ (including any
manually dropped-in World Aquatics CSV export -- see README "Failure Handling").
"""

import sys
import datetime as dt

from config.events import EVENTS, DEFAULT_EVENT_KEY
from src.ingestion.world_aquatics import fetch_event_history, IngestionError
from src.ingestion.load_to_db import load_event_to_db
from src.utilities.db import init_db


def main(event_keys=None):
    event_keys = event_keys or [DEFAULT_EVENT_KEY]
    print(f"=== Swimming Prediction Engine: data update ({dt.datetime.now().isoformat()}) ===")
    init_db()

    for key in event_keys:
        event = EVENTS[key]
        print(f"\n--- Event: {event.display_name} ({event.pool_type}) ---")
        print("Step 1/3: fetching from World Aquatics...")
        try:
            fetch_results = fetch_event_history(event)
            ok = sum(1 for r in fetch_results if r["status"] == "ok")
            failed = sum(1 for r in fetch_results if r["status"] == "failed")
            total_rows = sum(r["rows"] for r in fetch_results)
            print(f"  fetched {ok} slices OK, {failed} failed, {total_rows} raw rows pulled")
            for r in fetch_results:
                if r["status"] == "failed":
                    print(f"    FAILED gender={r['gender']} year={r['year']}: {r['message']}")
        except Exception as e:
            print(f"  Live ingestion raised an error, continuing with existing raw files: {e}")

        print("Step 2/3: cleaning + validating...")
        print("Step 3/3: loading into SQLite (duplicates skipped)...")
        result = load_event_to_db(event)
        print(f"  rows_fetched(clean)={result['rows_fetched']}  rows_newly_inserted={result['rows_inserted']}")

    print("\n=== Update complete ===")


if __name__ == "__main__":
    keys = sys.argv[1:] or None
    main(keys)
