# Swimming Prediction Engine (MVP)

A lightweight, statistically-driven prediction engine for competitive swimming.
MVP scope: **Men's & Women's 100m Freestyle, Long Course (50m pool)**.

Given a set of finalists, it estimates each athlete's expected time, uncertainty,
and (via Monte Carlo simulation) win/silver/bronze/top-3 probability and expected
finishing position — built on **official World Aquatics performance data**, not
arbitrary scores.

## Quick start

```bash
pip install -r requirements.txt
python update_data.py      # pulls official World Aquatics data into SQLite
streamlit run app.py
```

The app has 4 pages (sidebar navigation):
1. **Race Predictor** — pick an event/gender, add finalists, run the simulation.
2. **Athlete Profile** — recent form, PB, season best, AQUA points, trend, consistency.
3. **Model Validation** — backtested accuracy against real historical finals (leakage-checked).
4. **Data Status** — row counts, date coverage, flagged records, ingestion log.

## Data source

**Primary (and currently only) source: World Aquatics.**
Inspecting `https://www.worldaquatics.com/swimming/rankings` shows the "Download"
buttons point at a public, unauthenticated export endpoint:

```
https://api.worldaquatics.com/fina/rankings/swimming/report/csv
  ?gender=M|F&distance=100&stroke=FREESTYLE&poolConfiguration=LCM
  &year=YYYY&timesMode=ALL_TIMES&pageSize=200
```

This returns one row per swim (time, date, meet, athlete, country, AQUA points,
meet rank) — no HTML scraping required. `update_data.py` pulls the last 5 years
for both genders and stores the raw CSV responses, untouched, in `data/raw/`
(files are never overwritten — each pull gets a timestamped filename).

**Fallback:** if the live endpoint is unreachable, drop a World Aquatics CSV
export into `data/raw/` (same column layout) named
`worldaquatics_<event_key>_<gender>_<year>_<anything>.csv` and re-run
`python update_data.py` — the cleaning/loading steps work identically on a
manually-provided file.

## Project structure

```
app.py                     Streamlit entry point (Page 1: Race Predictor)
pages/                     Streamlit pages 2-4
update_data.py             `python update_data.py` — refresh the dataset
config/
  events.py                 Event registry (distance/stroke/pool/gender) — add
                             new events here only, no code changes needed elsewhere
  settings.py                Model weights, simulation count, file paths
data/
  raw/                        Untouched World Aquatics CSV pulls (timestamped)
  processed/                   (reserved for future intermediate artifacts)
  database/                     swimming.db (SQLite)
src/
  ingestion/                     World Aquatics client + SQLite loader
  cleaning/                       Time/date parsing, name/country standardisation, flagging
  features/                        Per-athlete feature engineering (recent form, trend, etc.)
  modelling/                        Expected-time model + prediction pipeline
  simulation/                        Monte Carlo race simulator
  validation/                        Backtesting + calibration
  utilities/                          DB schema/connection helper
tests/                                 Unit tests (cleaning, features, simulation)
```

## How the model works

1. **Expected time** = weighted blend of:
   - Recency-weighted recent races (35/25/18/12/10% by recency) — dominant signal
   - Current-season best
   - All-time personal best (supporting signal)
   - AQUA-points-implied time (small secondary signal, fit via linear regression
     of time vs. AQUA points across the full dataset — not hard-coded)
   - Small nudge from recent trend (linear slope of last 5 races)
   - Small (±0.15s max) adjustment for historical major-meet performance, only
     applied when an athlete has ≥3 major and ≥3 non-major swims on file
2. **Uncertainty (time_std)** = the athlete's own recent-race standard deviation
   (floor 0.12s, inflated 1.4x if fewer than 3 races on file).
3. **Monte Carlo simulation**: 100,000 draws from `Normal(expected_time, time_std)`
   per athlete per simulation; each simulation is ranked to get a finishing order.
   Gold/Silver/Bronze/Top-3 probabilities and expected position are the empirical
   frequencies/averages across all simulations.
4. **Backtesting**: for every historical "Major" meet final already in the
   database with a plausible field size (4-10 swimmers), the model is
   re-run using `as_of_date = that final's date`, so only pre-race data is
   used. Predicted probabilities are compared to actual outcomes to compute
   Brier score, log loss, top-1/top-3 hit rate, and calibration curves.

## Running tests

```bash
python -m unittest discover -s tests -v
```

## Extending to more events

Add an entry to `config/events.py` (distance/stroke/pool/gender), then run
`python update_data.py <new_event_key>` — ingestion, cleaning, features,
modelling, and simulation are all event-agnostic and read the event
configuration, not hard-coded event logic.
