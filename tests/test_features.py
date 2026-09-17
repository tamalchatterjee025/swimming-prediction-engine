import sys
import os
import unittest
import datetime as dt

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.athlete_features import (
    recent_weighted_time, consistency_std, trend_slope, filter_as_of, build_athlete_features,
)


def make_perf_df(times, dates, athlete_id="A1"):
    return pd.DataFrame({
        "athlete_id": [athlete_id] * len(times),
        "time_seconds": times,
        "competition_date": dates,
        "aqua_points": [900] * len(times),
        "competition_level": ["Other"] * len(times),
        "is_flagged": [0] * len(times),
    })


class TestConsistency(unittest.TestCase):
    def test_low_variance_vs_high_variance(self):
        # Spec example: these two groups have similar means but different spread.
        consistent = make_perf_df(
            [48.00, 48.02, 48.04, 48.01],
            [dt.date(2026, 1, 1), dt.date(2026, 2, 1), dt.date(2026, 3, 1), dt.date(2026, 4, 1)],
        ).sort_values("competition_date", ascending=False)
        volatile = make_perf_df(
            [47.70, 48.35, 47.95, 48.20],
            [dt.date(2026, 1, 1), dt.date(2026, 2, 1), dt.date(2026, 3, 1), dt.date(2026, 4, 1)],
        ).sort_values("competition_date", ascending=False)

        std_consistent = consistency_std(consistent)
        std_volatile = consistency_std(volatile)
        self.assertLess(std_consistent, std_volatile)


class TestTrend(unittest.TestCase):
    def test_improving_vs_slowing(self):
        improving = make_perf_df(
            [47.98, 48.03, 48.15, 48.22, 48.40],
            [dt.date(2026, 5, 1), dt.date(2026, 4, 1), dt.date(2026, 3, 1), dt.date(2026, 2, 1), dt.date(2026, 1, 1)],
        )  # already sorted most-recent-first, fastest most recent
        slowing = make_perf_df(
            [48.22, 48.15, 48.08, 48.00, 47.90],
            [dt.date(2026, 5, 1), dt.date(2026, 4, 1), dt.date(2026, 3, 1), dt.date(2026, 2, 1), dt.date(2026, 1, 1)],
        )
        slope_improving = trend_slope(improving)
        slope_slowing = trend_slope(slowing)
        self.assertLess(slope_improving, 0)   # getting faster over time -> negative slope
        self.assertGreater(slope_slowing, 0)  # getting slower over time -> positive slope


class TestLeakageGuard(unittest.TestCase):
    def test_future_races_excluded(self):
        df = make_perf_df(
            [48.0, 47.5, 46.9],
            [dt.date(2026, 1, 1), dt.date(2026, 6, 1), dt.date(2026, 12, 1)],
        )
        as_of = dt.date(2026, 7, 1)
        filtered = filter_as_of(df, as_of)
        # The 46.9 on 2026-12-01 is AFTER as_of and must not appear.
        self.assertNotIn(46.9, filtered["time_seconds"].tolist())
        self.assertEqual(len(filtered), 2)


class TestRecentWeighted(unittest.TestCase):
    def test_weights_favour_most_recent(self):
        df = make_perf_df(
            [50.0, 46.0],
            [dt.date(2026, 6, 1), dt.date(2026, 1, 1)],
        ).sort_values("competition_date", ascending=False)
        weighted = recent_weighted_time(df, weights=[0.8, 0.2])
        # weight is heavily on the more recent (50.0), so result should be closer to 50 than to 46
        self.assertGreater(weighted, 48.0)


if __name__ == "__main__":
    unittest.main()
