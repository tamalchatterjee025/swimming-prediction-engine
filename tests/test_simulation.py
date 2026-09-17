import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.simulation.monte_carlo import simulate_race


class TestMonteCarlo(unittest.TestCase):
    def setUp(self):
        self.athletes = [
            {"athlete_id": "A", "athlete_name": "A", "expected_time": 47.90, "time_std": 0.25},
            {"athlete_id": "B", "athlete_name": "B", "expected_time": 48.00, "time_std": 0.25},
            {"athlete_id": "C", "athlete_name": "C", "expected_time": 48.50, "time_std": 0.25},
        ]

    def test_probabilities_sum_to_one_per_position(self):
        out = simulate_race(self.athletes, n_simulations=50000, seed=1)
        self.assertAlmostEqual(out["win_probability"].sum(), 1.0, places=2)
        self.assertAlmostEqual(out["silver_probability"].sum(), 1.0, places=2)
        self.assertAlmostEqual(out["bronze_probability"].sum(), 1.0, places=2)

    def test_fastest_expected_time_has_highest_win_probability(self):
        out = simulate_race(self.athletes, n_simulations=50000, seed=1)
        best = out.sort_values("win_probability", ascending=False).iloc[0]
        self.assertEqual(best["athlete_id"], "A")

    def test_top3_probability_is_one_for_three_athlete_field(self):
        out = simulate_race(self.athletes, n_simulations=20000, seed=1)
        self.assertTrue((out["top3_probability"] > 0.999).all())

    def test_lower_uncertainty_favours_leader(self):
        tight = [
            {"athlete_id": "A", "athlete_name": "A", "expected_time": 47.90, "time_std": 0.05},
            {"athlete_id": "B", "athlete_name": "B", "expected_time": 48.00, "time_std": 0.05},
        ]
        loose = [
            {"athlete_id": "A", "athlete_name": "A", "expected_time": 47.90, "time_std": 0.6},
            {"athlete_id": "B", "athlete_name": "B", "expected_time": 48.00, "time_std": 0.6},
        ]
        tight_out = simulate_race(tight, n_simulations=50000, seed=1)
        loose_out = simulate_race(loose, n_simulations=50000, seed=1)
        tight_win = tight_out[tight_out["athlete_id"] == "A"]["win_probability"].iloc[0]
        loose_win = loose_out[loose_out["athlete_id"] == "A"]["win_probability"].iloc[0]
        self.assertGreater(tight_win, loose_win)


if __name__ == "__main__":
    unittest.main()
