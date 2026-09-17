import sys
import os
import unittest
import datetime as dt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cleaning.clean import time_str_to_seconds, parse_date, standardise_name, classify_meet_level, make_athlete_id


class TestTimeParsing(unittest.TestCase):
    def test_seconds_only(self):
        self.assertEqual(time_str_to_seconds("46.40"), 46.40)

    def test_minutes_seconds(self):
        self.assertEqual(time_str_to_seconds("1:46.40"), 106.40)

    def test_invalid(self):
        self.assertIsNone(time_str_to_seconds("DNS"))
        self.assertIsNone(time_str_to_seconds(""))
        self.assertIsNone(time_str_to_seconds(None))


class TestDateParsing(unittest.TestCase):
    def test_dmy(self):
        self.assertEqual(parse_date("31/07/2024"), dt.date(2024, 7, 31))

    def test_invalid(self):
        self.assertIsNone(parse_date("not-a-date"))
        self.assertIsNone(parse_date(""))
        self.assertIsNone(parse_date(None))


class TestNameStandardisation(unittest.TestCase):
    def test_lastname_firstname(self):
        self.assertEqual(standardise_name("PAN, Zhanle"), "Pan, Zhanle")

    def test_extra_whitespace(self):
        self.assertEqual(standardise_name("PAN,   Zhanle  "), "Pan, Zhanle")


class TestMeetLevel(unittest.TestCase):
    def test_major(self):
        self.assertEqual(classify_meet_level("Olympic Games Paris 2024"), "Major")
        self.assertEqual(classify_meet_level("World Aquatics Championships - Singapore 2025"), "Major")

    def test_other(self):
        self.assertEqual(classify_meet_level("Some Regional Open Meet"), "Other")


class TestAthleteId(unittest.TestCase):
    def test_stable(self):
        id1 = make_athlete_id("Pan, Zhanle", "CHN", "04/08/2004")
        id2 = make_athlete_id("Pan, Zhanle", "CHN", "04/08/2004")
        self.assertEqual(id1, id2)

    def test_different_athlete(self):
        id1 = make_athlete_id("Pan, Zhanle", "CHN", "04/08/2004")
        id2 = make_athlete_id("Popovici, David", "ROU", "15/09/2004")
        self.assertNotEqual(id1, id2)


if __name__ == "__main__":
    unittest.main()
