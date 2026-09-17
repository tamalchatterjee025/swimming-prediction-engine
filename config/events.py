"""
Event configuration registry.

Adding a new event/distance/stroke/pool combination for future versions only
requires adding an entry here -- no other code should hard-code event details.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EventConfig:
    key: str                 # internal identifier, e.g. "100_FREE_LCM"
    display_name: str        # e.g. "100m Freestyle"
    distance: int            # metres, e.g. 100
    stroke: str              # World Aquatics stroke code, e.g. "FREESTYLE"
    pool_type: str           # "LCM" (50m long course) or "SCM" (25m short course)


# World Aquatics stroke codes used by the rankings API.
STROKE_CODES = {
    "FREESTYLE": "FREESTYLE",
    "BACKSTROKE": "BACKSTROKE",
    "BREASTSTROKE": "BREASTSTROKE",
    "BUTTERFLY": "BUTTERFLY",
    "MEDLEY": "MEDLEY",
}

GENDER_CODES = {
    "MEN": "M",
    "WOMEN": "F",
}

# MVP event: 100m Freestyle, Long Course (50m pool).
# Additional events can be appended without touching ingestion/model code.
EVENTS = {
    "100_FREE_LCM": EventConfig(
        key="100_FREE_LCM",
        display_name="100m Freestyle",
        distance=100,
        stroke="FREESTYLE",
        pool_type="LCM",
    ),
    # Future events (not populated with data yet, but supported by the schema):
    "50_FREE_LCM": EventConfig("50_FREE_LCM", "50m Freestyle", 50, "FREESTYLE", "LCM"),
    "200_FREE_LCM": EventConfig("200_FREE_LCM", "200m Freestyle", 200, "FREESTYLE", "LCM"),
    "400_FREE_LCM": EventConfig("400_FREE_LCM", "400m Freestyle", 400, "FREESTYLE", "LCM"),
    "100_BACK_LCM": EventConfig("100_BACK_LCM", "100m Backstroke", 100, "BACKSTROKE", "LCM"),
    "100_BREAST_LCM": EventConfig("100_BREAST_LCM", "100m Breaststroke", 100, "BREASTSTROKE", "LCM"),
    "100_FLY_LCM": EventConfig("100_FLY_LCM", "100m Butterfly", 100, "BUTTERFLY", "LCM"),
    "200_IM_LCM": EventConfig("200_IM_LCM", "200m Individual Medley", 200, "MEDLEY", "LCM"),
}

DEFAULT_EVENT_KEY = "100_FREE_LCM"

# Meet-level classification, used for the (small) "major meet performance"
# feature component. Matching is a case-insensitive substring search against
# the meet_name / full_desc fields returned by World Aquatics.
MAJOR_MEET_KEYWORDS = [
    "Olympic Games",
    "World Championships",
    "World Aquatics Championships",
    "Asian Games",
    "Commonwealth Games",
    "Pan Pacific",
    "European Championships",
    "European Aquatics Championships",
]
