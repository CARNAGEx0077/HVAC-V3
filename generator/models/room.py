"""
Room Geometry and Fixed Scene Configuration.

Defines the fixed physical room geometry, 4 thermal zones,
fixed AC locations, and fixed computer placement across all scenarios.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


ZONES: List[str] = ["ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"]

# Adjacency definitions for heat conduction
ZONE_ADJACENCIES: Dict[str, Dict[str, str]] = {
    "ZONE_1": {"east": "ZONE_2", "south": "ZONE_4", "diagonal": "ZONE_3"},
    "ZONE_2": {"west": "ZONE_1", "south": "ZONE_3", "diagonal": "ZONE_4"},
    "ZONE_3": {"north": "ZONE_2", "west": "ZONE_4", "diagonal": "ZONE_1"},
    "ZONE_4": {"north": "ZONE_1", "east": "ZONE_3", "diagonal": "ZONE_2"},
}

# Fixed AC definitions: ID, Wall, and spatial zone influence distribution
# Total influence per AC sums to 1.0
AC_DEFINITIONS: Dict[str, Dict] = {
    "AC-1": {
        "wall": "north",
        "position": (5.0, 10.0),
        "zone_influence": {
            "ZONE_1": 0.45,
            "ZONE_2": 0.45,
            "ZONE_3": 0.05,
            "ZONE_4": 0.05,
        },
    },
    "AC-2": {
        "wall": "east",
        "position": (10.0, 5.0),
        "zone_influence": {
            "ZONE_1": 0.05,
            "ZONE_2": 0.45,
            "ZONE_3": 0.45,
            "ZONE_4": 0.05,
        },
    },
    "AC-3": {
        "wall": "south",
        "position": (5.0, 0.0),
        "zone_influence": {
            "ZONE_1": 0.05,
            "ZONE_2": 0.05,
            "ZONE_3": 0.45,
            "ZONE_4": 0.45,
        },
    },
    "AC-4": {
        "wall": "west",
        "position": (0.0, 5.0),
        "zone_influence": {
            "ZONE_1": 0.45,
            "ZONE_2": 0.05,
            "ZONE_3": 0.05,
            "ZONE_4": 0.45,
        },
    },
}

# Fixed 10 Computer placements across all scenarios
# Computers 1-4 are tightly clustered in ZONE_1 (North-West)
# Computers 5-6 in ZONE_2 (North-East)
# Computers 7-8 in ZONE_3 (South-East)
# Computers 9-10 in ZONE_4 (South-West)
COMPUTER_DEFINITIONS: Dict[str, Dict] = {
    "Computer 1": {"zone": "ZONE_1", "position": (1.5, 8.5)},
    "Computer 2": {"zone": "ZONE_1", "position": (3.0, 8.5)},
    "Computer 3": {"zone": "ZONE_1", "position": (1.5, 7.0)},
    "Computer 4": {"zone": "ZONE_1", "position": (3.0, 7.0)},
    "Computer 5": {"zone": "ZONE_2", "position": (7.0, 8.5)},
    "Computer 6": {"zone": "ZONE_2", "position": (8.5, 7.0)},
    "Computer 7": {"zone": "ZONE_3", "position": (8.5, 2.5)},
    "Computer 8": {"zone": "ZONE_3", "position": (7.0, 1.5)},
    "Computer 9": {"zone": "ZONE_4", "position": (2.5, 1.5)},
    "Computer 10": {"zone": "ZONE_4", "position": (1.5, 3.0)},
}


@dataclass(frozen=True)
class RoomScene:
    """Immutable representation of the fixed room physical scene."""
    zones: Tuple[str, ...] = tuple(ZONES)
    computers: Tuple[str, ...] = tuple(COMPUTER_DEFINITIONS.keys())
    acs: Tuple[str, ...] = tuple(AC_DEFINITIONS.keys())

    @staticmethod
    def get_computer_zone(computer_id: str) -> str:
        return COMPUTER_DEFINITIONS[computer_id]["zone"]

    @staticmethod
    def get_ac_wall(ac_id: str) -> str:
        return AC_DEFINITIONS[ac_id]["wall"]

    @staticmethod
    def get_ac_influence(ac_id: str) -> Dict[str, float]:
        return dict(AC_DEFINITIONS[ac_id]["zone_influence"])


FIXED_SCENE = RoomScene()
