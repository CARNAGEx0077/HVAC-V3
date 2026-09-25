"""
Fixed Room Geometry and Physical Component Positioning for HVEAC V3.

Defines the 2D layout:
- Room: 12.0m (W) x 10.0m (L) x 3.0m (H)
- 4 Thermal Zones:
    ZONE_1: North-West [X: 0..6, Y: 5..10]
    ZONE_2: North-East [X: 6..12, Y: 5..10]
    ZONE_3: South-East [X: 6..12, Y: 0..5]
    ZONE_4: South-West [X: 0..6, Y: 0..5]
- 4 AC Units on Perimeter Walls:
    AC-1: North Wall (x=6.0, y=10.0)
    AC-2: East Wall  (x=12.0, y=5.0)
    AC-3: South Wall (x=6.0, y=0.0)
    AC-4: West Wall  (x=0.0, y=5.0)
- 10 Fixed Computer Desks (Computers 1-4 form a tight cluster in ZONE_1).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


ZONE_IDS = ("ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4")
AC_IDS = ("AC-1", "AC-2", "AC-3", "AC-4")


@dataclass(frozen=True)
class ComputerLocation:
    """Fixed physical placement of a computer."""
    computer_id: int
    name: str
    x: float
    y: float
    zone: str


@dataclass(frozen=True)
class AcUnitLocation:
    """Fixed physical mounting of an AC unit."""
    ac_id: str
    wall: str
    x: float
    y: float


@dataclass(frozen=True)
class ZoneDefinition:
    """Spatial boundaries of a thermal zone."""
    zone_id: str
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    adjacent_zones: Tuple[str, ...]


# Fixed 10 Computers: Placed at fixed coordinates across all scenarios.
# Computers 1-4 form a physical cluster in ZONE_1 (NW corner).
FIXED_COMPUTERS: Tuple[ComputerLocation, ...] = (
    ComputerLocation(1, "Computer 1", 2.0, 8.0, "ZONE_1"),
    ComputerLocation(2, "Computer 2", 4.0, 8.0, "ZONE_1"),
    ComputerLocation(3, "Computer 3", 2.0, 6.5, "ZONE_1"),
    ComputerLocation(4, "Computer 4", 4.0, 6.5, "ZONE_1"),
    ComputerLocation(5, "Computer 5", 8.0, 8.0, "ZONE_2"),
    ComputerLocation(6, "Computer 6", 10.0, 6.5, "ZONE_2"),
    ComputerLocation(7, "Computer 7", 10.0, 3.5, "ZONE_3"),
    ComputerLocation(8, "Computer 8", 8.0, 2.0, "ZONE_3"),
    ComputerLocation(9, "Computer 9", 4.0, 2.0, "ZONE_4"),
    ComputerLocation(10, "Computer 10", 2.0, 3.5, "ZONE_4"),
)


# Fixed 4 AC units on perimeter walls
FIXED_AC_UNITS: Tuple[AcUnitLocation, ...] = (
    AcUnitLocation("AC-1", "north", 6.0, 10.0),
    AcUnitLocation("AC-2", "east", 12.0, 5.0),
    AcUnitLocation("AC-3", "south", 6.0, 0.0),
    AcUnitLocation("AC-4", "west", 0.0, 5.0),
)


# Fixed Zone Boundaries and Adjacency
FIXED_ZONES: Dict[str, ZoneDefinition] = {
    "ZONE_1": ZoneDefinition(
        zone_id="ZONE_1",
        name="North-West Zone",
        x_min=0.0,
        x_max=6.0,
        y_min=5.0,
        y_max=10.0,
        adjacent_zones=("ZONE_2", "ZONE_4"),
    ),
    "ZONE_2": ZoneDefinition(
        zone_id="ZONE_2",
        name="North-East Zone",
        x_min=6.0,
        x_max=12.0,
        y_min=5.0,
        y_max=10.0,
        adjacent_zones=("ZONE_1", "ZONE_3"),
    ),
    "ZONE_3": ZoneDefinition(
        zone_id="ZONE_3",
        name="South-East Zone",
        x_min=6.0,
        x_max=12.0,
        y_min=0.0,
        y_max=5.0,
        adjacent_zones=("ZONE_2", "ZONE_4"),
    ),
    "ZONE_4": ZoneDefinition(
        zone_id="ZONE_4",
        name="South-West Zone",
        x_min=0.0,
        x_max=6.0,
        y_min=0.0,
        y_max=5.0,
        adjacent_zones=("ZONE_1", "ZONE_3"),
    ),
}


class RoomLayout:
    """Provides querying and spatial mapping for the fixed room."""

    def __init__(self):
        self.zones = FIXED_ZONES
        self.computers = FIXED_COMPUTERS
        self.ac_units = FIXED_AC_UNITS
        self._comp_by_id = {c.computer_id: c for c in self.computers}
        self._ac_by_id = {a.ac_id: a for a in self.ac_units}
        self._comps_by_zone = {z: [] for z in ZONE_IDS}
        for c in self.computers:
            self._comps_by_zone[c.zone].append(c)

    def get_computer(self, computer_id: int) -> ComputerLocation:
        return self._comp_by_id[computer_id]

    def get_computers_in_zone(self, zone_id: str) -> List[ComputerLocation]:
        return list(self._comps_by_zone.get(zone_id, []))

    def get_ac(self, ac_id: str) -> AcUnitLocation:
        return self._ac_by_id[ac_id]

    def locate_zone(self, x: float, y: float) -> str:
        """Resolve (x, y) coordinates to zone ID."""
        for zone_id, z in self.zones.items():
            if z.x_min <= x <= z.x_max and z.y_min <= y <= z.y_max:
                return zone_id
        # Fallback clamped
        x_clamped = min(max(x, 0.0), 12.0)
        y_clamped = min(max(y, 0.0), 10.0)
        if x_clamped < 6.0:
            return "ZONE_1" if y_clamped >= 5.0 else "ZONE_4"
        else:
            return "ZONE_2" if y_clamped >= 5.0 else "ZONE_3"
