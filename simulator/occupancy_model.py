"""
Occupancy Model for HVEAC V3 Thermal Simulation.

Represents human occupancy strictly through deterministic zone counts.
Calculates sensible metabolic heat release per zone using ASHRAE standard
sedentary heat factors (~85 W / person).
"""

from dataclasses import dataclass
from typing import Dict, Optional
from simulator.config import OccupancyHeatConfig
from simulator.room import ZONE_IDS


@dataclass
class OccupancyState:
    """Snapshot of occupancy distribution and metabolic heat output."""
    zone_occupancy: Dict[str, int]
    total_occupancy: int
    zone_heat_watts: Dict[str, float]
    total_heat_watts: float


class OccupancyModel:
    """Manages zone occupancy levels and sensible metabolic heat calculation."""

    def __init__(self, config: Optional[OccupancyHeatConfig] = None):
        self.config = config or OccupancyHeatConfig()
        self._counts: Dict[str, int] = {z: 0 for z in ZONE_IDS}

    def set_zone_counts(self, counts: Dict[str, int]) -> OccupancyState:
        """Update occupancy count per zone."""
        for z in ZONE_IDS:
            if z in counts:
                self._counts[z] = max(int(counts[z]), 0)
        return self.get_state()

    def get_state(self) -> OccupancyState:
        zone_heat = {
            z: round(self._counts[z] * self.config.sensible_heat_per_person_w, 2)
            for z in ZONE_IDS
        }
        total_heat = round(sum(zone_heat.values()), 2)
        total_occ = sum(self._counts.values())
        return OccupancyState(
            zone_occupancy=dict(self._counts),
            total_occupancy=total_occ,
            zone_heat_watts=zone_heat,
            total_heat_watts=total_heat,
        )
