"""
Air Conditioning (HVAC) System Model for HVEAC V3.

Simulates 4 perimeter AC units:
- AC-1: North Wall (primarily cools Zone 1 and Zone 2)
- AC-2: East Wall  (primarily cools Zone 2 and Zone 3)
- AC-3: South Wall (primarily cools Zone 3 and Zone 4)
- AC-4: West Wall  (primarily cools Zone 4 and Zone 1)

Implements continuous spatial cooling distribution through a spatial influence matrix.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from simulator.config import HvacConfig
from simulator.room import AC_IDS, FIXED_AC_UNITS, ZONE_IDS


@dataclass
class SingleAcState:
    """Operating state of an individual AC unit."""
    ac_id: str
    wall: str
    state: str                 # 'OFF', 'COOLING', 'IDLE'
    cooling_level: float       # 0.0 (OFF) to 1.0 (MAX)
    setpoint_c: float
    delivered_cooling_w: float


@dataclass
class HvacSystemState:
    """Overall HVAC system output and spatial cooling distribution."""
    ac_units: Dict[str, SingleAcState]
    zone_cooling_w: Dict[str, float]
    total_cooling_w: float


class HvacSystemModel:
    """Manages the 4 perimeter air conditioner units."""

    def __init__(self, config: Optional[HvacConfig] = None):
        self.config = config or HvacConfig()
        self._weights = self.config.spatial_influence_weights
        self._ac_order = [u.ac_id for u in FIXED_AC_UNITS]
        self._zone_order = list(ZONE_IDS)

        self._units: Dict[str, Dict] = {}
        for loc in FIXED_AC_UNITS:
            self._units[loc.ac_id] = {
                "wall": loc.wall,
                "cooling_level": 0.0,
                "setpoint_c": self.config.default_setpoint_c,
                "state": "OFF",
            }

    def set_unit_control(self, ac_id: str, cooling_level: float, setpoint_c: Optional[float] = None):
        """Set cooling level (0.0 to 1.0) and optional setpoint for an AC unit."""
        if ac_id not in self._units:
            return
        level = min(max(float(cooling_level), 0.0), 1.0)
        self._units[ac_id]["cooling_level"] = round(level, 4)
        if setpoint_c is not None:
            self._units[ac_id]["setpoint_c"] = round(float(setpoint_c), 2)
        self._units[ac_id]["state"] = "OFF" if level == 0.0 else "COOLING"

    def set_all_controls(self, cooling_levels: Dict[str, float], setpoints: Optional[Dict[str, float]] = None):
        """Set controls across all units simultaneously."""
        for ac_id, lvl in cooling_levels.items():
            sp = setpoints.get(ac_id) if setpoints else None
            self.set_unit_control(ac_id, lvl, sp)

    def calculate_cooling_distribution(self) -> HvacSystemState:
        """
        Calculates heat removal delivered to each zone via spatial influence weights:
        Q_hvac,z = sum_k (W_k,z * cooling_level_k * Q_nominal)
        """
        unit_states: Dict[str, SingleAcState] = {}
        ac_cooling_w: List[float] = []

        for idx, ac_id in enumerate(self._ac_order):
            unit = self._units[ac_id]
            lvl = unit["cooling_level"]
            delivered = round(lvl * self.config.nominal_cooling_capacity_w, 2)
            ac_cooling_w.append(delivered)
            unit_states[ac_id] = SingleAcState(
                ac_id=ac_id,
                wall=unit["wall"],
                state=unit["state"],
                cooling_level=lvl,
                setpoint_c=unit["setpoint_c"],
                delivered_cooling_w=delivered,
            )

        # Distribute across zones: zone_j = sum_k (weights[k][j] * ac_cooling[k])
        zone_cooling: Dict[str, float] = {}
        for j, z_id in enumerate(self._zone_order):
            q_z = 0.0
            for k in range(len(self._ac_order)):
                q_z += self._weights[k][j] * ac_cooling_w[k]
            zone_cooling[z_id] = round(q_z, 2)

        total_cooling = round(sum(ac_cooling_w), 2)

        return HvacSystemState(
            ac_units=unit_states,
            zone_cooling_w=zone_cooling,
            total_cooling_w=total_cooling,
        )
