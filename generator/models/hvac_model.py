"""
HVAC Equipment & Physical Actuator Model.

Models 4 AC units located on North, East, South, and West walls.
Simulates spatial zone influence, inverter modulation (cooling_level 0.0 to 1.0),
actuator ramp-rate limits, and minimum ON/OFF dwell times to prevent unrealistic oscillation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from generator.config import HVACConfig
from generator.models.room import AC_DEFINITIONS, ZONES


@dataclass
class ACState:
    id: str
    wall: str
    setpoint_c: float
    state: str  # "ON" or "OFF"
    cooling_level: float  # 0.0 to 1.0
    cooling_capacity_watts: float
    zone_influence: Dict[str, float]
    time_in_state_seconds: float = 0.0


class HVACClusterModel:
    """Controls and simulates the 4 room AC units."""

    def __init__(self, config: HVACConfig = None):
        self.config = config or HVACConfig()
        self.units: Dict[str, ACState] = {}
        self._init_units()

    def _init_units(self):
        """Initialize all 4 AC units based on fixed physical layout."""
        for ac_id, info in AC_DEFINITIONS.items():
            self.units[ac_id] = ACState(
                id=ac_id,
                wall=info["wall"],
                setpoint_c=24.0,
                state="ON",
                cooling_level=0.3,
                cooling_capacity_watts=self.config.cooling_capacity_watts,
                zone_influence=dict(info["zone_influence"]),
                time_in_state_seconds=300.0,
            )

    def set_unit_setpoint(self, ac_id: str, setpoint_c: float):
        """Set target setpoint respecting valid control bounds."""
        if ac_id not in self.units:
            raise KeyError(f"Unknown AC ID: {ac_id}")
        clamped_sp = float(np.clip(setpoint_c, min(self.config.candidate_setpoints), max(self.config.candidate_setpoints)))
        self.units[ac_id].setpoint_c = round(clamped_sp, 1)

    def set_all_setpoints(self, setpoint_c: float):
        """Set uniform setpoint across all ACs."""
        for ac_id in self.units:
            self.set_unit_setpoint(ac_id, setpoint_c)

    def set_unit_state(self, ac_id: str, new_state: str, force: bool = False):
        """Change AC state between ON and OFF respecting minimum dwell time unless forced."""
        unit = self.units[ac_id]
        new_state = new_state.upper()
        if new_state == unit.state:
            return

        if not force:
            if unit.state == "ON" and unit.time_in_state_seconds < self.config.min_on_seconds:
                return  # Blocked by min ON time
            if unit.state == "OFF" and unit.time_in_state_seconds < self.config.min_off_seconds:
                return  # Blocked by min OFF time

        unit.state = new_state
        unit.time_in_state_seconds = 0.0
        if new_state == "OFF":
            unit.cooling_level = 0.0

    def compute_desired_cooling_level(self, ac_id: str, zone_temps: Dict[str, float]) -> float:
        """Inverter control logic: determine target cooling level from temperature error."""
        unit = self.units[ac_id]
        if unit.state == "OFF":
            return 0.0

        # Weighted temperature of the zones influenced by this AC
        effective_temp = sum(unit.zone_influence[z] * zone_temps[z] for z in ZONES)
        temp_error = effective_temp - unit.setpoint_c

        if temp_error <= -0.5:
            # Overcooling: throttle to idle or minimal capacity
            return 0.05
        elif temp_error <= 0.0:
            return 0.15
        else:
            # Proportional ramp with baseline
            target = 0.20 + (temp_error * self.config.proportional_gain)
            return float(np.clip(target, 0.0, self.config.max_cooling_level))

    def step(self, zone_temps: Dict[str, float], timestep_seconds: float):
        """Advance AC states and ramp cooling levels realistically toward target."""
        max_level_delta = self.config.ramp_rate_per_second * timestep_seconds

        for ac_id, unit in self.units.items():
            unit.time_in_state_seconds += timestep_seconds
            if unit.state == "OFF":
                unit.cooling_level = 0.0
                continue

            target_level = self.compute_desired_cooling_level(ac_id, zone_temps)
            level_diff = target_level - unit.cooling_level

            # Enforce actuator ramp rate
            if abs(level_diff) <= max_level_delta:
                unit.cooling_level = target_level
            else:
                unit.cooling_level += np.sign(level_diff) * max_level_delta

            unit.cooling_level = float(np.clip(unit.cooling_level, 0.0, self.config.max_cooling_level))
            unit.cooling_level = round(unit.cooling_level, 4)

    def get_cooling_watts_by_zone(self) -> Dict[str, float]:
        """Calculate total cooling heat removal rate (in Watts) per thermal zone."""
        cooling_by_zone = {z: 0.0 for z in ZONES}
        for unit in self.units.values():
            if unit.state == "ON":
                total_unit_watts = unit.cooling_capacity_watts * unit.cooling_level
                for zone, infl in unit.zone_influence.items():
                    cooling_by_zone[zone] += total_unit_watts * infl

        return {z: round(w, 2) for z, w in cooling_by_zone.items()}

    def get_total_cooling_watts(self) -> float:
        """Total cooling thermal extraction across all 4 units."""
        return round(sum(self.get_cooling_watts_by_zone().values()), 2)
