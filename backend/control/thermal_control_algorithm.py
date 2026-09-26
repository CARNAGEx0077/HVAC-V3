"""
Thermal Control Algorithm — Deterministic Prototype HVAC Controller.

This is the PROTOTYPE substitute for a future trained HVEAC AI model.
It uses three primary physical signals to determine per-zone cooling demand:

    A. TEMPERATURE ERROR — deviation from comfort target
    B. THERMAL LOAD — computer + occupancy heat in each zone
    C. TEMPERATURE TREND — rate of temperature change (rising = more cooling)

Runtime architecture:
    SIMULATION STATE → THERMAL CONTROL ALGORITHM → SAFETY GOVERNOR → HVAC ACTUATORS

This module does NOT:
    - Use machine learning
    - Load model files
    - Perform candidate search
    - Optimize a cost function
    - Compare multiple actions

It IS:
    - Deterministic
    - Transparent
    - Spatially aware (per-zone, per-AC)
    - Directly replaceable by a future HVEAC Brain
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hveac.thermal_control")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

COMFORT_TARGET_C = 23.5
DEADBAND_C = 0.3

# Demand calculation parameters
TEMP_ERROR_RANGE_C = 2.5       # temperature_factor saturates at this error
TEMP_DEMAND_MAX = 60.0         # max contribution from temperature signal

HEAT_LOAD_RANGE_W = 2000.0    # heat_factor saturates at this load
HEAT_DEMAND_MAX = 30.0         # max contribution from heat load signal

TREND_RANGE_C_PER_MIN = 0.20  # trend_factor saturates at this rate
TREND_DEMAND_MAX = 10.0        # max contribution from trend signal

# Setpoint mapping
SETPOINT_HIGH_C = 24.0         # setpoint when cooling_percent = 0%
SETPOINT_RANGE_C = 2.0         # setpoint swing (24.0 → 22.0)


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneControlState:
    """Computed control state for a single thermal zone."""
    zone_id: str
    temperature_c: float
    heat_load_w: float
    temperature_trend_c_per_min: float
    temperature_demand: float
    heat_demand: float
    trend_demand: float
    cooling_demand: float   # 0–100%


@dataclass
class HVACDecision:
    """
    Complete HVAC control decision output.
    This is the interface that a future HVEAC Brain would also produce.
    """
    # Per-zone control
    zone_states: Dict[str, ZoneControlState] = field(default_factory=dict)

    # Per-AC output
    ac_cooling_percent: Dict[str, float] = field(default_factory=dict)  # 0–100
    ac_setpoint_c: Dict[str, float] = field(default_factory=dict)

    # Room-level summary (UI display only)
    room_temperature_c: float = 0.0
    room_cooling_demand: float = 0.0
    room_setpoint_c: float = 24.0

    # Metadata
    controller_name: str = "THERMAL CONTROL ALGORITHM"
    is_fallback: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Decision Provider Interface
# ─────────────────────────────────────────────────────────────────────────────

class HVACDecisionProvider:
    """
    Abstract interface for HVAC decision providers.

    Current implementation: ThermalControlAlgorithm
    Future implementation: HVEACBrain

    This allows drop-in replacement without redesigning the HVAC simulator.
    """
    def compute(self, simulation_state: Dict[str, Any]) -> HVACDecision:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Thermal Control Algorithm
# ─────────────────────────────────────────────────────────────────────────────

class ThermalControlAlgorithm(HVACDecisionProvider):
    """
    Deterministic zone-based thermal control algorithm.

    PROTOTYPE CONTROL ALGORITHM — substitutes for a future AI model.

    Reads live simulation state and produces one HVAC control decision.
    No optimization, no candidate search, no cost function.
    """

    def __init__(self, spatial_influence_weights: Optional[List[List[float]]] = None):
        """
        Args:
            spatial_influence_weights: 4x4 matrix [AC][Zone] influence weights.
                Defaults to the standard HVEAC V3 layout.
        """
        # AC-to-Zone spatial influence matrix
        # Rows: AC-1(North), AC-2(East), AC-3(South), AC-4(West)
        # Cols: ZONE_1(NW), ZONE_2(NE), ZONE_3(SE), ZONE_4(SW)
        if spatial_influence_weights is not None:
            self._weights = spatial_influence_weights
        else:
            self._weights = [
                [0.45, 0.45, 0.05, 0.05],  # AC-1 (North)
                [0.05, 0.45, 0.45, 0.05],  # AC-2 (East)
                [0.05, 0.05, 0.45, 0.45],  # AC-3 (South)
                [0.45, 0.05, 0.05, 0.45],  # AC-4 (West)
            ]

        self._ac_ids = ["AC-1", "AC-2", "AC-3", "AC-4"]
        self._zone_ids = ["ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"]

        # Temperature trend tracking (per zone)
        self._prev_zone_temps: Dict[str, float] = {}
        self._trend_c_per_min: Dict[str, float] = {z: 0.0 for z in self._zone_ids}
        self._step_count: int = 0

    def reset(self):
        """Clear all controller state (call on scenario switch or reset)."""
        self._prev_zone_temps = {}
        self._trend_c_per_min = {z: 0.0 for z in self._zone_ids}
        self._step_count = 0
        logger.info("[THERMAL CONTROL] Controller state reset")

    def compute(self, simulation_state: Dict[str, Any]) -> HVACDecision:
        """
        Compute a single HVAC control decision from current simulation state.

        Args:
            simulation_state: Dictionary containing current thermal, computer,
                occupancy, and HVAC telemetry from the simulation.

        Returns:
            HVACDecision with per-zone demands and per-AC cooling commands.
        """
        # Validate input — fallback on invalid state
        if not simulation_state or not isinstance(simulation_state, dict):
            logger.warning("[THERMAL CONTROL] Invalid simulation state — fallback")
            return self._fallback_decision("Invalid simulation state")

        try:
            return self._compute_decision(simulation_state)
        except Exception as e:
            logger.error(f"[THERMAL CONTROL] Error during compute — fallback: {e}")
            return self._fallback_decision(f"Compute error: {e}")

    def _compute_decision(self, state: Dict[str, Any]) -> HVACDecision:
        """Core deterministic control computation."""
        # ─── Extract zone temperatures ───
        thermal = state.get("thermal", {})
        zone_temps_raw = thermal.get("zone_temperatures_c", {})

        zone_temps: Dict[str, float] = {}
        for z in self._zone_ids:
            key_lower = z.lower()
            raw_val = zone_temps_raw.get(key_lower, zone_temps_raw.get(z, None))
            if raw_val is None:
                raw_val = thermal.get("room_average_temperature_c", 22.5)
            val = float(raw_val)
            if not math.isfinite(val):
                logger.warning(f"[THERMAL CONTROL] Non-finite temperature for {z}: {raw_val} — fallback")
                return self._fallback_decision(f"Non-finite temperature: {z}={raw_val}")
            zone_temps[z] = val

        # ─── Extract zone heat loads ───
        zone_heat: Dict[str, float] = {}
        computers = state.get("computers", [])
        for z in self._zone_ids:
            zone_heat[z] = 0.0

        # Sum computer heat by zone
        for comp in computers:
            comp_zone = comp.get("zone", "")
            # Normalize zone key
            z_key = comp_zone.upper().replace(" ", "_")
            if z_key in zone_heat:
                zone_heat[z_key] += float(comp.get("heat_watts", 0.0))

        # Add occupancy heat (95W per person)
        occupancy = state.get("occupancy", {})
        occ_zones = occupancy.get("zones", {})
        for z in self._zone_ids:
            z_lower = z.lower()
            occ_count = int(occ_zones.get(z_lower, occ_zones.get(z, 0)))
            occupancy_heat = occ_count * 95.0  # W per person
            zone_heat[z] += occupancy_heat

        # ─── Calculate temperature trends ───
        self._step_count += 1
        timestep_s = float(state.get("timestep_seconds", 10.0))
        timestep_min = timestep_s / 60.0 if timestep_s > 0 else 1.0 / 6.0

        for z in self._zone_ids:
            if z in self._prev_zone_temps and self._step_count > 1:
                delta_t = zone_temps[z] - self._prev_zone_temps[z]
                # Exponential moving average for trend smoothing
                new_trend = delta_t / timestep_min if timestep_min > 0 else 0.0
                # EMA with alpha=0.3 for stability
                self._trend_c_per_min[z] = 0.3 * new_trend + 0.7 * self._trend_c_per_min[z]
            self._prev_zone_temps[z] = zone_temps[z]

        # ─── Compute per-zone cooling demand ───
        zone_states: Dict[str, ZoneControlState] = {}

        for z in self._zone_ids:
            temp = zone_temps[z]
            heat = zone_heat[z]
            trend = self._trend_c_per_min[z]

            # A. Temperature Demand
            temperature_error = temp - COMFORT_TARGET_C
            temperature_factor = _clamp(temperature_error / TEMP_ERROR_RANGE_C, 0.0, 1.0)
            temperature_demand = temperature_factor * TEMP_DEMAND_MAX

            # B. Thermal Load Demand
            heat_factor = _clamp(heat / HEAT_LOAD_RANGE_W, 0.0, 1.0)
            heat_demand = heat_factor * HEAT_DEMAND_MAX

            # C. Temperature Trend Demand (only positive trend increases demand)
            rising_temperature = max(trend, 0.0)
            trend_factor = _clamp(rising_temperature / TREND_RANGE_C_PER_MIN, 0.0, 1.0)
            trend_demand = trend_factor * TREND_DEMAND_MAX

            # Final zone demand
            cooling_demand = _clamp(
                temperature_demand + heat_demand + trend_demand,
                0.0, 100.0
            )

            zone_states[z] = ZoneControlState(
                zone_id=z,
                temperature_c=round(temp, 2),
                heat_load_w=round(heat, 1),
                temperature_trend_c_per_min=round(trend, 4),
                temperature_demand=round(temperature_demand, 2),
                heat_demand=round(heat_demand, 2),
                trend_demand=round(trend_demand, 2),
                cooling_demand=round(cooling_demand, 2),
            )

        # ─── Map zone demands to AC cooling levels ───
        ac_cooling: Dict[str, float] = {}
        ac_setpoints: Dict[str, float] = {}

        for ac_idx, ac_id in enumerate(self._ac_ids):
            # AC demand = weighted sum of zone demands using spatial influence
            ac_demand = 0.0
            for zone_idx, z in enumerate(self._zone_ids):
                weight = self._weights[ac_idx][zone_idx]
                ac_demand += zone_states[z].cooling_demand * weight

            ac_demand = _clamp(ac_demand, 0.0, 100.0)
            ac_cooling[ac_id] = round(ac_demand, 2)

            # Deterministic setpoint: 24.0°C at 0%, 22.0°C at 100%
            setpoint = SETPOINT_HIGH_C - (ac_demand / 100.0) * SETPOINT_RANGE_C
            ac_setpoints[ac_id] = round(setpoint, 2)

        # ─── Room-level summary (UI display only) ───
        temps_list = list(zone_temps.values())
        demands_list = [zone_states[z].cooling_demand for z in self._zone_ids]

        room_temperature = sum(temps_list) / len(temps_list) if temps_list else 22.5
        room_cooling_demand = sum(demands_list) / len(demands_list) if demands_list else 0.0
        room_setpoint = SETPOINT_HIGH_C - (room_cooling_demand / 100.0) * SETPOINT_RANGE_C

        # ─── Log control decision ───
        self._log_decision(zone_states, ac_cooling, ac_setpoints, room_temperature)

        return HVACDecision(
            zone_states=zone_states,
            ac_cooling_percent=ac_cooling,
            ac_setpoint_c=ac_setpoints,
            room_temperature_c=round(room_temperature, 2),
            room_cooling_demand=round(room_cooling_demand, 2),
            room_setpoint_c=round(room_setpoint, 2),
            controller_name="THERMAL CONTROL ALGORITHM",
            is_fallback=False,
        )

    def _fallback_decision(self, reason: str) -> HVACDecision:
        """Produce a safe baseline fallback with minimal cooling."""
        logger.warning(f"[THERMAL CONTROL] Fallback activated: {reason}")
        zone_states = {}
        for z in self._zone_ids:
            zone_states[z] = ZoneControlState(
                zone_id=z,
                temperature_c=22.5,
                heat_load_w=0.0,
                temperature_trend_c_per_min=0.0,
                temperature_demand=0.0,
                heat_demand=0.0,
                trend_demand=0.0,
                cooling_demand=0.0,
            )
        return HVACDecision(
            zone_states=zone_states,
            ac_cooling_percent={ac: 0.0 for ac in self._ac_ids},
            ac_setpoint_c={ac: SETPOINT_HIGH_C for ac in self._ac_ids},
            room_temperature_c=22.5,
            room_cooling_demand=0.0,
            room_setpoint_c=SETPOINT_HIGH_C,
            controller_name="THERMAL CONTROL ALGORITHM",
            is_fallback=True,
        )

    def _log_decision(
        self,
        zones: Dict[str, ZoneControlState],
        ac_cooling: Dict[str, float],
        ac_setpoints: Dict[str, float],
        room_temp: float,
    ):
        """Log the control decision at debug level."""
        lines = [f"\n[THERMAL CONTROL] Room: {room_temp:.1f}°C"]
        for z_id, zs in zones.items():
            lines.append(
                f"  {z_id}: Temp={zs.temperature_c:.1f}°C "
                f"Heat={zs.heat_load_w:.0f}W "
                f"Trend={zs.temperature_trend_c_per_min:+.2f}°C/min "
                f"Demand={zs.cooling_demand:.0f}%"
            )
        for ac_id in self._ac_ids:
            lines.append(
                f"  {ac_id}: Cooling={ac_cooling[ac_id]:.0f}% "
                f"Setpoint={ac_setpoints[ac_id]:.1f}°C"
            )
        logger.debug("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(value: float, low: float, high: float) -> float:
    """Clamp a value to [low, high], handling NaN/Inf gracefully."""
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))
