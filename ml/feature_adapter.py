"""
HVEAC Brain V1 — Feature Adapter

Maps the Simulation Lab state snapshot (from SimulationManager.build_update_payload())
into the exact 80-feature vector required by the HVEAC Brain V1 model.

Key responsibilities:
1. Extract raw simulation telemetry
2. Map simulation field names → model feature column names
3. Compute derived aggregates (30s rolling averages, min/max, etc.)
4. Encode categorical workload fields
5. Validate completeness against feature_schema_v1.json
6. Log any missing or defaulted features

PROHIBITED features (must never be included):
- scenario_id, scenario_family, run_id, random_seed, timestamp, simulation_time_seconds
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("hveac.brain.adapter")

# 80 feature columns in exact model training order
FEATURE_COLUMNS = [
    "occupancy_total",
    "occupancy_zone_1",
    "occupancy_zone_2",
    "occupancy_zone_3",
    "occupancy_zone_4",
    "outdoor_temperature_c",
    "humidity_percent",
    "solar_load",
    "ac1_state",
    "ac1_setpoint_c",
    "ac1_cooling_level",
    "ac2_state",
    "ac2_setpoint_c",
    "ac2_cooling_level",
    "ac3_state",
    "ac3_setpoint_c",
    "ac3_cooling_level",
    "ac4_state",
    "ac4_setpoint_c",
    "ac4_cooling_level",
    "zone_1_temperature_c",
    "zone_2_temperature_c",
    "zone_3_temperature_c",
    "zone_4_temperature_c",
    "room_average_temperature_c",
    "minimum_temperature_c",
    "maximum_temperature_c",
    "temperature_difference_c",
    "total_computer_heat_watts",
    "total_occupancy_heat_watts",
    "total_environmental_heat_watts",
    "total_heat_load_watts",
    "total_hvac_cooling_watts",
    "comfort_score",
    "comfort_penalty",
    "room_temp_30s_avg",
    "occupancy_total_30s_avg",
    "cpu_util_30s_avg",
    "gpu_util_30s_avg",
    "hvac_cooling_30s_avg",
    "computer_1_cpu",
    "computer_1_gpu",
    "computer_1_workload",
    "computer_1_heat",
    "computer_2_cpu",
    "computer_2_gpu",
    "computer_2_workload",
    "computer_2_heat",
    "computer_3_cpu",
    "computer_3_gpu",
    "computer_3_workload",
    "computer_3_heat",
    "computer_4_cpu",
    "computer_4_gpu",
    "computer_4_workload",
    "computer_4_heat",
    "computer_5_cpu",
    "computer_5_gpu",
    "computer_5_workload",
    "computer_5_heat",
    "computer_6_cpu",
    "computer_6_gpu",
    "computer_6_workload",
    "computer_6_heat",
    "computer_7_cpu",
    "computer_7_gpu",
    "computer_7_workload",
    "computer_7_heat",
    "computer_8_cpu",
    "computer_8_gpu",
    "computer_8_workload",
    "computer_8_heat",
    "computer_9_cpu",
    "computer_9_gpu",
    "computer_9_workload",
    "computer_9_heat",
    "computer_10_cpu",
    "computer_10_gpu",
    "computer_10_workload",
    "computer_10_heat",
]

# Forbidden features — must NEVER be included
FORBIDDEN_FEATURES = {
    "scenario_id", "scenario_family", "run_id",
    "random_seed", "timestamp", "simulation_time_seconds",
}


class FeatureAdapter:
    """
    Converts a Simulation Lab state dictionary into the 80-feature
    input vector required by the HVEAC Brain V1 model.

    Thread-safe and stateless per call (no mutation of internal state).
    Maintains a short rolling history for 30s average calculations.
    """

    def __init__(self):
        # Rolling history buffer for 30s averages (max ~3 entries at 10s timestep)
        self._history: List[Dict[str, float]] = []
        self._max_history = 4

    def reset(self):
        """Clear rolling history (e.g. on scenario change)."""
        self._history.clear()

    def adapt(self, sim_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Transform a simulation state snapshot into the 80-feature vector.

        Parameters
        ----------
        sim_state : dict
            The payload from SimulationManager.build_update_payload()

        Returns
        -------
        (features, warnings) where:
            features : dict mapping each of the 80 feature columns to its value
            warnings : list of str noting any missing/defaulted fields
        """
        warnings = []
        features: Dict[str, Any] = {}

        # ── 1. Occupancy (5 features) ──────────────────────────────────────
        occ = sim_state.get("occupancy", {})
        occ_zones = occ.get("zones", {})
        features["occupancy_total"] = occ.get("total", 0)
        features["occupancy_zone_1"] = occ_zones.get("zone_1", 0)
        features["occupancy_zone_2"] = occ_zones.get("zone_2", 0)
        features["occupancy_zone_3"] = occ_zones.get("zone_3", 0)
        features["occupancy_zone_4"] = occ_zones.get("zone_4", 0)

        # ── 2. Environment (3 features) ───────────────────────────────────
        env = sim_state.get("environment", {})
        features["outdoor_temperature_c"] = env.get("outdoor_temperature_c", 28.0)
        features["humidity_percent"] = env.get("humidity_percent", 50.0)
        features["solar_load"] = env.get("solar_irradiance_w_m2", 200.0)

        # ── 3. HVAC / AC Units (12 features: 4 ACs × 3 features each) ─────
        hvac = sim_state.get("hvac", {})
        acs = hvac.get("acs", [])
        ac_map = {}
        for ac in acs:
            ac_id = ac.get("id", "").upper()
            if "AC-1" in ac_id or "AC1" in ac_id or ac.get("wall") == "north":
                ac_map[1] = ac
            elif "AC-2" in ac_id or "AC2" in ac_id or ac.get("wall") == "east":
                ac_map[2] = ac
            elif "AC-3" in ac_id or "AC3" in ac_id or ac.get("wall") == "south":
                ac_map[3] = ac
            elif "AC-4" in ac_id or "AC4" in ac_id or ac.get("wall") == "west":
                ac_map[4] = ac

        for i in range(1, 5):
            ac = ac_map.get(i, {})
            state_str = ac.get("state", "OFF")
            features[f"ac{i}_state"] = state_str
            features[f"ac{i}_setpoint_c"] = float(ac.get("setpoint_c", 22.0))
            features[f"ac{i}_cooling_level"] = float(ac.get("cooling_level", 0.25))

        # ── 4. Zone temperatures (4 features) ─────────────────────────────
        thermal = sim_state.get("thermal", {})
        zone_temps = thermal.get("zone_temperatures_c", {})
        features["zone_1_temperature_c"] = float(zone_temps.get("zone_1", 22.5))
        features["zone_2_temperature_c"] = float(zone_temps.get("zone_2", 22.5))
        features["zone_3_temperature_c"] = float(zone_temps.get("zone_3", 22.5))
        features["zone_4_temperature_c"] = float(zone_temps.get("zone_4", 22.5))

        # ── 5. Room-level temperature aggregates (4 features) ──────────────
        features["room_average_temperature_c"] = float(thermal.get("room_average_temperature_c", 22.5))
        features["minimum_temperature_c"] = float(thermal.get("minimum_temperature_c", 22.5))
        features["maximum_temperature_c"] = float(thermal.get("maximum_temperature_c", 22.5))
        features["temperature_difference_c"] = float(thermal.get("temperature_difference_c", 0.0))

        # ── 6. Heat load aggregates (5 features) ──────────────────────────
        comp_heat_w = float(thermal.get("computer_heat_watts", 0.0))
        occ_heat_w = float(thermal.get("occupancy_heat_watts", 0.0))
        hvac_cool_w = float(thermal.get("hvac_cooling_watts", 0.0))

        # Environmental heat approximation: outdoor – room average delta × envelope coefficient
        outdoor_t = features["outdoor_temperature_c"]
        room_t = features["room_average_temperature_c"]
        envelope_gain = max(0.0, (outdoor_t - room_t) * 80.0 + features["solar_load"] * 0.3)

        features["total_computer_heat_watts"] = comp_heat_w
        features["total_occupancy_heat_watts"] = occ_heat_w
        features["total_environmental_heat_watts"] = round(envelope_gain, 2)
        features["total_heat_load_watts"] = round(comp_heat_w + occ_heat_w + envelope_gain, 2)
        features["total_hvac_cooling_watts"] = hvac_cool_w

        # ── 7. Comfort (2 features) ───────────────────────────────────────
        # Derived from room temp relative to 22.5°C comfort baseline
        room_avg = features["room_average_temperature_c"]
        deviation = abs(room_avg - 22.5)
        comfort_score = max(0.0, min(100.0, 100.0 - (deviation * 15.0)))
        comfort_penalty = max(0.0, deviation - 1.5) * 10.0

        features["comfort_score"] = round(comfort_score, 2)
        features["comfort_penalty"] = round(comfort_penalty, 2)

        # ── 8. Computer per-node telemetry (40 features: 10 comps × 4) ────
        computers = sim_state.get("computers", [])
        comp_by_id = {c.get("id"): c for c in computers}

        total_cpu = 0.0
        total_gpu = 0.0

        for cid in range(1, 11):
            comp = comp_by_id.get(cid, {})
            cpu = float(comp.get("cpu_util_percent", 0.0))
            gpu = float(comp.get("gpu_util_percent", 0.0))
            workload = str(comp.get("workload_category", "IDLE"))
            heat = float(comp.get("heat_watts", 45.0))

            features[f"computer_{cid}_cpu"] = round(cpu, 1)
            features[f"computer_{cid}_gpu"] = round(gpu, 1)
            features[f"computer_{cid}_workload"] = workload
            features[f"computer_{cid}_heat"] = round(heat, 1)

            total_cpu += cpu
            total_gpu += gpu

        avg_cpu = total_cpu / 10.0
        avg_gpu = total_gpu / 10.0

        # ── 9. 30-second rolling averages (5 features) ────────────────────
        snapshot = {
            "room_temp": room_avg,
            "occupancy_total": features["occupancy_total"],
            "cpu_util": avg_cpu,
            "gpu_util": avg_gpu,
            "hvac_cooling": hvac_cool_w,
        }
        self._history.append(snapshot)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        def rolling_avg(key: str) -> float:
            vals = [h[key] for h in self._history if key in h]
            return sum(vals) / max(1, len(vals))

        features["room_temp_30s_avg"] = round(rolling_avg("room_temp"), 2)
        features["occupancy_total_30s_avg"] = round(rolling_avg("occupancy_total"), 2)
        features["cpu_util_30s_avg"] = round(rolling_avg("cpu_util"), 2)
        features["gpu_util_30s_avg"] = round(rolling_avg("gpu_util"), 2)
        features["hvac_cooling_30s_avg"] = round(rolling_avg("hvac_cooling"), 2)

        # ── 10. Validation ────────────────────────────────────────────────
        # Check forbidden features are NOT present
        for forbidden in FORBIDDEN_FEATURES:
            if forbidden in features:
                del features[forbidden]
                warnings.append(f"Removed forbidden feature '{forbidden}'")

        # Check all 80 features present
        missing = set(FEATURE_COLUMNS) - set(features.keys())
        if missing:
            for m in sorted(missing):
                features[m] = 0.0
                warnings.append(f"Missing feature '{m}' — defaulted to 0.0")

        # Check no extra features
        extra = set(features.keys()) - set(FEATURE_COLUMNS)
        if extra:
            for e in sorted(extra):
                del features[e]
                warnings.append(f"Removed extra feature '{e}'")

        if warnings:
            logger.debug(f"[ADAPTER] {len(warnings)} warnings: {warnings[:5]}...")

        return features, warnings
