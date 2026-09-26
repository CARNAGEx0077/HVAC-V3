"""
HVEAC Brain V1 — Shadow Predictor Service

Orchestrates the complete shadow prediction lifecycle:
1. Receives simulation state
2. Adapts to 80-feature vector via FeatureAdapter
3. Runs inference via predict_room_setpoint
4. Tracks latency, metrics, history
5. Asserts HVAC immutability (AI never controls HVAC)

This is SHADOW MODE ONLY:
- All predictions are for observation and comparison.
- There is NO path from prediction to HVAC control.
"""

import copy
import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ml.feature_adapter import FeatureAdapter, FEATURE_COLUMNS
from ml.inference import predict_room_setpoint, get_model_status

logger = logging.getLogger("hveac.brain.shadow")

# Maximum shadow history entries in memory
MAX_HISTORY = 500

# Feature groups for debug inspector
FEATURE_GROUPS = {
    "ENVIRONMENT": [
        "outdoor_temperature_c", "humidity_percent", "solar_load",
    ],
    "OCCUPANCY": [
        "occupancy_total", "occupancy_zone_1", "occupancy_zone_2",
        "occupancy_zone_3", "occupancy_zone_4",
    ],
    "HVAC STATE": [
        "ac1_state", "ac1_setpoint_c", "ac1_cooling_level",
        "ac2_state", "ac2_setpoint_c", "ac2_cooling_level",
        "ac3_state", "ac3_setpoint_c", "ac3_cooling_level",
        "ac4_state", "ac4_setpoint_c", "ac4_cooling_level",
    ],
    "ZONE TEMPERATURES": [
        "zone_1_temperature_c", "zone_2_temperature_c",
        "zone_3_temperature_c", "zone_4_temperature_c",
    ],
    "ROOM AGGREGATES": [
        "room_average_temperature_c", "minimum_temperature_c",
        "maximum_temperature_c", "temperature_difference_c",
    ],
    "HEAT LOADS": [
        "total_computer_heat_watts", "total_occupancy_heat_watts",
        "total_environmental_heat_watts", "total_heat_load_watts",
        "total_hvac_cooling_watts",
    ],
    "COMFORT": [
        "comfort_score", "comfort_penalty",
    ],
    "ROLLING AVERAGES (30s)": [
        "room_temp_30s_avg", "occupancy_total_30s_avg",
        "cpu_util_30s_avg", "gpu_util_30s_avg", "hvac_cooling_30s_avg",
    ],
}

# Computer telemetry features (10 computers × 4 features each)
for cid in range(1, 11):
    group_name = f"COMPUTER {cid}"
    FEATURE_GROUPS[group_name] = [
        f"computer_{cid}_cpu", f"computer_{cid}_gpu",
        f"computer_{cid}_workload", f"computer_{cid}_heat",
    ]

# Units for display
FEATURE_UNITS = {
    "outdoor_temperature_c": "°C", "humidity_percent": "%",
    "solar_load": "W/m²",
    "occupancy_total": "people", "occupancy_zone_1": "people",
    "occupancy_zone_2": "people", "occupancy_zone_3": "people",
    "occupancy_zone_4": "people",
    "zone_1_temperature_c": "°C", "zone_2_temperature_c": "°C",
    "zone_3_temperature_c": "°C", "zone_4_temperature_c": "°C",
    "room_average_temperature_c": "°C", "minimum_temperature_c": "°C",
    "maximum_temperature_c": "°C", "temperature_difference_c": "°C",
    "total_computer_heat_watts": "W", "total_occupancy_heat_watts": "W",
    "total_environmental_heat_watts": "W", "total_heat_load_watts": "W",
    "total_hvac_cooling_watts": "W",
    "comfort_score": "pts", "comfort_penalty": "pts",
    "room_temp_30s_avg": "°C", "occupancy_total_30s_avg": "people",
    "cpu_util_30s_avg": "%", "gpu_util_30s_avg": "%",
    "hvac_cooling_30s_avg": "W",
}
for cid in range(1, 11):
    FEATURE_UNITS[f"computer_{cid}_cpu"] = "%"
    FEATURE_UNITS[f"computer_{cid}_gpu"] = "%"
    FEATURE_UNITS[f"computer_{cid}_workload"] = ""
    FEATURE_UNITS[f"computer_{cid}_heat"] = "W"
for i in range(1, 5):
    FEATURE_UNITS[f"ac{i}_state"] = ""
    FEATURE_UNITS[f"ac{i}_setpoint_c"] = "°C"
    FEATURE_UNITS[f"ac{i}_cooling_level"] = ""


class ShadowMetrics:
    """Accumulates agreement metrics between AI predictions and simulator control."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._pairs: List[Tuple[float, float]] = []  # (ai, sim) pairs
        self._latencies: List[float] = []
        self._prediction_changes = 0
        self._last_prediction: Optional[float] = None
        self._scenario_id: Optional[int] = None

    def record(self, ai_setpoint: float, sim_setpoint: float,
               latency_ms: float, scenario_id: int):
        self._pairs.append((ai_setpoint, sim_setpoint))
        self._latencies.append(latency_ms)
        self._scenario_id = scenario_id

        if self._last_prediction is not None and ai_setpoint != self._last_prediction:
            self._prediction_changes += 1
        self._last_prediction = ai_setpoint

    @property
    def count(self) -> int:
        return len(self._pairs)

    def compute(self) -> Dict[str, Any]:
        """Returns computed agreement metrics."""
        if not self._pairs:
            return {"count": 0, "status": "NO_DATA"}

        n = len(self._pairs)
        errors = [abs(ai - sim) for ai, sim in self._pairs]
        sq_errors = [(ai - sim) ** 2 for ai, sim in self._pairs]

        mae = sum(errors) / n
        rmse = math.sqrt(sum(sq_errors) / n)
        exact_agree = sum(1 for ai, sim in self._pairs if ai == sim) / n
        within_05 = sum(1 for e in errors if e <= 0.5) / n
        max_dev = max(errors)

        # Mean prediction and mean simulator setpoint
        mean_ai = sum(ai for ai, _ in self._pairs) / n
        mean_sim = sum(sim for _, sim in self._pairs) / n

        # Latency stats
        lat = sorted(self._latencies) if self._latencies else [0]
        p95_idx = min(int(len(lat) * 0.95), len(lat) - 1)

        return {
            "count": n,
            "exact_agreement_pct": round(exact_agree * 100, 2),
            "mae_c": round(mae, 4),
            "rmse_c": round(rmse, 4),
            "within_05c_pct": round(within_05 * 100, 2),
            "max_deviation_c": round(max_dev, 2),
            "prediction_changes": self._prediction_changes,
            "mean_prediction_c": round(mean_ai, 2),
            "mean_simulator_setpoint_c": round(mean_sim, 2),
            "latency_min_ms": round(min(lat), 2),
            "latency_mean_ms": round(sum(lat) / len(lat), 2),
            "latency_p95_ms": round(lat[p95_idx], 2),
            "latency_max_ms": round(max(lat), 2),
            "scenario_id": self._scenario_id,
        }


class ShadowPredictor:
    """
    Stateful shadow prediction service.

    Maintains:
    - Feature adapter with rolling history
    - Current shadow prediction state
    - Bounded prediction history for trend visualization
    - Per-scenario agreement metrics
    - HVAC immutability safety assertion

    SAFETY: This class NEVER writes to any HVAC control variable.
    """

    def __init__(self):
        self._adapter = FeatureAdapter()
        self._history: deque = deque(maxlen=MAX_HISTORY)
        self._metrics = ShadowMetrics()
        self._current: Optional[Dict[str, Any]] = None
        self._last_step_idx: int = -1
        self._enabled: bool = True
        self._model_status_cache: Optional[Dict[str, Any]] = None

        # Feature snapshot for debug inspector
        self._last_features: Optional[Dict[str, Any]] = None
        self._last_feature_warnings: List[str] = []

        logger.info("[HVEAC BRAIN] Shadow predictor service initialized")

    @property
    def adapter(self) -> FeatureAdapter:
        return self._adapter

    def reset(self):
        """Reset on scenario change."""
        self._adapter.reset()
        self._history.clear()
        self._metrics.reset()
        self._current = None
        self._last_step_idx = -1
        self._last_features = None
        self._last_feature_warnings = []
        logger.info("[HVEAC BRAIN] Shadow predictor reset for new scenario")

    def get_model_status(self) -> Dict[str, Any]:
        """Cached model status."""
        if self._model_status_cache is None:
            self._model_status_cache = get_model_status()
        return self._model_status_cache

    def predict(self, sim_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Run shadow prediction on the current simulation state.

        SAFETY: This method ONLY reads simulation state. It NEVER writes
        to any HVAC control variable. An immutability assertion verifies
        HVAC state is unchanged after every prediction.

        Returns the shadow prediction result, or the cached result if
        the simulation step has not changed.
        """
        if not self._enabled:
            return self._current

        if not sim_state:
            return None

        step_idx = sim_state.get("step_index", 0)

        # Only predict on step change to avoid redundant computation
        if step_idx == self._last_step_idx:
            return self._current

        self._last_step_idx = step_idx

        # ── HVAC Immutability: Snapshot HVAC state BEFORE inference ──
        hvac_before = self._snapshot_hvac(sim_state)

        # ── Run inference ──
        t_start = time.perf_counter()

        try:
            # 1. Adapt simulation state → 80 features
            features, adapter_warnings = self._adapter.adapt(sim_state)
            self._last_features = dict(features)
            self._last_feature_warnings = list(adapter_warnings)

            # 2. Run model prediction
            result = predict_room_setpoint(features)

            t_end = time.perf_counter()
            latency_ms = round((t_end - t_start) * 1000, 2)

            # 3. Get simulator's own recommendation for comparison
            targets = sim_state.get("targets", {})
            sim_setpoint = targets.get("optimal_temperature_c")
            sim_action = targets.get("optimal_hvac_action")

            # 4. Extract prediction
            ai_setpoint = result.get("predicted_class")

            # 5. Compute agreement
            agreement = None
            deviation = None
            if ai_setpoint is not None and sim_setpoint is not None:
                deviation = round(abs(ai_setpoint - sim_setpoint), 2)
                agreement = deviation <= 0.5

                # Record metrics
                scenario_id = sim_state.get("scenario_id", 0)
                self._metrics.record(
                    ai_setpoint, sim_setpoint, latency_ms, scenario_id
                )

            # 6. Build shadow result
            shadow = {
                "enabled": True,
                "mode": "SHADOW",
                "model_version": "hveac_brain_v1",
                "status": result.get("status", "ERROR"),
                "predicted_setpoint_c": ai_setpoint,
                "confidence": result.get("confidence"),
                "class_probabilities": result.get("class_probabilities"),
                "simulator_setpoint_c": sim_setpoint,
                "simulator_action": sim_action,
                "agreement": agreement,
                "deviation_c": deviation,
                "inference_latency_ms": latency_ms,
                "feature_count": len(features),
                "feature_warnings": adapter_warnings + result.get("feature_warnings", []),
                "simulation_time_s": sim_state.get("simulation_time_seconds", 0),
                "step_index": step_idx,
                "control_path": False,  # EXPLICIT: no HVAC control
            }

            self._current = shadow

            # 7. Append to history
            if ai_setpoint is not None:
                self._history.append({
                    "time": sim_state.get("simulation_time_seconds", 0),
                    "ai": ai_setpoint,
                    "sim": sim_setpoint,
                    "latency_ms": latency_ms,
                    "step": step_idx,
                })

            # ── HVAC Immutability: Verify HVAC state UNCHANGED ──
            hvac_after = self._snapshot_hvac(sim_state)
            if hvac_before != hvac_after:
                logger.critical(
                    "[HVEAC BRAIN] SAFETY VIOLATION: HVAC state changed "
                    "during inference! Disabling shadow predictor."
                )
                self._enabled = False
                shadow["status"] = "SAFETY_VIOLATION"

            # Structured log
            if ai_setpoint is not None:
                sign = "+" if deviation and ai_setpoint >= (sim_setpoint or 0) else ""
                logger.debug(
                    f"[HVEAC BRAIN] status={shadow['status']} "
                    f"model=hveac_brain_v1 "
                    f"prediction={ai_setpoint} "
                    f"simulator_setpoint={sim_setpoint} "
                    f"delta={sign}{deviation} "
                    f"latency={latency_ms}ms"
                )

            return shadow

        except Exception as e:
            t_end = time.perf_counter()
            latency_ms = round((t_end - t_start) * 1000, 2)
            logger.error(
                f"[HVEAC BRAIN] Shadow prediction error: {e}", exc_info=True
            )

            self._current = {
                "enabled": True,
                "mode": "SHADOW",
                "model_version": "hveac_brain_v1",
                "status": "ERROR",
                "predicted_setpoint_c": None,
                "confidence": None,
                "class_probabilities": None,
                "simulator_setpoint_c": None,
                "simulator_action": None,
                "agreement": None,
                "deviation_c": None,
                "inference_latency_ms": latency_ms,
                "feature_count": 0,
                "feature_warnings": [str(e)],
                "simulation_time_s": sim_state.get("simulation_time_seconds", 0),
                "step_index": step_idx,
                "control_path": False,
                "error": str(e),
            }
            return self._current

    def get_current(self) -> Optional[Dict[str, Any]]:
        """Returns the last shadow prediction without re-running inference."""
        return self._current

    def get_history(self) -> List[Dict[str, Any]]:
        """Returns the bounded prediction history for trend visualization."""
        return list(self._history)

    def get_metrics(self) -> Dict[str, Any]:
        """Returns computed agreement metrics."""
        return self._metrics.compute()

    def get_feature_snapshot(self) -> Optional[Dict[str, Any]]:
        """Returns the last 80-feature snapshot for the debug inspector."""
        return self._last_features

    def get_grouped_features(self) -> Dict[str, List[Dict[str, Any]]]:
        """Returns features grouped by category for the debug inspector UI."""
        if not self._last_features:
            return {}

        grouped = {}
        for group_name, feature_names in FEATURE_GROUPS.items():
            items = []
            for fname in feature_names:
                val = self._last_features.get(fname)
                items.append({
                    "feature": fname,
                    "value": val,
                    "unit": FEATURE_UNITS.get(fname, ""),
                })
            grouped[group_name] = items
        return grouped

    def get_feature_warnings(self) -> List[str]:
        """Returns warnings from the last feature adaptation."""
        return self._last_feature_warnings

    def get_full_shadow_state(self) -> Dict[str, Any]:
        """Returns complete shadow state for API/WS payload."""
        model_st = self.get_model_status()
        current = self._current

        if current is None:
            return {
                "enabled": self._enabled,
                "mode": "SHADOW",
                "model_version": "hveac_brain_v1",
                "model_status": model_st.get("status", "UNAVAILABLE"),
                "model_architecture": model_st.get("architecture", "Unknown"),
                "model_feature_count": model_st.get("feature_count", 0),
                "model_class_count": model_st.get("class_count", 0),
                "model_test_accuracy": model_st.get("test_accuracy"),
                "status": "WAITING",
                "predicted_setpoint_c": None,
                "confidence": None,
                "class_probabilities": None,
                "simulator_setpoint_c": None,
                "deviation_c": None,
                "agreement": None,
                "inference_latency_ms": None,
                "control_path": False,
            }

        return {
            **current,
            "model_status": model_st.get("status", "UNAVAILABLE"),
            "model_architecture": model_st.get("architecture", "Unknown"),
            "model_feature_count": model_st.get("feature_count", 0),
            "model_class_count": model_st.get("class_count", 0),
            "model_test_accuracy": model_st.get("test_accuracy"),
        }

    @staticmethod
    def _snapshot_hvac(sim_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-copy HVAC-related state for immutability verification."""
        hvac = sim_state.get("hvac", {})
        targets = sim_state.get("targets", {})
        return {
            "hvac": copy.deepcopy(hvac),
            "targets": copy.deepcopy(targets),
        }
