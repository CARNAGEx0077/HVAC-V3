"""
HVEAC Brain V1 — AI Closed-Loop Controller Adapter

Coordinates the complete closed-loop control flow:
Simulation State
      ↓
Feature Adapter (80 features)
      ↓
HVEAC Brain v1 (Random Forest)
      ↓
Safety Governor (Bounds, Rate Limit, Dwell, Fail-Safe)
      ↓
AI Control Adapter
      ↓
Existing Simulator HVAC Controller
      ↓
New Thermal State
      ↓
Next Simulation Step

SAFETY INVARIANTS:
1. SIMULATION ONLY: No real HVAC control, no physical actuator interfaces.
2. Control path is strictly asserted as SIMULATOR_ONLY.
3. Three distinct control modes:
   - BASELINE: Existing simulator optimizer is authoritative.
   - SHADOW: Existing simulator is authoritative; AI observes and predicts.
   - AI_CONTROL: HVEAC Brain controls simulated room target through Safety Governor.
4. Default mode is BASELINE.
5. In AI_CONTROL mode, parallel baseline calculation continues for real-time comparison.
"""

import copy
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ml.feature_adapter import FeatureAdapter
from ml.inference import get_model_status, predict_room_setpoint
from ml.safety_governor import (
    CONTROL_PATH_SIMULATOR_ONLY,
    STATUS_APPLIED,
    STATUS_FALLBACK_BASELINE,
    STATUS_UNAVAILABLE,
    SafetyGovernor,
    SafetyGovernorConfig,
    SafetyResult,
)

logger = logging.getLogger("hveac.brain.controller")

ALLOWED_CONTROL_MODES = ("BASELINE", "SHADOW", "AI_CONTROL")
DEFAULT_CONTROL_MODE = "BASELINE"


class ClosedLoopMetrics:
    """Calculates thermal performance, energy, comfort, and agreement metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count: int = 0
        self.ai_setpoints: List[float] = []
        self.baseline_setpoints: List[float] = []
        self.applied_setpoints: List[float] = []
        self.room_temps: List[float] = []
        self.max_zone_temps: List[float] = []
        self.min_zone_temps: List[float] = []
        self.gradients: List[float] = []
        self.cooling_watts: List[float] = []
        self.cooling_levels: List[float] = []
        
        # Comfort tracking: [21.0, 24.0] comfort band
        self.comfort_time_s: float = 0.0
        self.overheating_time_s: float = 0.0  # > 25.0 C
        self.overcooling_time_s: float = 0.0   # < 20.0 C
        self.total_sim_time_s: float = 0.0

        # Dynamics
        self.setpoint_changes: int = 0
        self.last_applied: Optional[float] = None
        self.latencies_ms: List[float] = []

    def record_step(
        self,
        ai_setpoint: Optional[float],
        safe_setpoint: float,
        applied_setpoint: float,
        baseline_setpoint: float,
        room_avg_temp: float,
        max_zone_temp: float,
        min_zone_temp: float,
        gradient: float,
        total_cooling_w: float,
        avg_cooling_level: float,
        dt_seconds: float,
        total_latency_ms: float,
    ):
        self.count += 1
        self.total_sim_time_s += dt_seconds

        if ai_setpoint is not None:
            self.ai_setpoints.append(ai_setpoint)
        self.applied_setpoints.append(applied_setpoint)
        self.baseline_setpoints.append(baseline_setpoint)
        self.room_temps.append(room_avg_temp)
        self.max_zone_temps.append(max_zone_temp)
        self.min_zone_temps.append(min_zone_temp)
        self.gradients.append(gradient)
        self.cooling_watts.append(total_cooling_w)
        self.cooling_levels.append(avg_cooling_level)
        self.latencies_ms.append(total_latency_ms)

        # Comfort band checks
        if 21.0 <= room_avg_temp <= 24.0:
            self.comfort_time_s += dt_seconds
        if room_avg_temp > 25.0:
            self.overheating_time_s += dt_seconds
        elif room_avg_temp < 20.0:
            self.overcooling_time_s += dt_seconds

        # Setpoint changes count
        if self.last_applied is not None and abs(applied_setpoint - self.last_applied) > 0.01:
            self.setpoint_changes += 1
        self.last_applied = applied_setpoint

    def compute(self) -> Dict[str, Any]:
        if not self.applied_setpoints:
            return {
                "count": 0,
                "mean_room_temp_c": 0.0,
                "max_room_temp_c": 0.0,
                "min_room_temp_c": 0.0,
                "mean_gradient_c": 0.0,
                "comfort_band_pct": 0.0,
                "overheating_time_s": 0.0,
                "overcooling_time_s": 0.0,
                "total_cooling_energy_kwh": 0.0,
                "avg_cooling_watts": 0.0,
                "avg_cooling_level": 0.0,
                "setpoint_changes": 0,
                "mae_vs_baseline": 0.0,
                "rmse_vs_baseline": 0.0,
                "within_05c_pct": 0.0,
                "exact_agreement_pct": 0.0,
                "mean_latency_ms": 0.0,
            }

        n = len(self.applied_setpoints)
        diffs = [abs(a - b) for a, b in zip(self.applied_setpoints, self.baseline_setpoints)]
        mae = sum(diffs) / n
        rmse = math.sqrt(sum(d * d for d in diffs) / n)
        exact_pct = (sum(1 for d in diffs if d < 0.01) / n) * 100.0
        within_05_pct = (sum(1 for d in diffs if d <= 0.50 + 1e-4) / n) * 100.0

        # Energy in kWh: sum(Watts * seconds) / (3600 * 1000)
        dt_avg = self.total_sim_time_s / max(1, n)
        energy_joules = sum(w * dt_avg for w in self.cooling_watts)
        energy_kwh = energy_joules / 3_600_000.0

        return {
            "count": n,
            "mean_room_temp_c": round(sum(self.room_temps) / n, 2),
            "max_room_temp_c": round(max(self.max_zone_temps), 2) if self.max_zone_temps else 0.0,
            "min_room_temp_c": round(min(self.min_zone_temps), 2) if self.min_zone_temps else 0.0,
            "mean_gradient_c": round(sum(self.gradients) / n, 2),
            "comfort_band_pct": round((self.comfort_time_s / max(1.0, self.total_sim_time_s)) * 100.0, 2),
            "overheating_time_s": round(self.overheating_time_s, 1),
            "overcooling_time_s": round(self.overcooling_time_s, 1),
            "total_cooling_energy_kwh": round(energy_kwh, 4),
            "avg_cooling_watts": round(sum(self.cooling_watts) / n, 1),
            "avg_cooling_level": round(sum(self.cooling_levels) / n, 3),
            "setpoint_changes": self.setpoint_changes,
            "mae_vs_baseline": round(mae, 4),
            "rmse_vs_baseline": round(rmse, 4),
            "within_05c_pct": round(within_05_pct, 2),
            "exact_agreement_pct": round(exact_pct, 2),
            "mean_latency_ms": round(sum(self.latencies_ms) / max(1, len(self.latencies_ms)), 2),
        }


class AiClosedLoopController:
    """
    Main closed-loop AI controller service.
    
    Provides:
    - Control mode switching (BASELINE, SHADOW, AI_CONTROL)
    - Full pipeline execution: FeatureAdapter → Brain → SafetyGovernor → HVAC Target
    - Parallel baseline reference tracking
    - Bounded event logging
    - Live comparative performance metrics
    """

    def __init__(
        self,
        governor_config: Optional[SafetyGovernorConfig] = None,
        control_mode: str = DEFAULT_CONTROL_MODE,
    ):
        mode_cleaned = control_mode.strip().upper()
        if mode_cleaned not in ALLOWED_CONTROL_MODES:
            raise ValueError(f"Invalid mode '{control_mode}'. Allowed: {ALLOWED_CONTROL_MODES}")
        self.control_mode: str = mode_cleaned
        self.governor = SafetyGovernor(governor_config)
        self.safety_governor = self.governor
        self.adapter = FeatureAdapter()
        self.metrics = ClosedLoopMetrics()
        
        # Last decision state
        self._last_result: Optional[SafetyResult] = None
        self._last_prediction: Optional[Dict[str, Any]] = None
        self._last_step_idx: int = -1
        self._last_inference_latency_ms: float = 0.0
        self._last_controller_latency_ms: float = 0.0

        logger.info(f"[AI CONTROLLER] Initialized with mode={self.control_mode}")

    @property
    def events(self) -> List[Any]:
        return self.governor.events

    def step(
        self,
        current_state_snapshot: Dict[str, Any],
        baseline_setpoint_c: float = 22.5,
        sim_time_s: float = 0.0,
        hvac_available: bool = True,
    ) -> SafetyResult:
        """Alias for execute_control_cycle for concise stepping."""
        snapshot = dict(current_state_snapshot)
        if "simulation_time_seconds" not in snapshot:
            snapshot["simulation_time_seconds"] = sim_time_s
        return self.execute_control_cycle(
            sim_state=snapshot,
            baseline_setpoint_c=baseline_setpoint_c,
            hvac_available=hvac_available,
        )

    def set_control_mode(self, mode: str) -> str:
        """
        Switches the operational control mode.
        Allowed: 'BASELINE', 'SHADOW', 'AI_CONTROL'.
        """
        cleaned = mode.strip().upper()
        if cleaned not in ALLOWED_CONTROL_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: {ALLOWED_CONTROL_MODES}")

        old_mode = self.control_mode
        self.control_mode = cleaned
        logger.info(f"[AI CONTROLLER] Control mode transition: {old_mode} → {self.control_mode}")
        return self.control_mode

    def reset(self, initial_setpoint: Optional[float] = 22.5):
        """Reset controller state, safety governor, and metrics on scenario change."""
        self.adapter.reset()
        self.governor.reset(initial_setpoint)
        self.metrics.reset()
        self._last_result = None
        self._last_prediction = None
        self._last_step_idx = -1
        self._last_inference_latency_ms = 0.0
        self._last_controller_latency_ms = 0.0
        logger.info("[AI CONTROLLER] Reset complete")

    def execute_control_cycle(
        self,
        sim_state: Dict[str, Any],
        baseline_setpoint_c: float = 22.5,
        hvac_available: bool = True,
    ) -> SafetyResult:
        """
        Executes one control cycle:
        1. In BASELINE: Returns baseline target without running AI.
        2. In SHADOW: Runs AI inference for observation, applies baseline.
        3. In AI_CONTROL: Runs AI inference, validates through SafetyGovernor, applies safe target.
        """
        t_cycle_start = time.perf_counter()
        sim_time = float(sim_state.get("simulation_time_seconds", 0.0))
        step_idx = int(sim_state.get("step_index", 0))

        # Check if model is loaded
        model_st = get_model_status()
        model_ready = (model_st.get("status") == "READY")

        # -------------------------------------------------------------
        # Mode: BASELINE
        # -------------------------------------------------------------
        if self.control_mode == "BASELINE":
            res = SafetyResult(
                mode="BASELINE",
                ai_requested_setpoint_c=None,
                safe_setpoint_c=baseline_setpoint_c,
                applied_setpoint_c=baseline_setpoint_c,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_APPLIED,
                reason="Baseline optimizer authoritative",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=0.0,
                timestamp_sim_s=sim_time,
            )
            self._last_result = res
            return res

        # -------------------------------------------------------------
        # Modes: SHADOW or AI_CONTROL — Run Inference
        # -------------------------------------------------------------
        ai_requested: Optional[float] = None
        feature_warnings: List[str] = []
        inference_status = "OK"

        t_inf_start = time.perf_counter()
        try:
            features, adapter_warnings = self.adapter.adapt(sim_state)
            feature_warnings.extend(adapter_warnings)

            pred_res = predict_room_setpoint(features)
            self._last_prediction = pred_res
            inference_status = pred_res.get("status", "ERROR")
            if inference_status == "OK":
                ai_requested = pred_res.get("predicted_class")
            else:
                feature_warnings.append(pred_res.get("error", "Inference returned non-OK status"))
        except Exception as ex:
            logger.error(f"[AI CONTROLLER] Inference cycle error: {ex}", exc_info=True)
            inference_status = "ERROR"
            feature_warnings.append(str(ex))

        t_inf_end = time.perf_counter()
        self._last_inference_latency_ms = (t_inf_end - t_inf_start) * 1000.0

        # -------------------------------------------------------------
        # Pass to Safety Governor
        # -------------------------------------------------------------
        try:
            res = self.governor.validate_command(
                ai_requested_setpoint_c=ai_requested,
                simulation_time_s=sim_time,
                baseline_setpoint_c=baseline_setpoint_c,
                control_mode=self.control_mode,
                model_status="READY" if model_ready else "UNAVAILABLE",
                inference_status=inference_status,
                hvac_available=hvac_available,
                feature_warnings=feature_warnings,
            )
        except Exception as gov_ex:
            logger.error(f"[AI CONTROLLER] Governor validation error: {gov_ex}", exc_info=True)
            res = SafetyResult(
                mode=self.control_mode,
                ai_requested_setpoint_c=ai_requested,
                safe_setpoint_c=baseline_setpoint_c,
                applied_setpoint_c=baseline_setpoint_c,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_FALLBACK_BASELINE,
                reason=f"Safety governor failure: {gov_ex} (falling back to baseline)",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=0.0,
                timestamp_sim_s=sim_time,
            )

        t_cycle_end = time.perf_counter()
        self._last_controller_latency_ms = (t_cycle_end - t_cycle_start) * 1000.0
        self._last_result = res
        self._last_step_idx = step_idx

        return res

    def get_state(self) -> Dict[str, Any]:
        """Returns complete structured control state for WebSocket/API payload."""
        res = self._last_result
        pred = self._last_prediction or {}
        model_st = get_model_status()

        if res is None:
            return {
                "control_mode": self.control_mode,
                "ai_requested_setpoint_c": None,
                "ai_safe_setpoint_c": None,
                "ai_applied_setpoint_c": None,
                "baseline_setpoint_c": 22.5,
                "setpoint_difference_c": None,
                "safety_status": "READY",
                "safety_reason": f"Initialized in {self.control_mode} mode",
                "control_path": CONTROL_PATH_SIMULATOR_ONLY,
                "model_status": model_st.get("status", "UNAVAILABLE"),
                "model_version": model_st.get("model_version", "hveac_brain_v1"),
                "inference_latency_ms": round(self._last_inference_latency_ms, 2),
                "controller_latency_ms": round(self._last_controller_latency_ms, 2),
                "confidence": None,
                "class_probabilities": None,
            }

        diff_c = None
        if res.applied_setpoint_c is not None and res.baseline_setpoint_c is not None:
            diff_c = round(res.applied_setpoint_c - res.baseline_setpoint_c, 2)

        return {
            "control_mode": self.control_mode,
            "ai_requested_setpoint_c": res.ai_requested_setpoint_c,
            "ai_safe_setpoint_c": res.safe_setpoint_c,
            "ai_applied_setpoint_c": res.applied_setpoint_c,
            "baseline_setpoint_c": res.baseline_setpoint_c,
            "setpoint_difference_c": diff_c,
            "safety_status": res.status,
            "safety_reason": res.reason,
            "control_path": CONTROL_PATH_SIMULATOR_ONLY,
            "model_status": model_st.get("status", "UNAVAILABLE"),
            "model_version": model_st.get("model_version", "hveac_brain_v1"),
            "inference_latency_ms": round(self._last_inference_latency_ms, 2),
            "controller_latency_ms": round(self._last_controller_latency_ms, 2),
            "confidence": pred.get("confidence"),
            "class_probabilities": pred.get("class_probabilities"),
        }

    def get_events(self) -> List[Dict[str, Any]]:
        return self.governor.get_events()

    def get_metrics(self) -> Dict[str, Any]:
        cm = self.metrics.compute()
        gm = self.governor.get_metrics()
        return {
            "performance": cm,
            "governor": gm,
            "control_mode": self.control_mode,
            "control_path": CONTROL_PATH_SIMULATOR_ONLY,
        }
