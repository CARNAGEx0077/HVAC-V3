"""
HVEAC Brain V1 — Safety / Constraint Governor

Dedicated safety component sitting strictly between HVEAC Brain inference
and the simulated HVAC controller:

HVEAC Brain
    ↓
Safety Governor
    ↓
AI Control Adapter
    ↓
Existing Simulator HVAC Controller

SAFETY INVARIANTS:
1. Validates every AI control command against thermal, rate, and dwell bounds.
2. The model NEVER directly calls the HVAC actuator.
3. If ANY check fails, safely falls back to BASELINE simulator control.
4. Operates strictly in SIMULATION ONLY mode.
   ABSOLUTE RULE: NO REAL HVAC CONTROL, NO PHYSICAL ACTUATORS.
"""

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hveac.brain.safety")

# Default valid class set from trained model
VALID_MODEL_CLASSES = (24.5, 25.0, 25.5, 26.0, 26.5)

# Safety Status Constants
STATUS_APPLIED = "APPLIED"
STATUS_RATE_LIMITED = "RATE_LIMITED"
STATUS_HYSTERESIS_HOLD = "HYSTERESIS_HOLD"
STATUS_FALLBACK_BASELINE = "FALLBACK_BASELINE"
STATUS_REJECTED = "REJECTED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

CONTROL_PATH_SIMULATOR_ONLY = "SIMULATOR_ONLY"


@dataclass(frozen=True)
class SafetyGovernorConfig:
    """Configurable safety constraints for AI HVAC control."""
    min_setpoint_c: float = 24.5          # Minimum allowed room setpoint
    max_setpoint_c: float = 26.5          # Maximum allowed room setpoint
    max_rate_step_c: float = 0.50         # Max setpoint step jump per transition
    min_dwell_seconds: float = 60.0       # Minimum dwell time between changes (sim time)
    hysteresis_deadband_c: float = 0.25   # Deadband to suppress chatter on minor fluctuations
    max_event_log_size: int = 100         # Bounded event buffer capacity
    valid_classes: Tuple[float, ...] = VALID_MODEL_CLASSES
    strict_class_enforcement: bool = True # Must match one of valid target classes
    clamp_out_of_bounds: bool = True      # Clamp rather than hard-reject out-of-bounds


@dataclass
class SafetyResult:
    """Structured decision output from SafetyGovernor."""
    mode: str                             # "AI_CONTROL" | "BASELINE" | "SHADOW"
    ai_requested_setpoint_c: Optional[float]
    safe_setpoint_c: float
    applied_setpoint_c: float
    status: str                           # APPLIED, RATE_LIMITED, HYSTERESIS_HOLD, etc.
    reason: str
    control_path: str = CONTROL_PATH_SIMULATOR_ONLY
    validation_latency_ms: float = 0.0
    timestamp_sim_s: float = 0.0
    baseline_setpoint_c: float = 22.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "ai_requested_setpoint_c": self.ai_requested_setpoint_c,
            "safe_setpoint_c": self.safe_setpoint_c,
            "applied_setpoint_c": self.applied_setpoint_c,
            "baseline_setpoint_c": self.baseline_setpoint_c,
            "status": self.status,
            "reason": self.reason,
            "control_path": self.control_path,
            "validation_latency_ms": round(self.validation_latency_ms, 3),
            "timestamp_sim_s": round(self.timestamp_sim_s, 1),
        }


@dataclass
class ControlEvent:
    """Compact record for the engineering control event log."""
    sim_time_s: float
    formatted_time: str
    ai_requested_c: Optional[float]
    applied_c: float
    baseline_c: float
    status: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_time_s": round(self.sim_time_s, 1),
            "time": self.formatted_time,
            "ai_requested_c": self.ai_requested_c,
            "applied_c": self.applied_c,
            "baseline_c": self.baseline_c,
            "status": self.status,
            "reason": self.reason,
        }


class SafetyGovernor:
    """
    Enforces all safety bounds, dwell times, rate limits, and fail-safes
    on AI room setpoint requests before reaching the simulator HVAC controller.
    """

    def __init__(self, config: Optional[SafetyGovernorConfig] = None):
        self.config = config or SafetyGovernorConfig()
        self._last_applied_setpoint: Optional[float] = None
        self._last_change_sim_time_s: float = -9999.0
        self._events: List[ControlEvent] = []
        self._fallback_count: int = 0
        self._rate_limit_count: int = 0
        self._dwell_hold_count: int = 0
        self._applied_count: int = 0

    @property
    def last_applied_setpoint(self) -> Optional[float]:
        return self._last_applied_setpoint

    @property
    def last_applied_setpoint_c(self) -> Optional[float]:
        return self._last_applied_setpoint

    @last_applied_setpoint_c.setter
    def last_applied_setpoint_c(self, val: Optional[float]):
        self._last_applied_setpoint = val

    @property
    def last_transition_time_s(self) -> float:
        return self._last_change_sim_time_s

    @last_transition_time_s.setter
    def last_transition_time_s(self, val: float):
        self._last_change_sim_time_s = val

    @property
    def events(self) -> List[ControlEvent]:
        return self._events

    def reset(self, initial_setpoint: Optional[float] = None):
        """Reset governor state on simulation reset or scenario switch."""
        self._last_applied_setpoint = initial_setpoint
        self._last_change_sim_time_s = -9999.0
        self._events.clear()
        self._fallback_count = 0
        self._rate_limit_count = 0
        self._dwell_hold_count = 0
        self._applied_count = 0
        logger.info("[SAFETY GOVERNOR] State reset")

    def validate_command(
        self,
        ai_requested_setpoint_c: Optional[float],
        baseline_setpoint_c: float = 22.5,
        simulation_time_s: float = 0.0,
        control_mode: str = "AI_CONTROL",
        model_status: str = "READY",
        inference_status: str = "OK",
        hvac_available: bool = True,
        feature_warnings: Optional[List[str]] = None,
        model_available: bool = True,
        timestamp_sim_s: Optional[float] = None,
    ) -> SafetyResult:
        """
        Validates an AI setpoint command through the 10 safety checks:
        1. Model availability
        2. Schema / feature validation
        3. Finite-value validation
        4. Valid class validation
        5. Setpoint bounds
        6. Rate limit
        7. Minimum dwell
        8. Hysteresis
        9. Simulation HVAC availability
        10. Control-mode verification

        Returns a structured SafetyResult with applied_setpoint_c and reasoning.
        """
        if isinstance(model_status, bool):
            model_available = model_status
            model_status = "READY" if model_available else "UNAVAILABLE"
        if not model_available:
            model_status = "UNAVAILABLE"
        if timestamp_sim_s is not None:
            simulation_time_s = timestamp_sim_s
        t_start = time.perf_counter()

        # Helper to construct fallback result
        def make_fallback(reason: str, status: str = STATUS_FALLBACK_BASELINE) -> SafetyResult:
            t_end = time.perf_counter()
            self._fallback_count += 1
            res = SafetyResult(
                mode=control_mode,
                ai_requested_setpoint_c=ai_requested_setpoint_c,
                safe_setpoint_c=baseline_setpoint_c,
                applied_setpoint_c=baseline_setpoint_c,
                baseline_setpoint_c=baseline_setpoint_c,
                status=status,
                reason=reason,
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=(t_end - t_start) * 1000.0,
                timestamp_sim_s=simulation_time_s,
            )
            self._record_event(res)
            return res

        # -------------------------------------------------------------
        # CHECK 10: Control Mode Verification
        # -------------------------------------------------------------
        if control_mode != "AI_CONTROL":
            t_end = time.perf_counter()
            res = SafetyResult(
                mode=control_mode,
                ai_requested_setpoint_c=ai_requested_setpoint_c,
                safe_setpoint_c=baseline_setpoint_c,
                applied_setpoint_c=baseline_setpoint_c,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_APPLIED if control_mode == "BASELINE" else "SHADOW_OBSERVATION",
                reason=f"Operating in {control_mode} mode; existing optimizer is authoritative",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=(t_end - t_start) * 1000.0,
                timestamp_sim_s=simulation_time_s,
            )
            return res

        # -------------------------------------------------------------
        # CHECK 1: Model Availability
        # -------------------------------------------------------------
        if model_status != "READY" or inference_status != "OK":
            return make_fallback(
                f"Model unavailable (status={model_status}, inference={inference_status})",
                status=STATUS_UNAVAILABLE,
            )

        # -------------------------------------------------------------
        # CHECK 9: Simulation HVAC Availability
        # -------------------------------------------------------------
        if not hvac_available:
            return make_fallback("Simulator HVAC system unavailable", status=STATUS_UNAVAILABLE)

        # -------------------------------------------------------------
        # CHECK 2: Feature Warnings / Adapter Failure
        # -------------------------------------------------------------
        if feature_warnings:
            # Check for fatal warnings like NaN inputs or missing core features
            fatal_warns = [w for w in feature_warnings if "NaN" in w or "missing" in w.lower()]
            if fatal_warns:
                return make_fallback(f"Feature adapter validation failure: {fatal_warns[0]}")

        # -------------------------------------------------------------
        # CHECK 3: Finite-Value Validation
        # -------------------------------------------------------------
        if ai_requested_setpoint_c is None:
            return make_fallback("AI setpoint is None")
        try:
            val = float(ai_requested_setpoint_c)
            if math.isnan(val) or math.isinf(val):
                return make_fallback("AI setpoint is NaN or Infinite")
        except (ValueError, TypeError):
            return make_fallback(f"AI setpoint is non-numeric: {ai_requested_setpoint_c}")

        # -------------------------------------------------------------
        # CHECK 4: Valid Class Validation
        # -------------------------------------------------------------
        # Verify requested setpoint matches or is close to trained model classes
        is_valid_class = any(abs(val - c) < 0.05 for c in self.config.valid_classes)
        if not is_valid_class:
            if self.config.strict_class_enforcement:
                # Snap to closest valid class or reject
                closest = min(self.config.valid_classes, key=lambda c: abs(c - val))
                logger.warning(
                    f"[SAFETY GOVERNOR] AI requested non-class setpoint {val}°C. "
                    f"Snapping to closest valid class {closest}°C"
                )
                val = closest

        # -------------------------------------------------------------
        # CHECK 5: Setpoint Bounds Enforcement
        # -------------------------------------------------------------
        bounded_val = val
        bound_violation = False
        if bounded_val < self.config.min_setpoint_c:
            bound_violation = True
            if self.config.clamp_out_of_bounds:
                bounded_val = self.config.min_setpoint_c
                logger.warning(
                    f"[SAFETY GOVERNOR] Clamped setpoint {val}°C to MIN {self.config.min_setpoint_c}°C"
                )
            else:
                return make_fallback(
                    f"Requested {val}°C below MIN bound {self.config.min_setpoint_c}°C",
                    status=STATUS_REJECTED,
                )
        elif bounded_val > self.config.max_setpoint_c:
            bound_violation = True
            if self.config.clamp_out_of_bounds:
                bounded_val = self.config.max_setpoint_c
                logger.warning(
                    f"[SAFETY GOVERNOR] Clamped setpoint {val}°C to MAX {self.config.max_setpoint_c}°C"
                )
            else:
                return make_fallback(
                    f"Requested {val}°C above MAX bound {self.config.max_setpoint_c}°C",
                    status=STATUS_REJECTED,
                )

        # Current reference for rate and dwell checks
        current_setpoint = self._last_applied_setpoint
        if current_setpoint is None:
            # First application: adopt bounded setpoint immediately
            self._last_applied_setpoint = bounded_val
            self._last_change_sim_time_s = simulation_time_s
            self._applied_count += 1
            t_end = time.perf_counter()
            res = SafetyResult(
                mode=control_mode,
                ai_requested_setpoint_c=ai_requested_setpoint_c,
                safe_setpoint_c=bounded_val,
                applied_setpoint_c=bounded_val,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_APPLIED if not bound_violation else STATUS_RATE_LIMITED,
                reason="Initial setpoint established" if not bound_violation else "Clamped to valid bounds",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=(t_end - t_start) * 1000.0,
                timestamp_sim_s=simulation_time_s,
            )
            self._record_event(res)
            return res

        # -------------------------------------------------------------
        # CHECK 8: Hysteresis Deadband (Suppress chatter)
        # -------------------------------------------------------------
        delta = abs(bounded_val - current_setpoint)
        if delta < self.config.hysteresis_deadband_c:
            # Command is essentially unchanged or within noise floor: retain current
            t_end = time.perf_counter()
            res = SafetyResult(
                mode=control_mode,
                ai_requested_setpoint_c=ai_requested_setpoint_c,
                safe_setpoint_c=current_setpoint,
                applied_setpoint_c=current_setpoint,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_HYSTERESIS_HOLD,
                reason=f"Delta ({delta:.2f}°C) within hysteresis deadband ({self.config.hysteresis_deadband_c}°C)",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=(t_end - t_start) * 1000.0,
                timestamp_sim_s=simulation_time_s,
            )
            self._record_event(res)
            return res

        # -------------------------------------------------------------
        # CHECK 7: Minimum Dwell Time (Simulated time)
        # -------------------------------------------------------------
        time_since_change = simulation_time_s - self._last_change_sim_time_s
        if time_since_change < self.config.min_dwell_seconds:
            self._dwell_hold_count += 1
            t_end = time.perf_counter()
            remaining_dwell = self.config.min_dwell_seconds - time_since_change
            res = SafetyResult(
                mode=control_mode,
                ai_requested_setpoint_c=ai_requested_setpoint_c,
                safe_setpoint_c=current_setpoint,
                applied_setpoint_c=current_setpoint,
                baseline_setpoint_c=baseline_setpoint_c,
                status=STATUS_HYSTERESIS_HOLD,
                reason=f"Minimum dwell active ({time_since_change:.1f}s / {self.config.min_dwell_seconds:.0f}s, {remaining_dwell:.1f}s remaining)",
                control_path=CONTROL_PATH_SIMULATOR_ONLY,
                validation_latency_ms=(t_end - t_start) * 1000.0,
                timestamp_sim_s=simulation_time_s,
            )
            self._record_event(res)
            return res

        # -------------------------------------------------------------
        # CHECK 6: Setpoint Rate Limit (Max step jump)
        # -------------------------------------------------------------
        target_step = bounded_val - current_setpoint
        rate_limited = False
        safe_val = bounded_val

        if abs(target_step) > self.config.max_rate_step_c + 1e-4:
            rate_limited = True
            self._rate_limit_count += 1
            step_sign = 1.0 if target_step > 0 else -1.0
            safe_val = current_setpoint + (step_sign * self.config.max_rate_step_c)
            safe_val = round(safe_val, 2)
            reason = f"Rate limited: Jump of {abs(target_step):.2f}°C clamped to {self.config.max_rate_step_c}°C step"
        else:
            reason = "Validation passed"

        # Apply validated setpoint
        self._last_applied_setpoint = safe_val
        self._last_change_sim_time_s = simulation_time_s
        self._applied_count += 1

        t_end = time.perf_counter()
        res = SafetyResult(
            mode=control_mode,
            ai_requested_setpoint_c=ai_requested_setpoint_c,
            safe_setpoint_c=safe_val,
            applied_setpoint_c=safe_val,
            baseline_setpoint_c=baseline_setpoint_c,
            status=STATUS_RATE_LIMITED if rate_limited else STATUS_APPLIED,
            reason=reason,
            control_path=CONTROL_PATH_SIMULATOR_ONLY,
            validation_latency_ms=(t_end - t_start) * 1000.0,
            timestamp_sim_s=simulation_time_s,
        )
        self._record_event(res)
        return res

    def _record_event(self, res: SafetyResult):
        """Append event to bounded rolling buffer."""
        # Format simulation time (HH:MM:SS)
        s = int(res.timestamp_sim_s)
        hh = s // 3600
        mm = (s % 3600) // 60
        ss = s % 60
        fmt_time = f"{hh:02d}:{mm:02d}:{ss:02d}"

        evt = ControlEvent(
            sim_time_s=res.timestamp_sim_s,
            formatted_time=fmt_time,
            ai_requested_c=res.ai_requested_setpoint_c,
            applied_c=res.applied_setpoint_c,
            baseline_c=res.baseline_setpoint_c,
            status=res.status,
            reason=res.reason,
        )
        self._events.append(evt)
        if len(self._events) > self.config.max_event_log_size:
            self._events.pop(0)

    def get_events(self) -> List[Dict[str, Any]]:
        """Return list of recent control events."""
        return [e.to_dict() for e in self._events]

    def get_metrics(self) -> Dict[str, Any]:
        """Return safety intervention counts."""
        total = self._applied_count + self._rate_limit_count + self._dwell_hold_count + self._fallback_count
        return {
            "applied_count": self._applied_count,
            "rate_limit_count": self._rate_limit_count,
            "dwell_hold_count": self._dwell_hold_count,
            "fallback_count": self._fallback_count,
            "total_decisions": total,
            "rate_limited_pct": round((self._rate_limit_count / max(1, total)) * 100, 2),
            "fallback_pct": round((self._fallback_count / max(1, total)) * 100, 2),
            "control_path": CONTROL_PATH_SIMULATOR_ONLY,
        }
