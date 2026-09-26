"""
Safety Governor — Deterministic HVAC Safety Validation Layer.

Runtime Architecture:
    SIMULATION STATE
          ↓
    THERMAL CONTROL ALGORITHM
          ↓
    SAFETY GOVERNOR
          ↓
    HVAC ACTUATORS
          ↓
    THERMAL PHYSICS
          ↓
    NEXT SIMULATION STATE

Safety Governor is allowed to:
    - Validate numeric values (reject NaN, Inf)
    - Clamp invalid ranges to safe bounds
    - Enforce actuator limits (0–100% cooling, physical setpoint limits)
    - Enforce minimum dwell or rate limits if applicable
    - Activate safe baseline fallback when inputs are corrupted

Safety Governor must NOT:
    - Optimize
    - Search candidate actions
    - Compare multiple commands
    - Calculate a new 'best' HVAC action
    - Act as a secondary decision-maker
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hveac.safety_governor")

# Actuator Limits
MIN_COOLING_PCT = 0.0
MAX_COOLING_PCT = 100.0
MIN_SETPOINT_C = 18.0
MAX_SETPOINT_C = 28.0
SAFE_FALLBACK_SETPOINT_C = 24.0


@dataclass
class SafetyValidationReport:
    """Detailed report from the Safety Governor validation pass."""
    status: str = "ACTIVE"         # ACTIVE, CLAMPED, FALLBACK
    reason: str = "Safety Governor validated all commands"
    is_safe: bool = True
    clamped_fields: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)


class SafetyGovernor:
    """
    Deterministic Safety Governor for HVAC Actuation Commands.
    Validates and constrains commands produced by ThermalControlAlgorithm.
    """

    def __init__(
        self,
        min_cooling_pct: float = MIN_COOLING_PCT,
        max_cooling_pct: float = MAX_COOLING_PCT,
        min_setpoint_c: float = MIN_SETPOINT_C,
        max_setpoint_c: float = MAX_SETPOINT_C,
    ):
        self.min_cooling_pct = min_cooling_pct
        self.max_cooling_pct = max_cooling_pct
        self.min_setpoint_c = min_setpoint_c
        self.max_setpoint_c = max_setpoint_c
        self._validation_count: int = 0
        self._fallback_count: int = 0
        self._last_report: Optional[SafetyValidationReport] = None

    def reset(self):
        """Reset governor state counters."""
        self._validation_count = 0
        self._fallback_count = 0
        self._last_report = None
        logger.info("[SAFETY GOVERNOR] Reset completed")

    @property
    def last_report(self) -> Optional[SafetyValidationReport]:
        return self._last_report

    def validate_decision(self, decision: Any, sim_time_s: float = 0.0) -> Any:
        """
        Validate and constrain an HVACDecision instance.

        Args:
            decision: HVACDecision from ThermalControlAlgorithm
            sim_time_s: Current simulation timestamp in seconds

        Returns:
            The validated and safely constrained decision object.
        """
        self._validation_count += 1
        clamped_fields: List[str] = []
        violations: List[str] = []

        if decision is None:
            self._fallback_count += 1
            report = SafetyValidationReport(
                status="FALLBACK",
                reason="Null decision received; safety fallback activated",
                is_safe=False,
                violations=["Null decision object"],
            )
            self._last_report = report
            logger.warning("[SAFETY GOVERNOR] Null decision received -> Fallback activated")
            return decision

        # 1. Validate and constrain per-AC cooling percentages and setpoints
        for ac_id, cooling_pct in list(decision.ac_cooling_percent.items()):
            # Numeric sanity check
            if not isinstance(cooling_pct, (int, float)) or not math.isfinite(cooling_pct):
                violations.append(f"{ac_id} cooling_pct non-finite: {cooling_pct}")
                decision.ac_cooling_percent[ac_id] = 0.0
                decision.is_fallback = True
                continue

            # Actuator bounds check
            if cooling_pct < self.min_cooling_pct:
                decision.ac_cooling_percent[ac_id] = self.min_cooling_pct
                clamped_fields.append(f"{ac_id} cooling clamped to {self.min_cooling_pct}%")
            elif cooling_pct > self.max_cooling_pct:
                decision.ac_cooling_percent[ac_id] = self.max_cooling_pct
                clamped_fields.append(f"{ac_id} cooling clamped to {self.max_cooling_pct}%")

        for ac_id, setpoint_c in list(decision.ac_setpoint_c.items()):
            # Numeric sanity check
            if not isinstance(setpoint_c, (int, float)) or not math.isfinite(setpoint_c):
                violations.append(f"{ac_id} setpoint_c non-finite: {setpoint_c}")
                decision.ac_setpoint_c[ac_id] = SAFE_FALLBACK_SETPOINT_C
                decision.is_fallback = True
                continue

            # Setpoint bounds check
            if setpoint_c < self.min_setpoint_c:
                decision.ac_setpoint_c[ac_id] = self.min_setpoint_c
                clamped_fields.append(f"{ac_id} setpoint clamped to {self.min_setpoint_c}°C")
            elif setpoint_c > self.max_setpoint_c:
                decision.ac_setpoint_c[ac_id] = self.max_setpoint_c
                clamped_fields.append(f"{ac_id} setpoint clamped to {self.max_setpoint_c}°C")

        # 2. Determine safety status
        if violations or decision.is_fallback:
            self._fallback_count += 1
            status = "FALLBACK"
            reason = f"Safety Governor engaged fallback: {'; '.join(violations) if violations else 'Fallback decision'}"
        elif clamped_fields:
            status = "CLAMPED"
            reason = f"Safety Governor clamped actuators: {'; '.join(clamped_fields)}"
        else:
            status = "ACTIVE"
            reason = "Safety Governor validated all commands"

        report = SafetyValidationReport(
            status=status,
            reason=reason,
            is_safe=len(violations) == 0,
            clamped_fields=clamped_fields,
            violations=violations,
        )
        self._last_report = report

        return decision
