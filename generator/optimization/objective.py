"""
Multi-Objective Cost Function and Label Attribution (V2 Authoritative).

Calculates the explicit scalar cost combining:
1. Thermal safety & Overheating risk
2. Acceptable comfort (PMV-based)
3. Overcooling risk
4. Directional control coherence (State A-E penalty)
5. Actuator switching penalty
6. Energy consumption

Enforces the core HVEAC principle:
When a zone is too hot or moving toward overheating, cooling action must
increase or cooling setpoint target must move downward. Energy saving
must NEVER override thermal control correctness.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from generator.config import OptimizationWeights


@dataclass
class CostBreakdown:
    comfort_cost: float
    overheat_cost: float
    overcool_cost: float
    directional_cost: float
    energy_cost: float
    switch_cost: float
    total_cost: float
    comfort_score: float
    energy_score: float
    temperature_error: float
    predicted_error: float
    thermal_state: str
    label_reason: str


class ObjectiveEvaluator:
    """Evaluates multi-objective trade-offs for candidate HVAC control actions with directional coherence."""

    def __init__(self, weights: OptimizationWeights = None):
        self.weights = weights or OptimizationWeights()

    def get_objective_weights_doc(self) -> str:
        """Explicitly returns the mathematical formulation and active weights."""
        return (
            "TOTAL_COST = "
            f"{self.weights.comfort_weight:.2f} * comfort_penalty + "
            f"{self.weights.overheat_weight:.2f} * overheating_penalty + "
            f"{self.weights.overcool_weight:.2f} * overcooling_penalty + "
            f"{self.weights.directional_weight:.2f} * directional_penalty + "
            f"{self.weights.energy_weight:.2f} * energy_penalty + "
            f"{self.weights.switch_weight:.2f} * switching_penalty"
        )

    def classify_thermal_state(
        self,
        current_temp_c: float,
        predicted_temp_c: float,
        trend_c_per_min: float,
        upper_threshold_c: float = 24.5,
        lower_threshold_c: float = 21.0,
    ) -> str:
        """Classify the operative thermal condition into States A through E (Section 4)."""
        # State A: Too hot / overheating
        if current_temp_c >= upper_threshold_c:
            return "STATE_A"

        # State C: Too cold / overcooled
        if current_temp_c < lower_threshold_c:
            return "STATE_C"

        # State D: Heating toward upper thermal boundary
        if predicted_temp_c >= upper_threshold_c or trend_c_per_min >= 0.20 or (current_temp_c >= 24.0 and trend_c_per_min > 0.05):
            return "STATE_D"

        # State E: Cooling toward lower thermal boundary
        if predicted_temp_c <= lower_threshold_c or trend_c_per_min <= -0.30:
            return "STATE_E"

        # State B: Comfortable / stable
        return "STATE_B"

    def evaluate(
        self,
        comfort_penalty: float,
        overheating_penalty: float,
        overcooling_penalty: float,
        comfort_score: float,
        total_cooling_watts: float,
        max_possible_cooling_watts: float,
        current_setpoint_c: float,
        candidate_setpoint_c: float,
        occupancy_total: int,
        computer_heat_total: float,
        zone_temp_diff: float,
        mean_zone_temp: float,
        current_room_temp: float = None,
        predicted_future_temp: float = None,
        temperature_trend_c_per_min: float = 0.0,
        target_reference_c: float = 22.5,
        upper_comfort_threshold_c: float = 24.5,
        lower_comfort_threshold_c: float = 21.0,
    ) -> CostBreakdown:
        """Calculate weighted objective cost with directional coherence and attribute label reason."""
        curr_t = current_room_temp if current_room_temp is not None else mean_zone_temp
        pred_t = predicted_future_temp if predicted_future_temp is not None else mean_zone_temp

        # 1. Explicit thermal control errors (Section 10)
        temperature_error = round(curr_t - target_reference_c, 3)
        predicted_error = round(pred_t - target_reference_c, 3)

        # 2. 5-State Thermal Control Classification (Section 4)
        thermal_state = self.classify_thermal_state(
            current_temp_c=curr_t,
            predicted_temp_c=pred_t,
            trend_c_per_min=temperature_trend_c_per_min,
            upper_threshold_c=upper_comfort_threshold_c,
            lower_threshold_c=lower_comfort_threshold_c,
        )

        # 3. Directional Coherence Penalty (Section 3, 7, 9)
        # Authoritative rule:
        # If State A or D: target must NOT be above current room temp.
        # If State C or E: target must NOT be colder than current room temp.
        directional_penalty = 0.0
        if thermal_state == "STATE_A":
            if candidate_setpoint_c > curr_t:
                directional_penalty += (candidate_setpoint_c - curr_t + 0.5) ** 2
            # Penalize targets that fail to cool adequately during overheating
            if candidate_setpoint_c > 24.0:
                directional_penalty += 0.5 * (candidate_setpoint_c - 24.0) ** 2

        elif thermal_state == "STATE_D":
            if candidate_setpoint_c > curr_t:
                directional_penalty += (candidate_setpoint_c - curr_t) ** 2
            if temperature_trend_c_per_min >= 0.30 and candidate_setpoint_c > 23.5:
                directional_penalty += 0.35 * (candidate_setpoint_c - 23.5) ** 2

        elif thermal_state == "STATE_C":
            if candidate_setpoint_c < curr_t:
                directional_penalty += (curr_t - candidate_setpoint_c + 0.5) ** 2
            if candidate_setpoint_c < 23.0:
                directional_penalty += 0.5 * (23.0 - candidate_setpoint_c) ** 2

        elif thermal_state == "STATE_E":
            if candidate_setpoint_c < curr_t:
                directional_penalty += (curr_t - candidate_setpoint_c) ** 2

        # 4. Normalized energy penalty: fraction of full capacity utilization
        energy_penalty = total_cooling_watts / max(1.0, max_possible_cooling_watts)
        energy_score = round(max(0.0, 100.0 * (1.0 - energy_penalty)), 2)

        # 5. Switching penalty: normalized difference from current operational setpoint
        setpoint_delta = abs(candidate_setpoint_c - current_setpoint_c)
        switching_penalty = setpoint_delta / 5.0

        # Compute cost components
        c_comf = self.weights.comfort_weight * comfort_penalty
        c_overheat = self.weights.overheat_weight * overheating_penalty
        c_overcool = self.weights.overcool_weight * overcooling_penalty
        c_dir = self.weights.directional_weight * directional_penalty

        # Energy saving is attenuated during active overheating to prevent overriding safety
        if thermal_state in ["STATE_A", "STATE_D"]:
            c_energy = (self.weights.energy_weight * 0.5) * (energy_penalty ** 1.2)
        else:
            c_energy = self.weights.energy_weight * (energy_penalty ** 1.2)

        c_switch = self.weights.switch_weight * switching_penalty

        total_cost = c_comf + c_overheat + c_overcool + c_dir + c_energy + c_switch

        # Determine machine-readable label reason
        reason = self._determine_label_reason(
            thermal_state=thermal_state,
            occupancy_total=occupancy_total,
            computer_heat_total=computer_heat_total,
            zone_temp_diff=zone_temp_diff,
            candidate_setpoint_c=candidate_setpoint_c,
            current_setpoint_c=current_setpoint_c,
        )

        return CostBreakdown(
            comfort_cost=round(c_comf, 4),
            overheat_cost=round(c_overheat, 4),
            overcool_cost=round(c_overcool, 4),
            directional_cost=round(c_dir, 4),
            energy_cost=round(c_energy, 4),
            switch_cost=round(c_switch, 4),
            total_cost=round(total_cost, 4),
            comfort_score=round(comfort_score, 2),
            energy_score=energy_score,
            temperature_error=temperature_error,
            predicted_error=predicted_error,
            thermal_state=thermal_state,
            label_reason=reason,
        )

    def _determine_label_reason(
        self,
        thermal_state: str,
        occupancy_total: int,
        computer_heat_total: float,
        zone_temp_diff: float,
        candidate_setpoint_c: float = 24.0,
        current_setpoint_c: float = 24.0,
    ) -> str:
        """Assign standardized machine-readable justification according to V2 control regimes."""
        if thermal_state == "STATE_A":
            return "STATE_A_OVERHEATING_COOLING"
        if thermal_state == "STATE_D":
            return "STATE_D_PREVENTIVE_COOLING"
        if thermal_state == "STATE_C":
            return "STATE_C_OVERCOOLED_RECOVERY"
        if thermal_state == "STATE_E":
            return "STATE_E_PREVENTIVE_WARMING"

        # In State B (Comfortable / Stable):
        if computer_heat_total >= 2500.0:
            return "HIGH_COMPUTE_LOAD"
        if occupancy_total >= 18:
            return "HIGH_OCCUPANCY"
        if zone_temp_diff >= 1.6:
            return "LOCALIZED_ZONE_COOLING"
        if candidate_setpoint_c > current_setpoint_c:
            return "STATE_B_ENERGY_OPTIMIZED"
        return "STATE_B_COMFORT_ENERGY_BALANCE"

