"""
HVAC Action Optimizer.

Evaluates candidate HVAC setpoints over a forward prediction horizon, calculates
thermal comfort, overheating/overcooling penalties, energy usage, and switching costs,
and determines the optimal HVAC control action and labeled targets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from generator.config import HVACConfig, OptimizationWeights, RoomGeometryConfig
from generator.models.hvac_model import HVACClusterModel, AC_DEFINITIONS
from generator.models.room import ZONES
from generator.models.thermal_model import ThermalModel
from generator.optimization.comfort_model import ComfortModel, FangerPMVComfortModel
from generator.optimization.objective import CostBreakdown, ObjectiveEvaluator


@dataclass
class CandidateEvaluation:
    candidate_setpoint_c: float
    predicted_avg_temp_c: float
    predicted_zone_temps: Dict[str, float]
    comfort_score: float
    comfort_penalty: float
    overheat_cost: float
    overcool_cost: float
    directional_cost: float
    energy_cost: float
    switch_cost: float
    total_cost: float
    thermal_state: str
    label_reason: str
    selected: bool


@dataclass
class OptimalActionLabel:
    optimal_room_setpoint_c: float
    optimal_ac1_setpoint_c: float
    optimal_ac2_setpoint_c: float
    optimal_ac3_setpoint_c: float
    optimal_ac4_setpoint_c: float
    optimal_ac1_cooling_level: float
    optimal_ac2_cooling_level: float
    optimal_ac3_cooling_level: float
    optimal_ac4_cooling_level: float
    comfort_score: float
    energy_score: float
    optimization_cost: float
    label_reason: str


class HVACOptimizer:
    """Multi-objective model-predictive optimizer for room and unit HVAC actions (V2 Authoritative)."""

    def __init__(
        self,
        comfort_model: Optional[ComfortModel] = None,
        objective_evaluator: Optional[ObjectiveEvaluator] = None,
        hvac_config: Optional[HVACConfig] = None,
        geometry_config: Optional[RoomGeometryConfig] = None,
        weights: Optional[OptimizationWeights] = None,
    ):
        self.comfort_model = comfort_model or FangerPMVComfortModel()
        self.objective_evaluator = objective_evaluator or ObjectiveEvaluator(weights)
        self.hvac_config = hvac_config or HVACConfig()
        self.geometry_config = geometry_config or RoomGeometryConfig()
        self.candidate_setpoints = self.hvac_config.candidate_setpoints

    def optimize_timestep(
        self,
        current_zone_temps: Dict[str, float],
        current_ac_setpoints: Dict[str, float],
        current_ac_cooling_levels: Dict[str, float],
        current_ac_states: Dict[str, str],
        computer_heat_by_zone: Dict[str, float],
        occupancy_heat_by_zone: Dict[str, float],
        occupancy_counts_by_zone: Dict[str, int],
        outdoor_temp_c: float,
        humidity_percent: float,
        solar_loads_by_zone: Dict[str, float],
        temperature_trend_c_per_min: float = 0.0,
        evaluation_horizon_seconds: float = 120.0,
        horizon_timestep_seconds: float = 15.0,
        prev_optimal_action: Optional[OptimalActionLabel] = None,
        current_dwell_seconds: float = 0.0,
        print_evaluation_table: bool = False,
    ) -> Tuple[OptimalActionLabel, List[CandidateEvaluation]]:
        """Evaluate candidate setpoints over lookahead horizon and pick best action with temporal stability."""
        evaluations: List[CandidateEvaluation] = []
        cand_breakdowns: List[Tuple[float, CostBreakdown]] = []
        best_cost = float("inf")
        best_candidate = None
        best_breakdown = None

        current_avg_sp = float(np.mean(list(current_ac_setpoints.values())))
        current_room_temp = float(np.mean(list(current_zone_temps.values())))
        total_occupancy = sum(occupancy_counts_by_zone.values())
        total_comp_heat = sum(computer_heat_by_zone.values())
        max_cooling_watts = self.hvac_config.cooling_capacity_watts * 4.0

        # Enforce maximum room setpoint transition constraint
        max_sp_delta = self.hvac_config.maximum_setpoint_change_per_step_c
        if prev_optimal_action is not None:
            prev_sp = prev_optimal_action.optimal_room_setpoint_c
            allowed_candidates = [
                sp for sp in self.candidate_setpoints
                if abs(sp - prev_sp) <= (max_sp_delta + 1e-4)
            ]
            if not allowed_candidates:
                allowed_candidates = [prev_sp]
        else:
            prev_sp = None
            allowed_candidates = list(self.candidate_setpoints)

        for candidate_sp in self.candidate_setpoints:
            # Simulate forward thermal evolution over prediction horizon
            pred_temps, pred_levels = self._forward_rollout(
                initial_zone_temps=current_zone_temps,
                initial_cooling_levels=current_ac_cooling_levels,
                ac_states=current_ac_states,
                candidate_setpoint_c=candidate_sp,
                computer_heat_by_zone=computer_heat_by_zone,
                occupancy_heat_by_zone=occupancy_heat_by_zone,
                occupancy_counts_by_zone=occupancy_counts_by_zone,
                outdoor_temp_c=outdoor_temp_c,
                solar_loads_by_zone=solar_loads_by_zone,
                horizon_seconds=evaluation_horizon_seconds,
                dt=horizon_timestep_seconds,
            )

            # Evaluate comfort on predicted end-state
            comfort_res = self.comfort_model.evaluate_room(pred_temps, humidity_percent)

            # Calculate total cooling watts delivered at end-state
            total_cooling = sum(
                pred_levels[ac_id] * self.hvac_config.cooling_capacity_watts
                for ac_id in current_ac_states
                if current_ac_states[ac_id] == "ON"
            )

            avg_pred_temp = float(np.mean(list(pred_temps.values())))
            diff_pred_temp = float(np.ptp(list(pred_temps.values())))

            cost_breakdown = self.objective_evaluator.evaluate(
                comfort_penalty=comfort_res.room_comfort_penalty,
                overheating_penalty=comfort_res.room_overheating_penalty,
                overcooling_penalty=comfort_res.room_overcooling_penalty,
                comfort_score=comfort_res.room_comfort_score,
                total_cooling_watts=total_cooling,
                max_possible_cooling_watts=max_cooling_watts,
                current_setpoint_c=current_avg_sp,
                candidate_setpoint_c=candidate_sp,
                occupancy_total=total_occupancy,
                computer_heat_total=total_comp_heat,
                zone_temp_diff=diff_pred_temp,
                mean_zone_temp=avg_pred_temp,
                current_room_temp=current_room_temp,
                predicted_future_temp=avg_pred_temp,
                temperature_trend_c_per_min=temperature_trend_c_per_min,
            )

            cand_breakdowns.append((candidate_sp, cost_breakdown))

            # Only candidates within allowable step ramp can be selected
            if candidate_sp in allowed_candidates:
                if cost_breakdown.total_cost < best_cost:
                    best_cost = cost_breakdown.total_cost
                    best_candidate = candidate_sp
                    best_breakdown = cost_breakdown

            cand_eval = CandidateEvaluation(
                candidate_setpoint_c=candidate_sp,
                predicted_avg_temp_c=round(avg_pred_temp, 2),
                predicted_zone_temps={z: round(t, 2) for z, t in pred_temps.items()},
                comfort_score=cost_breakdown.comfort_score,
                comfort_penalty=cost_breakdown.comfort_cost,
                overheat_cost=cost_breakdown.overheat_cost,
                overcool_cost=cost_breakdown.overcool_cost,
                directional_cost=cost_breakdown.directional_cost,
                energy_cost=cost_breakdown.energy_cost,
                switch_cost=cost_breakdown.switch_cost,
                total_cost=cost_breakdown.total_cost,
                thermal_state=cost_breakdown.thermal_state,
                label_reason=cost_breakdown.label_reason,
                selected=False,  # updated after dwell/hysteresis resolution
            )
            evaluations.append(cand_eval)

        # Apply minimum dwell time and hysteresis constraints to prevent setpoint chattering
        if prev_optimal_action is not None and prev_sp is not None:
            prev_eval_item = next((e for e in evaluations if e.candidate_setpoint_c == prev_sp), None)
            if prev_eval_item is not None:
                # 1. Dwell time check: If dwell time has not elapsed, retain previous setpoint
                # unless previous setpoint suffers directional or severe thermal violation
                if current_dwell_seconds < self.hvac_config.minimum_setpoint_dwell_seconds:
                    if prev_eval_item.directional_cost <= 0.1 and prev_eval_item.overheat_cost <= 0.5 and prev_eval_item.overcool_cost <= 0.5 and prev_eval_item.comfort_score >= 70.0:
                        best_candidate = prev_sp
                        best_breakdown = next(b for s, b in cand_breakdowns if s == prev_sp)

                # 2. Hysteresis check: If dwell elapsed, only switch if cost improvement exceeds threshold
                elif best_candidate != prev_sp:
                    cost_improvement = prev_eval_item.total_cost - best_cost
                    if cost_improvement < self.hvac_config.setpoint_hysteresis_cost:
                        best_candidate = prev_sp
                        best_breakdown = next(b for s, b in cand_breakdowns if s == prev_sp)

        # Mark selected candidate
        for cand in evaluations:
            if cand.candidate_setpoint_c == best_candidate:
                cand.selected = True
                break

        # Compute zone-tailored optimal setpoints and cooling levels with actuator ramp limiting
        optimal_label = self._build_optimal_label(
            best_candidate=best_candidate,
            current_zone_temps=current_zone_temps,
            best_breakdown=best_breakdown,
            ac_states=current_ac_states,
            prev_optimal_action=prev_optimal_action,
        )

        if print_evaluation_table:
            self._print_inspection_table(evaluations, current_zone_temps, total_comp_heat, total_occupancy)

        return optimal_label, evaluations

    def _forward_rollout(
        self,
        initial_zone_temps: Dict[str, float],
        initial_cooling_levels: Dict[str, float],
        ac_states: Dict[str, str],
        candidate_setpoint_c: float,
        computer_heat_by_zone: Dict[str, float],
        occupancy_heat_by_zone: Dict[str, float],
        occupancy_counts_by_zone: Dict[str, int],
        outdoor_temp_c: float,
        solar_loads_by_zone: Dict[str, float],
        horizon_seconds: float = 120.0,
        dt: float = 15.0,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Closed-loop predictive forward thermal rollout across control horizon."""
        alpha = 0.70  # Effective projection factor over thermal control horizon
        u_wall = self.geometry_config.exterior_wall_conductance_w_per_k
        cap_per_ac = self.hvac_config.cooling_capacity_watts

        pred_temps = {}
        for z in ZONES:
            q_load = (
                computer_heat_by_zone.get(z, 0.0)
                + occupancy_heat_by_zone.get(z, 0.0)
                + solar_loads_by_zone.get(z, 0.0)
            )
            # Localized zone thermal offset based on heat intensity
            bias = (q_load - 500.0) / 3200.0
            target_t = candidate_setpoint_c + bias
            pred_t = initial_zone_temps[z] + alpha * (target_t - initial_zone_temps[z])
            pred_temps[z] = float(round(pred_t, 3))

        # Required inverter cooling levels to sustain predicted thermal states
        levels = {}
        for ac_id, info in AC_DEFINITIONS.items():
            if ac_states.get(ac_id, "ON") == "OFF":
                levels[ac_id] = 0.0
                continue

            eff_pred = sum(info["zone_influence"][z] * pred_temps[z] for z in ZONES)
            eff_load = sum(
                info["zone_influence"][z]
                * (computer_heat_by_zone.get(z, 0.0) + occupancy_heat_by_zone.get(z, 0.0))
                for z in ZONES
            )
            env_gain = max(0.0, u_wall * (outdoor_temp_c - eff_pred))
            req_w = eff_load + env_gain
            # Modulate inverter level proportionally
            lvl = (req_w / cap_per_ac) * (1.0 + max(0.0, eff_pred - candidate_setpoint_c) * 0.20)
            levels[ac_id] = float(np.clip(round(lvl, 3), 0.05, self.hvac_config.max_cooling_level))

        return pred_temps, levels

    def _build_optimal_label(
        self,
        best_candidate: float,
        current_zone_temps: Dict[str, float],
        best_breakdown: CostBreakdown,
        ac_states: Dict[str, str],
        prev_optimal_action: Optional[OptimalActionLabel] = None,
    ) -> OptimalActionLabel:
        """Construct individual AC optimal setpoints and cooling levels respecting actuator constraints."""
        ac_setpoints = {}
        ac_levels = {}

        for ac_id, info in AC_DEFINITIONS.items():
            if ac_states.get(ac_id, "ON") == "OFF":
                ac_setpoints[ac_id] = best_candidate
                ac_levels[ac_id] = 0.0
                continue

            eff_temp = sum(info["zone_influence"][z] * current_zone_temps[z] for z in ZONES)
            temp_err = eff_temp - best_candidate

            # Localized trim for hot zones: if zone is unusually warm compared to room
            if temp_err > 1.2:
                sp_trim = -0.5  # aggressive local cooling
            elif temp_err < -1.0:
                sp_trim = +0.5  # mild trim
            else:
                sp_trim = 0.0

            ac_sp = float(np.clip(best_candidate + sp_trim, min(self.candidate_setpoints), max(self.candidate_setpoints)))
            ac_setpoints[ac_id] = round(ac_sp, 1)

            # Inverter target level for optimal state
            opt_err = eff_temp - ac_sp
            if opt_err <= -0.5:
                raw_lvl = 0.05
            elif opt_err <= 0.0:
                raw_lvl = 0.15
            else:
                raw_lvl = min(1.0, 0.20 + opt_err * self.hvac_config.proportional_gain)

            # Enforce actuator cooling ramp limit: |level[t] - level[t-1]| <= maximum_cooling_change_per_step
            if prev_optimal_action is not None:
                ac_num = ac_id.lower().replace("-", "")
                prev_lvl = getattr(prev_optimal_action, f"optimal_{ac_num}_cooling_level", raw_lvl)
                max_step = self.hvac_config.maximum_cooling_change_per_step
                delta = raw_lvl - prev_lvl
                clamped_delta = float(np.clip(delta, -max_step, max_step))
                lvl = prev_lvl + clamped_delta
            else:
                lvl = raw_lvl

            lvl = float(np.clip(round(lvl, 3), 0.05, self.hvac_config.max_cooling_level))
            ac_levels[ac_id] = round(lvl, 3)

        return OptimalActionLabel(
            optimal_room_setpoint_c=round(best_candidate, 1),
            optimal_ac1_setpoint_c=ac_setpoints.get("AC-1", best_candidate),
            optimal_ac2_setpoint_c=ac_setpoints.get("AC-2", best_candidate),
            optimal_ac3_setpoint_c=ac_setpoints.get("AC-3", best_candidate),
            optimal_ac4_setpoint_c=ac_setpoints.get("AC-4", best_candidate),
            optimal_ac1_cooling_level=ac_levels.get("AC-1", 0.0),
            optimal_ac2_cooling_level=ac_levels.get("AC-2", 0.0),
            optimal_ac3_cooling_level=ac_levels.get("AC-3", 0.0),
            optimal_ac4_cooling_level=ac_levels.get("AC-4", 0.0),
            comfort_score=best_breakdown.comfort_score,
            energy_score=best_breakdown.energy_score,
            optimization_cost=best_breakdown.total_cost,
            label_reason=best_breakdown.label_reason,
        )

    def _print_inspection_table(
        self,
        evaluations: List[CandidateEvaluation],
        current_zone_temps: Dict[str, float],
        total_comp_heat: float,
        total_occupancy: int,
    ):
        """Format and print candidate setpoint evaluation inspection table (as required by Section 37)."""
        avg_temp = float(np.mean(list(current_zone_temps.values())))
        print(f"\n==================== CANDIDATE EVALUATION INSPECTION ====================")
        print(f"Current State: Room Temp={avg_temp:.2f}°C | Comp Heat={total_comp_heat:.1f}W | Occupancy={total_occupancy}")
        print(f"{'Candidate':<11} | {'Pred Temp':<10} | {'Comfort':<8} | {'Energy Cost':<12} | {'Total Obj':<10} | {'Decision'}")
        print("-" * 75)
        for ev in evaluations:
            status = "SELECTED" if ev.selected else "rejected"
            print(
                f"{ev.candidate_setpoint_c:>5.1f} °C    | "
                f"{ev.predicted_avg_temp_c:>6.2f} °C  | "
                f"{ev.comfort_score:>6.1f}   | "
                f"{ev.energy_cost:>10.4f}   | "
                f"{ev.total_cost:>8.4f}   | "
                f"{status}"
            )
        print("=" * 75 + "\n")
