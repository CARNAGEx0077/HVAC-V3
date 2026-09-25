"""
Multi-Objective HVAC Target Optimizer for HVEAC V3.

Deterministically determines the optimal HVAC control action and setpoint for each timestep.
Formulates a multi-objective cost function balancing:
1. Thermal Comfort: Tracking 22.5 C target
2. Safety Bounds: Penalizing overheating (>25 C) and overcooling (<20 C)
3. Zone Balance: Minimizing temperature gradient across zones (max T - min T)
4. Energy Efficiency: Minimizing AC power expenditure
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import minimize

from simulator.config import HvacConfig, OptimizationConfig, ThermalPhysicsConfig
from simulator.room import AC_IDS, ZONE_IDS


@dataclass
class OptimalHvacTarget:
    """Optimal control action label and targets generated for ML training."""
    optimal_cooling_ac1: float
    optimal_cooling_ac2: float
    optimal_cooling_ac3: float
    optimal_cooling_ac4: float
    optimal_temperature_c: float
    optimal_hvac_action: str
    predicted_next_avg_temp: float
    objective_cost: float


class HvacTargetOptimizer:
    """Evaluates candidate HVAC actions and computes optimal settings."""

    def __init__(
        self,
        opt_config: Optional[OptimizationConfig] = None,
        hvac_config: Optional[HvacConfig] = None,
        physics_config: Optional[ThermalPhysicsConfig] = None,
    ):
        self.opt = opt_config or OptimizationConfig()
        self.hvac = hvac_config or HvacConfig()
        self.physics = physics_config or ThermalPhysicsConfig()
        self.weights = np.array(self.hvac.spatial_influence_weights)  # 4x4 matrix (AC to Zone)
        self.q_nominal = self.hvac.nominal_cooling_capacity_w
        self.c_zone = self.physics.zone_heat_capacity_j_k

    def optimize_action(
        self,
        current_temperatures: Dict[str, float],
        zone_disturbance_heat_w: Dict[str, float],
        dt_seconds: float = 60.0,
    ) -> OptimalHvacTarget:
        """
        Solves for optimal continuous cooling vector u = [u1, u2, u3, u4] in [0, 1]^4
        minimizing multi-objective loss function J(u).
        """
        t_current = np.array([current_temperatures[z] for z in ZONE_IDS], dtype=float)
        q_dist = np.array([zone_disturbance_heat_w.get(z, 0.0) for z in ZONE_IDS], dtype=float)

        t_target = self.opt.target_temperature_c
        t_safe_max = self.opt.safe_max_temperature_c
        t_safe_min = self.opt.safe_min_temperature_c

        w_comfort = self.opt.w_comfort
        w_penalty = self.opt.w_safety_penalty
        w_gradient = self.opt.w_zone_gradient
        w_energy = self.opt.w_energy

        def loss_function(u: np.ndarray) -> float:
            # u is shape (4,), in [0, 1]
            # Zone cooling: Q_cooling_z = sum_k (W_k,z * u_k * Q_nominal)
            # W is 4x4: u @ W gives zone cooling fractions of Q_nominal
            q_cool_zone = (u @ self.weights) * self.q_nominal
            q_net_zone = q_dist - q_cool_zone

            # Predicted temperature forward lookahead
            t_pred = t_current + (q_net_zone * dt_seconds) / self.c_zone

            # 1. Comfort deviation from 22.5 C
            loss_comfort = np.sum((t_pred - t_target) ** 2)

            # 2. Overheating and overcooling penalties (one-sided quadratic)
            overheat = np.maximum(0.0, t_pred - t_safe_max)
            overcool = np.maximum(0.0, t_safe_min - t_pred)
            loss_safety = np.sum(overheat ** 2 + overcool ** 2)

            # 3. Zone temperature disparity
            loss_gradient = (np.max(t_pred) - np.min(t_pred)) ** 2

            # 4. Energy cost
            loss_energy = np.sum(u ** 2)

            return float(
                w_comfort * loss_comfort
                + w_penalty * loss_safety
                + w_gradient * loss_gradient
                + w_energy * loss_energy
            )

        # Initial guess based on average thermal load
        avg_load = np.mean(q_dist)
        u0_val = min(max(avg_load / (self.q_nominal * 2.0), 0.1), 0.7)
        u0 = np.full(4, u0_val)
        bounds = [(0.0, 1.0)] * 4

        res = minimize(
            loss_function,
            u0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 40, "ftol": 1e-5},
        )

        u_opt = np.clip(res.x, 0.0, 1.0)
        obj_cost = float(res.fun)

        # Predict resulting temperature with optimal action
        q_cool_opt = (u_opt @ self.weights) * self.q_nominal
        q_net_opt = q_dist - q_cool_opt
        t_pred_opt = t_current + (q_net_opt * dt_seconds) / self.c_zone
        next_avg = float(np.mean(t_pred_opt))

        # Categorize action deterministically
        action_name = self._categorize_action(u_opt)

        return OptimalHvacTarget(
            optimal_cooling_ac1=round(float(u_opt[0]), 3),
            optimal_cooling_ac2=round(float(u_opt[1]), 3),
            optimal_cooling_ac3=round(float(u_opt[2]), 3),
            optimal_cooling_ac4=round(float(u_opt[3]), 3),
            optimal_temperature_c=round(self.opt.target_temperature_c, 2),
            optimal_hvac_action=action_name,
            predicted_next_avg_temp=round(next_avg, 3),
            objective_cost=round(obj_cost, 4),
        )

    def _categorize_action(self, u: np.ndarray) -> str:
        """Assigns a clear categorical label to the optimal control vector."""
        u1, u2, u3, u4 = u
        max_u = float(np.max(u))
        mean_u = float(np.mean(u))
        u_std = float(np.std(u))

        if max_u < 0.12:
            return "ECO_MAINTAIN"

        # Check if cooling is relatively uniform
        if u_std < 0.15:
            if mean_u < 0.45:
                return "MODERATE_UNIFORM_COOL"
            else:
                return "MAX_UNIFORM_COOL"

        # Asymmetric targeted cooling
        west_bias = (u1 + u4) - (u2 + u3)
        east_bias = (u2 + u3) - (u1 + u4)

        if west_bias > 0.35:
            return "TARGETED_WEST_COMPUTE_COOL"
        elif east_bias > 0.35:
            return "TARGETED_EAST_OCCUPANCY_COOL"
        elif (u1 > 0.4 and u3 > 0.4) or (u2 > 0.4 and u4 > 0.4):
            return "OPPOSING_ZONE_COMPENSATE"
        else:
            return "TARGETED_HOTSPOT_COOL"
