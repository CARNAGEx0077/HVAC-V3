"""
Scenario Time-Series Simulation Engine.

Simulates complete time-series thermal scenarios, computes lagged features
without future leakage, evaluates candidate HVAC actions at every timestep,
and generates ML-ready labeled rows.
"""

from collections import deque
from datetime import datetime, timezone
from typing import Dict, Generator, List, Optional
import numpy as np

from generator.config import SimulationConfig
from generator.models.computer_model import ComputerClusterModel, WorkloadCategory
from generator.models.environment_model import EnvironmentModel
from generator.models.hvac_model import HVACClusterModel, AC_DEFINITIONS
from generator.models.occupancy_model import OccupancyModel
from generator.models.room import ZONES, COMPUTER_DEFINITIONS
from generator.models.thermal_model import ThermalModel
from generator.optimization.comfort_model import FangerPMVComfortModel
from generator.optimization.optimizer import HVACOptimizer, OptimalActionLabel
from generator.scenario_templates import ScenarioInstance, TemporalTransition


class ScenarioSimulator:
    """Executes a single continuous scenario simulation producing a labeled time-series."""

    def __init__(self, instance: ScenarioInstance, config: SimulationConfig):
        self.instance = instance
        self.config = config

        # Seeded local RNG for exact reproducibility
        self.rng = np.random.Generator(np.random.PCG64(self.instance.random_seed))

        # Instantiate physical models
        self.computers = ComputerClusterModel(config=self.config.computer)
        self.occupancy = OccupancyModel(config=self.config.occupancy)
        self.environment = EnvironmentModel(
            config=self.config.environment,
            base_temp_c=self.instance.initial_outdoor_temp_c,
            base_humidity_pct=self.instance.initial_humidity_pct,
            base_solar_watts=self.instance.initial_solar_watts,
        )
        self.hvac = HVACClusterModel(config=self.config.hvac)
        self.thermal = ThermalModel(
            config=self.config.room,
            initial_temp_c=float(np.mean(list(self.instance.initial_zone_temps.values()))),
        )
        self.comfort_model = FangerPMVComfortModel(config=self.config.comfort)
        self.optimizer = HVACOptimizer(
            comfort_model=self.comfort_model,
            hvac_config=self.config.hvac,
            geometry_config=self.config.room,
            weights=self.config.weights,
        )

        self._apply_initial_conditions()

        # Rolling buffers for lagged features (30s window = 3 steps at dt=10s)
        self.history_room_temp = deque(maxlen=3)
        self.history_occupancy = deque(maxlen=3)
        self.history_cpu = deque(maxlen=3)
        self.history_gpu = deque(maxlen=3)
        self.history_cooling = deque(maxlen=3)

        # Temporal control state tracking for label optimization
        self.prev_optimal_label: Optional[OptimalActionLabel] = None
        self.setpoint_dwell_seconds: float = 0.0

    def _apply_initial_conditions(self):
        """Set up all models according to sampled ScenarioInstance."""
        # Computers
        for comp_id, (cat, cpu, gpu) in self.instance.computer_workloads.items():
            self.computers.set_computer_state(comp_id, cat, cpu, gpu)

        # Occupancy
        occ = self.instance.initial_occupancy
        self.occupancy.set_occupancy(occ["ZONE_1"], occ["ZONE_2"], occ["ZONE_3"], occ["ZONE_4"])

        # HVAC
        for ac_id, sp in self.instance.initial_ac_setpoints.items():
            self.hvac.set_unit_setpoint(ac_id, sp)
            state = self.instance.initial_ac_states.get(ac_id, "ON")
            self.hvac.set_unit_state(ac_id, state, force=True)
            self.hvac.units[ac_id].cooling_level = self.instance.initial_cooling_levels.get(ac_id, 0.25)

        # Thermal zones
        self.thermal.set_zone_temperatures(self.instance.initial_zone_temps)

    def _handle_transitions(self, elapsed_seconds: float):
        """Execute scheduled temporal transitions at the matching timestamp."""
        for tr in self.instance.transitions:
            if abs(elapsed_seconds - tr.time_seconds) < (self.config.timestep_seconds / 2.0):
                if tr.transition_type == "OCCUPANCY_SPIKE":
                    zone = tr.payload["zone"]
                    curr = self.occupancy.state.as_dict()
                    curr[zone] = min(self.config.occupancy.max_zone_occupancy, curr[zone] + tr.payload["count_delta"])
                    self.occupancy.set_occupancy(curr["ZONE_1"], curr["ZONE_2"], curr["ZONE_3"], curr["ZONE_4"])

                elif tr.transition_type == "OCCUPANCY_SHIFT":
                    fz, tz, people = tr.payload["from_zone"], tr.payload["to_zone"], tr.payload["people"]
                    curr = self.occupancy.state.as_dict()
                    actual_shift = min(curr[fz], people)
                    curr[fz] -= actual_shift
                    curr[tz] += actual_shift
                    self.occupancy.set_occupancy(curr["ZONE_1"], curr["ZONE_2"], curr["ZONE_3"], curr["ZONE_4"])

                elif tr.transition_type == "WORKLOAD_BURST":
                    for c_id in tr.payload["computers"]:
                        cat = tr.payload["target_category"]
                        self.computers.set_computer_state(c_id, cat, 95.0, 92.0)

                elif tr.transition_type == "OUTDOOR_WARMING":
                    self.environment.base_temp_c += tr.payload["temp_increase_c"]

    def run(self) -> Generator[Dict, None, None]:
        """Execute simulation generator yielding ML-ready labeled row dictionaries."""
        dt = float(self.config.timestep_seconds)
        total_steps = int(self.config.duration_seconds / dt)

        base_time_iso = datetime.now(timezone.utc).isoformat()

        for step_idx in range(total_steps):
            elapsed_seconds = step_idx * dt
            self._handle_transitions(elapsed_seconds)

            # 1. READ CURRENT PRE-ACTION STATE
            thermal_summary = self.thermal.get_summary()
            env_state = self.environment.step(elapsed_seconds, dt, self.rng)
            solar_loads = env_state.zone_solar_loads()

            comp_heat_by_zone = self.computers.get_heat_by_zone()
            total_comp_heat = self.computers.get_total_heat_watts()

            occ_heat_by_zone = self.occupancy.get_heat_by_zone()
            total_occ_heat = self.occupancy.get_total_heat_watts()
            occ_counts = self.occupancy.state.as_dict()
            total_occupancy = self.occupancy.state.total_occupancy

            # Total environmental envelope heat gain
            total_env_heat = sum(
                self.config.room.exterior_wall_conductance_w_per_k * (env_state.outdoor_temperature_c - t) + solar_loads[z]
                for z, t in self.thermal.zone_temperatures.items()
            )
            total_env_heat = round(float(total_env_heat), 2)
            total_heat_load = round(total_comp_heat + total_occ_heat + total_env_heat, 2)

            cooling_by_zone = self.hvac.get_cooling_watts_by_zone()
            total_cooling = self.hvac.get_total_cooling_watts()

            # Current comfort metrics
            curr_comfort = self.comfort_model.evaluate_room(
                self.thermal.zone_temperatures,
                env_state.humidity_percent,
            )

            # Update rolling history (past & current, strictly no future leakage)
            avg_cpu = float(np.mean([c.cpu_utilization_percent for c in self.computers.computers.values()]))
            avg_gpu = float(np.mean([c.gpu_utilization_percent for c in self.computers.computers.values()]))
            self.history_room_temp.append(thermal_summary.room_average_temperature_c)
            self.history_occupancy.append(total_occupancy)
            self.history_cpu.append(avg_cpu)
            self.history_gpu.append(avg_gpu)
            self.history_cooling.append(total_cooling)

            # Current AC states before optimization
            current_setpoints = {ac_id: u.setpoint_c for ac_id, u in self.hvac.units.items()}
            current_levels = {ac_id: u.cooling_level for ac_id, u in self.hvac.units.items()}
            current_states = {ac_id: u.state for ac_id, u in self.hvac.units.items()}

            # Physical temperature trend calculation (deg C / min) strictly from past history
            if len(self.history_room_temp) >= 2:
                hist = list(self.history_room_temp)
                dt_span = (len(hist) - 1) * dt
                temp_trend_c_per_min = round(((hist[-1] - hist[0]) / max(1.0, dt_span)) * 60.0, 3)
            else:
                temp_trend_c_per_min = 0.0

            # 2. RUN OPTIMIZER FOR LABELS (purely based on available state)
            optimal_label, _ = self.optimizer.optimize_timestep(
                current_zone_temps=self.thermal.zone_temperatures,
                current_ac_setpoints=current_setpoints,
                current_ac_cooling_levels=current_levels,
                current_ac_states=current_states,
                computer_heat_by_zone=comp_heat_by_zone,
                occupancy_heat_by_zone=occ_heat_by_zone,
                occupancy_counts_by_zone=occ_counts,
                outdoor_temp_c=env_state.outdoor_temperature_c,
                humidity_percent=env_state.humidity_percent,
                solar_loads_by_zone=solar_loads,
                temperature_trend_c_per_min=temp_trend_c_per_min,
                evaluation_horizon_seconds=self.config.weights.evaluation_horizon_seconds,
                prev_optimal_action=self.prev_optimal_label,
                current_dwell_seconds=self.setpoint_dwell_seconds,
            )

            # Update dwell tracking for subsequent timesteps
            if self.prev_optimal_label is not None and optimal_label.optimal_room_setpoint_c == self.prev_optimal_label.optimal_room_setpoint_c:
                self.setpoint_dwell_seconds += dt
            else:
                self.setpoint_dwell_seconds = 0.0
            self.prev_optimal_label = optimal_label

            # 3. CONSTRUCT ML-READY RECORD
            row = {
                # METADATA
                "scenario_id": self.instance.scenario_id,
                "scenario_family": self.instance.family.name,
                "run_id": self.instance.run_id,
                "random_seed": self.instance.random_seed,
                "timestamp": base_time_iso,
                "simulation_time_seconds": int(elapsed_seconds),

                # OCCUPANCY
                "occupancy_total": total_occupancy,
                "occupancy_zone_1": occ_counts["ZONE_1"],
                "occupancy_zone_2": occ_counts["ZONE_2"],
                "occupancy_zone_3": occ_counts["ZONE_3"],
                "occupancy_zone_4": occ_counts["ZONE_4"],

                # ENVIRONMENT
                "outdoor_temperature_c": env_state.outdoor_temperature_c,
                "humidity_percent": env_state.humidity_percent,
                "solar_load": env_state.solar_load_watts,

                # CURRENT HVAC
                "ac1_state": current_states["AC-1"],
                "ac1_setpoint_c": current_setpoints["AC-1"],
                "ac1_cooling_level": current_levels["AC-1"],

                "ac2_state": current_states["AC-2"],
                "ac2_setpoint_c": current_setpoints["AC-2"],
                "ac2_cooling_level": current_levels["AC-2"],

                "ac3_state": current_states["AC-3"],
                "ac3_setpoint_c": current_setpoints["AC-3"],
                "ac3_cooling_level": current_levels["AC-3"],

                "ac4_state": current_states["AC-4"],
                "ac4_setpoint_c": current_setpoints["AC-4"],
                "ac4_cooling_level": current_levels["AC-4"],

                # CURRENT THERMAL
                "zone_1_temperature_c": thermal_summary.zone_1_temperature_c,
                "zone_2_temperature_c": thermal_summary.zone_2_temperature_c,
                "zone_3_temperature_c": thermal_summary.zone_3_temperature_c,
                "zone_4_temperature_c": thermal_summary.zone_4_temperature_c,
                "room_average_temperature_c": thermal_summary.room_average_temperature_c,
                "minimum_temperature_c": thermal_summary.minimum_temperature_c,
                "maximum_temperature_c": thermal_summary.maximum_temperature_c,
                "temperature_difference_c": thermal_summary.temperature_difference_c,

                # HEAT LOADS
                "total_computer_heat_watts": total_comp_heat,
                "total_occupancy_heat_watts": total_occ_heat,
                "total_environmental_heat_watts": total_env_heat,
                "total_heat_load_watts": total_heat_load,
                "total_hvac_cooling_watts": total_cooling,

                # COMFORT
                "comfort_score": curr_comfort.room_comfort_score,
                "comfort_penalty": curr_comfort.room_comfort_penalty,

                # LAGGED / TIME-SERIES CONTEXT
                "room_temp_30s_avg": round(sum(self.history_room_temp) / len(self.history_room_temp), 3),
                "occupancy_total_30s_avg": round(sum(self.history_occupancy) / len(self.history_occupancy), 1),
                "cpu_util_30s_avg": round(sum(self.history_cpu) / len(self.history_cpu), 2),
                "gpu_util_30s_avg": round(sum(self.history_gpu) / len(self.history_gpu), 2),
                "hvac_cooling_30s_avg": round(sum(self.history_cooling) / len(self.history_cooling), 2),
            }

            # 10 COMPUTERS (Section 24 specification)
            for idx in range(1, 11):
                cid = f"Computer {idx}"
                comp = self.computers.computers[cid]
                row[f"computer_{idx}_cpu"] = comp.cpu_utilization_percent
                row[f"computer_{idx}_gpu"] = comp.gpu_utilization_percent
                cat_val = comp.workload_category.value if hasattr(comp.workload_category, "value") else str(comp.workload_category)
                row[f"computer_{idx}_workload"] = cat_val
                row[f"computer_{idx}_heat"] = comp.simulated_heat_watts

            # LABELS (Target Actions and machine-readable attribution)
            row["optimal_room_setpoint_c"] = optimal_label.optimal_room_setpoint_c
            row["optimal_ac1_setpoint_c"] = optimal_label.optimal_ac1_setpoint_c
            row["optimal_ac2_setpoint_c"] = optimal_label.optimal_ac2_setpoint_c
            row["optimal_ac3_setpoint_c"] = optimal_label.optimal_ac3_setpoint_c
            row["optimal_ac4_setpoint_c"] = optimal_label.optimal_ac4_setpoint_c

            row["optimal_ac1_cooling_level"] = optimal_label.optimal_ac1_cooling_level
            row["optimal_ac2_cooling_level"] = optimal_label.optimal_ac2_cooling_level
            row["optimal_ac3_cooling_level"] = optimal_label.optimal_ac3_cooling_level
            row["optimal_ac4_cooling_level"] = optimal_label.optimal_ac4_cooling_level

            row["optimization_cost"] = optimal_label.optimization_cost
            row["label_reason"] = optimal_label.label_reason

            yield row

            # 4. ADVANCE PHYSICAL MODELS TO NEXT TIMESTEP
            self.thermal.step(
                computer_heat=comp_heat_by_zone,
                occupancy_heat=occ_heat_by_zone,
                occupancy_counts=occ_counts,
                cooling_watts=cooling_by_zone,
                outdoor_temp_c=env_state.outdoor_temperature_c,
                solar_loads=solar_loads,
                timestep_seconds=dt,
            )
            self.hvac.step(self.thermal.zone_temperatures, dt)
            self.computers.step(self.rng)
