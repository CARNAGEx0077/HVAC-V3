"""
ScenarioEngine for HVEAC V3 Thermal Simulation.

Coordinates models, steps through time, executes physical energy balance,
computes optimal HVAC targets, and produces machine-readable records.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from simulator.config import SimulationEngineConfig
from simulator.computer_model import ComputerClusterModel
from simulator.environment_model import EnvironmentModel
from simulator.hvac_model import HvacSystemModel
from simulator.occupancy_model import OccupancyModel
from simulator.optimizer import HvacTargetOptimizer
from simulator.room import RoomLayout, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario
from simulator.thermal_model import ThermalModel


class ScenarioEngine:
    """Master simulation engine for generating ML training data."""

    def __init__(self, config: Optional[SimulationEngineConfig] = None):
        self.config = config or SimulationEngineConfig()
        self.room = RoomLayout()
        self.computer_cluster = ComputerClusterModel(self.config.computer)
        self.occupancy_model = OccupancyModel(self.config.occupancy)
        self.environment_model = EnvironmentModel(self.config.physics)
        self.hvac_model = HvacSystemModel(self.config.hvac)
        self.thermal_model = ThermalModel(self.config.physics)
        self.optimizer = HvacTargetOptimizer(
            self.config.optimization,
            self.config.hvac,
            self.config.physics,
        )

    def run_scenario(
        self,
        scenario: BaseScenario,
        run_idx: int = 0,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Executes a complete scenario simulation run and returns list of flat records.
        """
        # Seed generator deterministically for this scenario and run
        # Seed formula: seed + (scenario_id * 10000) + (run_idx * 100)
        run_seed = seed + (scenario.scenario_id * 10000) + (run_idx * 100)
        rng = np.random.default_rng(run_seed)

        # 1. Initialize temperatures
        init_temps = scenario.get_initial_temperatures(run_idx=run_idx, rng=rng)
        self.thermal_model.reset(init_temps)

        dt = scenario.timestep_seconds
        duration = scenario.duration_seconds
        num_timesteps = int(duration / dt)

        records: List[Dict[str, Any]] = []

        for step_idx in range(num_timesteps):
            t_sec = round(step_idx * dt, 1)

            # A. Get scenario drivers for this timestep
            ts_input = scenario.get_timestep_input(
                t_seconds=t_sec,
                current_temperatures=self.thermal_model.temperatures,
                run_idx=run_idx,
                rng=rng,
            )

            # B. Apply computer workloads & calculate computational heat
            self.computer_cluster.set_workloads(ts_input.computer_workloads)
            comp_states = self.computer_cluster.get_all_states()
            comp_heat_by_zone = self.computer_cluster.get_zone_heat_watts()

            # C. Apply occupancy & calculate metabolic sensible heat
            self.occupancy_model.set_zone_counts(ts_input.zone_occupancy)
            occ_state = self.occupancy_model.get_state()

            # D. Apply environment conditions & calculate envelope heat
            self.environment_model.set_conditions(
                outdoor_temp_c=ts_input.outdoor_temp_c,
                relative_humidity=ts_input.relative_humidity,
                solar_irradiance_w_m2=ts_input.solar_irradiance_w_m2,
            )
            env_state = self.environment_model.calculate_envelope_heat(self.thermal_model.temperatures)

            # E. Apply current baseline HVAC controls & calculate delivered cooling
            self.hvac_model.set_all_controls(
                ts_input.hvac_cooling_levels,
                ts_input.hvac_setpoints,
            )
            hvac_state = self.hvac_model.calculate_cooling_distribution()

            # F. Total envelope heat by zone (transmission + solar)
            total_env_by_zone = {
                z: round(env_state.zone_transmission_heat_w[z] + env_state.zone_solar_heat_w[z], 2)
                for z in ZONE_IDS
            }

            # G. Advance thermal dynamic state
            thermal_state = self.thermal_model.step(
                dt_seconds=dt,
                comp_heat_by_zone=comp_heat_by_zone,
                occ_heat_by_zone=occ_state.zone_heat_watts,
                env_heat_by_zone=total_env_by_zone,
                hvac_cooling_by_zone=hvac_state.zone_cooling_w,
            )

            # H. Total disturbance heat currently acting on zones (excluding HVAC cooling)
            disturbance_heat_by_zone = {
                z: round(
                    comp_heat_by_zone[z]
                    + occ_state.zone_heat_watts[z]
                    + total_env_by_zone[z]
                    + thermal_state.zone_interzone_heat_w[z],
                    2,
                )
                for z in ZONE_IDS
            }

            # I. Compute optimal target control action for future ML training
            optimal_target = self.optimizer.optimize_action(
                current_temperatures=thermal_state.zone_temperatures,
                zone_disturbance_heat_w=disturbance_heat_by_zone,
                dt_seconds=60.0,
            )

            # J. Assemble flat, fully-formed record
            rec: Dict[str, Any] = {
                # Metadata
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.name,
                "run_id": run_idx,
                "timestep_index": step_idx,
                "timestamp_seconds": t_sec,
                # Occupancy features
                "total_occupancy": occ_state.total_occupancy,
                "occupancy_zone_1": occ_state.zone_occupancy["ZONE_1"],
                "occupancy_zone_2": occ_state.zone_occupancy["ZONE_2"],
                "occupancy_zone_3": occ_state.zone_occupancy["ZONE_3"],
                "occupancy_zone_4": occ_state.zone_occupancy["ZONE_4"],
            }

            # Computer features (Computers 1 to 10)
            for c in comp_states:
                cid = c.computer_id
                rec[f"cpu_util_comp_{cid}"] = c.cpu_utilization_percent
                rec[f"gpu_util_comp_{cid}"] = c.gpu_utilization_percent
                rec[f"workload_cat_comp_{cid}"] = c.workload_category
                rec[f"heat_w_comp_{cid}"] = c.synthetic_heat_w

            # Environment features
            rec["outdoor_temperature_c"] = env_state.outdoor_temperature_c
            rec["relative_humidity_percent"] = env_state.relative_humidity_percent
            rec["solar_irradiance_w_m2"] = env_state.solar_irradiance_w_m2

            # Baseline HVAC operational state
            for ac_id in ("AC-1", "AC-2", "AC-3", "AC-4"):
                key_suffix = ac_id.lower().replace("-", "")
                rec[f"cooling_level_{key_suffix}"] = hvac_state.ac_units[ac_id].cooling_level
                rec[f"setpoint_{key_suffix}"] = hvac_state.ac_units[ac_id].setpoint_c
                rec[f"state_{key_suffix}"] = hvac_state.ac_units[ac_id].state

            # Resulting thermal state
            rec["zone_1_temperature_c"] = thermal_state.zone_temperatures["ZONE_1"]
            rec["zone_2_temperature_c"] = thermal_state.zone_temperatures["ZONE_2"]
            rec["zone_3_temperature_c"] = thermal_state.zone_temperatures["ZONE_3"]
            rec["zone_4_temperature_c"] = thermal_state.zone_temperatures["ZONE_4"]
            rec["room_average_temperature_c"] = thermal_state.room_average_temperature
            rec["max_zone_temperature_c"] = thermal_state.max_zone_temperature
            rec["min_zone_temperature_c"] = thermal_state.min_zone_temperature
            rec["temperature_gradient_c"] = thermal_state.temperature_gradient

            # Heat loads by zone (Watts)
            rec["comp_heat_zone_1_w"] = thermal_state.zone_comp_heat_w["ZONE_1"]
            rec["comp_heat_zone_2_w"] = thermal_state.zone_comp_heat_w["ZONE_2"]
            rec["comp_heat_zone_3_w"] = thermal_state.zone_comp_heat_w["ZONE_3"]
            rec["comp_heat_zone_4_w"] = thermal_state.zone_comp_heat_w["ZONE_4"]

            rec["occ_heat_zone_1_w"] = thermal_state.zone_occ_heat_w["ZONE_1"]
            rec["occ_heat_zone_2_w"] = thermal_state.zone_occ_heat_w["ZONE_2"]
            rec["occ_heat_zone_3_w"] = thermal_state.zone_occ_heat_w["ZONE_3"]
            rec["occ_heat_zone_4_w"] = thermal_state.zone_occ_heat_w["ZONE_4"]

            rec["env_heat_zone_1_w"] = thermal_state.zone_env_heat_w["ZONE_1"]
            rec["env_heat_zone_2_w"] = thermal_state.zone_env_heat_w["ZONE_2"]
            rec["env_heat_zone_3_w"] = thermal_state.zone_env_heat_w["ZONE_3"]
            rec["env_heat_zone_4_w"] = thermal_state.zone_env_heat_w["ZONE_4"]

            rec["interzone_heat_zone_1_w"] = thermal_state.zone_interzone_heat_w["ZONE_1"]
            rec["interzone_heat_zone_2_w"] = thermal_state.zone_interzone_heat_w["ZONE_2"]
            rec["interzone_heat_zone_3_w"] = thermal_state.zone_interzone_heat_w["ZONE_3"]
            rec["interzone_heat_zone_4_w"] = thermal_state.zone_interzone_heat_w["ZONE_4"]

            rec["hvac_cooling_zone_1_w"] = thermal_state.zone_hvac_cooling_w["ZONE_1"]
            rec["hvac_cooling_zone_2_w"] = thermal_state.zone_hvac_cooling_w["ZONE_2"]
            rec["hvac_cooling_zone_3_w"] = thermal_state.zone_hvac_cooling_w["ZONE_3"]
            rec["hvac_cooling_zone_4_w"] = thermal_state.zone_hvac_cooling_w["ZONE_4"]

            rec["net_heat_zone_1_w"] = thermal_state.zone_net_heat_w["ZONE_1"]
            rec["net_heat_zone_2_w"] = thermal_state.zone_net_heat_w["ZONE_2"]
            rec["net_heat_zone_3_w"] = thermal_state.zone_net_heat_w["ZONE_3"]
            rec["net_heat_zone_4_w"] = thermal_state.zone_net_heat_w["ZONE_4"]

            # Cluster totals (Watts)
            rec["total_computational_heat_w"] = thermal_state.total_comp_heat_w
            rec["total_occupancy_heat_w"] = thermal_state.total_occ_heat_w
            rec["total_envelope_heat_w"] = thermal_state.total_env_heat_w
            rec["total_hvac_cooling_w"] = thermal_state.total_hvac_cooling_w
            rec["total_net_heat_w"] = thermal_state.total_net_heat_w

            # Optimal target / labels for ML
            rec["optimal_cooling_ac1"] = optimal_target.optimal_cooling_ac1
            rec["optimal_cooling_ac2"] = optimal_target.optimal_cooling_ac2
            rec["optimal_cooling_ac3"] = optimal_target.optimal_cooling_ac3
            rec["optimal_cooling_ac4"] = optimal_target.optimal_cooling_ac4
            rec["optimal_temperature_c"] = optimal_target.optimal_temperature_c
            rec["optimal_hvac_action"] = optimal_target.optimal_hvac_action
            rec["target_objective_cost"] = optimal_target.objective_cost

            records.append(rec)

        return records
