"""
Directional Control Monotonicity and Coherence Audit (V2 Authoritative).

Audits control decisions across physical regimes:
- TEST A: Hot + Rising temp -> Cooling target <= current temp
- TEST B: Stable + Low load -> Energy optimization allowed
- TEST C: 23°C + Low load -> No cooling escalation
- TEST D: 27°C + High load -> Strong cooling response
- TEST E: 25°C + Rising trend -> Setpoint != 26°C (must cool)
- TEST F: 22°C + Stable -> Cooling reduced
- Monotonicity audits across load and temperature sweeps
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

from generator.config import SimulationConfig, ComfortConfig, HVACConfig, OptimizationWeights
from generator.models.hvac_model import AC_DEFINITIONS
from generator.models.room import ZONES
from generator.optimization.comfort_model import FangerPMVComfortModel
from generator.optimization.objective import ObjectiveEvaluator
from generator.optimization.optimizer import HVACOptimizer, OptimalActionLabel


class DirectionalityAuditor:
    """Audits directional coherence and physical monotonicity of HVAC control decisions."""

    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.optimizer = HVACOptimizer(
            comfort_model=FangerPMVComfortModel(self.config.comfort),
            hvac_config=self.config.hvac,
            geometry_config=self.config.room,
            weights=self.config.weights,
        )

    def run_unit_tests(self) -> Dict[str, Any]:
        """Execute automated tests A through F specified by Section 17."""
        results = {}

        # Default state template
        def_ac_sp = {ac: 24.0 for ac in AC_DEFINITIONS}
        def_ac_lvl = {ac: 0.25 for ac in AC_DEFINITIONS}
        def_ac_st = {ac: "ON" for ac in AC_DEFINITIONS}
        def_solar = {z: 50.0 for z in ZONES}

        # TEST A: Current zone = 25.0°C, Rising trend, Predicted > 24.5°C, High load
        # Expected: target <= current zone temp (cooling oriented)
        z_temps_a = {z: 25.0 for z in ZONES}
        comp_heat_a = {z: 800.0 for z in ZONES}  # 3200W total
        occ_heat_a = {z: 200.0 for z in ZONES}   # 800W total
        occ_cnt_a = {z: 3 for z in ZONES}
        label_a, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_a,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_a,
            occupancy_heat_by_zone=occ_heat_a,
            occupancy_counts_by_zone=occ_cnt_a,
            outdoor_temp_c=36.0,
            humidity_percent=65.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=+0.35,
        )
        passed_a = (label_a.optimal_room_setpoint_c <= 25.0)
        results["TEST_A"] = {
            "name": "Hot + Rising Trend Cooling Orientation",
            "current_temp": 25.0,
            "trend": "+0.35 C/min",
            "optimal_target": label_a.optimal_room_setpoint_c,
            "condition": "optimal_target <= 25.0",
            "passed": passed_a,
        }

        # TEST B: Current zone = 24.0°C, Stable, Low load
        # Expected: energy-efficient control may maintain or relax setpoint
        z_temps_b = {z: 24.0 for z in ZONES}
        comp_heat_b = {z: 200.0 for z in ZONES}  # 800W total
        occ_heat_b = {z: 50.0 for z in ZONES}
        occ_cnt_b = {z: 1 for z in ZONES}
        label_b, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_b,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_b,
            occupancy_heat_by_zone=occ_heat_b,
            occupancy_counts_by_zone=occ_cnt_b,
            outdoor_temp_c=30.0,
            humidity_percent=55.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=0.0,
        )
        passed_b = (label_b.optimal_room_setpoint_c >= 23.5)
        results["TEST_B"] = {
            "name": "Comfortable + Stable Energy Optimization",
            "current_temp": 24.0,
            "trend": "0.0 C/min",
            "optimal_target": label_b.optimal_room_setpoint_c,
            "condition": "optimal_target >= 23.5",
            "passed": passed_b,
        }

        # TEST C: Current zone = 23.0°C, Low heat load
        # Expected: no unnecessary cooling escalation (target >= 23.0°C)
        z_temps_c = {z: 23.0 for z in ZONES}
        label_c, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_c,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_b,
            occupancy_heat_by_zone=occ_heat_b,
            occupancy_counts_by_zone=occ_cnt_b,
            outdoor_temp_c=30.0,
            humidity_percent=55.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=0.0,
        )
        passed_c = (label_c.optimal_room_setpoint_c >= 23.0)
        results["TEST_C"] = {
            "name": "Cool Room No Cooling Escalation",
            "current_temp": 23.0,
            "trend": "0.0 C/min",
            "optimal_target": label_c.optimal_room_setpoint_c,
            "condition": "optimal_target >= 23.0",
            "passed": passed_c,
        }

        # TEST D: Current zone = 27.0°C, High thermal load
        # Expected: strong cooling response (optimal_target <= 23.0°C)
        z_temps_d = {z: 27.0 for z in ZONES}
        label_d, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_d,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_a,
            occupancy_heat_by_zone=occ_heat_a,
            occupancy_counts_by_zone=occ_cnt_a,
            outdoor_temp_c=40.0,
            humidity_percent=70.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=+0.40,
        )
        passed_d = (label_d.optimal_room_setpoint_c <= 23.0 and label_d.optimal_ac1_cooling_level >= 0.50)
        results["TEST_D"] = {
            "name": "Severe Overheating Strong Cooling",
            "current_temp": 27.0,
            "trend": "+0.40 C/min",
            "optimal_target": label_d.optimal_room_setpoint_c,
            "cooling_level": label_d.optimal_ac1_cooling_level,
            "condition": "optimal_target <= 23.0 and cooling_level >= 0.50",
            "passed": passed_d,
        }

        # TEST E: Current zone = 25.0°C, Trend = +0.4°C/min
        # Expected: target must NOT be 26.0°C (must be <= 25.0°C)
        z_temps_e = {z: 25.0 for z in ZONES}
        label_e, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_e,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_a,
            occupancy_heat_by_zone=occ_heat_a,
            occupancy_counts_by_zone=occ_cnt_a,
            outdoor_temp_c=36.0,
            humidity_percent=65.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=+0.40,
        )
        passed_e = (label_e.optimal_room_setpoint_c <= 25.0 and label_e.optimal_room_setpoint_c != 26.0)
        results["TEST_E"] = {
            "name": "Imminent Overheating Target Restriction",
            "current_temp": 25.0,
            "trend": "+0.40 C/min",
            "optimal_target": label_e.optimal_room_setpoint_c,
            "condition": "optimal_target <= 25.0 and optimal_target != 26.0",
            "passed": passed_e,
        }

        # TEST F: Current zone = 22.0°C, Temperature stable
        # Expected: cooling should be reduced rather than increased (target >= 23.0°C)
        z_temps_f = {z: 22.0 for z in ZONES}
        label_f, _ = self.optimizer.optimize_timestep(
            current_zone_temps=z_temps_f,
            current_ac_setpoints=def_ac_sp,
            current_ac_cooling_levels=def_ac_lvl,
            current_ac_states=def_ac_st,
            computer_heat_by_zone=comp_heat_b,
            occupancy_heat_by_zone=occ_heat_b,
            occupancy_counts_by_zone=occ_cnt_b,
            outdoor_temp_c=28.0,
            humidity_percent=50.0,
            solar_loads_by_zone=def_solar,
            temperature_trend_c_per_min=0.0,
        )
        passed_f = (label_f.optimal_room_setpoint_c >= 23.0 and label_f.optimal_ac1_cooling_level <= 0.20)
        results["TEST_F"] = {
            "name": "Overcooled Room Cooling Reduction",
            "current_temp": 22.0,
            "trend": "0.0 C/min",
            "optimal_target": label_f.optimal_room_setpoint_c,
            "cooling_level": label_f.optimal_ac1_cooling_level,
            "condition": "optimal_target >= 23.0 and cooling_level <= 0.20",
            "passed": passed_f,
        }

        all_passed = all(r["passed"] for r in results.values())
        return {
            "status": "PASS" if all_passed else "FAIL",
            "tests": results,
        }

    def run_monotonicity_sweep(self) -> Dict[str, Any]:
        """Perform controlled perturbation sweeps verifying physical monotonicity (Section 18)."""
        def_ac_sp = {ac: 24.0 for ac in AC_DEFINITIONS}
        def_ac_lvl = {ac: 0.25 for ac in AC_DEFINITIONS}
        def_ac_st = {ac: "ON" for ac in AC_DEFINITIONS}
        def_solar = {z: 50.0 for z in ZONES}

        # Sweep 1: Temperature Sweep from 21.0 to 28.0 C (with fixed load 1800W)
        temp_sweep = []
        for t_val in [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0]:
            z_temps = {z: t_val for z in ZONES}
            comp_heat = {z: 350.0 for z in ZONES}
            occ_heat = {z: 100.0 for z in ZONES}
            occ_cnt = {z: 2 for z in ZONES}
            trend = 0.0 if t_val <= 24.0 else (t_val - 24.0) * 0.1
            lbl, _ = self.optimizer.optimize_timestep(
                current_zone_temps=z_temps,
                current_ac_setpoints=def_ac_sp,
                current_ac_cooling_levels=def_ac_lvl,
                current_ac_states=def_ac_st,
                computer_heat_by_zone=comp_heat,
                occupancy_heat_by_zone=occ_heat,
                occupancy_counts_by_zone=occ_cnt,
                outdoor_temp_c=34.0,
                humidity_percent=60.0,
                solar_loads_by_zone=def_solar,
                temperature_trend_c_per_min=trend,
            )
            temp_sweep.append({
                "temp": t_val,
                "setpoint": lbl.optimal_room_setpoint_c,
                "cooling": lbl.optimal_ac1_cooling_level,
            })

        # Verify: higher temperature never results in higher setpoint (cooling must intensify or maintain)
        temp_monotonic = True
        for i in range(1, len(temp_sweep)):
            if temp_sweep[i]["setpoint"] > temp_sweep[i - 1]["setpoint"] + 1e-4:
                # Monotonicity violation: hotter temp requested warmer setpoint!
                temp_monotonic = False
                break

        # Sweep 2: Thermal Load Sweep from 800W to 4400W (with fixed temp 24.5 C)
        load_sweep = []
        for load_w in [800.0, 1600.0, 2400.0, 3200.0, 4000.0]:
            z_temps = {z: 24.5 for z in ZONES}
            comp_h = {z: load_w / 4.0 for z in ZONES}
            occ_h = {z: 50.0 for z in ZONES}
            occ_cnt = {z: 1 for z in ZONES}
            trend = (load_w - 1600.0) / 8000.0
            lbl, _ = self.optimizer.optimize_timestep(
                current_zone_temps=z_temps,
                current_ac_setpoints=def_ac_sp,
                current_ac_cooling_levels=def_ac_lvl,
                current_ac_states=def_ac_st,
                computer_heat_by_zone=comp_h,
                occupancy_heat_by_zone=occ_h,
                occupancy_counts_by_zone=occ_cnt,
                outdoor_temp_c=35.0,
                humidity_percent=60.0,
                solar_loads_by_zone=def_solar,
                temperature_trend_c_per_min=trend,
            )
            load_sweep.append({
                "load_w": load_w,
                "setpoint": lbl.optimal_room_setpoint_c,
                "cooling": lbl.optimal_ac1_cooling_level,
            })

        # Verify: higher load never results in warmer setpoint or reduced cooling
        load_monotonic = True
        for i in range(1, len(load_sweep)):
            if load_sweep[i]["setpoint"] > load_sweep[i - 1]["setpoint"] + 1e-4:
                load_monotonic = False
                break
            if load_sweep[i]["cooling"] < load_sweep[i - 1]["cooling"] - 0.05:
                load_monotonic = False
                break

        return {
            "status": "PASS" if (temp_monotonic and load_monotonic) else "FAIL",
            "temperature_monotonic": temp_monotonic,
            "load_monotonic": load_monotonic,
            "temperature_sweep": temp_sweep,
            "load_sweep": load_sweep,
        }

    def audit_dataset_file(self, csv_path: Path) -> Dict[str, Any]:
        """Audit an existing generated dataset CSV for directional control coherence (Section 26)."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        total_rows = 0
        violations_hot_warming = 0
        violations_overcooled_escalation = 0

        sample_cases = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                curr_t = float(row["room_average_temperature_c"])
                sp = float(row["optimal_room_setpoint_c"])
                cooling = float(row["optimal_ac1_cooling_level"])
                heat = float(row["total_heat_load_watts"])

                # Check 1: If current temp >= 24.5°C, setpoint must NOT exceed current temp
                if curr_t >= 24.5:
                    if sp > curr_t + 0.01:
                        violations_hot_warming += 1
                        if len(sample_cases) < 10:
                            sample_cases.append({
                                "type": "VIOLATION_HOT_WARMING",
                                "row": total_rows,
                                "temp": curr_t,
                                "target": sp,
                                "heat": heat,
                            })

                # Check 2: If current temp <= 21.0°C, setpoint should not be aggressively lowered
                if curr_t <= 21.0:
                    if sp < curr_t - 0.5:
                        violations_overcooled_escalation += 1

        passed = (violations_hot_warming == 0 and violations_overcooled_escalation == 0)
        return {
            "status": "PASS" if passed else "FAIL",
            "total_rows_audited": total_rows,
            "violations_hot_warming": violations_hot_warming,
            "violations_overcooled_escalation": violations_overcooled_escalation,
            "sample_cases": sample_cases,
        }
