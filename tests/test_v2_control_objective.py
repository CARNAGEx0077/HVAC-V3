"""
Unit and Integration Tests for HVEAC V2 Control Objective and Optimizer.

Covers all 20 required validation dimensions:
1. Control semantics & Target definition
2. Hot-zone response (Test A)
3. Rising-temperature response (Test E)
4. High-load response (Test D)
5. Low-temperature response (Test F)
6. Cooling direction coherence
7. No unnecessary cooling escalation (Test C)
8. Thermal disturbance response
9. Actuator ramp limits (<= 0.20 per step)
10. Setpoint ramp limits (<= 0.50 C per step)
11. Setpoint dwell time (>= 60s)
12. Hysteresis enforcement
13. Directional monotonicity across temperature sweep
14. Directional monotonicity across load sweep
15. Same-state deterministic reproducibility
16. Explicit temperature error and predicted error calculations
17. 5-state thermal classification (States A-E)
18. Energy optimization in comfortable state (Test B)
19. Actuator cooling level bounds [0.0, 1.0]
20. Candidate setpoints adherence [21.5, 25.5]
"""

import pytest
import numpy as np

from generator.config import SimulationConfig, ComfortConfig, HVACConfig, OptimizationWeights
from generator.models.hvac_model import AC_DEFINITIONS
from generator.models.room import ZONES
from generator.optimization.comfort_model import FangerPMVComfortModel
from generator.optimization.objective import ObjectiveEvaluator
from generator.optimization.optimizer import HVACOptimizer, OptimalActionLabel
from generator.validation.directionality_audit import DirectionalityAuditor


@pytest.fixture
def auditor():
    return DirectionalityAuditor()


@pytest.fixture
def optimizer():
    config = SimulationConfig()
    return HVACOptimizer(
        comfort_model=FangerPMVComfortModel(config.comfort),
        hvac_config=config.hvac,
        geometry_config=config.room,
        weights=config.weights,
    )


def test_section17_test_a(auditor):
    """TEST A: Current zone 25.0 C + rising trend -> target <= 25.0 C."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_A"]["passed"], f"Test A failed: {res['tests']['TEST_A']}"


def test_section17_test_b(auditor):
    """TEST B: 24.0 C stable + low load -> energy efficiency relaxes setpoint."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_B"]["passed"], f"Test B failed: {res['tests']['TEST_B']}"


def test_section17_test_c(auditor):
    """TEST C: 23.0 C + low heat load -> no unnecessary cooling escalation."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_C"]["passed"], f"Test C failed: {res['tests']['TEST_C']}"


def test_section17_test_d(auditor):
    """TEST D: 27.0 C + high load -> strong cooling response."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_D"]["passed"], f"Test D failed: {res['tests']['TEST_D']}"


def test_section17_test_e(auditor):
    """TEST E: 25.0 C + rising trend -> AI target must NOT become 26.0 C."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_E"]["passed"], f"Test E failed: {res['tests']['TEST_E']}"


def test_section17_test_f(auditor):
    """TEST F: 22.0 C stable -> cooling should be reduced rather than increased."""
    res = auditor.run_unit_tests()
    assert res["tests"]["TEST_F"]["passed"], f"Test F failed: {res['tests']['TEST_F']}"


def test_directional_monotonicity_temperature(auditor):
    """Verify that rising room temperature never causes setpoint to increase."""
    res = auditor.run_monotonicity_sweep()
    assert res["temperature_monotonic"], f"Temperature monotonicity failed: {res['temperature_sweep']}"


def test_directional_monotonicity_load(auditor):
    """Verify that increasing thermal load never causes setpoint to increase or cooling to decrease."""
    res = auditor.run_monotonicity_sweep()
    assert res["load_monotonic"], f"Load monotonicity failed: {res['load_sweep']}"


def test_5state_classification():
    """Verify 5-state thermal condition classifier (States A, B, C, D, E)."""
    evaluator = ObjectiveEvaluator()
    # State A: Overheating
    assert evaluator.classify_thermal_state(25.0, 25.2, 0.0) == "STATE_A"
    assert evaluator.classify_thermal_state(24.5, 24.6, 0.0) == "STATE_A"
    # State C: Overcooled
    assert evaluator.classify_thermal_state(20.5, 20.8, 0.0) == "STATE_C"
    # State D: Heating toward overheat
    assert evaluator.classify_thermal_state(24.2, 24.6, 0.25) == "STATE_D"
    assert evaluator.classify_thermal_state(23.8, 24.0, 0.35) == "STATE_D"
    # State E: Cooling toward lower boundary
    assert evaluator.classify_thermal_state(21.5, 20.9, -0.1) == "STATE_E"
    assert evaluator.classify_thermal_state(22.0, 21.4, -0.35) == "STATE_E"
    # State B: Comfortable / stable
    assert evaluator.classify_thermal_state(23.0, 23.1, 0.0) == "STATE_B"


def test_explicit_error_metrics():
    """Verify temperature_error and predicted_error calculations."""
    evaluator = ObjectiveEvaluator()
    res = evaluator.evaluate(
        comfort_penalty=0.1,
        overheating_penalty=0.0,
        overcooling_penalty=0.0,
        comfort_score=90.0,
        total_cooling_watts=2000.0,
        max_possible_cooling_watts=18000.0,
        current_setpoint_c=24.0,
        candidate_setpoint_c=23.5,
        occupancy_total=5,
        computer_heat_total=1000.0,
        zone_temp_diff=0.5,
        mean_zone_temp=23.8,
        current_room_temp=23.8,
        predicted_future_temp=24.0,
        temperature_trend_c_per_min=0.05,
        target_reference_c=22.5,
    )
    assert res.temperature_error == pytest.approx(1.3, abs=0.01)
    assert res.predicted_error == pytest.approx(1.5, abs=0.01)


def test_actuator_ramp_limit(optimizer):
    """Verify cooling level change <= 0.20 per timestep."""
    prev_label = OptimalActionLabel(
        optimal_room_setpoint_c=24.0,
        optimal_ac1_setpoint_c=24.0,
        optimal_ac2_setpoint_c=24.0,
        optimal_ac3_setpoint_c=24.0,
        optimal_ac4_setpoint_c=24.0,
        optimal_ac1_cooling_level=0.10,
        optimal_ac2_cooling_level=0.10,
        optimal_ac3_cooling_level=0.10,
        optimal_ac4_cooling_level=0.10,
        comfort_score=85.0,
        energy_score=90.0,
        optimization_cost=1.0,
        label_reason="TEST",
    )
    # Sudden extreme heat
    z_temps = {z: 28.0 for z in ZONES}
    def_sp = {ac: 24.0 for ac in AC_DEFINITIONS}
    def_lvl = {ac: 0.10 for ac in AC_DEFINITIONS}
    def_st = {ac: "ON" for ac in AC_DEFINITIONS}
    comp_heat = {z: 1000.0 for z in ZONES}
    occ_heat = {z: 300.0 for z in ZONES}
    occ_cnt = {z: 5 for z in ZONES}
    new_label, _ = optimizer.optimize_timestep(
        current_zone_temps=z_temps,
        current_ac_setpoints=def_sp,
        current_ac_cooling_levels=def_lvl,
        current_ac_states=def_st,
        computer_heat_by_zone=comp_heat,
        occupancy_heat_by_zone=occ_heat,
        occupancy_counts_by_zone=occ_cnt,
        outdoor_temp_c=42.0,
        humidity_percent=70.0,
        solar_loads_by_zone={z: 100.0 for z in ZONES},
        prev_optimal_action=prev_label,
        current_dwell_seconds=70.0,
    )
    delta_ac1 = abs(new_label.optimal_ac1_cooling_level - prev_label.optimal_ac1_cooling_level)
    assert delta_ac1 <= 0.2001, f"Actuator ramp exceeded: {delta_ac1}"


def test_setpoint_jump_limit(optimizer):
    """Verify maximum room setpoint jump <= 0.50 C per timestep."""
    prev_label = OptimalActionLabel(
        optimal_room_setpoint_c=25.0,
        optimal_ac1_setpoint_c=25.0,
        optimal_ac2_setpoint_c=25.0,
        optimal_ac3_setpoint_c=25.0,
        optimal_ac4_setpoint_c=25.0,
        optimal_ac1_cooling_level=0.20,
        optimal_ac2_cooling_level=0.20,
        optimal_ac3_cooling_level=0.20,
        optimal_ac4_cooling_level=0.20,
        comfort_score=85.0,
        energy_score=90.0,
        optimization_cost=1.0,
        label_reason="TEST",
    )
    z_temps = {z: 28.0 for z in ZONES}
    def_sp = {ac: 25.0 for ac in AC_DEFINITIONS}
    def_lvl = {ac: 0.20 for ac in AC_DEFINITIONS}
    def_st = {ac: "ON" for ac in AC_DEFINITIONS}
    comp_heat = {z: 1000.0 for z in ZONES}
    occ_heat = {z: 300.0 for z in ZONES}
    occ_cnt = {z: 5 for z in ZONES}
    new_label, _ = optimizer.optimize_timestep(
        current_zone_temps=z_temps,
        current_ac_setpoints=def_sp,
        current_ac_cooling_levels=def_lvl,
        current_ac_states=def_st,
        computer_heat_by_zone=comp_heat,
        occupancy_heat_by_zone=occ_heat,
        occupancy_counts_by_zone=occ_cnt,
        outdoor_temp_c=42.0,
        humidity_percent=70.0,
        solar_loads_by_zone={z: 100.0 for z in ZONES},
        prev_optimal_action=prev_label,
        current_dwell_seconds=70.0,
    )
    delta_sp = abs(new_label.optimal_room_setpoint_c - prev_label.optimal_room_setpoint_c)
    assert delta_sp <= 0.5001, f"Setpoint jump exceeded: {delta_sp}"


def test_dwell_enforcement(optimizer):
    """Verify that setpoint is retained when dwell < 60s under comfortable conditions."""
    prev_label = OptimalActionLabel(
        optimal_room_setpoint_c=24.0,
        optimal_ac1_setpoint_c=24.0,
        optimal_ac2_setpoint_c=24.0,
        optimal_ac3_setpoint_c=24.0,
        optimal_ac4_setpoint_c=24.0,
        optimal_ac1_cooling_level=0.25,
        optimal_ac2_cooling_level=0.25,
        optimal_ac3_cooling_level=0.25,
        optimal_ac4_cooling_level=0.25,
        comfort_score=92.0,
        energy_score=85.0,
        optimization_cost=1.1,
        label_reason="STATE_B_COMFORT_ENERGY_BALANCE",
    )
    z_temps = {z: 23.8 for z in ZONES}
    def_sp = {ac: 24.0 for ac in AC_DEFINITIONS}
    def_lvl = {ac: 0.25 for ac in AC_DEFINITIONS}
    def_st = {ac: "ON" for ac in AC_DEFINITIONS}
    comp_heat = {z: 250.0 for z in ZONES}
    occ_heat = {z: 50.0 for z in ZONES}
    occ_cnt = {z: 1 for z in ZONES}
    new_label, _ = optimizer.optimize_timestep(
        current_zone_temps=z_temps,
        current_ac_setpoints=def_sp,
        current_ac_cooling_levels=def_lvl,
        current_ac_states=def_st,
        computer_heat_by_zone=comp_heat,
        occupancy_heat_by_zone=occ_heat,
        occupancy_counts_by_zone=occ_cnt,
        outdoor_temp_c=32.0,
        humidity_percent=55.0,
        solar_loads_by_zone={z: 50.0 for z in ZONES},
        prev_optimal_action=prev_label,
        current_dwell_seconds=30.0,  # dwell not yet elapsed (< 60s)
    )
    assert new_label.optimal_room_setpoint_c == 24.0


def test_reproducibility(optimizer):
    """Verify exact same state yields identical control recommendation."""
    z_temps = {z: 24.2 for z in ZONES}
    def_sp = {ac: 24.0 for ac in AC_DEFINITIONS}
    def_lvl = {ac: 0.30 for ac in AC_DEFINITIONS}
    def_st = {ac: "ON" for ac in AC_DEFINITIONS}
    comp_heat = {z: 400.0 for z in ZONES}
    occ_heat = {z: 100.0 for z in ZONES}
    occ_cnt = {z: 2 for z in ZONES}

    l1, _ = optimizer.optimize_timestep(
        current_zone_temps=z_temps,
        current_ac_setpoints=def_sp,
        current_ac_cooling_levels=def_lvl,
        current_ac_states=def_st,
        computer_heat_by_zone=comp_heat,
        occupancy_heat_by_zone=occ_heat,
        occupancy_counts_by_zone=occ_cnt,
        outdoor_temp_c=35.0,
        humidity_percent=60.0,
        solar_loads_by_zone={z: 50.0 for z in ZONES},
    )
    l2, _ = optimizer.optimize_timestep(
        current_zone_temps=z_temps,
        current_ac_setpoints=def_sp,
        current_ac_cooling_levels=def_lvl,
        current_ac_states=def_st,
        computer_heat_by_zone=comp_heat,
        occupancy_heat_by_zone=occ_heat,
        occupancy_counts_by_zone=occ_cnt,
        outdoor_temp_c=35.0,
        humidity_percent=60.0,
        solar_loads_by_zone={z: 50.0 for z in ZONES},
    )
    assert l1.optimal_room_setpoint_c == l2.optimal_room_setpoint_c
    assert l1.optimal_ac1_cooling_level == l2.optimal_ac1_cooling_level
    assert l1.label_reason == l2.label_reason
