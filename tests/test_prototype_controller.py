"""
Automated Test Suite for HVEAC Prototype Control & Deterministic Controller Substitute.

Covers all 26 test requirements from Section 38, plus:
- Section 39: Critical Optimizer-Absence Test
- Section 40: Runtime Control Test (Strict single decision-maker pipeline)
"""

import math
from unittest.mock import patch, MagicMock
import pytest

from backend.control.thermal_control_algorithm import (
    ThermalControlAlgorithm,
    HVACDecision,
    ZoneControlState,
    COMFORT_TARGET_C,
)
from backend.control.safety_governor import (
    SafetyGovernor,
    MIN_COOLING_PCT,
    MAX_COOLING_PCT,
    MIN_SETPOINT_C,
    MAX_SETPOINT_C,
)
from backend.simulation.manager import SimulationManager
import simulator.optimizer


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONTROLLER INITIALIZES
# ─────────────────────────────────────────────────────────────────────────────
def test_1_controller_initializes():
    controller = ThermalControlAlgorithm()
    assert controller is not None
    assert controller._weights is not None
    assert len(controller._weights) == 4
    assert len(controller._weights[0]) == 4


# ─────────────────────────────────────────────────────────────────────────────
# 2. LOW TEMPERATURE (Below Target)
# ─────────────────────────────────────────────────────────────────────────────
def test_2_low_temperature():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {
            "zone_temperatures_c": {"zone_1": 21.0, "zone_2": 21.0, "zone_3": 21.0, "zone_4": 21.0},
            "room_average_temperature_c": 21.0,
        },
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    assert decision is not None
    for z in ("ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"):
        assert decision.zone_states[z].temperature_demand == 0.0
        assert decision.zone_states[z].cooling_demand == 0.0
    for ac in ("AC-1", "AC-2", "AC-3", "AC-4"):
        assert decision.ac_cooling_percent[ac] == 0.0
        assert decision.ac_setpoint_c[ac] == 24.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMAL TEMPERATURE (At Comfort Target)
# ─────────────────────────────────────────────────────────────────────────────
def test_3_normal_temperature():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {
            "zone_temperatures_c": {
                "zone_1": COMFORT_TARGET_C,
                "zone_2": COMFORT_TARGET_C,
                "zone_3": COMFORT_TARGET_C,
                "zone_4": COMFORT_TARGET_C,
            },
            "room_average_temperature_c": COMFORT_TARGET_C,
        },
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    for z in ("ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"):
        assert decision.zone_states[z].temperature_demand == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. HOT ZONE (Elevated Temperature)
# ─────────────────────────────────────────────────────────────────────────────
def test_4_hot_zone():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {
            "zone_temperatures_c": {"zone_1": 26.5, "zone_2": 23.0, "zone_3": 23.0, "zone_4": 23.0},
            "room_average_temperature_c": 23.8,
        },
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    # Zone 1 is hot -> high temperature demand
    assert decision.zone_states["ZONE_1"].cooling_demand > decision.zone_states["ZONE_2"].cooling_demand
    assert decision.zone_states["ZONE_1"].temperature_demand > 0.0
    # AC-1 and AC-4 have high spatial influence on Zone 1 (NW)
    assert decision.ac_cooling_percent["AC-1"] > decision.ac_cooling_percent["AC-3"]


# ─────────────────────────────────────────────────────────────────────────────
# 5. HIGH COMPUTER HEAT
# ─────────────────────────────────────────────────────────────────────────────
def test_5_high_computer_heat():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {
            "zone_temperatures_c": {"zone_1": 23.5, "zone_2": 23.5, "zone_3": 23.5, "zone_4": 23.5},
            "room_average_temperature_c": 23.5,
        },
        "computers": [
            {"id": "COMP-01", "zone": "ZONE_1", "heat_watts": 800.0},
            {"id": "COMP-02", "zone": "ZONE_1", "heat_watts": 800.0},
        ],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    assert decision.zone_states["ZONE_1"].heat_demand > 0.0
    assert decision.zone_states["ZONE_1"].cooling_demand > 0.0
    assert decision.zone_states["ZONE_3"].heat_demand == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. OCCUPANCY HEAT IF AVAILABLE
# ─────────────────────────────────────────────────────────────────────────────
def test_6_occupancy_heat():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {
            "zone_temperatures_c": {"zone_1": 23.5, "zone_2": 23.5, "zone_3": 23.5, "zone_4": 23.5},
            "room_average_temperature_c": 23.5,
        },
        "computers": [],
        "occupancy": {"zones": {"zone_2": 4, "zone_1": 0, "zone_3": 0, "zone_4": 0}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    # Zone 2 has 4 people * 95W = 380W
    assert decision.zone_states["ZONE_2"].heat_load_w >= 380.0
    assert decision.zone_states["ZONE_2"].heat_demand > 0.0
    assert decision.zone_states["ZONE_2"].cooling_demand > decision.zone_states["ZONE_1"].cooling_demand


# ─────────────────────────────────────────────────────────────────────────────
# 7. POSITIVE TEMPERATURE TREND (Rising Temperature)
# ─────────────────────────────────────────────────────────────────────────────
def test_7_positive_temperature_trend():
    controller = ThermalControlAlgorithm()
    # Step 1: initial baseline
    state1 = {
        "thermal": {"zone_temperatures_c": {"zone_1": 23.5, "zone_2": 23.5, "zone_3": 23.5, "zone_4": 23.5}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    controller.compute(state1)

    # Step 2: Zone 1 temperature rises rapidly (+0.2°C in 10s)
    state2 = {
        "thermal": {"zone_temperatures_c": {"zone_1": 23.7, "zone_2": 23.5, "zone_3": 23.5, "zone_4": 23.5}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state2)
    assert decision.zone_states["ZONE_1"].temperature_trend_c_per_min > 0.0
    assert decision.zone_states["ZONE_1"].trend_demand > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. NEGATIVE TEMPERATURE TREND (Falling Temperature)
# ─────────────────────────────────────────────────────────────────────────────
def test_8_negative_temperature_trend():
    controller = ThermalControlAlgorithm()
    # Step 1: initial baseline
    state1 = {
        "thermal": {"zone_temperatures_c": {"zone_1": 24.5, "zone_2": 24.5, "zone_3": 24.5, "zone_4": 24.5}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    controller.compute(state1)

    # Step 2: Zone 1 cools down
    state2 = {
        "thermal": {"zone_temperatures_c": {"zone_1": 24.2, "zone_2": 24.5, "zone_3": 24.5, "zone_4": 24.5}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state2)
    # Negative trend must not add cooling demand
    assert decision.zone_states["ZONE_1"].trend_demand == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 9. LOW-LOAD CONDITION
# ─────────────────────────────────────────────────────────────────────────────
def test_9_low_load_condition():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": 22.0, "zone_2": 22.0, "zone_3": 22.0, "zone_4": 22.0}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    for ac in ("AC-1", "AC-2", "AC-3", "AC-4"):
        assert decision.ac_cooling_percent[ac] == 0.0
        assert decision.ac_setpoint_c[ac] == 24.0


# ─────────────────────────────────────────────────────────────────────────────
# 10. SPATIAL ZONE DEMAND
# ─────────────────────────────────────────────────────────────────────────────
def test_10_spatial_zone_demand():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": 27.0, "zone_2": 22.0, "zone_3": 22.0, "zone_4": 22.0}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    # Zone 1 demand is high, others are zero
    assert decision.zone_states["ZONE_1"].cooling_demand > 50.0
    assert decision.zone_states["ZONE_3"].cooling_demand == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 11. AC MAPPING
# ─────────────────────────────────────────────────────────────────────────────
def test_11_ac_mapping():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": 27.0, "zone_2": 22.0, "zone_3": 22.0, "zone_4": 22.0}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    # AC-1 (North, covers Z1/Z2) and AC-4 (West, covers Z1/Z4) must have much higher cooling than AC-3 (South)
    assert decision.ac_cooling_percent["AC-1"] > decision.ac_cooling_percent["AC-3"]
    assert decision.ac_cooling_percent["AC-4"] > decision.ac_cooling_percent["AC-3"]


# ─────────────────────────────────────────────────────────────────────────────
# 12. ACTUATOR LIMITS
# ─────────────────────────────────────────────────────────────────────────────
def test_12_actuator_limits():
    controller = ThermalControlAlgorithm()
    gov = SafetyGovernor()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": 45.0, "zone_2": 45.0, "zone_3": 45.0, "zone_4": 45.0}},
        "computers": [{"id": f"C{i}", "zone": "ZONE_1", "heat_watts": 5000.0} for i in range(10)],
        "occupancy": {"zones": {"zone_1": 50}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    safe_decision = gov.validate_decision(decision)
    for ac in ("AC-1", "AC-2", "AC-3", "AC-4"):
        assert MIN_COOLING_PCT <= safe_decision.ac_cooling_percent[ac] <= MAX_COOLING_PCT
        assert MIN_SETPOINT_C <= safe_decision.ac_setpoint_c[ac] <= MAX_SETPOINT_C


# ─────────────────────────────────────────────────────────────────────────────
# 13. INVALID VALUES
# ─────────────────────────────────────────────────────────────────────────────
def test_13_invalid_values():
    controller = ThermalControlAlgorithm()
    # Malformed state
    decision = controller.compute({})
    assert decision.is_fallback is True
    assert decision.room_setpoint_c == 24.0


# ─────────────────────────────────────────────────────────────────────────────
# 14. NaN HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def test_14_nan_handling():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": float("nan"), "zone_2": 23.0, "zone_3": 23.0, "zone_4": 23.0}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    assert decision.is_fallback is True
    gov = SafetyGovernor()
    safe_decision = gov.validate_decision(decision)
    assert all(math.isfinite(val) for val in safe_decision.ac_cooling_percent.values())


# ─────────────────────────────────────────────────────────────────────────────
# 15. INFINITY HANDLING
# ─────────────────────────────────────────────────────────────────────────────
def test_15_infinity_handling():
    controller = ThermalControlAlgorithm()
    state = {
        "thermal": {"zone_temperatures_c": {"zone_1": float("inf"), "zone_2": 23.0, "zone_3": 23.0, "zone_4": 23.0}},
        "computers": [],
        "occupancy": {"zones": {}},
        "timestep_seconds": 10.0,
    }
    decision = controller.compute(state)
    assert decision.is_fallback is True


# ─────────────────────────────────────────────────────────────────────────────
# 16. BASELINE FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def test_16_baseline_fallback():
    controller = ThermalControlAlgorithm()
    fallback = controller._fallback_decision("Test trigger")
    assert fallback.is_fallback is True
    assert all(fallback.ac_cooling_percent[ac] == 0.0 for ac in ("AC-1", "AC-2", "AC-3", "AC-4"))
    assert all(fallback.ac_setpoint_c[ac] == 24.0 for ac in ("AC-1", "AC-2", "AC-3", "AC-4"))


# ─────────────────────────────────────────────────────────────────────────────
# 17-21. FIVE SCENARIOS (F1 - F5)
# ─────────────────────────────────────────────────────────────────────────────
def test_17_scenario_f1():
    """F1: Localized Heavy Compute (Zone 1 computers heavy)"""
    mgr = SimulationManager()
    mgr.scenario_id = 1
    mgr.current_step_idx = 10
    payload = mgr.build_update_payload()
    assert payload["scenario_id"] == 1
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"
    assert payload["control_authority"] == "THERMAL CONTROL ALGORITHM"
    assert "acs" in payload["hvac"]
    assert len(payload["hvac"]["acs"]) == 4


def test_18_scenario_f2():
    """F2: Occupancy Concentration"""
    mgr = SimulationManager()
    mgr.scenario_id = 2
    mgr.current_step_idx = 10
    payload = mgr.build_update_payload()
    assert payload["scenario_id"] == 2
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"


def test_19_scenario_f3():
    """F3: Distributed Heavy Compute"""
    mgr = SimulationManager()
    mgr.scenario_id = 3
    mgr.current_step_idx = 10
    payload = mgr.build_update_payload()
    assert payload["scenario_id"] == 3
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"


def test_20_scenario_f4():
    """F4: High Occupancy / Low Computer Load"""
    mgr = SimulationManager()
    mgr.scenario_id = 4
    mgr.current_step_idx = 10
    payload = mgr.build_update_payload()
    assert payload["scenario_id"] == 4
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"


def test_21_scenario_f5():
    """F5: Opposing Thermal Zones"""
    mgr = SimulationManager()
    mgr.scenario_id = 5
    mgr.current_step_idx = 10
    payload = mgr.build_update_payload()
    assert payload["scenario_id"] == 5
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"


# ─────────────────────────────────────────────────────────────────────────────
# 22. OPTIMIZER IS NEVER CALLED AT RUNTIME
# ─────────────────────────────────────────────────────────────────────────────
def test_22_optimizer_is_never_called():
    with patch.object(simulator.optimizer.HvacTargetOptimizer, "optimize_action") as mock_opt:
        mgr = SimulationManager()
        mgr.current_step_idx = 5
        mgr.build_update_payload()
        mock_opt.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 23. OPTIMIZER IS NOT INSTANTIATED IN RUNTIME PATH
# ─────────────────────────────────────────────────────────────────────────────
def test_23_optimizer_is_not_instantiated():
    with patch("simulator.optimizer.HvacTargetOptimizer.__init__", return_value=None) as mock_init:
        mgr = SimulationManager()
        mgr.current_step_idx = 5
        mgr.build_update_payload()
        mock_init.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# 24. PROTOTYPE STARTS WITHOUT ML
# ─────────────────────────────────────────────────────────────────────────────
def test_24_prototype_starts_without_ml():
    mgr = SimulationManager()
    payload = mgr.build_update_payload()
    assert payload["model_status"] == "NOT_INTEGRATED"
    assert payload["control_mode"] == "PROTOTYPE_CONTROL"


# ─────────────────────────────────────────────────────────────────────────────
# 25. RESET
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_25_reset():
    mgr = SimulationManager()
    mgr.sim_time_seconds = 120.0
    mgr.current_step_idx = 12
    await mgr.reset()
    assert mgr.sim_time_seconds == 0.0
    assert mgr.current_step_idx == 0
    assert mgr.status == "READY"


# ─────────────────────────────────────────────────────────────────────────────
# 26. SCENARIO SWITCH
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.anyio
async def test_26_scenario_switch():
    mgr = SimulationManager()
    await mgr.select_scenario(3)
    assert mgr.scenario_id == 3
    assert mgr.current_step_idx == 0
    assert mgr.sim_time_seconds == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 39. CRITICAL OPTIMIZER-ABSENCE TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_39_critical_optimizer_absence_test():
    """
    SECTION 39: Proves the runtime path does not contain or invoke HvacTargetOptimizer.
    The test must fail if HvacTargetOptimizer is instantiated or called.
    """
    call_tracker = {"instantiated": False, "called": False}

    orig_init = simulator.optimizer.HvacTargetOptimizer.__init__
    orig_call = simulator.optimizer.HvacTargetOptimizer.optimize_action

    def tracked_init(self, *args, **kwargs):
        call_tracker["instantiated"] = True
        return orig_init(self, *args, **kwargs)

    def tracked_call(self, *args, **kwargs):
        call_tracker["called"] = True
        return orig_call(self, *args, **kwargs)

    with patch.object(simulator.optimizer.HvacTargetOptimizer, "__init__", side_effect=tracked_init):
        with patch.object(simulator.optimizer.HvacTargetOptimizer, "optimize_action", side_effect=tracked_call):
            mgr = SimulationManager()
            for step in range(10):
                mgr.current_step_idx = step
                mgr.sim_time_seconds = step * 10.0
                mgr.build_update_payload()

    assert not call_tracker["instantiated"], "CRITICAL FAILURE: HvacTargetOptimizer was instantiated during simulation!"
    assert not call_tracker["called"], "CRITICAL FAILURE: HvacTargetOptimizer.optimize_action was called during simulation!"


# ─────────────────────────────────────────────────────────────────────────────
# 40. RUNTIME CONTROL TEST
# ─────────────────────────────────────────────────────────────────────────────
def test_40_runtime_control_pipeline():
    """
    SECTION 40: Proves the runtime control architecture:
        SIMULATION STATE -> ThermalControlAlgorithm -> Safety Governor -> HVAC ACTUATORS
    No other decision-making component may appear in the path.
    """
    mgr = SimulationManager()
    assert hasattr(mgr, "_controller"), "SimulationManager must have _controller"
    assert isinstance(mgr._controller, ThermalControlAlgorithm), "_controller must be ThermalControlAlgorithm"
    assert hasattr(mgr, "_safety_governor"), "SimulationManager must have _safety_governor"
    assert isinstance(mgr._safety_governor, SafetyGovernor), "_safety_governor must be SafetyGovernor"

    # Verify pipeline execution
    with patch.object(mgr._controller, "compute", wraps=mgr._controller.compute) as spy_alg:
        with patch.object(mgr._safety_governor, "validate_decision", wraps=mgr._safety_governor.validate_decision) as spy_gov:
            payload = mgr.build_update_payload()
            assert spy_alg.called, "ThermalControlAlgorithm.compute was not called in runtime path!"
            assert spy_gov.called, "SafetyGovernor.validate_decision was not called in runtime path!"

    # Verify single authority in telemetry
    assert payload["control_authority"] == "THERMAL CONTROL ALGORITHM"
    assert payload["control_path"] == "THERMAL_CONTROL_ALGORITHM → SAFETY_GOVERNOR → HVAC"
    assert payload["safety_status"] in ("ACTIVE", "CLAMPED", "FALLBACK")
    assert payload["model_status"] == "NOT_INTEGRATED"
