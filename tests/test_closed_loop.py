"""
Comprehensive Test Suite for HVEAC Brain V1 Closed-Loop Simulation Control.

Covers:
- Section 31: Test Failure Injection (20 failure injection scenarios)
- Section 32: Control Path & Hardware Boundary verification
- Section 33: Closed-Loop Stability & Convergence verification
- Mode transitions, Safety Governor validation, fail-safe fallbacks
"""

import math
import pytest
from pathlib import Path
from typing import Any, Dict

from ml.safety_governor import SafetyGovernor, SafetyGovernorConfig
from ml.ai_controller import AiClosedLoopController, ClosedLoopMetrics
from simulator.config import SimulationEngineConfig
from simulator.engine import ScenarioEngine
from simulator.optimizer import HvacTargetOptimizer
from simulator.scenarios import ALL_SCENARIOS, Scenario1LocalizedCompute


# ==============================================================================
# 1. SAFETY GOVERNOR TESTS (Sections 6, 7, 8, 9)
# ==============================================================================

def test_governor_valid_prediction():
    """Verify governor accepts valid setpoints within trained domain."""
    gov = SafetyGovernor()
    res = gov.validate_command(
        ai_requested_setpoint_c=25.5,
        baseline_setpoint_c=22.5,
        timestamp_sim_s=0.0,
        control_mode="AI_CONTROL",
        model_available=True,
    )
    assert res.status == "APPLIED"
    assert res.safe_setpoint_c == 25.5
    assert res.applied_setpoint_c == 25.5
    assert res.control_path == "SIMULATOR_ONLY"


def test_governor_bounds_enforcement():
    """Verify setpoints outside [24.5, 26.5]°C are clamped or rejected."""
    gov = SafetyGovernor()

    # Above max
    res_high = gov.validate_command(
        ai_requested_setpoint_c=28.0,
        baseline_setpoint_c=22.5,
        timestamp_sim_s=0.0,
        control_mode="AI_CONTROL",
        model_available=True,
    )
    assert res_high.safe_setpoint_c == 26.5
    assert res_high.status in ("APPLIED", "RATE_LIMITED")

    # Below min
    gov_low = SafetyGovernor()
    res_low = gov_low.validate_command(
        ai_requested_setpoint_c=20.0,
        baseline_setpoint_c=22.5,
        timestamp_sim_s=0.0,
        control_mode="AI_CONTROL",
        model_available=True,
    )
    assert res_low.safe_setpoint_c == 24.5


def test_governor_rate_limit_enforcement():
    """Verify setpoint step changes cannot exceed max_step_change_c (0.50°C)."""
    gov = SafetyGovernor()

    # Step 1: Initial setpoint at 24.5°C
    gov.validate_command(24.5, 22.5, 0.0, "AI_CONTROL", True)

    # Step 2: Request jump to 26.0°C (jump of 1.5°C > 0.50°C) after dwell expired
    res = gov.validate_command(26.0, 22.5, 120.0, "AI_CONTROL", True)
    assert res.status == "RATE_LIMITED"
    assert res.safe_setpoint_c == 25.0  # 24.5 + 0.5
    assert "0.5" in res.reason


def test_governor_dwell_time_enforcement():
    """Verify setpoint cannot change before min_dwell_seconds (60s sim time)."""
    gov = SafetyGovernor()

    # Step 1: at t=0s set to 25.0°C
    gov.validate_command(25.0, 22.5, 0.0, "AI_CONTROL", True)

    # Step 2: at t=30s (<60s dwell), request 25.5°C
    res = gov.validate_command(25.5, 22.5, 30.0, "AI_CONTROL", True)
    assert res.status == "HYSTERESIS_HOLD"
    assert res.safe_setpoint_c == 25.0  # Kept previous target
    assert "dwell" in res.reason.lower()


def test_governor_hysteresis_deadband():
    """Verify small perturbations within deadband (0.25°C) hold current target."""
    gov = SafetyGovernor()

    # Step 1: set to 25.5°C at t=0s
    gov.validate_command(25.5, 22.5, 0.0, "AI_CONTROL", True)

    # Step 2: at t=100s, request 25.6°C (delta 0.10°C < 0.25°C deadband)
    res = gov.validate_command(25.6, 22.5, 100.0, "AI_CONTROL", True)
    assert res.status == "HYSTERESIS_HOLD"
    assert res.safe_setpoint_c == 25.5
    assert "hysteresis deadband" in res.reason


def test_governor_nan_and_inf_handling():
    """Verify NaN and Inf predictions are rejected and fall back safely."""
    gov = SafetyGovernor()

    # NaN
    res_nan = gov.validate_command(float("nan"), 22.5, 0.0, "AI_CONTROL", True)
    assert res_nan.status == "FALLBACK_BASELINE"
    assert res_nan.safe_setpoint_c == 22.5

    # Inf
    res_inf = gov.validate_command(float("inf"), 23.0, 10.0, "AI_CONTROL", True)
    assert res_inf.status == "FALLBACK_BASELINE"
    assert res_inf.safe_setpoint_c == 23.0


def test_governor_model_unavailable_fallback():
    """Verify governor falls back to baseline if model is unavailable."""
    gov = SafetyGovernor()
    res = gov.validate_command(None, 22.5, 0.0, "AI_CONTROL", model_available=False)
    assert res.status in ("FALLBACK_BASELINE", "UNAVAILABLE")
    assert res.safe_setpoint_c == 22.5
    assert "unavailable" in res.reason.lower()


# ==============================================================================
# 2. MODE TRANSITIONS & AUTHORITY TESTS (Sections 3, 4, 11, 30)
# ==============================================================================

def test_control_mode_baseline_authoritative():
    """In BASELINE mode, governor returns baseline and does not apply AI target."""
    gov = SafetyGovernor()
    res = gov.validate_command(26.0, 22.5, 0.0, control_mode="BASELINE", model_available=True)
    assert res.status == "APPLIED"
    assert res.applied_setpoint_c == 22.5  # Baseline setpoint is applied


def test_control_mode_shadow_observational():
    """In SHADOW mode, governor marks prediction observational and applies baseline."""
    gov = SafetyGovernor()
    res = gov.validate_command(26.0, 22.5, 0.0, control_mode="SHADOW", model_available=True)
    assert res.status in ("APPLIED", "SHADOW_OBSERVATION")
    assert res.applied_setpoint_c == 22.5  # Baseline applied
    assert "observational" in res.reason.lower() or "shadow" in res.reason.lower()


def test_mode_transitions_deterministic():
    """Verify transitions between BASELINE -> SHADOW -> AI_CONTROL -> BASELINE."""
    ctrl = AiClosedLoopController(control_mode="BASELINE")
    assert ctrl.control_mode == "BASELINE"

    ctrl.set_control_mode("SHADOW")
    assert ctrl.control_mode == "SHADOW"

    ctrl.set_control_mode("AI_CONTROL")
    assert ctrl.control_mode == "AI_CONTROL"

    ctrl.set_control_mode("BASELINE")
    assert ctrl.control_mode == "BASELINE"

    with pytest.raises(ValueError):
        ctrl.set_control_mode("INVALID_MODE")


def test_reset_clears_controller_and_governor_state():
    """Verify reset clears dwell timers, history, and metrics."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")
    ctrl.safety_governor.last_applied_setpoint_c = 25.5
    ctrl.safety_governor.last_transition_time_s = 500.0

    ctrl.reset(initial_setpoint=None)
    assert ctrl.safety_governor.last_applied_setpoint_c is None
    assert ctrl.safety_governor.last_transition_time_s < 0.0
    assert len(ctrl.events) == 0


# ==============================================================================
# 3. CONTROLLER & ADAPTER FAILURE INJECTION (Section 31)
# ==============================================================================

def test_controller_missing_features_fallback():
    """Verify incomplete snapshot falls back gracefully without crashing."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")
    bad_snapshot = {"thermal": {}}  # Missing all computers, environment, etc.

    cmd = ctrl.step(
        current_state_snapshot=bad_snapshot,
        baseline_setpoint_c=22.5,
        sim_time_s=10.0,
    )
    assert cmd.status in ("FALLBACK_BASELINE", "APPLIED")
    assert cmd.applied_setpoint_c is not None


def test_controller_governor_failure_failsafe():
    """Verify that if the governor encounters an unexpected error, controller fails safe."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")

    # Sabotage governor validate_command to raise an exception
    def broken_validate(*args, **kwargs):
        raise RuntimeError("Injected Governor Failure")

    ctrl.safety_governor.validate_command = broken_validate

    # Valid step snapshot
    snapshot = {
        "scenario_id": 1,
        "thermal": {"room_average_temperature_c": 23.0},
        "occupancy": {"total": 5},
    }
    cmd = ctrl.step(snapshot, baseline_setpoint_c=22.5, sim_time_s=10.0)

    assert cmd.status == "FALLBACK_BASELINE"
    assert cmd.applied_setpoint_c == 22.5
    assert "Injected Governor Failure" in cmd.reason


# ==============================================================================
# 4. ARCHITECTURAL BOUNDARY & NO HARDWARE CONTROL (Section 11, 32)
# ==============================================================================

def test_no_real_hardware_interfaces():
    """
    CRITICAL SAFETY AUDIT:
    Verify no real-world physical HVAC, BACnet, Modbus, GPIO, or serial libraries
    are imported or referenced anywhere in ml/ or simulator/.
    """
    import inspect
    import ml.safety_governor
    import ml.ai_controller
    import simulator.engine
    import simulator.optimizer

    forbidden_terms = [
        "bacnet", "modbus", "pyserial", "serial.Serial", "RPi.GPIO",
        "gpiozero", "real_hvac", "physical_actuator", "building_automation_system"
    ]

    for module in (ml.safety_governor, ml.ai_controller, simulator.engine, simulator.optimizer):
        source = inspect.getsource(module).lower()
        for term in forbidden_terms:
            assert term not in source, f"Forbidden hardware interface '{term}' detected in {module.__name__}!"


def test_ai_control_path_label():
    """Verify control_path always explicitly reports SIMULATOR_ONLY."""
    gov = SafetyGovernor()
    res = gov.validate_command(25.5, 22.5, 0.0, "AI_CONTROL", True)
    assert res.control_path == "SIMULATOR_ONLY"


# ==============================================================================
# 5. SIMULATOR HVAC INTEGRATION & PHYSICS PRESERVATION (Sections 2, 12)
# ==============================================================================

def test_hvac_optimizer_respects_target_setpoint():
    """Verify existing HvacTargetOptimizer accepts room setpoint target and optimizes perimeter ACs."""
    optimizer = HvacTargetOptimizer()
    cur_temps = {"zone_1": 23.0, "zone_2": 23.0, "zone_3": 23.0, "zone_4": 23.0}
    dist_heat = {"zone_1": 400.0, "zone_2": 400.0, "zone_3": 400.0, "zone_4": 400.0}

    # Baseline action
    base_action = optimizer.optimize_action(cur_temps, dist_heat, dt_seconds=60.0, target_temperature_c=22.5)
    assert base_action.optimal_temperature_c == 22.5
    assert 0.0 <= base_action.optimal_cooling_ac1 <= 1.0

    # AI target action
    ai_action = optimizer.optimize_action(cur_temps, dist_heat, dt_seconds=60.0, target_temperature_c=25.5)
    assert ai_action.optimal_temperature_c == 25.5
    assert 0.0 <= ai_action.optimal_cooling_ac1 <= 1.0


def test_applied_target_equals_validated_target():
    """Verify the setpoint applied to the simulator equals the validated governor setpoint."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")
    snapshot = {"scenario_id": 1, "thermal": {"room_average_temperature_c": 23.0}}
    cmd = ctrl.step(snapshot, baseline_setpoint_c=22.5, sim_time_s=10.0)
    assert cmd.applied_setpoint_c == cmd.safe_setpoint_c


# ==============================================================================
# 6. CLOSED-LOOP STABILITY TEST (Section 33)
# ==============================================================================

def test_closed_loop_short_simulation_stability():
    """
    Run 30 simulation steps (300 seconds) in AI_CONTROL mode.
    Verify temperature remains strictly bounded, no runaway heating/cooling,
    and no numerical NaN or crashes.
    """
    config = SimulationEngineConfig()
    engine = ScenarioEngine(config)
    scen = Scenario1LocalizedCompute(duration_seconds=300.0, timestep_seconds=10.0)

    records = engine.run_scenario(scen, seed=42, control_mode="AI_CONTROL")
    assert len(records) == 30

    for r in records:
        temp = r["room_average_temperature_c"]
        assert not math.isnan(temp), "Room temperature became NaN!"
        assert 15.0 < temp < 40.0, f"Thermal runaway detected: {temp}°C!"
        assert r["safety_status"] in ("APPLIED", "RATE_LIMITED", "HYSTERESIS_HOLD", "FALLBACK_BASELINE")
        assert r["ai_applied_setpoint_c"] == r["ai_safe_setpoint_c"]


def test_reproducibility_same_seed():
    """Verify running the same scenario with the same seed produces bit-exact reproducibility."""
    config = SimulationEngineConfig()
    engine = ScenarioEngine(config)
    scen = Scenario1LocalizedCompute(duration_seconds=120.0, timestep_seconds=10.0)

    run1 = engine.run_scenario(scen, seed=42, control_mode="AI_CONTROL")
    run2 = engine.run_scenario(scen, seed=42, control_mode="AI_CONTROL")

    for r1, r2 in zip(run1, run2):
        assert r1["room_average_temperature_c"] == r2["room_average_temperature_c"]
        assert r1["ai_applied_setpoint_c"] == r2["ai_applied_setpoint_c"]


def test_invalid_feature_values_sanitization():
    """Verify invalid or extreme feature inputs do not crash inference or control."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")
    extreme_snapshot = {
        "scenario_id": 1,
        "thermal": {
            "room_average_temperature_c": 999.0,  # Extreme temp
            "zone_temperatures_c": {"zone_1": -100.0, "zone_2": 500.0, "zone_3": 22.0, "zone_4": 22.0}
        },
        "computers": [{"id": 1, "cpu_util_percent": -50.0, "gpu_util_percent": 250.0}],
    }
    cmd = ctrl.step(extreme_snapshot, baseline_setpoint_c=22.5, sim_time_s=10.0)
    assert cmd.applied_setpoint_c is not None
    assert 20.0 <= cmd.applied_setpoint_c <= 30.0


def test_scenario_switching_cleans_controller():
    """Verify switching scenario resets controller state."""
    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")
    ctrl.safety_governor.last_applied_setpoint_c = 26.5
    ctrl.safety_governor.last_transition_time_s = 150.0

    # Switching to new scenario triggers reset
    ctrl.reset(initial_setpoint=22.5)
    assert ctrl.safety_governor.last_transition_time_s < 0.0
    assert len(ctrl.events) == 0


def test_simulation_continues_after_injected_ai_crash():
    """
    Verify the simulation continues stepping smoothly even if the AI model
    throws an unexpected runtime exception mid-run.
    """
    config = SimulationEngineConfig()
    engine = ScenarioEngine(config)
    scen = Scenario1LocalizedCompute(duration_seconds=100.0, timestep_seconds=10.0)

    ctrl = AiClosedLoopController(control_mode="AI_CONTROL")

    # Hook into adapter to inject a failure after step 3
    step_count = 0
    orig_adapt = ctrl.adapter.adapt

    def failing_adapt(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        if step_count > 3:
            raise RuntimeError("Injected ML Inference Failure")
        return orig_adapt(*args, **kwargs)

    ctrl.adapter.adapt = failing_adapt

    records = engine.run_scenario(scen, seed=42, control_mode="AI_CONTROL", ai_controller=ctrl)
    assert len(records) == 10
    # Steps after 3 fell back safely to baseline
    for r in records[4:]:
        assert r["safety_status"] in ("FALLBACK_BASELINE", "UNAVAILABLE")
        assert r["room_average_temperature_c"] > 15.0


def test_all_five_scenario_families_dataset_verified():
    """
    Verify that all 5 scenario family runs exist in simulation_dataset/
    and have valid records and thermal metrics.
    """
    import json
    dataset_dir = Path("simulation_dataset")
    for sid in (1, 2, 3, 4, 5):
        ai_file = dataset_dir / f"scenario_{sid}_ai.jsonl"
        assert ai_file.exists(), f"Missing AI dataset file {ai_file}!"
        with open(ai_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) == 720, f"Scenario {sid} AI dataset has {len(lines)} records, expected 720!"
        # Check thermodynamic keys
        for rec in lines[:10]:
            assert "room_average_temperature_c" in rec
            assert "optimal_temperature_c" in rec
            assert "total_hvac_cooling_w" in rec
            assert not math.isnan(rec["room_average_temperature_c"])


def test_closed_loop_reports_generated():
    """Verify all 4 required report artifacts exist in ml/reports/."""
    reports_dir = Path("ml/reports")
    assert (reports_dir / "closed_loop_report.md").exists()
    assert (reports_dir / "closed_loop_metrics.json").exists()
    assert (reports_dir / "scenario_comparison.csv").exists()
    assert (reports_dir / "control_events.csv").exists()

