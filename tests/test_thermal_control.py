"""
Deterministic Control Validation & Physics Tests for HVEAC Thermal Control Audit.

Covers:
- Section 11: Control Validation Tests (TEST A through TEST F)
- Section 12: Closed-Loop Step Response Test
- Section 13: Cooling Sign Convention & Heat Flux Balance
- Section 14: AI_CONTROL Default Disabled Verification
"""

import numpy as np
import pytest

from simulator.config import HvacConfig, OptimizationConfig, ThermalPhysicsConfig
from simulator.optimizer import HvacTargetOptimizer
from simulator.thermal_model import ThermalModel
from ml.safety_governor import SafetyGovernor, SafetyGovernorConfig
from backend.simulation.manager import SimulationManager


# ==============================================================================
# SECTION 11: CONTROL VALIDATION TESTS (TESTS A - F)
# ==============================================================================

class TestControlSemanticsValidation:
    """Deterministic validation of physical HVAC control law and thermostat semantics."""

    @pytest.fixture
    def optimizer(self):
        return HvacTargetOptimizer()

    def test_test_a_cooling_demand_when_above_target(self, optimizer):
        """
        TEST A:
        Zone = 25.3°C, Target = 22.5°C
        Expected: cooling demand > 0
        """
        temps = {"ZONE_1": 25.3, "ZONE_2": 25.3, "ZONE_3": 25.3, "ZONE_4": 25.3}
        q_dist = {"ZONE_1": 1000.0, "ZONE_2": 1000.0, "ZONE_3": 1000.0, "ZONE_4": 1000.0}

        res = optimizer.optimize_action(temps, q_dist, target_temperature_c=22.5)

        levels = [res.optimal_cooling_ac1, res.optimal_cooling_ac2, res.optimal_cooling_ac3, res.optimal_cooling_ac4]
        assert all(u > 0.0 for u in levels), f"Expected cooling demand > 0, got {levels}"
        assert res.optimal_hvac_action != "ALL_OFF"

    def test_test_b_no_cooling_when_below_target(self, optimizer):
        """
        TEST B:
        Zone = 25.3°C, Target = 26.0°C
        Expected: cooling demand should NOT be produced merely because target is 26°C.
        Cooling demand must be 0 or reduce to 0 since current temperature <= target.
        """
        temps = {"ZONE_1": 25.3, "ZONE_2": 25.3, "ZONE_3": 25.3, "ZONE_4": 25.3}
        q_dist = {"ZONE_1": 1000.0, "ZONE_2": 1000.0, "ZONE_3": 1000.0, "ZONE_4": 1000.0}

        res = optimizer.optimize_action(temps, q_dist, target_temperature_c=26.0)

        levels = [res.optimal_cooling_ac1, res.optimal_cooling_ac2, res.optimal_cooling_ac3, res.optimal_cooling_ac4]
        assert all(u == 0.0 for u in levels), f"Expected cooling demand == 0 when room <= target, got {levels}"
        assert res.optimal_hvac_action == "ALL_OFF"

    def test_test_c_no_unnecessary_cooling(self, optimizer):
        """
        TEST C:
        Zone = 23.0°C, Target = 24.0°C
        Expected: no unnecessary cooling (cooling demand == 0).
        """
        temps = {"ZONE_1": 23.0, "ZONE_2": 23.0, "ZONE_3": 23.0, "ZONE_4": 23.0}
        q_dist = {"ZONE_1": 500.0, "ZONE_2": 500.0, "ZONE_3": 500.0, "ZONE_4": 500.0}

        res = optimizer.optimize_action(temps, q_dist, target_temperature_c=24.0)

        levels = [res.optimal_cooling_ac1, res.optimal_cooling_ac2, res.optimal_cooling_ac3, res.optimal_cooling_ac4]
        assert all(u == 0.0 for u in levels), f"Expected no unnecessary cooling, got {levels}"
        assert res.optimal_hvac_action == "ALL_OFF"

    def test_test_d_cooling_increases_when_hot(self, optimizer):
        """
        TEST D:
        Zone = 27.0°C, Target = 24.0°C
        Expected: cooling demand should increase subject to actuator constraints.
        """
        temps = {"ZONE_1": 27.0, "ZONE_2": 27.0, "ZONE_3": 27.0, "ZONE_4": 27.0}
        q_dist = {"ZONE_1": 1000.0, "ZONE_2": 1000.0, "ZONE_3": 1000.0, "ZONE_4": 1000.0}

        res = optimizer.optimize_action(temps, q_dist, target_temperature_c=24.0)

        levels = [res.optimal_cooling_ac1, res.optimal_cooling_ac2, res.optimal_cooling_ac3, res.optimal_cooling_ac4]
        assert all(u > 0.5 for u in levels), f"Expected high cooling demand when hot (27C > 24C), got {levels}"
        assert all(u <= 1.0 for u in levels), f"Actuator constraint violation: {levels}"

    def test_test_e_monotonic_target_response(self, optimizer):
        """
        TEST E:
        AI prediction changes from 25.5°C → 26.0°C.
        Verify: system does not incorrectly increase cooling merely because the numeric target increased.
        """
        temps = {"ZONE_1": 25.8, "ZONE_2": 25.8, "ZONE_3": 25.8, "ZONE_4": 25.8}
        q_dist = {"ZONE_1": 1000.0, "ZONE_2": 1000.0, "ZONE_3": 1000.0, "ZONE_4": 1000.0}

        res_255 = optimizer.optimize_action(temps, q_dist, target_temperature_c=25.5)
        res_260 = optimizer.optimize_action(temps, q_dist, target_temperature_c=26.0)

        sum_255 = sum([res_255.optimal_cooling_ac1, res_255.optimal_cooling_ac2, res_255.optimal_cooling_ac3, res_255.optimal_cooling_ac4])
        sum_260 = sum([res_260.optimal_cooling_ac1, res_260.optimal_cooling_ac2, res_260.optimal_cooling_ac3, res_260.optimal_cooling_ac4])

        assert sum_260 <= sum_255, (
            f"Cooling must not increase when setpoint increases from 25.5 to 26.0! "
            f"Got {sum_255} -> {sum_260}"
        )

    def test_test_f_comfort_band_metric_consistency(self):
        """
        TEST F:
        Zone = 25.3°C, Comfort band = 21–24°C.
        Expected: this temperature is OUTSIDE the stated band. The metric must reflect this.
        """
        zone_temp = 25.3
        comfort_min = 21.0
        comfort_max = 24.0

        is_compliant = (comfort_min <= zone_temp <= comfort_max)
        assert not is_compliant, f"{zone_temp}°C must be classified as OUTSIDE [{comfort_min}, {comfort_max}]°C"

        # Check compliance percentage calculation
        def calculate_zone_compliance(temps):
            in_band = sum(1 for t in temps if comfort_min <= t <= comfort_max)
            return (in_band / len(temps)) * 100.0

        assert calculate_zone_compliance([25.3]) == 0.0
        assert calculate_zone_compliance([25.3, 24.4, 22.5, 23.0]) == 50.0


# ==============================================================================
# SECTION 12: CLOSED-LOOP STEP TEST
# ==============================================================================

class TestClosedLoopStepResponse:
    """Verifies that closed-loop cooling actuation drives temperature in the physical direction."""

    def test_closed_loop_step_temperature_decrease(self):
        """
        Initial: Zone temperature = 27°C, Target = 24°C
        Run 15 simulation steps.
        Verify: T0 > T_target -> cooling starts, then T1 < T0, T2 < T1, ...
        """
        physics = ThermalPhysicsConfig()
        hvac = HvacConfig()
        weights = np.array(hvac.spatial_influence_weights)
        q_nominal = hvac.nominal_cooling_capacity_w

        tm = ThermalModel(
            physics,
            initial_temperatures={"ZONE_1": 27.0, "ZONE_2": 27.0, "ZONE_3": 27.0, "ZONE_4": 27.0}
        )
        target = 24.0
        assert tm.temperatures["ZONE_1"] > target

        # Heat loads
        comp_heat = {"ZONE_1": 200.0, "ZONE_2": 200.0, "ZONE_3": 200.0, "ZONE_4": 200.0}
        occ_heat = {"ZONE_1": 50.0, "ZONE_2": 50.0, "ZONE_3": 50.0, "ZONE_4": 50.0}
        env_heat = {"ZONE_1": 50.0, "ZONE_2": 50.0, "ZONE_3": 50.0, "ZONE_4": 50.0}

        # Active cooling at 70% modulation
        u = np.array([0.7, 0.7, 0.7, 0.7])
        q_cool_z = (u @ weights) * q_nominal
        hvac_cooling = {f"ZONE_{i+1}": float(q_cool_z[i]) for i in range(4)}

        trajectory = [tm.temperatures["ZONE_1"]]
        for _ in range(15):
            state = tm.step(
                dt_seconds=10.0,
                comp_heat_by_zone=comp_heat,
                occ_heat_by_zone=occ_heat,
                env_heat_by_zone=env_heat,
                hvac_cooling_by_zone=hvac_cooling,
            )
            trajectory.append(state.zone_temperatures["ZONE_1"])

        # Verify strict monotonic decrease during active cooling
        is_strictly_decreasing = all(trajectory[i] > trajectory[i+1] for i in range(len(trajectory)-1))
        assert is_strictly_decreasing, f"Trajectory was not strictly decreasing: {trajectory}"
        assert trajectory[-1] < trajectory[0]


# ==============================================================================
# SECTION 13: COOLING SIGN CONVENTION AUDIT
# ==============================================================================

class TestCoolingSignConvention:
    """Verifies that thermal physics sign convention is strictly obeyed."""

    def test_heat_flux_sign_convention(self):
        """
        Verify:
        Computer heat: +
        Occupancy heat: +
        Envelope heat: +
        HVAC cooling: -
        Net heat = internal/environmental gains - HVAC cooling.
        """
        physics = ThermalPhysicsConfig()
        tm = ThermalModel(
            physics,
            initial_temperatures={"ZONE_1": 24.0, "ZONE_2": 24.0, "ZONE_3": 24.0, "ZONE_4": 24.0}
        )

        comp = {"ZONE_1": 600.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}
        occ = {"ZONE_1": 170.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}
        env = {"ZONE_1": 50.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}
        cool = {"ZONE_1": 500.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}

        state = tm.step(
            dt_seconds=10.0,
            comp_heat_by_zone=comp,
            occ_heat_by_zone=occ,
            env_heat_by_zone=env,
            hvac_cooling_by_zone=cool,
        )

        expected_net = (600.0 + 170.0 + 50.0) - 500.0  # 320.0 W (interzone is 0 since all zones start at 24C)
        assert abs(state.zone_net_heat_w["ZONE_1"] - expected_net) < 1.0, (
            f"Net heat sign violation: expected ~{expected_net}, got {state.zone_net_heat_w['ZONE_1']}"
        )

    def test_higher_cooling_capacity_yields_lower_temperature(self):
        """
        Regression test: Verify that increasing cooling capacity never accidentally increases temperature.
        """
        physics = ThermalPhysicsConfig()
        tm_low = ThermalModel(physics, {"ZONE_1": 25.0, "ZONE_2": 25.0, "ZONE_3": 25.0, "ZONE_4": 25.0})
        tm_high = ThermalModel(physics, {"ZONE_1": 25.0, "ZONE_2": 25.0, "ZONE_3": 25.0, "ZONE_4": 25.0})

        heat = {"ZONE_1": 500.0, "ZONE_2": 500.0, "ZONE_3": 500.0, "ZONE_4": 500.0}
        zero = {"ZONE_1": 0.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}

        s_low = tm_low.step(10.0, heat, zero, zero, {"ZONE_1": 500.0, "ZONE_2": 500.0, "ZONE_3": 500.0, "ZONE_4": 500.0})
        s_high = tm_high.step(10.0, heat, zero, zero, {"ZONE_1": 1500.0, "ZONE_2": 1500.0, "ZONE_3": 1500.0, "ZONE_4": 1500.0})

        t_low = s_low.zone_temperatures["ZONE_1"]
        t_high = s_high.zone_temperatures["ZONE_1"]

        assert t_high < t_low, f"Higher cooling produced higher temp! low={t_low}, high={t_high}"


# ==============================================================================
# SECTION 14: AI CONTROL DISABLED UNTIL VALIDATED
# ==============================================================================

class TestAiControlDisabledDefault:
    """Verifies AI_CONTROL is strictly disabled pending validation."""

    def test_manager_allowed_modes_does_not_contain_ai_control(self):
        mgr = SimulationManager()
        assert "AI_CONTROL" not in mgr.allowed_control_modes, (
            f"AI_CONTROL must not be in allowed_control_modes: {mgr.allowed_control_modes}"
        )
        assert "BASELINE" in mgr.allowed_control_modes
        assert "SHADOW" in mgr.allowed_control_modes

    def test_manager_has_disabled_control_modes(self):
        mgr = SimulationManager()
        assert hasattr(mgr, "disabled_control_modes")
        assert "AI_CONTROL" in mgr.disabled_control_modes
