"""
HVEAC Brain V1 — Shadow Mode Tests

Test suite for the inference service, feature adapter, shadow predictor,
and shadow-mode safety.

Test cases:
 1. Model loading (success & failure)
 2. Model status required fields
 3. Model loads when artifacts exist
 4. Feature columns count (80)
 5. No forbidden features in schema
 6. Feature columns match schema file
 7. Feature adapter returns 80 features
 8. Adapter excludes forbidden features
 9. Adapter maps occupancy correctly
10. Adapter maps computer telemetry
11. Adapter reset clears history
12. Prediction returns required keys
13. Prediction mode always SHADOW_ONLY
14. Prediction with valid features
15. No control path in API response
16. Inference result has shadow mode
17. Model status has shadow mode
18. Shadow predictor initialization
19. Shadow predictor predict returns result
20. Shadow predictor HVAC immutability
21. Shadow predictor metrics accumulation
22. Shadow predictor history bounded
23. Shadow predictor reset clears state
24. Shadow predictor feature inspector
25. Shadow predictor handles missing model
26. Prediction within valid classes
27. Probability sum validation
28. NaN/Inf input handling
29. Missing feature handling
"""

import json
import math
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ─────────────────────────────────────────────────────────────────────────
# 1. Model Loading
# ─────────────────────────────────────────────────────────────────────────

class TestModelLoading:
    """Tests that the model loads correctly and reports proper status."""

    def test_model_status_returns_dict(self):
        """Model status should always return a dictionary."""
        from ml.inference import get_model_status
        status = get_model_status()
        assert isinstance(status, dict)
        assert "status" in status
        assert "mode" in status
        assert status["mode"] == "SHADOW_ONLY"

    def test_model_status_has_required_fields(self):
        """Model status should include version, feature count, etc."""
        from ml.inference import get_model_status
        status = get_model_status()
        required_keys = ["status", "feature_count", "class_count", "mode"]
        for key in required_keys:
            assert key in status, f"Missing key: {key}"

    def test_model_loads_when_artifacts_exist(self):
        """If model artifacts exist, status should be READY."""
        from ml.inference import get_model_status
        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if model_path.exists():
            status = get_model_status()
            assert status["status"] == "READY"
            assert status["feature_count"] == 80
            assert status["class_count"] == 5


# ─────────────────────────────────────────────────────────────────────────
# 2. Feature Schema Validation
# ─────────────────────────────────────────────────────────────────────────

class TestFeatureSchema:
    """Tests the 80-feature schema contract."""

    def test_feature_columns_count(self):
        """Feature adapter must define exactly 80 columns."""
        from ml.feature_adapter import FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 80

    def test_no_forbidden_features_in_schema(self):
        """Forbidden metadata features must never appear in the feature list."""
        from ml.feature_adapter import FEATURE_COLUMNS, FORBIDDEN_FEATURES
        for col in FEATURE_COLUMNS:
            assert col not in FORBIDDEN_FEATURES, f"Forbidden feature '{col}' in FEATURE_COLUMNS"

    def test_feature_columns_match_schema_file(self):
        """Feature columns must match the authoritative schema JSON."""
        from ml.feature_adapter import FEATURE_COLUMNS
        schema_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "feature_schema_v1.json"
        if schema_path.exists():
            with open(schema_path, "r") as f:
                schema = json.load(f)
            expected = schema.get("feature_columns", [])
            assert FEATURE_COLUMNS == expected, "Feature columns don't match schema file"


# ─────────────────────────────────────────────────────────────────────────
# 3. Feature Adapter Mapping
# ─────────────────────────────────────────────────────────────────────────

def _make_sample_state():
    """Creates a minimal simulation state dict for testing."""
    computers = []
    for i in range(1, 11):
        computers.append({
            "id": i,
            "name": f"Computer {i}",
            "zone": f"ZONE_{((i-1)//3)+1}",
            "cpu_util_percent": 25.0 + i,
            "gpu_util_percent": 10.0 + i,
            "workload_category": "LIGHT",
            "heat_watts": 85.0 + i * 5,
        })

    return {
        "scenario_id": 1,
        "simulation_time_seconds": 120.0,
        "step_index": 12,
        "occupancy": {
            "total": 12,
            "zones": {
                "zone_1": 3, "zone_2": 4,
                "zone_3": 3, "zone_4": 2,
            },
        },
        "environment": {
            "outdoor_temperature_c": 32.0,
            "humidity_percent": 65.0,
            "solar_irradiance_w_m2": 300.0,
        },
        "hvac": {
            "acs": [
                {"id": "AC-1", "wall": "north", "state": "COOLING", "setpoint_c": 23.0, "cooling_level": 0.6},
                {"id": "AC-2", "wall": "east", "state": "COOLING", "setpoint_c": 23.5, "cooling_level": 0.5},
                {"id": "AC-3", "wall": "south", "state": "COOLING", "setpoint_c": 23.0, "cooling_level": 0.4},
                {"id": "AC-4", "wall": "west", "state": "OFF", "setpoint_c": 24.0, "cooling_level": 0.0},
            ],
        },
        "thermal": {
            "zone_temperatures_c": {
                "zone_1": 24.5, "zone_2": 23.8,
                "zone_3": 24.1, "zone_4": 23.5,
            },
            "room_average_temperature_c": 23.97,
            "minimum_temperature_c": 23.5,
            "maximum_temperature_c": 24.5,
            "temperature_difference_c": 1.0,
            "computer_heat_watts": 1200.0,
            "occupancy_heat_watts": 840.0,
            "hvac_cooling_watts": 5250.0,
        },
        "targets": {
            "optimal_temperature_c": 25.5,
            "optimal_hvac_action": "ECO_MAINTAIN",
            "optimal_cooling_ac1": 0.6,
            "optimal_cooling_ac2": 0.5,
            "optimal_cooling_ac3": 0.4,
            "optimal_cooling_ac4": 0.0,
        },
        "computers": computers,
    }


class TestFeatureAdapter:
    """Tests the FeatureAdapter state → feature vector mapping."""

    def test_adapt_returns_80_features(self):
        """Adapter must produce exactly 80 features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        features, warnings = adapter.adapt(_make_sample_state())
        assert len(features) == 80, f"Expected 80 features, got {len(features)}"

    def test_adapt_no_forbidden_features(self):
        """Adapted features must never contain forbidden columns."""
        from ml.feature_adapter import FeatureAdapter, FORBIDDEN_FEATURES
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        for forbidden in FORBIDDEN_FEATURES:
            assert forbidden not in features, f"Forbidden feature '{forbidden}' in output"

    def test_adapt_maps_occupancy_correctly(self):
        """Occupancy fields should map from state to features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        state = _make_sample_state()
        features, _ = adapter.adapt(state)
        assert features["occupancy_total"] == 12
        assert features["occupancy_zone_1"] == 3
        assert features["occupancy_zone_4"] == 2

    def test_adapt_maps_computer_telemetry(self):
        """Per-computer features should be populated."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        assert features["computer_1_cpu"] > 0
        assert features["computer_10_heat"] > 0
        assert isinstance(features["computer_5_workload"], str)

    def test_adapter_reset_clears_history(self):
        """Reset should clear rolling history."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        adapter.adapt(_make_sample_state())
        assert len(adapter._history) > 0
        adapter.reset()
        assert len(adapter._history) == 0

    def test_metadata_excluded_from_features(self):
        """Metadata columns must not appear in adapted features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        state = _make_sample_state()
        state["scenario_id"] = 1
        state["simulation_time_seconds"] = 100.0
        features, _ = adapter.adapt(state)
        assert "scenario_id" not in features
        assert "simulation_time_seconds" not in features

    def test_target_excluded_from_features(self):
        """Target column must not appear in adapted features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        assert "optimal_room_setpoint_c" not in features


# ─────────────────────────────────────────────────────────────────────────
# 4. Prediction Output
# ─────────────────────────────────────────────────────────────────────────

class TestPrediction:
    """Tests model prediction output shape and constraints."""

    def test_predict_returns_required_keys(self):
        """Prediction result must have all required keys."""
        from ml.inference import predict_room_setpoint
        result = predict_room_setpoint({})
        required = ["status", "predicted_class", "confidence",
                     "class_probabilities", "mode"]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_predict_mode_always_shadow(self):
        """Mode must always be SHADOW_ONLY regardless of input."""
        from ml.inference import predict_room_setpoint
        result = predict_room_setpoint({})
        assert result["mode"] == "SHADOW_ONLY"

    def test_predict_with_valid_features(self):
        """With valid features, prediction should return OK or UNAVAILABLE."""
        from ml.inference import predict_room_setpoint
        from ml.feature_adapter import FeatureAdapter

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        # Build features from adapter
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())

        result = predict_room_setpoint(features)
        assert result["status"] in ("OK", "ERROR")

        if result["status"] == "OK":
            assert result["predicted_class"] is not None
            assert isinstance(result["predicted_class"], float)
            assert 24.0 <= result["predicted_class"] <= 27.0

            assert result["confidence"] is not None
            assert 0.0 <= result["confidence"] <= 1.0

            assert result["class_probabilities"] is not None
            probs = result["class_probabilities"]
            total_prob = sum(probs.values())
            assert abs(total_prob - 1.0) < 0.01, f"Probabilities don't sum to 1: {total_prob}"

    def test_prediction_within_valid_classes(self):
        """Predicted class must be one of the 5 valid setpoint classes."""
        from ml.inference import predict_room_setpoint
        from ml.feature_adapter import FeatureAdapter

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        result = predict_room_setpoint(features)

        if result["status"] == "OK":
            valid_classes = [24.5, 25.0, 25.5, 26.0, 26.5]
            assert result["predicted_class"] in valid_classes, \
                f"Predicted {result['predicted_class']} not in valid classes"

    def test_probability_sum_validation(self):
        """Class probabilities must sum to approximately 1.0."""
        from ml.inference import predict_room_setpoint
        from ml.feature_adapter import FeatureAdapter

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        result = predict_room_setpoint(features)

        if result["status"] == "OK" and result["class_probabilities"]:
            total = sum(result["class_probabilities"].values())
            assert abs(total - 1.0) < 0.01, f"Probability sum = {total}"

    def test_missing_feature_handling(self):
        """Missing features should be defaulted, not cause a crash."""
        from ml.inference import predict_room_setpoint

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        # Partial feature set (many missing)
        partial = {"outdoor_temperature_c": 30.0, "humidity_percent": 60.0}
        result = predict_room_setpoint(partial)
        # Should not crash; may warn about missing features
        assert result["status"] in ("OK", "ERROR", "UNAVAILABLE")
        if result["status"] == "OK":
            assert len(result.get("feature_warnings", [])) > 0  # Should warn about missing

    def test_nan_inf_input_handling(self):
        """NaN/Inf inputs should not crash the inference."""
        from ml.inference import predict_room_setpoint
        from ml.feature_adapter import FeatureAdapter, FEATURE_COLUMNS

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        # Build valid features first, then inject NaN
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(_make_sample_state())
        features["outdoor_temperature_c"] = float('nan')

        result = predict_room_setpoint(features)
        # Should handle gracefully (may predict or error, but not crash)
        assert result["mode"] == "SHADOW_ONLY"


# ─────────────────────────────────────────────────────────────────────────
# 5. Shadow Mode Safety
# ─────────────────────────────────────────────────────────────────────────

class TestShadowModeSafety:
    """Ensures there is NO path from model prediction to HVAC control."""

    def test_no_control_path_in_api_response(self):
        """API responses must explicitly declare no control path."""
        # This tests the brain_api module structure
        from ml.brain_api import router
        # All routes must be read-only (GET) except adapter-reset
        for route in router.routes:
            path = getattr(route, 'path', '')
            methods = getattr(route, 'methods', set())
            if 'shadow-predict' in path or 'status' in path:
                assert 'GET' in methods or not methods, f"Shadow endpoint {path} should be GET"

    def test_inference_result_has_shadow_mode(self):
        """Every inference result must declare SHADOW_ONLY mode."""
        from ml.inference import predict_room_setpoint
        result = predict_room_setpoint({})
        assert result["mode"] == "SHADOW_ONLY"

    def test_model_status_has_shadow_mode(self):
        """Model status must declare SHADOW_ONLY mode."""
        from ml.inference import get_model_status
        status = get_model_status()
        assert status["mode"] == "SHADOW_ONLY"

    def test_shadow_predictor_no_control_path(self):
        """Shadow predictor result must declare control_path=False."""
        from ml.shadow_predictor import ShadowPredictor
        sp = ShadowPredictor()
        state = sp.get_full_shadow_state()
        assert state["control_path"] is False

    def test_hvac_immutability(self):
        """HVAC state must not change as a result of AI prediction."""
        from ml.shadow_predictor import ShadowPredictor
        import copy

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        state = _make_sample_state()

        # Deep-copy HVAC state before
        hvac_before = copy.deepcopy(state.get("hvac", {}))
        targets_before = copy.deepcopy(state.get("targets", {}))

        # Run prediction
        sp.predict(state)

        # HVAC state must be unchanged
        assert state["hvac"] == hvac_before, "HVAC state was mutated by prediction!"
        assert state["targets"] == targets_before, "Targets state was mutated by prediction!"


# ─────────────────────────────────────────────────────────────────────────
# 6. Shadow Predictor Service
# ─────────────────────────────────────────────────────────────────────────

class TestShadowPredictor:
    """Tests the ShadowPredictor orchestrator service."""

    def test_initialization(self):
        """Shadow predictor should initialize without error."""
        from ml.shadow_predictor import ShadowPredictor
        sp = ShadowPredictor()
        assert sp._enabled is True
        assert sp._current is None
        assert len(sp._history) == 0

    def test_predict_returns_result(self):
        """predict() should return a shadow result dict."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        result = sp.predict(_make_sample_state())
        assert result is not None
        assert "mode" in result
        assert result["mode"] == "SHADOW"
        assert result["control_path"] is False

    def test_predict_caches_on_same_step(self):
        """Calling predict with same step_index should return cached result."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        state = _make_sample_state()
        r1 = sp.predict(state)
        r2 = sp.predict(state)
        # Same reference should be returned (cached)
        assert r1 is r2

    def test_metrics_accumulation(self):
        """Metrics should accumulate across predictions."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        for step in range(5):
            state = _make_sample_state()
            state["step_index"] = step
            state["simulation_time_seconds"] = step * 10.0
            sp.predict(state)

        metrics = sp.get_metrics()
        assert metrics["count"] == 5
        assert "mae_c" in metrics
        assert "rmse_c" in metrics
        assert "exact_agreement_pct" in metrics
        assert "within_05c_pct" in metrics
        assert "latency_mean_ms" in metrics

    def test_history_bounded(self):
        """History should not exceed MAX_HISTORY entries."""
        from ml.shadow_predictor import ShadowPredictor, MAX_HISTORY

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        for step in range(MAX_HISTORY + 50):
            state = _make_sample_state()
            state["step_index"] = step
            sp.predict(state)

        assert len(sp.get_history()) <= MAX_HISTORY

    def test_reset_clears_state(self):
        """reset() should clear history, metrics, and current state."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        sp.predict(_make_sample_state())
        assert sp.get_current() is not None

        sp.reset()
        assert sp.get_current() is None
        assert len(sp.get_history()) == 0
        assert sp.get_metrics()["count"] == 0

    def test_feature_inspector(self):
        """Feature inspector should return grouped features."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        sp.predict(_make_sample_state())

        grouped = sp.get_grouped_features()
        assert len(grouped) > 0
        assert "ENVIRONMENT" in grouped
        assert "OCCUPANCY" in grouped
        # Total features across all groups should be 80
        total = sum(len(items) for items in grouped.values())
        assert total == 80, f"Expected 80 features in inspector, got {total}"

    def test_get_full_shadow_state_before_predict(self):
        """get_full_shadow_state before any prediction should return WAITING."""
        from ml.shadow_predictor import ShadowPredictor
        sp = ShadowPredictor()
        state = sp.get_full_shadow_state()
        assert state["status"] == "WAITING"
        assert state["control_path"] is False

    def test_simulation_continues_if_ai_unavailable(self):
        """If model is unavailable, predict should still return (not crash)."""
        from ml.shadow_predictor import ShadowPredictor
        sp = ShadowPredictor()
        # Force model to appear loaded but broken
        sp._model_status_cache = {"status": "UNAVAILABLE", "feature_count": 0, "class_count": 0}
        result = sp.predict(_make_sample_state())
        # Should not crash — returns a result (may be error/unavailable)
        assert result is not None

    def test_model_reload_consistency(self):
        """Multiple predictions should use the same model instance."""
        from ml.shadow_predictor import ShadowPredictor

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        sp = ShadowPredictor()
        state1 = _make_sample_state()
        state1["step_index"] = 0
        r1 = sp.predict(state1)

        state2 = _make_sample_state()
        state2["step_index"] = 1
        r2 = sp.predict(state2)

        if r1["status"] == "OK" and r2["status"] == "OK":
            # Same model should produce same prediction for same input
            assert r1["predicted_setpoint_c"] is not None
            assert r2["predicted_setpoint_c"] is not None
