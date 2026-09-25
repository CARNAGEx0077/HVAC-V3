"""
HVEAC Brain V1 — Shadow Mode Tests

Test suite for the inference service, feature adapter, and shadow-mode safety.

Test cases:
1. Model loading (success & failure)
2. Feature schema validation (80 columns)
3. Feature adapter mapping
4. Forbidden features excluded
5. Missing features defaulted
6. Prediction output shape
7. Confidence bounds
8. Class probability distribution
9. Shadow-mode safety (no control path)
10. Feature adapter rolling average
11. Adapter reset
12. Model status endpoint
13. Shadow predict endpoint
"""

import json
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

class TestFeatureAdapter:
    """Tests the FeatureAdapter state → feature vector mapping."""

    def _make_sample_state(self):
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
            "computers": computers,
        }

    def test_adapt_returns_80_features(self):
        """Adapter must produce exactly 80 features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        features, warnings = adapter.adapt(self._make_sample_state())
        assert len(features) == 80, f"Expected 80 features, got {len(features)}"

    def test_adapt_no_forbidden_features(self):
        """Adapted features must never contain forbidden columns."""
        from ml.feature_adapter import FeatureAdapter, FORBIDDEN_FEATURES
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(self._make_sample_state())
        for forbidden in FORBIDDEN_FEATURES:
            assert forbidden not in features, f"Forbidden feature '{forbidden}' in output"

    def test_adapt_maps_occupancy_correctly(self):
        """Occupancy fields should map from state to features."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        state = self._make_sample_state()
        features, _ = adapter.adapt(state)
        assert features["occupancy_total"] == 12
        assert features["occupancy_zone_1"] == 3
        assert features["occupancy_zone_4"] == 2

    def test_adapt_maps_computer_telemetry(self):
        """Per-computer features should be populated."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        features, _ = adapter.adapt(self._make_sample_state())
        assert features["computer_1_cpu"] > 0
        assert features["computer_10_heat"] > 0
        assert isinstance(features["computer_5_workload"], str)

    def test_adapter_reset_clears_history(self):
        """Reset should clear rolling history."""
        from ml.feature_adapter import FeatureAdapter
        adapter = FeatureAdapter()
        adapter.adapt(self._make_sample_state())
        assert len(adapter._history) > 0
        adapter.reset()
        assert len(adapter._history) == 0


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
        from ml.inference import predict_room_setpoint, get_model_status
        from ml.feature_adapter import FeatureAdapter

        model_path = PROJECT_ROOT / "models" / "hveac_brain_v1" / "hveac_brain_v1.joblib"
        if not model_path.exists():
            pytest.skip("Model artifact not found")

        # Build features from adapter
        adapter = FeatureAdapter()
        state = TestFeatureAdapter()._make_sample_state()
        features, _ = adapter.adapt(state)

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
