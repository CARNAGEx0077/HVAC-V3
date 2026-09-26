"""
Test Suite for HVEAC Brain V2 — Machine Learning Training, Inference & Validation Pipeline.

Strictly verifies all 20 dimensions mandated by Section 30 of the Specification:
1. V2 dataset loads correctly
2. Correct feature count (80 physical features)
3. No metadata leakage
4. No target leakage
5. Preprocessing fits only on train
6. Model training succeeds
7. Inference succeeds
8. Prediction range is valid ([22.0, 25.0]°C)
9. Probabilities sum correctly where available (sum ≈ 1.0)
10. Saved model reloads
11. Reloaded model produces consistent predictions
12. Missing feature rejection
13. Unexpected feature rejection
14. NaN/Inf rejection
15. Directional audit executes
16. Rare-regime evaluation executes
17. Family evaluation executes
18. Scenario-level evaluation executes
19. Test remains scenario-disjoint
20. Model artifact is created
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from ml.config_v2 import (
    PathConfigV2,
    TARGET_COLUMN,
    TARGET_CLASSES,
    EXPECTED_FEATURE_COUNT,
    FORBIDDEN_INPUT_COLUMNS,
)
from ml.preprocessing_v2 import HVEACPreprocessorV2
from ml.inference_v2 import predict_hveac_v2, load_model_v2

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
PATHS = PathConfigV2()


@pytest.fixture(scope="module")
def dataset_splits():
    df_train = pd.read_csv(PATHS.train_csv)
    df_val = pd.read_csv(PATHS.val_csv)
    df_test = pd.read_csv(PATHS.test_csv)
    with open(PATHS.schema_json, "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_cols = schema.get("feature_columns", [])
    return {
        "train": df_train,
        "val": df_val,
        "test": df_test,
        "feature_cols": feature_cols,
    }


@pytest.fixture(scope="module")
def sample_test_features(dataset_splits):
    df_test = dataset_splits["test"]
    feat_cols = dataset_splits["feature_cols"]
    sample_row = df_test.iloc[0]
    return {col: sample_row[col] for col in feat_cols}


def test_01_v2_dataset_loads_correctly(dataset_splits):
    """Test 1: Verify V2 dataset splits load with exact expected row and scenario counts."""
    df_train = dataset_splits["train"]
    df_val = dataset_splits["val"]
    df_test = dataset_splits["test"]
    
    assert len(df_train) == 50400, f"Expected 50400 train rows, got {len(df_train)}"
    assert len(df_val) == 10800, f"Expected 10800 val rows, got {len(df_val)}"
    assert len(df_test) == 10800, f"Expected 10800 test rows, got {len(df_test)}"
    assert df_train["scenario_id"].nunique() == 70
    assert df_val["scenario_id"].nunique() == 15
    assert df_test["scenario_id"].nunique() == 15


def test_02_correct_feature_count(dataset_splits):
    """Test 2: Verify exactly 80 physical features are specified in schema and present in dataset."""
    feature_cols = dataset_splits["feature_cols"]
    assert len(feature_cols) == EXPECTED_FEATURE_COUNT, f"Expected {EXPECTED_FEATURE_COUNT}, got {len(feature_cols)}"
    for col in feature_cols:
        assert col in dataset_splits["train"].columns, f"Feature {col} missing from train split"


def test_03_no_metadata_leakage(dataset_splits):
    """Test 3: Assert no scenario identifiers, run IDs, seeds, or timestamps exist in features."""
    feature_cols = set(dataset_splits["feature_cols"])
    metadata_cols = {"scenario_id", "scenario_family", "run_id", "random_seed", "timestamp", "simulation_time_seconds"}
    leakage = feature_cols.intersection(metadata_cols)
    assert len(leakage) == 0, f"Metadata leakage detected: {leakage}"


def test_04_no_target_leakage(dataset_splits):
    """Test 4: Assert no target variables or optimization post-decision outputs exist in features."""
    feature_cols = set(dataset_splits["feature_cols"])
    forbidden = set(FORBIDDEN_INPUT_COLUMNS)
    leakage = feature_cols.intersection(forbidden)
    assert len(leakage) == 0, f"Target leakage detected: {leakage}"


def test_05_preprocessing_fits_only_on_train(dataset_splits):
    """Test 5: Verify preprocessor fits strictly on train split without using val/test data."""
    feat_cols = dataset_splits["feature_cols"]
    preprocessor = HVEACPreprocessorV2(feature_names=feat_cols)
    
    # Preprocessor is unfitted initially
    assert not preprocessor.is_fitted
    
    # Fit strictly on train
    preprocessor.fit(dataset_splits["train"][feat_cols])
    assert preprocessor.is_fitted
    
    # Numeric scaler means and stds are computed
    scaler = preprocessor.column_transformer_.named_transformers_["num"]
    assert scaler.mean_ is not None
    assert len(scaler.mean_) == len(preprocessor.numeric_features_)
    
    # Transform test without changing fitted params
    old_mean = scaler.mean_.copy()
    X_test_trans = preprocessor.transform(dataset_splits["test"][feat_cols])
    np.testing.assert_array_equal(scaler.mean_, old_mean)
    assert X_test_trans.shape[0] == len(dataset_splits["test"])


def test_06_model_training_succeeds():
    """Test 6: Verify trained model artifact exists and has valid classifier attributes."""
    model_path = PATHS.models_dir / "hveac_brain_v2.joblib"
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    model = joblib.load(model_path)
    assert hasattr(model, "predict"), "Trained model has no predict method"
    assert hasattr(model, "classes_"), "Trained model has no classes_ attribute"
    assert len(model.classes_) == len(TARGET_CLASSES)


def test_07_inference_succeeds(sample_test_features):
    """Test 7: Verify predict_hveac_v2 produces a valid SUCCESS response."""
    result = predict_hveac_v2(sample_test_features)
    assert result["status"] == "SUCCESS", f"Inference failed: {result.get('error')}"
    assert "optimal_room_setpoint_c" in result
    assert "class_probabilities" in result
    assert result["model_version"] == "hveac_brain_v2"
    assert result["inference_latency_ms"] >= 0.0


def test_08_prediction_range_is_valid(sample_test_features):
    """Test 8: Verify predicted setpoint is in operative range [22.0, 25.0]°C and is discrete."""
    result = predict_hveac_v2(sample_test_features)
    pred_sp = result["optimal_room_setpoint_c"]
    assert pred_sp in TARGET_CLASSES, f"Predicted setpoint {pred_sp} not in valid classes {TARGET_CLASSES}"
    assert 22.0 <= pred_sp <= 25.0


def test_09_probabilities_sum_correctly(sample_test_features):
    """Test 9: Verify predicted class probabilities sum to approximately 1.0."""
    result = predict_hveac_v2(sample_test_features)
    probs = result.get("class_probabilities", {})
    assert len(probs) == len(TARGET_CLASSES)
    total_prob = sum(probs.values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities do not sum to 1.0: {total_prob}"


def test_10_saved_model_reloads():
    """Test 10: Verify both model and preprocessor reload successfully from disk."""
    model_path = PATHS.models_dir / "hveac_brain_v2.joblib"
    prep_path = PATHS.models_dir / "preprocessing_v2.joblib"
    
    assert model_path.exists()
    assert prep_path.exists()
    
    model = joblib.load(model_path)
    prep = joblib.load(prep_path)
    assert model is not None
    assert prep.is_fitted


def test_11_reloaded_model_produces_consistent_predictions(dataset_splits, sample_test_features):
    """Test 11: Verify reloaded model predictions match inference service predictions exactly."""
    model = joblib.load(PATHS.models_dir / "hveac_brain_v2.joblib")
    prep = joblib.load(PATHS.models_dir / "preprocessing_v2.joblib")
    
    with open(PATHS.models_dir / "class_mapping.json", "r") as f:
        cmap = json.load(f)
    idx_to_class = {int(k): float(v) for k, v in cmap["index_to_class"].items()}
    
    df_row = pd.DataFrame([sample_test_features])
    X = prep.transform(df_row)
    direct_pred_idx = int(model.predict(X)[0])
    direct_pred_sp = idx_to_class[direct_pred_idx]
    
    api_result = predict_hveac_v2(sample_test_features)
    assert api_result["status"] == "SUCCESS"
    assert direct_pred_sp == api_result["optimal_room_setpoint_c"]


def test_12_missing_feature_rejection(sample_test_features):
    """Test 12: Verify inference service rejects input with missing required features."""
    bad_features = dict(sample_test_features)
    del bad_features["room_average_temperature_c"]
    result = predict_hveac_v2(bad_features)
    assert result["status"] == "ERROR"
    assert "Missing required features" in result["error"]


def test_13_unexpected_feature_rejection(sample_test_features):
    """Test 13: Verify inference service rejects unexpected or malicious extra keys."""
    bad_features = dict(sample_test_features)
    bad_features["unexpected_unauthorized_key"] = 999.0
    result = predict_hveac_v2(bad_features)
    assert result["status"] == "ERROR"
    assert "Unexpected features" in result["error"]


def test_14_nan_inf_rejection(sample_test_features):
    """Test 14: Verify inference service rejects NaN and Inf feature values."""
    nan_features = dict(sample_test_features)
    nan_features["room_average_temperature_c"] = float("nan")
    res_nan = predict_hveac_v2(nan_features)
    assert res_nan["status"] == "ERROR"
    assert "NaN or Inf" in res_nan["error"]
    
    inf_features = dict(sample_test_features)
    inf_features["room_average_temperature_c"] = float("inf")
    res_inf = predict_hveac_v2(inf_features)
    assert res_inf["status"] == "ERROR"
    assert "NaN or Inf" in res_inf["error"]


def test_15_directional_audit_executes():
    """Test 15: Verify directional control audit report exists and confirms PASS status."""
    report_file = PATHS.reports_dir / "brain_v2_directional_audit.md"
    assert report_file.exists(), f"Directional audit report missing at {report_file}"
    content = report_file.read_text(encoding="utf-8")
    assert "PASS" in content
    assert "CASE_A_HOT_RISING_HIGH_LOAD" in content
    assert "CASE_D_COOL_ROOM" in content
    assert "CASE_E_LOAD_MONOTONICITY" in content


def test_16_rare_regime_evaluation_executes():
    """Test 16: Verify rare regime metrics JSON exists and covers critical rare regimes."""
    rare_file = PATHS.reports_dir / "brain_v2_rare_regime_metrics.json"
    assert rare_file.exists(), f"Rare regime report missing at {rare_file}"
    with open(rare_file, "r", encoding="utf-8") as f:
        rare_data = json.load(f)
    assert "AGGRESSIVE_COOLING" in rare_data
    assert "HIGH_THERMAL_LOAD_GT_3500W" in rare_data
    assert "OPPOSING_ZONES_FAMILY_5" in rare_data


def test_17_family_evaluation_executes():
    """Test 17: Verify per-family metrics CSV exists and covers all 5 scenario families."""
    family_file = PATHS.reports_dir / "brain_v2_per_family_metrics.csv"
    assert family_file.exists(), f"Family metrics CSV missing at {family_file}"
    df_fam = pd.read_csv(family_file)
    assert len(df_fam) == 5
    expected_fams = {"FAMILY_1", "FAMILY_2", "FAMILY_3", "FAMILY_4", "FAMILY_5"}
    assert set(df_fam["scenario_family"].unique()) == expected_fams


def test_18_scenario_level_evaluation_executes():
    """Test 18: Verify per-scenario metrics CSV exists and covers all 15 test scenarios."""
    scen_file = PATHS.reports_dir / "brain_v2_per_scenario_metrics.csv"
    assert scen_file.exists(), f"Scenario metrics CSV missing at {scen_file}"
    df_scen = pd.read_csv(scen_file)
    assert len(df_scen) == 15
    assert (df_scen["row_count"] == 720).all()
    
    # Also verify scenario_metrics.csv satisfies Section 19 requirement
    scen_file_alt = PATHS.reports_dir / "scenario_metrics.csv"
    assert scen_file_alt.exists()


def test_19_test_remains_scenario_disjoint(dataset_splits):
    """Test 19: Verify zero scenario overlap between train, val, and test splits."""
    s_tr = set(dataset_splits["train"]["scenario_id"].unique())
    s_va = set(dataset_splits["val"]["scenario_id"].unique())
    s_te = set(dataset_splits["test"]["scenario_id"].unique())
    
    assert len(s_tr.intersection(s_te)) == 0, "Scenario overlap between Train and Test!"
    assert len(s_va.intersection(s_te)) == 0, "Scenario overlap between Val and Test!"
    assert len(s_tr.intersection(s_va)) == 0, "Scenario overlap between Train and Val!"


def test_20_model_artifact_is_created():
    """Test 20: Verify all 5 mandatory production model artifacts exist in models/hveac_brain_v2/."""
    m_dir = PATHS.models_dir
    assert (m_dir / "hveac_brain_v2.joblib").exists()
    assert (m_dir / "preprocessing_v2.joblib").exists()
    assert (m_dir / "feature_schema_v2.json").exists()
    assert (m_dir / "model_metadata.json").exists()
    assert (m_dir / "class_mapping.json").exists()
