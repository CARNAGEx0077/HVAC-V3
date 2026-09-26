"""
HVEAC Brain V2 — Inference Service (Offline / Shadow-Ready API)

Loads the trained Random Forest model and preprocessing pipeline for Brain V2,
providing a robust, high-performance predict_hveac_v2(features) function.

Strict Safety & Contract Enforcement:
- NO HVAC control path exists (SHADOW / OFFLINE ONLY).
- Validates exact 80 physical feature inputs from Dataset V2 feature schema.
- Rejects missing features, unexpected features, metadata columns, and target columns.
- Rejects NaNs, Infs, and out-of-bounds physical values.
- Returns predicted setpoint, calibrated class probabilities, inference latency, and version.

Artifacts:
- Model: models/hveac_brain_v2/hveac_brain_v2.joblib
- Preprocessor: models/hveac_brain_v2/preprocessing_v2.joblib
- Schema: models/hveac_brain_v2/feature_schema_v2.json
- Metadata: models/hveac_brain_v2/model_metadata.json
- Class Mapping: models/hveac_brain_v2/class_mapping.json
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("hveac.brain_v2.inference")

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hveac_brain_v2"

# Singleton state
_model = None
_preprocessor = None
_metadata: Dict[str, Any] = {}
_feature_columns: List[str] = []
_target_classes: List[float] = []
_class_to_index: Dict[str, int] = {}
_index_to_class: Dict[int, float] = {}
_is_loaded: bool = False
_load_error: Optional[str] = None

# Validation constants
METADATA_COLUMNS = {
    "scenario_id",
    "scenario_family",
    "run_id",
    "random_seed",
    "timestamp",
    "simulation_time_seconds",
}

FORBIDDEN_TARGET_COLUMNS = {
    "optimal_room_setpoint_c",
    "optimal_ac1_setpoint_c",
    "optimal_ac2_setpoint_c",
    "optimal_ac3_setpoint_c",
    "optimal_ac4_setpoint_c",
    "optimal_ac1_cooling_level",
    "optimal_ac2_cooling_level",
    "optimal_ac3_cooling_level",
    "optimal_ac4_cooling_level",
    "optimization_cost",
    "label_reason",
}

VALID_PHYSICAL_RANGES = {
    "room_average_temperature_c": (10.0, 50.0),
    "zone_1_temperature_c": (10.0, 50.0),
    "zone_2_temperature_c": (10.0, 50.0),
    "zone_3_temperature_c": (10.0, 50.0),
    "zone_4_temperature_c": (10.0, 50.0),
    "outdoor_temperature_c": (-20.0, 60.0),
    "humidity_percent": (0.0, 100.0),
    "occupancy_total": (0, 100),
    "total_heat_load_watts": (0.0, 50000.0),
    "ac1_cooling_level": (0.0, 1.0),
    "ac2_cooling_level": (0.0, 1.0),
    "ac3_cooling_level": (0.0, 1.0),
    "ac4_cooling_level": (0.0, 1.0),
}


def load_model_v2():
    """Load model, preprocessor, and metadata from disk safely."""
    global _model, _preprocessor, _metadata, _feature_columns
    global _target_classes, _class_to_index, _index_to_class
    global _is_loaded, _load_error

    if _is_loaded:
        return

    try:
        import joblib
    except ImportError:
        _load_error = "joblib not installed — cannot load model"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    model_path = MODEL_DIR / "hveac_brain_v2.joblib"
    preprocessor_path = MODEL_DIR / "preprocessing_v2.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    schema_path = MODEL_DIR / "feature_schema_v2.json"
    class_map_path = MODEL_DIR / "class_mapping.json"

    # 1. Load metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        _target_classes = [float(c) for c in _metadata.get("target_classes", [])]
        logger.info(f"[BRAIN_V2] Loaded metadata: v{_metadata.get('brain_version', '?')}")
    except Exception as e:
        _load_error = f"Failed to load metadata: {e}"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    # 2. Load feature schema
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        _feature_columns = schema.get("feature_columns", [])
        if len(_feature_columns) != 80:
            _load_error = f"Feature schema has {len(_feature_columns)} columns, expected 80"
            logger.error(f"[BRAIN_V2] {_load_error}")
            return
    except Exception as e:
        _load_error = f"Failed to load feature schema: {e}"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    # 3. Load class mapping
    try:
        with open(class_map_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
        _class_to_index = {str(k): int(v) for k, v in class_map.get("class_to_index", {}).items()}
        _index_to_class = {int(k): float(v) for k, v in class_map.get("index_to_class", {}).items()}
    except Exception as e:
        _load_error = f"Failed to load class mapping: {e}"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    # 4. Load preprocessor
    try:
        _preprocessor = joblib.load(str(preprocessor_path))
    except Exception as e:
        _load_error = f"Failed to load preprocessor: {e}"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    # 5. Load model
    try:
        _model = joblib.load(str(model_path))
    except Exception as e:
        _load_error = f"Failed to load model: {e}"
        logger.error(f"[BRAIN_V2] {_load_error}")
        return

    _is_loaded = True
    _load_error = None
    logger.info("[BRAIN_V2] Model loaded successfully and ready for inference.")


def validate_input_features(features: Dict[str, Any]) -> List[str]:
    """Validate input feature dictionary against V2 contract. Returns list of violation messages."""
    errors = []
    
    keys = set(features.keys())
    
    # 1. Check for metadata leakage
    meta_found = keys.intersection(METADATA_COLUMNS)
    if meta_found:
        errors.append(f"Forbidden metadata columns provided: {sorted(meta_found)}")
        
    # 2. Check for target leakage
    target_found = keys.intersection(FORBIDDEN_TARGET_COLUMNS)
    if target_found:
        errors.append(f"Forbidden target columns provided: {sorted(target_found)}")
        
    # 3. Check for missing required features
    missing = [c for c in _feature_columns if c not in keys]
    if missing:
        errors.append(f"Missing required features ({len(missing)}): {missing[:5]}")
        
    # 4. Check for unexpected keys
    unexpected = [c for c in keys if c not in _feature_columns and c not in METADATA_COLUMNS and c not in FORBIDDEN_TARGET_COLUMNS]
    if unexpected:
        errors.append(f"Unexpected features provided ({len(unexpected)}): {unexpected[:5]}")
        
    # 5. Null, NaN, Inf, and Range checks
    for col in _feature_columns:
        if col in features:
            val = features[col]
            if val is None:
                errors.append(f"Feature '{col}' is None")
                continue
            if isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    errors.append(f"Feature '{col}' contains NaN or Inf")
                if col in VALID_PHYSICAL_RANGES:
                    r_min, r_max = VALID_PHYSICAL_RANGES[col]
                    if val < r_min or val > r_max:
                        errors.append(f"Feature '{col}' value {val} out of valid range [{r_min}, {r_max}]")
                        
    return errors


def predict_hveac_v2(features: Dict[str, Any]) -> Dict[str, Any]:
    """Execute synchronous model inference for a single physical state record.
    
    Args:
        features: Dictionary containing exactly the 80 physical features defined in V2 schema.
        
    Returns:
        Dictionary containing:
        - status: "SUCCESS" | "ERROR"
        - optimal_room_setpoint_c: Recommended temperature setpoint in °C
        - class_probabilities: Dict mapping setpoint string to float probability
        - confidence: Highest predicted probability
        - model_version: "hveac_brain_v2"
        - target_name: "optimal_room_setpoint_c"
        - inference_latency_ms: Milliseconds taken for prediction
        - error: Error message if status == "ERROR"
    """
    t_start = time.perf_counter()
    
    load_model_v2()
    if not _is_loaded:
        return {
            "status": "ERROR",
            "error": f"Model not loaded: {_load_error}",
            "inference_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }
        
    violations = validate_input_features(features)
    if violations:
        return {
            "status": "ERROR",
            "error": f"Input validation failed: {'; '.join(violations)}",
            "violations": violations,
            "inference_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }
        
    try:
        # Build single-row DataFrame with exact feature columns
        df_row = pd.DataFrame([{col: features[col] for col in _feature_columns}])
        X_trans = _preprocessor.transform(df_row)
        
        pred_idx = int(_model.predict(X_trans)[0])
        pred_setpoint = _index_to_class.get(pred_idx, float(pred_idx))
        
        # Probabilities
        if hasattr(_model, "predict_proba"):
            probs = _model.predict_proba(X_trans)[0]
            # Map model classes to setpoints
            class_probs = {}
            for cls_idx, p in zip(_model.classes_, probs):
                sp_val = _index_to_class.get(int(cls_idx), float(cls_idx))
                class_probs[str(sp_val)] = round(float(p), 4)
            confidence = round(float(np.max(probs)), 4)
        else:
            class_probs = {str(pred_setpoint): 1.0}
            confidence = 1.0
            
        latency = round((time.perf_counter() - t_start) * 1000.0, 3)
        
        return {
            "status": "SUCCESS",
            "optimal_room_setpoint_c": float(pred_setpoint),
            "class_probabilities": class_probs,
            "confidence": confidence,
            "model_version": "hveac_brain_v2",
            "target_name": "optimal_room_setpoint_c",
            "inference_latency_ms": latency,
        }
    except Exception as e:
        logger.error(f"[BRAIN_V2] Inference execution failed: {e}")
        return {
            "status": "ERROR",
            "error": f"Inference execution failed: {e}",
            "inference_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        }
