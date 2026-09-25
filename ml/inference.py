"""
HVEAC Brain V1 — Inference Service

Loads the trained Random Forest model and preprocessing pipeline,
and provides a clean predict_room_setpoint(features) function.

This module is SHADOW-MODE ONLY:
- No HVAC control path exists.
- All predictions are for observation and comparison.

Model artifact: models/hveac_brain_v1/hveac_brain_v1.joblib
Preprocessor:   models/hveac_brain_v1/preprocessing_v1.joblib
Feature schema:  models/hveac_brain_v1/feature_schema_v1.json
Metadata:       models/hveac_brain_v1/model_metadata.json
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hveac.brain.inference")

# ── Model artifact directory ───────────────────────────────────────────────
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hveac_brain_v1"

# ── Singleton state ────────────────────────────────────────────────────────
_model = None
_preprocessor = None
_metadata: Dict[str, Any] = {}
_feature_columns: List[str] = []
_target_classes: List[float] = []
_is_loaded: bool = False
_load_error: Optional[str] = None


def _load_model():
    """Lazy-load model, preprocessor, and metadata from disk.
    Designed to fail safely — never crashes the application.
    """
    global _model, _preprocessor, _metadata, _feature_columns, _target_classes
    global _is_loaded, _load_error

    if _is_loaded:
        return

    try:
        import joblib
    except ImportError:
        _load_error = "joblib not installed — cannot load model"
        logger.error(f"[BRAIN] {_load_error}")
        return

    model_path = MODEL_DIR / "hveac_brain_v1.joblib"
    preprocessor_path = MODEL_DIR / "preprocessing_v1.joblib"
    metadata_path = MODEL_DIR / "model_metadata.json"
    schema_path = MODEL_DIR / "feature_schema_v1.json"

    # 1. Load metadata
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        _target_classes = [float(c) for c in _metadata.get("target_classes", [])]
        logger.info(f"[BRAIN] Loaded metadata: v{_metadata.get('brain_version', '?')}, "
                     f"{_metadata.get('feature_count', '?')} features, "
                     f"{len(_target_classes)} classes")
    except Exception as e:
        _load_error = f"Failed to load metadata: {e}"
        logger.error(f"[BRAIN] {_load_error}")
        return

    # 2. Load feature schema
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        _feature_columns = schema.get("feature_columns", [])
        if len(_feature_columns) != 80:
            _load_error = f"Feature schema has {len(_feature_columns)} columns, expected 80"
            logger.error(f"[BRAIN] {_load_error}")
            return
        logger.info(f"[BRAIN] Feature schema loaded: {len(_feature_columns)} columns")
    except Exception as e:
        _load_error = f"Failed to load feature schema: {e}"
        logger.error(f"[BRAIN] {_load_error}")
        return

    # 3. Load preprocessor
    try:
        _preprocessor = joblib.load(str(preprocessor_path))
        logger.info(f"[BRAIN] Preprocessing pipeline loaded from {preprocessor_path.name}")
    except Exception as e:
        _load_error = f"Failed to load preprocessor: {e}"
        logger.error(f"[BRAIN] {_load_error}")
        return

    # 4. Load model
    try:
        _model = joblib.load(str(model_path))
        logger.info(f"[BRAIN] Model loaded from {model_path.name} "
                     f"({_metadata.get('selected_model_architecture', 'Unknown')})")
    except Exception as e:
        _load_error = f"Failed to load model: {e}"
        logger.error(f"[BRAIN] {_load_error}")
        return

    _is_loaded = True
    _load_error = None
    logger.info("[BRAIN] ✓ HVEAC Brain V1 inference service ready (SHADOW MODE ONLY)")


def get_feature_columns() -> List[str]:
    """Returns the authoritative 80-feature column list."""
    _load_model()
    return list(_feature_columns)


def get_model_status() -> Dict[str, Any]:
    """Returns current model health and metadata."""
    _load_model()
    if not _is_loaded:
        return {
            "status": "UNAVAILABLE",
            "error": _load_error or "Model not loaded",
            "model_version": None,
            "feature_count": 0,
            "class_count": 0,
            "mode": "SHADOW_ONLY",
        }
    return {
        "status": "READY",
        "error": None,
        "model_version": _metadata.get("brain_version", "unknown"),
        "architecture": _metadata.get("selected_model_architecture", "Unknown"),
        "dataset_version": _metadata.get("dataset_version", "unknown"),
        "feature_count": len(_feature_columns),
        "class_count": len(_target_classes),
        "target_classes": _target_classes,
        "mode": "SHADOW_ONLY",
        "validation_accuracy": _metadata.get("validation_metrics", {}).get("accuracy"),
        "test_accuracy": _metadata.get("final_test_metrics", {}).get("accuracy"),
    }


def predict_room_setpoint(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run inference on a single feature vector.

    Parameters
    ----------
    features : dict
        Mapping of feature_name → value for all 80 input features.

    Returns
    -------
    dict with keys:
        status            : 'OK' | 'UNAVAILABLE' | 'ERROR'
        predicted_class   : float (e.g. 25.0) or None
        confidence        : float [0-1] or None
        class_probabilities : dict {class_label: probability} or None
        all_classes       : list of float
        feature_warnings  : list of str (missing/defaulted features)
        error             : str or None
        mode              : 'SHADOW_ONLY'
    """
    _load_model()

    # Fail safe if model unavailable
    if not _is_loaded or _model is None or _preprocessor is None:
        return {
            "status": "UNAVAILABLE",
            "predicted_class": None,
            "confidence": None,
            "class_probabilities": None,
            "all_classes": [],
            "feature_warnings": [],
            "error": _load_error or "Model not loaded",
            "mode": "SHADOW_ONLY",
        }

    try:
        import pandas as pd

        warnings_list = []

        # Build ordered feature vector, validating completeness
        row_data = {}
        for col in _feature_columns:
            if col in features:
                row_data[col] = features[col]
            else:
                # Default missing numeric features to 0.0, string to "IDLE"
                schema_path_local = MODEL_DIR / "feature_schema_v1.json"
                with open(schema_path_local, "r", encoding="utf-8") as f:
                    schema_full = json.load(f)
                col_detail = schema_full.get("feature_column_details", {}).get(col, {})
                dtype = col_detail.get("dtype", "float64")
                if dtype == "string":
                    row_data[col] = "IDLE"
                else:
                    row_data[col] = 0.0
                warnings_list.append(f"Missing feature '{col}' — defaulted")

        # Create single-row DataFrame with exact column order
        df = pd.DataFrame([row_data], columns=_feature_columns)

        # Apply preprocessing (handles encoding of categorical features)
        X_transformed = _preprocessor.transform(df)

        # Predict
        pred_encoded = _model.predict(X_transformed)[0]
        probas = _model.predict_proba(X_transformed)[0]

        # Map prediction back to temperature class
        if hasattr(_model, 'classes_'):
            classes = [float(c) for c in _model.classes_]
        else:
            classes = _target_classes

        predicted_class = float(classes[int(pred_encoded)] if isinstance(pred_encoded, (int, np.integer)) else pred_encoded)
        confidence = float(np.max(probas))

        class_probabilities = {}
        for i, cls in enumerate(classes):
            class_probabilities[str(cls)] = round(float(probas[i]), 4)

        return {
            "status": "OK",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities,
            "all_classes": classes,
            "feature_warnings": warnings_list,
            "error": None,
            "mode": "SHADOW_ONLY",
        }

    except Exception as e:
        logger.error(f"[BRAIN] Inference error: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "predicted_class": None,
            "confidence": None,
            "class_probabilities": None,
            "all_classes": [],
            "feature_warnings": [],
            "error": str(e),
            "mode": "SHADOW_ONLY",
        }
