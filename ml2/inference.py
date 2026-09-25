"""
Inference API and runtime validation interface for HVEAC Brain v1.

Provides standalone, offline prediction of the optimal room thermostat setpoint
from 80 physical and thermal state features.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from ml.artifact import load_brain_artifacts, HVEACBrainArtifact
from ml.config import (
    PathConfig,
    TARGET_CLASSES,
    EXPECTED_FEATURE_COUNT,
)


class HVEACBrainInference:
    """
    Offline inference engine for HVEAC Brain v1.
    Validates physical feature inputs and computes setpoint predictions and class probabilities.
    """

    def __init__(self, artifact_dir: Optional[Path] = None):
        if artifact_dir is None:
            # Default to production models directory or fallback to artifacts directory
            paths = PathConfig()
            if paths.models_dir.exists() and (paths.models_dir / "hveac_brain_v1.joblib").exists():
                artifact_dir = paths.models_dir
            else:
                artifact_dir = paths.artifacts_dir
                
        self.artifact: HVEACBrainArtifact = load_brain_artifacts(artifact_dir)
        self.feature_names: List[str] = self.artifact.schema.get("feature_columns", [])
        self.classes: List[str] = [str(c) for c in TARGET_CLASSES]

    def validate_features(self, features: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Rigorous input validation:
        - Exactly 80 required features present
        - No unexpected features
        - Physical range sanity checks
        """
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            df = pd.DataFrame([features.to_dict()])
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise TypeError(f"Unsupported features type: {type(features)}. Expected dict, Series, or DataFrame.")

        cols = list(df.columns)

        # 1. Missing feature validation
        missing = [f for f in self.feature_names if f not in cols]
        if missing:
            raise ValueError(
                f"Missing required input features ({len(missing)} missing): {missing[:5]}"
            )

        # 2. Unexpected feature validation
        unexpected = [c for c in cols if c not in self.feature_names]
        if unexpected:
            raise ValueError(
                f"Unexpected features provided to inference engine ({len(unexpected)} unexpected): {unexpected[:5]}"
            )

        # 3. Ensure columns are ordered strictly as expected
        df = df[self.feature_names].copy()

        # 4. Null / NaN check
        if df.isnull().values.any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"Null or NaN values detected in features: {nan_cols}")

        # 5. Domain physical range validation
        for idx, row in df.iterrows():
            if "room_average_temperature_c" in row and not (10.0 <= float(row["room_average_temperature_c"]) <= 55.0):
                raise ValueError(
                    f"Physical range violation: room_average_temperature_c={row['room_average_temperature_c']} (expected 10-55°C)"
                )
            if "humidity_percent" in row and not (0.0 <= float(row["humidity_percent"]) <= 100.0):
                raise ValueError(
                    f"Physical range violation: humidity_percent={row['humidity_percent']} (expected 0-100%)"
                )
            if "occupancy_total" in row and int(row["occupancy_total"]) < 0:
                raise ValueError(
                    f"Physical range violation: occupancy_total={row['occupancy_total']} (cannot be negative)"
                )

        return df

    def predict_room_setpoint(
        self,
        features: Union[Dict[str, Any], pd.DataFrame, pd.Series],
    ) -> Dict[str, Any]:
        """
        Predict optimal room thermostat setpoint for a single observation or batch.
        
        Returns:
            Dictionary containing predicted setpoint (°C), class probabilities,
            and max class probability.
        """
        df_valid = self.validate_features(features)
        
        # Transform through fitted preprocessor
        X_trans = self.artifact.preprocessor.transform(df_valid)
        
        # Predict class
        raw_pred = self.artifact.model.predict(X_trans)
        
        # Calculate class probabilities
        if hasattr(self.artifact.model, "predict_proba"):
            raw_proba = self.artifact.model.predict_proba(X_trans)
            # Match model.classes_ to TARGET_CLASSES
            model_classes = list(self.artifact.model.classes_)
        else:
            raw_proba = None
            model_classes = self.classes

        # For single sample return single dict; for batch return list of dicts
        results = []
        for i in range(len(df_valid)):
            pred_val = float(raw_pred[i])
            
            if raw_proba is not None:
                prob_dict = {}
                for cls_name in self.classes:
                    if cls_name in model_classes:
                        cls_idx = model_classes.index(cls_name)
                        prob_dict[cls_name] = round(float(raw_proba[i, cls_idx]), 4)
                    else:
                        prob_dict[cls_name] = 0.0
                max_prob = max(prob_dict.values())
            else:
                prob_dict = {cls_name: (1.0 if float(cls_name) == pred_val else 0.0) for cls_name in self.classes}
                max_prob = 1.0

            results.append({
                "predicted_setpoint_c": pred_val,
                "class_probabilities": prob_dict,
                "max_class_probability": max_prob,
                "model_version": self.artifact.metadata.get("brain_version", "hveac_brain_v1"),
            })

        if len(results) == 1:
            return results[0]
        return {"batch_predictions": results}


def predict_room_setpoint(features: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> Dict[str, Any]:
    """Convenience standalone inference function."""
    engine = HVEACBrainInference()
    return engine.predict_room_setpoint(features)
