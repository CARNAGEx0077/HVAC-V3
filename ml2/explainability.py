"""
Model Explainability, Feature Importance, and Physical Sanity Diagnostics.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from ml.config import TARGET_CLASSES


class FeatureAnalyzer:
    """
    Analyzes physical feature contributions and executes domain sanity diagnostics.
    """

    def __init__(self, model: Any, preprocessor: Any, feature_names: List[str]):
        self.model = model
        self.preprocessor = preprocessor
        self.raw_feature_names = feature_names

    def compute_feature_importance(self) -> pd.DataFrame:
        """
        Extract feature importances from the trained model and map them
        back to the 80 authoritative physical features.
        """
        encoded_names = self.preprocessor.get_feature_names_out()
        
        # 1. Extract raw importances/weights depending on model type
        if hasattr(self.model, "feature_importances_"):
            raw_importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            # Multi-class linear model: mean of absolute coefficients across classes
            raw_importances = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            # Fallback uniform
            raw_importances = np.zeros(len(encoded_names))

        # 2. Map encoded importances back to 80 raw physical features
        importance_by_raw_feature: Dict[str, float] = {feat: 0.0 for feat in self.raw_feature_names}
        
        for enc_name, imp in zip(encoded_names, raw_importances):
            # enc_name looks like "num__room_average_temperature_c" or "cat__ac1_state_ON"
            clean_name = enc_name.split("__", 1)[-1]
            
            # Find matching raw feature
            matched = False
            for raw_feat in self.raw_feature_names:
                if clean_name == raw_feat or clean_name.startswith(f"{raw_feat}_"):
                    importance_by_raw_feature[raw_feat] += float(imp)
                    matched = True
                    break
            if not matched:
                # Direct match fallback
                if clean_name in importance_by_raw_feature:
                    importance_by_raw_feature[clean_name] += float(imp)

        # Normalize to percentage / relative share
        total_imp = sum(importance_by_raw_feature.values())
        if total_imp > 0:
            rel_importances = {k: v / total_imp for k, v in importance_by_raw_feature.items()}
        else:
            rel_importances = importance_by_raw_feature

        rows = []
        for feat in self.raw_feature_names:
            rows.append({
                "feature_name": feat,
                "importance_score": round(importance_by_raw_feature[feat], 6),
                "importance_share_pct": round(rel_importances[feat] * 100.0, 3),
            })

        df = pd.DataFrame(rows).sort_values("importance_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        return df

    def run_physical_sanity_diagnostics(
        self,
        X_val: pd.DataFrame,
        y_val_pred: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Verify physical consistency of model predictions against thermal laws:
        1. Higher room temperature should correlate with lower (cooler) or equal setpoints.
        2. Higher total thermal load should correlate with increased cooling effort.
        """
        df = X_val.copy()
        df["predicted_setpoint_c"] = np.asarray(y_val_pred, dtype=float)

        results = {}

        # Diagnostic 1: Room Temperature vs Predicted Setpoint
        if "room_average_temperature_c" in df.columns:
            corr_room_temp = float(df["room_average_temperature_c"].corr(df["predicted_setpoint_c"]))
            
            # Group into temperature bins to inspect monotonic cooling trend
            df["temp_bin"] = pd.cut(df["room_average_temperature_c"], bins=5)
            mean_sp_by_temp = df.groupby("temp_bin", observed=False)["predicted_setpoint_c"].mean().to_dict()
            mean_sp_dict = {str(k): round(float(v), 3) for k, v in mean_sp_by_temp.items() if not np.isnan(v)}
            
            results["room_temperature_sensitivity"] = {
                "pearson_correlation": round(corr_room_temp, 4),
                "expected_direction": "Negative or flat (higher indoor temp -> cooler setpoint)",
                "observed_direction": "Negative" if corr_room_temp < 0 else "Positive",
                "mean_predicted_setpoint_by_temp_bin": mean_sp_dict,
                "physically_sound": corr_room_temp <= 0.05,  # allowing minor near-zero noise
            }

        # Diagnostic 2: Total Heat Load vs Predicted Setpoint
        if "total_heat_load_watts" in df.columns:
            corr_heat_load = float(df["total_heat_load_watts"].corr(df["predicted_setpoint_c"]))
            df["load_bin"] = pd.cut(df["total_heat_load_watts"], bins=5)
            mean_sp_by_load = df.groupby("load_bin", observed=False)["predicted_setpoint_c"].mean().to_dict()
            mean_load_dict = {str(k): round(float(v), 3) for k, v in mean_sp_by_load.items() if not np.isnan(v)}

            results["heat_load_sensitivity"] = {
                "pearson_correlation": round(corr_heat_load, 4),
                "expected_direction": "Negative or flat (higher heat load -> lower/equal setpoint for comfort)",
                "observed_direction": "Negative" if corr_heat_load < 0 else "Positive",
                "mean_predicted_setpoint_by_load_bin": mean_load_dict,
                "physically_sound": corr_heat_load <= 0.05,
            }

        # Diagnostic 3: Prediction range bound check
        min_p = float(np.min(df["predicted_setpoint_c"]))
        max_p = float(np.max(df["predicted_setpoint_c"]))
        results["range_bounds_check"] = {
            "min_prediction_c": min_p,
            "max_prediction_c": max_p,
            "within_admissible_bounds": min_p >= 24.5 and max_p <= 26.5,
        }

        return results
