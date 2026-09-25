"""
Comprehensive evaluation orchestrator for HVEAC Brain v1 models.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from ml.metrics import (
    compute_classification_metrics,
    compute_setpoint_physical_metrics,
    compute_per_class_metrics,
    compute_confusion_matrix_df,
    compute_family_metrics,
    compute_rare_regime_metrics,
    compute_scenario_level_metrics,
    compute_temporal_prediction_metrics,
)


@dataclass
class EvaluationResults:
    """Encapsulates full diagnostic evaluation results for a model on a split."""
    model_name: str
    split_name: str
    sample_count: int
    classification_metrics: Dict[str, float]
    physical_metrics: Dict[str, float]
    per_class_metrics: Dict[str, Dict[str, Any]]
    confusion_matrix_df: pd.DataFrame
    family_metrics_df: pd.DataFrame
    rare_regime_metrics: Dict[str, Any]
    scenario_metrics_df: pd.DataFrame
    temporal_metrics: Dict[str, Any]
    predictions: np.ndarray
    probabilities: Optional[np.ndarray]

    def to_summary_dict(self) -> Dict[str, Any]:
        """Flatten key performance indicators into a summary dictionary."""
        return {
            "model_name": self.model_name,
            "split_name": self.split_name,
            "sample_count": self.sample_count,
            "accuracy": self.classification_metrics["accuracy"],
            "balanced_accuracy": self.classification_metrics["balanced_accuracy"],
            "macro_f1": self.classification_metrics["macro_f1"],
            "weighted_f1": self.classification_metrics["weighted_f1"],
            "mae_deg_c": self.physical_metrics["mae_deg_c"],
            "rmse_deg_c": self.physical_metrics["rmse_deg_c"],
            "pct_within_0_5_c": self.physical_metrics["pct_within_0_5_c"],
            "pct_within_1_0_c": self.physical_metrics["pct_within_1_0_c"],
            "max_absolute_error_c": self.physical_metrics["maximum_absolute_error_c"],
            "median_absolute_error_c": self.physical_metrics["median_absolute_error_c"],
            "rare_24_5_recall": self.per_class_metrics.get("24.5°C", {}).get("recall", 0.0),
            "rare_26_5_recall": self.per_class_metrics.get("26.5°C", {}).get("recall", 0.0),
            "mean_changes_per_scenario": self.temporal_metrics.get("mean_prediction_changes_per_scenario", 0.0),
        }

    def save_reports(self, output_dir: Path, prefix: str = ""):
        """Export all evaluation tables, metrics, and matrices to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pref = f"{prefix}_" if prefix else ""
        
        # 1. Full metrics JSON
        full_json = {
            "model_name": self.model_name,
            "split_name": self.split_name,
            "sample_count": self.sample_count,
            "classification_metrics": self.classification_metrics,
            "physical_metrics": self.physical_metrics,
            "per_class_metrics": self.per_class_metrics,
            "rare_regime_metrics": self.rare_regime_metrics,
            "temporal_metrics": self.temporal_metrics,
        }
        with open(output_dir / f"{pref}{self.split_name}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_json, f, indent=2)
            
        # 2. Confusion matrix CSV
        self.confusion_matrix_df.to_csv(output_dir / f"{pref}confusion_matrix_{self.split_name}.csv")
        
        # 3. Family metrics CSV
        self.family_metrics_df.to_csv(output_dir / f"{pref}per_family_metrics_{self.split_name}.csv", index=False)
        
        # 4. Scenario metrics CSV
        self.scenario_metrics_df.to_csv(output_dir / f"{pref}per_scenario_metrics_{self.split_name}.csv", index=False)


class ModelEvaluator:
    """Runs standard multi-tier evaluation on trained models."""

    @staticmethod
    def evaluate(
        model: Any,
        preprocessor: Any,
        X: pd.DataFrame,
        y_true: pd.Series,
        meta_df: pd.DataFrame,
        model_name: str,
        split_name: str,
    ) -> EvaluationResults:
        """Execute full evaluation pipeline on given dataset partition."""
        # 1. Transform features if preprocessor provided
        if preprocessor is not None:
            X_trans = preprocessor.transform(X)
        else:
            X_trans = X

        # 2. Generate predictions and probabilities
        y_pred = model.predict(X_trans)
        
        probabilities = None
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(X_trans)
            except Exception:
                probabilities = None

        y_true_arr = y_true.values
        
        # 3. Calculate all metric components
        cls_metrics = compute_classification_metrics(y_true_arr, y_pred)
        phys_metrics = compute_setpoint_physical_metrics(y_true_arr, y_pred)
        per_class = compute_per_class_metrics(y_true_arr, y_pred)
        cm_df = compute_confusion_matrix_df(y_true_arr, y_pred)
        fam_df = compute_family_metrics(y_true_arr, y_pred, meta_df)
        rare_dict = compute_rare_regime_metrics(y_true_arr, y_pred, meta_df)
        scen_df = compute_scenario_level_metrics(y_true_arr, y_pred, meta_df)
        temp_dict = compute_temporal_prediction_metrics(y_pred, y_true_arr, meta_df)

        return EvaluationResults(
            model_name=model_name,
            split_name=split_name,
            sample_count=len(y_true_arr),
            classification_metrics=cls_metrics,
            physical_metrics=phys_metrics,
            per_class_metrics=per_class,
            confusion_matrix_df=cm_df,
            family_metrics_df=fam_df,
            rare_regime_metrics=rare_dict,
            scenario_metrics_df=scen_df,
            temporal_metrics=temp_dict,
            predictions=y_pred,
            probabilities=probabilities,
        )
