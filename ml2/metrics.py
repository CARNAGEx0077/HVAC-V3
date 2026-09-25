"""
Comprehensive evaluation metrics module for HVEAC Brain v1.

Calculates:
- Standard multi-class classification metrics
- Setpoint-specific physical error metrics (MAE, RMSE, tolerance accuracy)
- Per-class precision, recall, F1, and support
- Scenario-family grouped evaluations
- Rare-regime (24.5°C and 26.5°C) deep-dive evaluations
- Per-scenario diagnostics
- Temporal prediction stability and dwell analysis
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from ml.config import TARGET_CLASSES


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate multi-class classification performance metrics."""
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)
    classes_str = [str(c) for c in TARGET_CLASSES]

    return {
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_str, y_pred_str)),
        "macro_precision": float(precision_score(y_true_str, y_pred_str, labels=classes_str, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_str, y_pred_str, labels=classes_str, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_str, y_pred_str, labels=classes_str, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_str, y_pred_str, labels=classes_str, average="weighted", zero_division=0)),
    }


def compute_setpoint_physical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate physical setpoint metrics in degrees Celsius.
    Accounts for ordered ordinal distance between thermostat setpoints.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    diffs = np.abs(y_t - y_p)
    mae = float(np.mean(diffs))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    med_ae = float(np.median(diffs))
    max_ae = float(np.max(diffs))
    
    pct_0_5 = float(np.mean(diffs <= 0.5001) * 100.0)
    pct_1_0 = float(np.mean(diffs <= 1.0001) * 100.0)

    return {
        "mae_deg_c": round(mae, 4),
        "rmse_deg_c": round(rmse, 4),
        "median_absolute_error_c": round(med_ae, 4),
        "maximum_absolute_error_c": round(max_ae, 4),
        "pct_within_0_5_c": round(pct_0_5, 2),
        "pct_within_1_0_c": round(pct_1_0, 2),
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Compute precision, recall, f1, support per target setpoint class."""
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)
    classes_str = [str(c) for c in TARGET_CLASSES]

    p_per_class = precision_score(y_true_str, y_pred_str, labels=classes_str, average=None, zero_division=0)
    r_per_class = recall_score(y_true_str, y_pred_str, labels=classes_str, average=None, zero_division=0)
    f_per_class = f1_score(y_true_str, y_pred_str, labels=classes_str, average=None, zero_division=0)
    
    per_class_res = {}
    for i, cls_name in enumerate(classes_str):
        supp = int(np.sum(y_true_str == cls_name))
        pred_count = int(np.sum(y_pred_str == cls_name))
        per_class_res[f"{cls_name}°C"] = {
            "setpoint_c": float(cls_name),
            "precision": round(float(p_per_class[i]), 4),
            "recall": round(float(r_per_class[i]), 4),
            "f1": round(float(f_per_class[i]), 4),
            "support": supp,
            "predicted_count": pred_count,
        }
    return per_class_res


def compute_confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Generate confusion matrix DataFrame labeled by setpoint classes."""
    y_true_str = np.asarray(y_true, dtype=str)
    y_pred_str = np.asarray(y_pred, dtype=str)
    classes_str = [str(c) for c in TARGET_CLASSES]
    
    cm = confusion_matrix(y_true_str, y_pred_str, labels=classes_str)
    cols = [f"Pred_{c}°C" for c in classes_str]
    idx = [f"Actual_{c}°C" for c in classes_str]
    return pd.DataFrame(cm, index=idx, columns=cols)


def compute_family_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate performance grouped by scenario_family."""
    df = meta_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["abs_err"] = np.abs(df["y_true"].astype(float) - df["y_pred"].astype(float))
    df["sq_err"] = (df["y_true"].astype(float) - df["y_pred"].astype(float)) ** 2
    df["exact_match"] = (df["y_true"].astype(str) == df["y_pred"].astype(str)).astype(int)
    df["within_0_5"] = (df["abs_err"] <= 0.5001).astype(int)

    family_rows = []
    classes_str = [str(c) for c in TARGET_CLASSES]
    
    for fam, grp in df.groupby("scenario_family"):
        fam_scenarios = grp["scenario_id"].nunique()
        fam_samples = len(grp)
        fam_acc = float(accuracy_score(grp["y_true"].astype(str), grp["y_pred"].astype(str)))
        fam_mae = float(grp["abs_err"].mean())
        fam_rmse = float(np.sqrt(grp["sq_err"].mean()))
        fam_within_0_5 = float(grp["within_0_5"].mean() * 100.0)
        fam_f1 = float(f1_score(
            grp["y_true"].astype(str),
            grp["y_pred"].astype(str),
            labels=classes_str,
            average="macro",
            zero_division=0,
        ))

        family_rows.append({
            "scenario_family": fam,
            "scenario_count": fam_scenarios,
            "sample_count": fam_samples,
            "accuracy": round(fam_acc, 4),
            "mae_deg_c": round(fam_mae, 4),
            "rmse_deg_c": round(fam_rmse, 4),
            "pct_within_0_5_c": round(fam_within_0_5, 2),
            "macro_f1": round(fam_f1, 4),
        })

    return pd.DataFrame(family_rows).sort_values("scenario_family")


def compute_rare_regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Detailed evaluation of rare regimes 24.5°C and 26.5°C."""
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    
    rare_report = {}
    for target_val in [24.5, 26.5]:
        mask = (y_t == target_val)
        n_rows = int(np.sum(mask))
        
        if n_rows > 0:
            scen_count = int(meta_df[mask]["scenario_id"].nunique())
            acc = float(np.mean(y_t[mask] == y_p[mask]))
            mae = float(np.mean(np.abs(y_t[mask] - y_p[mask])))
            within_0_5 = float(np.mean(np.abs(y_t[mask] - y_p[mask]) <= 0.5001) * 100.0)
            
            # Confusion breakdown: what was predicted when actual was target_val?
            pred_counts = pd.Series(y_p[mask]).value_counts().to_dict()
            confusion_dict = {f"{k:.1f}°C": int(v) for k, v in pred_counts.items()}
        else:
            scen_count = 0
            acc = 0.0
            mae = 0.0
            within_0_5 = 0.0
            confusion_dict = {}

        rare_report[f"{target_val}°C"] = {
            "actual_setpoint_c": target_val,
            "row_count": n_rows,
            "scenario_count": scen_count,
            "accuracy": round(acc, 4),
            "mae_deg_c": round(mae, 4),
            "pct_within_0_5_c": round(within_0_5, 2),
            "predicted_distribution": confusion_dict,
        }

    return rare_report


def compute_scenario_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate diagnostic performance metrics per individual scenario trajectory."""
    df = meta_df.copy()
    df["y_true"] = np.asarray(y_true, dtype=float)
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["abs_err"] = np.abs(df["y_true"] - df["y_pred"])
    df["sq_err"] = (df["y_true"] - df["y_pred"]) ** 2
    df["exact_match"] = (df["y_true"] == df["y_pred"]).astype(int)
    df["within_0_5"] = (df["abs_err"] <= 0.5001).astype(int)

    scen_rows = []
    for s_id, grp in df.groupby("scenario_id"):
        fam = grp["scenario_family"].iloc[0] if "scenario_family" in grp.columns else "UNKNOWN"
        n_timesteps = len(grp)
        mae = float(grp["abs_err"].mean())
        rmse = float(np.sqrt(grp["sq_err"].mean()))
        acc = float(grp["exact_match"].mean())
        within_0_5 = float(grp["within_0_5"].mean() * 100.0)
        misclass = int(n_timesteps - grp["exact_match"].sum())
        
        mean_pred = float(grp["y_pred"].mean())
        mean_act = float(grp["y_true"].mean())
        min_pred = float(grp["y_pred"].min())
        max_pred = float(grp["y_pred"].max())

        scen_rows.append({
            "scenario_id": s_id,
            "scenario_family": fam,
            "timesteps": n_timesteps,
            "mae_deg_c": round(mae, 4),
            "rmse_deg_c": round(rmse, 4),
            "exact_accuracy": round(acc, 4),
            "pct_within_0_5_c": round(within_0_5, 2),
            "misclassified_timesteps": misclass,
            "mean_predicted_c": round(mean_pred, 3),
            "mean_actual_c": round(mean_act, 3),
            "min_predicted_c": round(min_pred, 1),
            "max_predicted_c": round(max_pred, 1),
        })

    return pd.DataFrame(scen_rows).sort_values("mae_deg_c")


def compute_temporal_prediction_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    meta_df: pd.DataFrame,
    timestep_seconds: float = 10.0,
) -> Dict[str, Any]:
    """
    Analyze the temporal dynamics and control stability of model predictions.
    Computes setpoint switches, dwell times, and potential chattering flags.
    """
    df = meta_df.copy()
    df["y_pred"] = np.asarray(y_pred, dtype=float)
    df["y_true"] = np.asarray(y_true, dtype=float)

    total_pred_changes = 0
    total_true_changes = 0
    max_jump = 0.0
    dwell_durations_sec = []
    scen_changes_list = []

    for s_id, grp in df.groupby("scenario_id"):
        # Sort by simulation time if available
        if "simulation_time_seconds" in grp.columns:
            grp = grp.sort_values("simulation_time_seconds")
            
        preds = grp["y_pred"].values
        trues = grp["y_true"].values
        
        # Count switches
        pred_diffs = np.abs(np.diff(preds))
        true_diffs = np.abs(np.diff(trues))
        
        n_p_changes = int(np.sum(pred_diffs > 0.001))
        n_t_changes = int(np.sum(true_diffs > 0.001))
        
        total_pred_changes += n_p_changes
        total_true_changes += n_t_changes
        scen_changes_list.append(n_p_changes)
        
        if len(pred_diffs) > 0 and np.max(pred_diffs) > max_jump:
            max_jump = float(np.max(pred_diffs))

        # Dwell time calculation
        curr_dwell = 1
        for i in range(1, len(preds)):
            if abs(preds[i] - preds[i-1]) < 0.001:
                curr_dwell += 1
            else:
                dwell_durations_sec.append(curr_dwell * timestep_seconds)
                curr_dwell = 1
        dwell_durations_sec.append(curr_dwell * timestep_seconds)

    n_scenarios = df["scenario_id"].nunique()
    mean_changes_per_scen = float(np.mean(scen_changes_list)) if scen_changes_list else 0.0
    mean_dwell = float(np.mean(dwell_durations_sec)) if dwell_durations_sec else 0.0
    
    # 7200 seconds = 2.0 hours per scenario
    mean_changes_per_hour = mean_changes_per_scen / 2.0

    return {
        "total_scenarios_evaluated": n_scenarios,
        "mean_prediction_changes_per_scenario": round(mean_changes_per_scen, 2),
        "mean_actual_changes_per_scenario": round(total_true_changes / max(n_scenarios, 1), 2),
        "mean_prediction_changes_per_hour": round(mean_changes_per_hour, 2),
        "max_single_step_prediction_jump_c": round(max_jump, 2),
        "average_prediction_dwell_seconds": round(mean_dwell, 1),
        "temporal_stability_status": "STABLE" if mean_changes_per_scen <= 15.0 and max_jump <= 1.0 else "OSCILLATING_WARNING",
    }
