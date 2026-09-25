"""
Visual reporting and engineering figure generation for HVEAC Brain v1.

Generates:
1. Confusion matrix heatmap
2. Actual vs predicted setpoint distribution
3. Per-class performance (Precision, Recall, F1)
4. Error distribution histogram
5. Per-family MAE comparison
6. Per-scenario MAE distribution
7. Top physical feature importances
8. Time-series trajectory: actual vs predicted setpoint over time for representative scenarios
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.config import TARGET_CLASSES


def generate_all_plots(
    eval_results: Any,
    feature_importance_df: Optional[pd.DataFrame],
    meta_df: pd.DataFrame,
    output_dir: Path,
):
    """Generate the complete suite of 8 engineering plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Confusion Matrix
    _plot_confusion_matrix(eval_results.confusion_matrix_df, output_dir / "confusion_matrix.png")

    # 2. Actual vs Predicted Distribution
    _plot_actual_vs_predicted_dist(eval_results, output_dir / "actual_vs_predicted_distribution.png")

    # 3. Per-Class Performance
    _plot_per_class_metrics(eval_results.per_class_metrics, output_dir / "per_class_performance.png")

    # 4. Error Distribution
    _plot_error_distribution(eval_results, output_dir / "error_distribution.png")

    # 5. Per-Family MAE
    _plot_per_family_mae(eval_results.family_metrics_df, output_dir / "per_family_mae.png")

    # 6. Per-Scenario MAE
    _plot_per_scenario_mae(eval_results.scenario_metrics_df, output_dir / "per_scenario_mae.png")

    # 7. Top Feature Importance
    if feature_importance_df is not None:
        _plot_feature_importance(feature_importance_df, output_dir / "feature_importance.png")

    # 8. Time-Series Trajectory Sample
    _plot_representative_trajectories(eval_results, meta_df, output_dir / "temporal_trajectory_sample.png")


def _plot_confusion_matrix(cm_df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    matrix = cm_df.values
    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    
    classes = [f"{c:.1f}°C" for c in TARGET_CLASSES]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel("Predicted Room Setpoint (°C)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Actual Optimal Setpoint (°C)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("HVEAC Brain v1 — Confusion Matrix", fontsize=12, fontweight="bold", pad=12)

    # Annotate counts and percentages
    total = np.sum(matrix)
    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            pct = (val / total) * 100.0 if total > 0 else 0
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_actual_vs_predicted_dist(eval_res: Any, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    classes = [f"{c:.1f}°C" for c in TARGET_CLASSES]
    
    actual_counts = [eval_res.per_class_metrics.get(c, {}).get("support", 0) for c in classes]
    pred_counts = [eval_res.per_class_metrics.get(c, {}).get("predicted_count", 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width/2, actual_counts, width, label="Actual Ground Truth", color="#2c3e50", alpha=0.9)
    ax.bar(x + width/2, pred_counts, width, label="Brain v1 Predicted", color="#2980b9", alpha=0.9)

    ax.set_xlabel("Room Setpoint Category", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observation Count", fontsize=11, fontweight="bold")
    ax.set_title("Actual vs. Predicted Room Setpoint Distribution", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_per_class_metrics(per_class_metrics: Dict[str, Any], save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    classes = [f"{c:.1f}°C" for c in TARGET_CLASSES]
    
    precision = [per_class_metrics.get(c, {}).get("precision", 0) for c in classes]
    recall = [per_class_metrics.get(c, {}).get("recall", 0) for c in classes]
    f1 = [per_class_metrics.get(c, {}).get("f1", 0) for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    ax.bar(x - width, precision, width, label="Precision", color="#3498db")
    ax.bar(x, recall, width, label="Recall", color="#2ecc71")
    ax.bar(x + width, f1, width, label="F1 Score", color="#e74c3c")

    ax.set_xlabel("Target Setpoint Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Class Classification Metrics (Precision, Recall, F1)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_error_distribution(eval_res: Any, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    # y_true - y_pred
    y_true_f = np.array([float(eval_res.per_class_metrics[f"{float(p):.1f}°C"]["setpoint_c"]) for p in eval_res.predictions])
    # Compute error directly from predictions
    errors = eval_res.predictions.astype(float) - y_true_f

    counts, bins, patches = ax.hist(errors, bins=np.arange(-2.25, 2.75, 0.5), color="#34495e", edgecolor="white", rwidth=0.85)

    ax.set_xlabel("Prediction Error (Predicted - Actual Setpoint in °C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Frequency (Rows)", fontsize=11, fontweight="bold")
    ax.set_title(f"Prediction Error Distribution (MAE = {eval_res.physical_metrics['mae_deg_c']:.3f}°C)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    # Annotate percentages
    total = len(errors)
    for count, patch in zip(counts, patches):
        if count > 0:
            pct = (count / total) * 100.0
            ax.text(patch.get_x() + patch.get_width() / 2, count + (total * 0.01), f"{pct:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_per_family_mae(family_df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    
    families = family_df["scenario_family"].tolist()
    maes = family_df["mae_deg_c"].tolist()
    accuracies = family_df["pct_within_0_5_c"].tolist()

    x = np.arange(len(families))
    bars = ax.bar(x, maes, color="#16a085", width=0.55, edgecolor="black", alpha=0.85)

    ax.set_xlabel("Scenario Family", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error (°C)", fontsize=11, fontweight="bold")
    ax.set_title("Generalization Error Across Scenario Families (MAE in °C)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha="right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with MAE and ±0.5°C accuracy
    for bar, mae, acc in zip(bars, maes, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{mae:.3f}°C\n(±0.5°: {acc:.0f}%)", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_per_scenario_mae(scen_df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    sorted_df = scen_df.sort_values("mae_deg_c", ascending=True).reset_index(drop=True)
    
    scenarios = [s.replace("_FAMILY_", " F") for s in sorted_df["scenario_id"]]
    maes = sorted_df["mae_deg_c"]

    colors = ["#27ae60" if m <= 0.10 else "#f39c12" if m <= 0.20 else "#c0392b" for m in maes]
    ax.barh(np.arange(len(scenarios)), maes, color=colors, height=0.7)

    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_xlabel("Mean Absolute Error (°C)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Scenario ID", fontsize=11, fontweight="bold")
    ax.set_title(f"Scenario-Level Generalization Performance (Median MAE = {np.median(maes):.3f}°C)", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Vertical reference line for median
    ax.axvline(np.median(maes), color="black", linestyle=":", label=f"Median MAE ({np.median(maes):.3f}°C)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_feature_importance(feat_df: pd.DataFrame, save_path: Path, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    top_df = feat_df.head(top_n).sort_values("importance_score", ascending=True)

    y_pos = np.arange(len(top_df))
    ax.barh(y_pos, top_df["importance_share_pct"], color="#2980b9", edgecolor="black", height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["feature_name"], fontsize=9)
    ax.set_xlabel("Relative Importance Contribution (%)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {top_n} Physical Features Influencing Optimal Room Setpoint", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for i, v in enumerate(top_df["importance_share_pct"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def _plot_representative_trajectories(eval_res: Any, meta_df: pd.DataFrame, save_path: Path):
    """Plot temporal actual vs predicted setpoint trajectory for 3 diverse scenarios."""
    df = meta_df.copy()
    df["y_true"] = np.asarray(meta_df["optimal_room_setpoint_c"] if "optimal_room_setpoint_c" in meta_df else [25.5]*len(meta_df), dtype=float)
    df["y_pred"] = eval_res.predictions.astype(float)
    
    scenarios = df["scenario_id"].unique()
    sample_scens = scenarios[:min(3, len(scenarios))]

    fig, axes = plt.subplots(len(sample_scens), 1, figsize=(10, 3 * len(sample_scens)), sharex=True, dpi=300)
    if len(sample_scens) == 1:
        axes = [axes]

    for ax, s_id in zip(axes, sample_scens):
        scen_data = df[df["scenario_id"] == s_id].sort_values("simulation_time_seconds")
        t_minutes = scen_data["simulation_time_seconds"] / 60.0
        
        ax.step(t_minutes, scen_data["y_true"], label="Actual Optimal (Simulator)", color="#2c3e50", linewidth=2.0, where="post")
        ax.step(t_minutes, scen_data["y_pred"], label="HVEAC Brain v1 Predicted", color="#e74c3c", linewidth=1.5, linestyle="--", where="post")
        
        fam = scen_data["scenario_family"].iloc[0] if "scenario_family" in scen_data.columns else ""
        ax.set_title(f"Trajectory: {s_id} ({fam})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Setpoint (°C)", fontsize=9, fontweight="bold")
        ax.set_ylim(24.0, 27.0)
        ax.set_yticks([24.5, 25.0, 25.5, 26.0, 26.5])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Simulation Elapsed Time (Minutes)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
