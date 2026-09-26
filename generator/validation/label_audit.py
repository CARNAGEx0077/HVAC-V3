"""
Automated Label Distribution Audit System.

Performs comprehensive audits on HVAC optimal control actions:
- Overall label distributions and dominant value statistics
- Family-level target distributions (most critical)
- Individual scenario-level statistics and temporal dynamics
- Train/Validation/Test split shift analysis via Jensen-Shannon Divergence
- "Same Answer" low discrimination detection across scenario families
- Machine-readable label reason attribution
- Feature-target correlation analysis
- Generates JSON, CSV, and human-readable Markdown reports
"""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, wasserstein_distance


# Audit target definitions
TEMPERATURE_TARGETS = [
    "optimal_room_setpoint_c",
    "optimal_ac1_setpoint_c",
    "optimal_ac2_setpoint_c",
    "optimal_ac3_setpoint_c",
    "optimal_ac4_setpoint_c",
]

COOLING_LEVEL_TARGETS = [
    "optimal_ac1_cooling_level",
    "optimal_ac2_cooling_level",
    "optimal_ac3_cooling_level",
    "optimal_ac4_cooling_level",
]

RELATIONSHIP_FEATURES = [
    "room_average_temperature_c",
    "maximum_temperature_c",
    "temperature_difference_c",
    "total_computer_heat_watts",
    "total_occupancy_heat_watts",
    "total_heat_load_watts",
    "outdoor_temperature_c",
    "humidity_percent",
]


class LabelAuditor:
    """Performs deep statistical audit and distribution shift analysis on generated labels."""

    def __init__(
        self,
        dataset_dir: Path,
        dominance_warning_threshold: float = 0.80,
        dominance_fail_threshold: float = 0.90,
        jsd_warning_threshold: float = 0.10,
        jsd_fail_threshold: float = 0.25,
        low_discrimination_threshold: float = 0.02,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.raw_csv = self.dataset_dir / "raw" / "all_scenarios.csv"
        self.train_csv = self.dataset_dir / "train" / "train.csv"
        self.val_csv = self.dataset_dir / "validation" / "validation.csv"
        self.test_csv = self.dataset_dir / "test" / "test.csv"
        self.metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"

        self.dom_warn_thresh = dominance_warning_threshold
        self.dom_fail_thresh = dominance_fail_threshold
        self.jsd_warn_thresh = jsd_warning_threshold
        self.jsd_fail_thresh = jsd_fail_threshold
        self.low_discrim_thresh = low_discrimination_threshold

    def run_full_audit(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Execute all audit phases and compile structured report."""
        out_dir = Path(output_dir) if output_dir else (self.dataset_dir / "metadata")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load dataset metadata & rows
        raw_rows = self._read_csv(self.raw_csv)
        train_rows = self._read_csv(self.train_csv)
        val_rows = self._read_csv(self.val_csv)
        test_rows = self._read_csv(self.test_csv)

        metadata = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        total_scenarios = len(set(r["scenario_id"] for r in raw_rows))
        total_rows = len(raw_rows)
        master_seed = metadata.get("random_seed", "unknown")

        warnings: List[str] = []
        failures: List[str] = []

        # 2. Split statistics
        train_scenarios = sorted(list(set(r["scenario_id"] for r in train_rows)))
        val_scenarios = sorted(list(set(r["scenario_id"] for r in val_rows)))
        test_scenarios = sorted(list(set(r["scenario_id"] for r in test_rows)))

        split_summary = {
            "train": {"scenarios": len(train_scenarios), "rows": len(train_rows)},
            "validation": {"scenarios": len(val_scenarios), "rows": len(val_rows)},
            "test": {"scenarios": len(test_scenarios), "rows": len(test_rows)},
        }

        # 3. Family representation
        family_rows = defaultdict(list)
        scenario_family_map = {}
        for r in raw_rows:
            fam = r["scenario_family"]
            s_id = r["scenario_id"]
            family_rows[fam].append(r)
            scenario_family_map[s_id] = fam

        family_scenario_counts = Counter(scenario_family_map.values())
        family_distribution = {
            fam: {
                "scenarios": family_scenario_counts[fam],
                "rows": len(rows),
                "row_percentage": round(len(rows) / max(1, total_rows) * 100.0, 2),
            }
            for fam, rows in sorted(family_rows.items())
        }

        # 4. Overall target distribution
        overall_targets = {}
        for col in TEMPERATURE_TARGETS + COOLING_LEVEL_TARGETS:
            stats = self._calc_numeric_distribution(raw_rows, col)
            status = self._evaluate_dominance_status(stats["dominant_percentage"])
            stats["dominance_status"] = status
            if status == "WARNING":
                warnings.append(f"Overall {col}: dominant value {stats['dominant_value']} has share {stats['dominant_percentage']:.1f}% >= {self.dom_warn_thresh*100:.0f}%")
            elif status == "FAIL":
                failures.append(f"Overall {col}: dominant value {stats['dominant_value']} has share {stats['dominant_percentage']:.1f}% >= {self.dom_fail_thresh*100:.0f}%")
            overall_targets[col] = stats

        # 5. Family-level distribution (most critical requirement)
        family_targets = {}
        for fam, f_rows in sorted(family_rows.items()):
            family_targets[fam] = {}
            for col in ["optimal_room_setpoint_c"] + COOLING_LEVEL_TARGETS:
                stats = self._calc_numeric_distribution(f_rows, col)
                status = self._evaluate_dominance_status(stats["dominant_percentage"])
                stats["dominance_status"] = status
                if status == "WARNING":
                    warnings.append(f"{fam} {col}: dominant value {stats['dominant_value']} has share {stats['dominant_percentage']:.1f}%")
                elif status == "FAIL":
                    failures.append(f"{fam} {col}: dominant value {stats['dominant_value']} has share {stats['dominant_percentage']:.1f}%")
                family_targets[fam][col] = stats

        # 6. Scenario-level distribution & Temporal Dynamics
        scenario_rows = defaultdict(list)
        for r in raw_rows:
            scenario_rows[r["scenario_id"]].append(r)

        scenario_stats = []
        temporal_behavior = []

        for s_id, s_rows in sorted(scenario_rows.items()):
            fam = s_rows[0]["scenario_family"]
            s_setpoints = [float(r["optimal_room_setpoint_c"]) for r in s_rows]
            sp_counts = Counter(s_setpoints)
            dom_val, dom_cnt = sp_counts.most_common(1)[0]
            dom_pct = round((dom_cnt / len(s_setpoints)) * 100.0, 2)

            scen_stat = {
                "scenario_id": s_id,
                "scenario_family": fam,
                "total_timesteps": len(s_setpoints),
                "dominant_setpoint": dom_val,
                "dominant_percentage": dom_pct,
                "mean_setpoint": round(float(np.mean(s_setpoints)), 3),
                "std_setpoint": round(float(np.std(s_setpoints)), 3),
                "min_setpoint": round(float(np.min(s_setpoints)), 1),
                "max_setpoint": round(float(np.max(s_setpoints)), 1),
                "unique_labels": len(sp_counts),
            }
            scenario_stats.append(scen_stat)

            # Temporal dynamics
            label_changes = 0
            longest_run = 1
            curr_run = 1
            for i in range(1, len(s_setpoints)):
                if s_setpoints[i] != s_setpoints[i - 1]:
                    label_changes += 1
                    longest_run = max(longest_run, curr_run)
                    curr_run = 1
                else:
                    curr_run += 1
            longest_run = max(longest_run, curr_run)

            duration_hours = (len(s_setpoints) * 10.0) / 3600.0
            changes_per_hour = round(label_changes / max(0.1, duration_hours), 2)
            longest_period_seconds = longest_run * 10

            temporal_behavior.append({
                "scenario_id": s_id,
                "scenario_family": fam,
                "label_changes": label_changes,
                "changes_per_hour": changes_per_hour,
                "longest_period_seconds": longest_period_seconds,
                "unique_labels": len(sp_counts),
                "first_label": s_setpoints[0],
                "final_label": s_setpoints[-1],
            })

        # 7. Train / Validation / Test Distribution Comparison
        split_distributions = {
            "train": self._calc_numeric_distribution(train_rows, "optimal_room_setpoint_c"),
            "validation": self._calc_numeric_distribution(val_rows, "optimal_room_setpoint_c"),
            "test": self._calc_numeric_distribution(test_rows, "optimal_room_setpoint_c"),
        }

        # 8. Distribution Shift Check via Jensen-Shannon Divergence
        train_probs = self._extract_probabilities(split_distributions["train"].get("frequency", {}))
        val_probs = self._extract_probabilities(split_distributions["validation"].get("frequency", {}), align_keys=train_probs.keys())
        test_probs = self._extract_probabilities(split_distributions["test"].get("frequency", {}), align_keys=train_probs.keys())

        # Align keys
        all_keys = sorted(list(set(train_probs.keys()) | set(val_probs.keys()) | set(test_probs.keys())))
        p_vec = np.array([train_probs.get(k, 0.0) for k in all_keys])
        q_val_vec = np.array([val_probs.get(k, 0.0) for k in all_keys])
        q_test_vec = np.array([test_probs.get(k, 0.0) for k in all_keys])

        if len(p_vec) > 0 and np.sum(p_vec) > 0 and np.sum(q_val_vec) > 0:
            jsd_val = float(jensenshannon(p_vec, q_val_vec, base=2) ** 2)
            js_dist_val = math.sqrt(max(0.0, jsd_val))
        else:
            jsd_val = 0.0
            js_dist_val = 0.0

        if len(p_vec) > 0 and np.sum(p_vec) > 0 and np.sum(q_test_vec) > 0:
            jsd_test = float(jensenshannon(p_vec, q_test_vec, base=2) ** 2)
            js_dist_test = math.sqrt(max(0.0, jsd_test))
        else:
            jsd_test = 0.0
            js_dist_test = 0.0

        train_vals = [float(r["optimal_room_setpoint_c"]) for r in train_rows if "optimal_room_setpoint_c" in r and r["optimal_room_setpoint_c"] != ""]
        val_vals = [float(r["optimal_room_setpoint_c"]) for r in val_rows if "optimal_room_setpoint_c" in r and r["optimal_room_setpoint_c"] != ""]
        test_vals = [float(r["optimal_room_setpoint_c"]) for r in test_rows if "optimal_room_setpoint_c" in r and r["optimal_room_setpoint_c"] != ""]

        wass_val = float(wasserstein_distance(train_vals, val_vals)) if (train_vals and val_vals) else 0.0
        wass_test = float(wasserstein_distance(train_vals, test_vals)) if (train_vals and test_vals) else 0.0

        val_shift_status = "PASS" if js_dist_val < self.jsd_warn_thresh else ("WARNING" if js_dist_val < self.jsd_fail_thresh else "FAIL")
        test_shift_status = "PASS" if js_dist_test < self.jsd_warn_thresh else ("WARNING" if js_dist_test < self.jsd_fail_thresh else "FAIL")

        if val_shift_status != "PASS":
            warnings.append(f"Train vs Validation shift: JS Distance={js_dist_val:.4f} ({val_shift_status})")
        if test_shift_status != "PASS":
            warnings.append(f"Train vs Test shift: JS Distance={js_dist_test:.4f} ({test_shift_status})")

        distribution_shift = {
            "train_vs_val": {
                "jensen_shannon_divergence": round(jsd_val, 5),
                "jensen_shannon_distance": round(js_dist_val, 4),
                "wasserstein_distance": round(wass_val, 4),
                "status": val_shift_status,
            },
            "train_vs_test": {
                "jensen_shannon_divergence": round(jsd_test, 5),
                "jensen_shannon_distance": round(js_dist_test, 4),
                "wasserstein_distance": round(wass_test, 4),
                "status": test_shift_status,
            },
        }

        # 9. Family Separation & "Same Answer" Detection
        family_separation_matrix, low_discrim_warnings = self._calc_family_separation(family_targets)
        warnings.extend(low_discrim_warnings)

        # 10. Label Reason Audit by Family
        label_reasons_by_family = {}
        all_reasons = sorted(list(set(r["label_reason"] for r in raw_rows)))
        for fam, f_rows in sorted(family_rows.items()):
            cnts = Counter(r["label_reason"] for r in f_rows)
            f_len = len(f_rows)
            label_reasons_by_family[fam] = {
                rsn: {
                    "count": cnts[rsn],
                    "percentage": round(cnts[rsn] / max(1, f_len) * 100.0, 2),
                }
                for rsn in all_reasons
            }

        # 11. Correlation / Relationship Audit
        relationships = self._calc_feature_relationships(raw_rows)

        # Determine overall audit status
        if failures:
            final_status = "FAIL"
        elif warnings:
            final_status = "WARNING"
        else:
            final_status = "PASS"

        report = {
            "audit_version": "1.0.0",
            "dataset_dir": str(self.dataset_dir),
            "total_scenarios": total_scenarios,
            "total_rows": total_rows,
            "master_seed": master_seed,
            "split_summary": split_summary,
            "family_distribution": family_distribution,
            "overall_target_distribution": overall_targets,
            "family_target_distribution": family_targets,
            "scenario_statistics": scenario_stats,
            "temporal_behavior": temporal_behavior,
            "split_distributions": split_distributions,
            "distribution_shift": distribution_shift,
            "family_separation_matrix": family_separation_matrix,
            "label_reasons_by_family": label_reasons_by_family,
            "feature_relationships": relationships,
            "warnings": warnings,
            "failures": failures,
            "final_audit_status": final_status,
        }

        # Save artifacts
        self._save_json_report(report, out_dir / "label_audit_report.json")
        self._save_csv_summary(report, out_dir / "label_audit_report.csv")
        self._save_markdown_report(report, out_dir / "label_audit_report.md")

        return report

    def _read_csv(self, path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        with open(path, mode="r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _calc_numeric_distribution(self, rows: List[Dict], col: str) -> Dict[str, Any]:
        """Calculates discrete and continuous statistical summary for a column."""
        vals = [float(r[col]) for r in rows if col in r and r[col] != ""]
        if not vals:
            return {
                "count": 0,
                "unique_count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "dominant_value": None,
                "dominant_percentage": 0.0,
                "frequency": {},
            }

        counts = Counter(vals)
        dom_val, dom_cnt = counts.most_common(1)[0]
        dom_pct = round((dom_cnt / len(vals)) * 100.0, 2)

        freq_breakdown = {
            str(k): {
                "count": v,
                "percentage": round((v / len(vals)) * 100.0, 2),
            }
            for k, v in sorted(counts.items())
        }

        return {
            "count": len(vals),
            "unique_count": len(counts),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "dominant_value": dom_val,
            "dominant_percentage": dom_pct,
            "frequency": freq_breakdown,
        }

    def _extract_probabilities(self, freq_dict: Optional[Dict[str, Any]], align_keys: Optional[Any] = None) -> Dict[float, float]:
        """Convert frequency dictionary to normalized probability dict."""
        probs = {}
        if freq_dict:
            for k, info in freq_dict.items():
                probs[float(k)] = info["percentage"] / 100.0
        if align_keys:
            for k in align_keys:
                if k not in probs:
                    probs[k] = 0.0
        return probs

    def _evaluate_dominance_status(self, dom_pct: float) -> str:
        """Classify dominance percentage into PASS, WARNING, or FAIL."""
        if dom_pct >= self.dom_fail_thresh * 100.0:
            return "FAIL"
        elif dom_pct >= self.dom_warn_thresh * 100.0:
            return "WARNING"
        return "PASS"

    def _calc_family_separation(
        self, family_targets: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
        """Calculates pairwise Jensen-Shannon distance matrix and checks for low discrimination."""
        families = sorted(list(family_targets.keys()))
        matrix = {f1: {} for f1 in families}
        warnings = []

        # Gather all support setpoints
        all_sps = set()
        for fam in families:
            freq = family_targets[fam]["optimal_room_setpoint_c"]["frequency"]
            for sp_str in freq:
                all_sps.add(float(sp_str))
        sorted_sps = sorted(list(all_sps))

        for i, f1 in enumerate(families):
            p1_dict = self._extract_probabilities(family_targets[f1]["optimal_room_setpoint_c"]["frequency"])
            p1 = np.array([p1_dict.get(sp, 0.0) for sp in sorted_sps])
            for j, f2 in enumerate(families):
                if i == j:
                    matrix[f1][f2] = 0.0
                    continue
                p2_dict = self._extract_probabilities(family_targets[f2]["optimal_room_setpoint_c"]["frequency"])
                p2 = np.array([p2_dict.get(sp, 0.0) for sp in sorted_sps])

                js_dist = float(jensenshannon(p1, p2, base=2))
                matrix[f1][f2] = round(js_dist, 4)

                # Check "Same Answer" problem (Section 12)
                if i < j and js_dist < self.low_discrim_thresh:
                    warnings.append(
                        f"POSSIBLE LOW LABEL DISCRIMINATION: {f1} and {f2} have nearly identical target distribution (JS Distance={js_dist:.4f} < {self.low_discrim_thresh})"
                    )

        return matrix, warnings

    def _calc_feature_relationships(self, rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Compute Pearson correlation between optimal setpoint and input features."""
        setpoints = [float(r["optimal_room_setpoint_c"]) for r in rows]
        relationships = {}

        for feat in RELATIONSHIP_FEATURES:
            feat_vals = [float(r[feat]) for r in rows if feat in r and r[feat] != ""]
            if len(feat_vals) == len(setpoints) and len(setpoints) > 1:
                r_val, p_val = pearsonr(feat_vals, setpoints)
                relationships[feat] = {
                    "pearson_r": round(float(r_val), 4),
                    "p_value": float(round(p_val, 6)),
                }

        return relationships

    def _save_json_report(self, report: Dict[str, Any], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _save_csv_summary(self, report: Dict[str, Any], path: Path):
        """Export tabular summary CSV for quick ingestion."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Section", "Item", "Metric", "Value"])

            # Split summary
            for split, info in report["split_summary"].items():
                writer.writerow(["Split", split, "scenarios", info["scenarios"]])
                writer.writerow(["Split", split, "rows", info["rows"]])

            # Family summary
            for fam, info in report["family_distribution"].items():
                writer.writerow(["Family", fam, "scenarios", info["scenarios"]])
                writer.writerow(["Family", fam, "rows", info["rows"]])
                writer.writerow(["Family", fam, "row_percentage", info["row_percentage"]])

            # Target statistics
            for target, s in report["overall_target_distribution"].items():
                writer.writerow(["OverallTarget", target, "mean", s.get("mean")])
                writer.writerow(["OverallTarget", target, "std", s.get("std")])
                writer.writerow(["OverallTarget", target, "dominant_value", s.get("dominant_value")])
                writer.writerow(["OverallTarget", target, "dominant_percentage", s.get("dominant_percentage")])

            # Distribution shifts
            for cmp_name, s in report["distribution_shift"].items():
                writer.writerow(["DistributionShift", cmp_name, "js_distance", s.get("jensen_shannon_distance")])
                writer.writerow(["DistributionShift", cmp_name, "status", s.get("status")])

            # Correlation relationships
            for feat, info in report["feature_relationships"].items():
                writer.writerow(["Correlation", feat, "pearson_r", info.get("pearson_r")])

            writer.writerow(["Audit", "FinalStatus", "status", report["final_audit_status"]])

    def _save_markdown_report(self, report: Dict[str, Any], path: Path):
        """Build Human-Readable Markdown report conforming strictly to Section 17 structure."""
        lines = []
        lines.append("# HVEAC Dataset Label Audit\n")

        # ## Dataset
        lines.append("## Dataset")
        lines.append(f"- **Total Scenarios:** {report['total_scenarios']}")
        lines.append(f"- **Total Simulation Rows:** {report['total_rows']:,}")
        lines.append(f"- **Master Seed:** {report['master_seed']}")
        lines.append(f"- **Audit Status:** **{report['final_audit_status']}**\n")

        # ## Split
        lines.append("## Split")
        lines.append("| Split | Scenarios | Rows | Percentage |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for s_name in ["train", "validation", "test"]:
            info = report["split_summary"][s_name]
            pct = round(info["rows"] / max(1, report["total_rows"]) * 100.0, 1)
            lines.append(f"| **{s_name.capitalize()}** | {info['scenarios']} | {info['rows']:,} | {pct:.1f}% |")
        lines.append("")

        # ## Family Distribution
        lines.append("## Family Distribution")
        lines.append("| Scenario Family | Scenarios | Rows | Share |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for fam, info in sorted(report["family_distribution"].items()):
            lines.append(f"| `{fam}` | {info['scenarios']} | {info['rows']:,} | {info['row_percentage']:.1f}% |")
        lines.append("")

        # ## Overall Target Distribution
        lines.append("## Overall Target Distribution")
        lines.append("| Target Variable | Min | Max | Mean | Std | Dominant Value | Dominant Share | Status |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for col, s in sorted(report["overall_target_distribution"].items()):
            lines.append(
                f"| `{col}` | {s['min']} | {s['max']} | {s['mean']:.2f} | {s['std']:.2f} | "
                f"{s['dominant_value']} | {s['dominant_percentage']:.1f}% | `{s['dominance_status']}` |"
            )
        lines.append("")

        # Breakdown of optimal_room_setpoint_c
        opt_freq = report["overall_target_distribution"]["optimal_room_setpoint_c"]["frequency"]
        lines.append("### Optimal Room Setpoint Breakdown")
        lines.append("| Candidate Setpoint | Timestep Count | Percentage |")
        lines.append("| :--- | :--- | :--- |")
        for sp, f_info in sorted(opt_freq.items(), key=lambda x: float(x[0])):
            lines.append(f"| {float(sp):.1f}°C | {f_info['count']:,} | {f_info['percentage']:.2f}% |")
        lines.append("")

        # ## Per-Family Target Distribution
        lines.append("## Per-Family Target Distribution")
        for fam, t_dict in sorted(report["family_target_distribution"].items()):
            lines.append(f"### {fam}")
            sp_info = t_dict["optimal_room_setpoint_c"]
            lines.append(f"- **Mean Optimal Setpoint:** {sp_info['mean']:.2f}°C (Std: {sp_info['std']:.2f}°C)")
            lines.append(f"- **Dominant Setpoint:** {sp_info['dominant_value']}°C ({sp_info['dominant_percentage']:.1f}%)")
            lines.append("\n| Setpoint | Count | Share |")
            lines.append("| :--- | :--- | :--- |")
            for sp, f_info in sorted(sp_info["frequency"].items(), key=lambda x: float(x[0])):
                lines.append(f"| {float(sp):.1f}°C | {f_info['count']:,} | {f_info['percentage']:.1f}% |")

            lines.append("\n*Primary AC Cooling Level Means:*")
            ac_means = [f"`AC-{i}`: {t_dict[f'optimal_ac{i}_cooling_level']['mean']:.3f}" for i in range(1, 5)]
            lines.append("- " + " | ".join(ac_means) + "\n")

        # ## Per-Scenario Target Statistics
        lines.append("## Per-Scenario Target Statistics")
        lines.append("| Scenario ID | Family | Dominant SP | Dom % | Mean SP | Std | Min | Max | Unique |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for sc in report["scenario_statistics"][:25]:  # sample representative first 25 scenarios in markdown
            lines.append(
                f"| `{sc['scenario_id']}` | {sc['scenario_family'][:8]} | {sc['dominant_setpoint']}°C | "
                f"{sc['dominant_percentage']:.1f}% | {sc['mean_setpoint']:.2f} | {sc['std_setpoint']:.2f} | "
                f"{sc['min_setpoint']} | {sc['max_setpoint']} | {sc['unique_labels']} |"
            )
        lines.append(f"\n*(Showing 25 of {len(report['scenario_statistics'])} scenarios. Complete list in label_audit_report.json)*\n")

        # ## Train/Validation/Test Distribution
        lines.append("## Train/Validation/Test Distribution")
        lines.append("| Split | Mean SP | Std | Min | Max | Dominant Value | Dominant Share |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for s_name in ["train", "validation", "test"]:
            s = report["split_distributions"][s_name]
            lines.append(
                f"| **{s_name.capitalize()}** | {s['mean']:.2f}°C | {s['std']:.2f} | {s['min']}°C | {s['max']}°C | "
                f"{s['dominant_value']}°C | {s['dominant_percentage']:.1f}% |"
            )
        lines.append("")

        # ## Distribution Shift
        lines.append("## Distribution Shift")
        lines.append("| Comparison | Jensen-Shannon Distance | Wasserstein Distance | Status |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for cmp_name, info in report["distribution_shift"].items():
            lines.append(f"| **{cmp_name}** | {info['jensen_shannon_distance']:.4f} | {info['wasserstein_distance']:.4f} | `{info['status']}` |")
        lines.append("")

        # ## Temporal Label Behavior
        lines.append("## Temporal Label Behavior")
        tb = report["temporal_behavior"]
        mean_changes = float(np.mean([t["label_changes"] for t in tb]))
        mean_changes_hr = float(np.mean([t["changes_per_hour"] for t in tb]))
        mean_longest_sec = float(np.mean([t["longest_period_seconds"] for t in tb]))

        lines.append(f"- **Mean Label Adjustments per Scenario:** {mean_changes:.2f}")
        lines.append(f"- **Mean Label Adjustments per Hour:** {mean_changes_hr:.2f} changes/hr")
        lines.append(f"- **Mean Longest Unchanged Period:** {mean_longest_sec:.1f}s ({mean_longest_sec/60.0:.1f} minutes)\n")
        lines.append("| Scenario ID | Family | Label Changes | Changes/hr | Longest Run (s) | First Label | Final Label |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for t in tb[:15]:
            lines.append(
                f"| `{t['scenario_id']}` | {t['scenario_family'][:8]} | {t['label_changes']} | {t['changes_per_hour']} | "
                f"{t['longest_period_seconds']}s | {t['first_label']}°C | {t['final_label']}°C |"
            )
        lines.append("")

        # ## Label Reasons
        lines.append("## Label Reasons")
        lines.append("| Family | Dominant Reason | Primary Reasons Breakdown |")
        lines.append("| :--- | :--- | :--- |")
        for fam, r_dict in sorted(report["label_reasons_by_family"].items()):
            sorted_rsns = sorted(r_dict.items(), key=lambda x: x[1]["count"], reverse=True)
            top_rsn = sorted_rsns[0][0]
            summary_str = ", ".join([f"{r}: {info['percentage']:.1f}%" for r, info in sorted_rsns if info['count'] > 0][:3])
            lines.append(f"| `{fam}` | **{top_rsn}** | {summary_str} |")
        lines.append("")

        # ## Family Separation
        lines.append("## Family Separation")
        lines.append("Pairwise Jensen-Shannon Distance Matrix across 5 Scenario Families:")
        fam_keys = sorted(list(report["family_separation_matrix"].keys()))
        short_names = [f"F{i+1}" for i in range(len(fam_keys))]

        lines.append("\n| Family | " + " | ".join(short_names) + " |")
        lines.append("| :--- | " + " | ".join([":---"] * len(fam_keys)) + " |")
        for i, f1 in enumerate(fam_keys):
            row_vals = [f"{report['family_separation_matrix'][f1][f2]:.4f}" for f2 in fam_keys]
            lines.append(f"| **F{i+1} ({f1[:8]})** | " + " | ".join(row_vals) + " |")
        lines.append("")

        # ## Relationship Audit
        lines.append("## Relationship Audit")
        lines.append("| Current State Feature | Pearson Correlation (r) | p-value | Responsiveness Note |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for feat, info in report["feature_relationships"].items():
            r_val = info["pearson_r"]
            note = "Strong response" if abs(r_val) > 0.3 else ("Moderate response" if abs(r_val) > 0.1 else "Mild/orthogonal response")
            lines.append(f"| `{feat}` | {r_val:+.4f} | {info['p_value']:.4e} | {note} |")
        lines.append("\n*Note: Correlation measures systematic response to physical drivers, not causality.*\n")

        # ## Warnings
        lines.append("## Warnings")
        if report["warnings"]:
            for w in report["warnings"]:
                lines.append(f"- [WARNING] {w}")
        else:
            lines.append("- None. No statistical dominance, shift, or low-discrimination warnings detected.")
        lines.append("")

        # ## Final Audit Status
        lines.append("## Final Audit Status")
        status_color = "**PASS**" if report["final_audit_status"] == "PASS" else f"**{report['final_audit_status']}**"
        lines.append(f"### {status_color}")
        lines.append(f"- Failures: {len(report['failures'])}")
        lines.append(f"- Warnings: {len(report['warnings'])}")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
