"""
Dataset Diversity and Feature Coverage Analyzer.

Analyzes distributions across occupancy, computer loads, climate, thermal states,
and optimal setpoints. Flags excessive statistical concentration as mandated by Section 28.
"""

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


class DatasetCoverageAnalyzer:
    """Computes distribution percentiles and concentration warnings for the dataset."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.raw_csv = self.dataset_dir / "raw" / "all_scenarios.csv"

    def analyze_coverage(self) -> Dict[str, Any]:
        """Perform comprehensive statistical distribution sweep across dataset rows."""
        occupancy_vals = []
        cpu_vals = []
        gpu_vals = []
        outdoor_temps = []
        humidities = []
        room_temps = []
        heat_loads = []
        optimal_setpoints = []
        family_counts = Counter()
        label_reasons = Counter()

        total_rows = 0
        with open(self.raw_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                occupancy_vals.append(int(row["occupancy_total"]))
                outdoor_temps.append(float(row["outdoor_temperature_c"]))
                humidities.append(float(row["humidity_percent"]))
                room_temps.append(float(row["room_average_temperature_c"]))
                heat_loads.append(float(row["total_heat_load_watts"]))
                opt_sp = float(row["optimal_room_setpoint_c"])
                optimal_setpoints.append(opt_sp)

                family_counts[row["scenario_family"]] += 1
                label_reasons[row["label_reason"]] += 1

                # Sample CPU and GPU from first and last computers for distribution metrics
                cpu_vals.append(float(row["computer_1_cpu"]))
                cpu_vals.append(float(row["computer_5_cpu"]))
                gpu_vals.append(float(row["computer_1_gpu"]))
                gpu_vals.append(float(row["computer_5_gpu"]))

        # Check for excessive concentration (> 80% in any 0.5-1.0 deg bin)
        warnings = []
        sp_counter = Counter(optimal_setpoints)
        for sp, count in sp_counter.items():
            pct = (count / total_rows) * 100.0
            if pct > 80.0:
                warnings.append(
                    f"Excessive concentration alert: optimal setpoint {sp}°C represents {pct:.1f}% of total timesteps."
                )

        # Multi-bin check: e.g. 24.0-24.5 combined
        count_24_to_245 = sp_counter[24.0] + sp_counter[24.5]
        pct_24_to_245 = (count_24_to_245 / total_rows) * 100.0
        if pct_24_to_245 > 80.0:
            warnings.append(
                f"Concentration review recommended: optimal setpoint distribution is {pct_24_to_245:.1f}% between 24.0-24.5°C"
            )

        report = {
            "total_rows_analyzed": total_rows,
            "scenario_family_distribution": dict(family_counts),
            "label_reason_distribution": dict(label_reasons),
            "metrics_summary": {
                "occupancy": self._calc_percentiles(occupancy_vals),
                "cpu_utilization_percent": self._calc_percentiles(cpu_vals),
                "gpu_utilization_percent": self._calc_percentiles(gpu_vals),
                "outdoor_temperature_c": self._calc_percentiles(outdoor_temps),
                "humidity_percent": self._calc_percentiles(humidities),
                "room_average_temperature_c": self._calc_percentiles(room_temps),
                "total_heat_load_watts": self._calc_percentiles(heat_loads),
                "optimal_room_setpoint_c": self._calc_percentiles(optimal_setpoints),
            },
            "optimal_setpoint_breakdown": {
                str(k): {"count": v, "percent": round(v / total_rows * 100.0, 2)}
                for k, v in sorted(sp_counter.items())
            },
            "concentration_warnings": warnings,
        }

        # Save to metadata/coverage_report.json
        coverage_path = self.dataset_dir / "metadata" / "coverage_report.json"
        with open(coverage_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _calc_percentiles(self, values: List[float]) -> Dict[str, float]:
        arr = np.array(values)
        return {
            "min": round(float(np.min(arr)), 2),
            "p25": round(float(np.percentile(arr, 25)), 2),
            "median": round(float(np.median(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "p75": round(float(np.percentile(arr, 75)), 2),
            "max": round(float(np.max(arr)), 2),
            "std": round(float(np.std(arr)), 2),
        }
