"""
Dataset Summary Statistics and Metrics Collector.

Accumulates streaming metrics to produce high-level summary printouts
as required by Section 38.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


class OnlineStatsAccumulator:
    """Computes streaming min, max, mean, and distributions without storing raw arrays."""

    def __init__(self):
        self.count = 0
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.sum_val = 0.0

    def update(self, val: float):
        self.count += 1
        self.sum_val += val
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

    @property
    def mean(self) -> float:
        return self.sum_val / self.count if self.count > 0 else 0.0

    def as_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0}
        return {
            "min": round(self.min_val, 2),
            "max": round(self.max_val, 2),
            "mean": round(self.mean, 2),
        }


class DatasetStatisticsCollector:
    """Tracks global statistics across all generated scenarios and simulation rows."""

    def __init__(self):
        self.total_scenarios = 0
        self.total_rows = 0

        self.family_counts: Dict[str, int] = Counter()
        self.label_reason_counts: Dict[str, int] = Counter()
        self.optimal_setpoint_distribution: Dict[float, int] = Counter()

        self.temp_stats = OnlineStatsAccumulator()
        self.occupancy_stats = OnlineStatsAccumulator()
        self.computer_heat_stats = OnlineStatsAccumulator()
        self.optimal_setpoint_stats = OnlineStatsAccumulator()

        self.comfort_violations = 0
        self.invalid_rows = 0

    def update_with_row(self, row: dict):
        """Update metrics with a single simulated row."""
        self.total_rows += 1

        # Family & Reason
        self.family_counts[row["scenario_family"]] += 1
        self.label_reason_counts[row["label_reason"]] += 1

        # Temperatures
        avg_temp = row["room_average_temperature_c"]
        self.temp_stats.update(avg_temp)

        # Occupancy
        occ = row["occupancy_total"]
        self.occupancy_stats.update(occ)

        # Computer Heat
        comp_heat = row["total_computer_heat_watts"]
        self.computer_heat_stats.update(comp_heat)

        # Optimal Setpoint
        opt_sp = row["optimal_room_setpoint_c"]
        self.optimal_setpoint_stats.update(opt_sp)
        self.optimal_setpoint_distribution[opt_sp] += 1

        # Comfort violations: PMV or temp out of acceptable bounds (e.g. > 27.5 or < 21.5)
        if avg_temp > 27.0 or avg_temp < 22.0 or row["comfort_penalty"] > 1.0:
            self.comfort_violations += 1

        # Check for NaN / None
        for k, v in row.items():
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                self.invalid_rows += 1
                break

    def print_quality_check(self, split_row_counts: Dict[str, int]):
        """Print Section 38 formatted quality check summary."""
        print("\n" + "=" * 60)
        print("                  QUALITY CHECK REPORT")
        print("=" * 60)
        print(f"Total scenarios:       {self.total_scenarios}")
        print(f"Total simulation rows: {self.total_rows:,}")
        print(f"Train rows:            {split_row_counts.get('train', 0):,}")
        print(f"Validation rows:       {split_row_counts.get('val', 0):,}")
        print(f"Test rows:             {split_row_counts.get('test', 0):,}")

        print("\nScenario family distribution (rows):")
        for fam, count in sorted(self.family_counts.items()):
            pct = (count / self.total_rows * 100) if self.total_rows > 0 else 0
            print(f"  {fam:<38}: {count:>8,} ({pct:>5.1f}%)")

        print("\nTemperature (°C):")
        t_stats = self.temp_stats.as_dict()
        print(f"  min:  {t_stats['min']:.2f}")
        print(f"  max:  {t_stats['max']:.2f}")
        print(f"  mean: {t_stats['mean']:.2f}")

        print("\nOccupancy:")
        o_stats = self.occupancy_stats.as_dict()
        print(f"  min:  {int(o_stats['min'])}")
        print(f"  max:  {int(o_stats['max'])}")
        print(f"  mean: {o_stats['mean']:.2f}")

        print("\nComputer heat (W):")
        c_stats = self.computer_heat_stats.as_dict()
        print(f"  min:  {c_stats['min']:.2f}")
        print(f"  max:  {c_stats['max']:.2f}")
        print(f"  mean: {c_stats['mean']:.2f}")

        print("\nOptimal setpoint (°C):")
        sp_stats = self.optimal_setpoint_stats.as_dict()
        print(f"  min:  {sp_stats['min']:.1f}")
        print(f"  max:  {sp_stats['max']:.1f}")
        print(f"  mean: {sp_stats['mean']:.2f}")
        print("  distribution:")
        for sp in sorted(self.optimal_setpoint_distribution.keys()):
            cnt = self.optimal_setpoint_distribution[sp]
            pct = (cnt / self.total_rows * 100) if self.total_rows > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {sp:>4.1f}°C: {cnt:>7,} ({pct:>5.1f}%) {bar}")

        print("\nLabel Reasons:")
        for rsn, cnt in sorted(self.label_reason_counts.items()):
            pct = (cnt / self.total_rows * 100) if self.total_rows > 0 else 0
            print(f"  {rsn:<25}: {cnt:>7,} ({pct:>5.1f}%)")

        print(f"\nComfort violations:   {self.comfort_violations:,}")
        print(f"Invalid rows:         {self.invalid_rows}")
        print("=" * 60 + "\n")
