"""
Dataset Serialization and Metadata Generation for HVEAC V3.

Exports simulation records to:
- CSV (ML-friendly flat tabular format, stable column names, numeric values)
- JSONL (Streaming newline-delimited JSON)
- dataset_metadata.json (Schema specifications, statistical summaries, target distributions)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class DatasetWriter:
    """Handles writing structured simulation records and metadata."""

    def __init__(self, output_dir: str = "simulation_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(self, records: List[Dict[str, Any]], filename: str) -> Path:
        """Writes records to a flat, ML-friendly CSV file."""
        if not records:
            raise ValueError("Cannot write empty dataset to CSV.")

        filepath = self.output_dir / filename
        fieldnames = list(records[0].keys())

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r)

        return filepath

    def write_jsonl(self, records: List[Dict[str, Any]], filename: str) -> Path:
        """Writes records to a newline-delimited JSON (JSONL) file."""
        filepath = self.output_dir / filename
        with open(filepath, mode="w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return filepath

    def write_metadata(
        self,
        all_records: List[Dict[str, Any]],
        scenario_records: Dict[int, List[Dict[str, Any]]],
        config_snapshot: Dict[str, Any],
        filename: str = "dataset_metadata.json",
    ) -> Path:
        """Generates comprehensive dataset metadata with schema and statistics."""
        filepath = self.output_dir / filename

        if not all_records:
            metadata = {"total_rows": 0, "status": "empty"}
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            return filepath

        columns = list(all_records[0].keys())

        # Determine column types
        column_types = {}
        for col in columns:
            val = all_records[0][col]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                column_types[col] = "float" if isinstance(val, float) else "int"
            elif isinstance(val, bool):
                column_types[col] = "bool"
            else:
                column_types[col] = "categorical"

        # Key numeric statistics
        stat_cols = [
            "room_average_temperature_c",
            "zone_1_temperature_c",
            "zone_2_temperature_c",
            "zone_3_temperature_c",
            "zone_4_temperature_c",
            "temperature_gradient_c",
            "total_computational_heat_w",
            "total_occupancy_heat_w",
            "total_hvac_cooling_w",
            "total_net_heat_w",
            "optimal_cooling_ac1",
            "optimal_cooling_ac2",
            "optimal_cooling_ac3",
            "optimal_cooling_ac4",
        ]

        summary_stats = {}
        for col in stat_cols:
            if col in columns:
                vals = [r[col] for r in all_records if col in r]
                arr = np.array(vals, dtype=float)
                summary_stats[col] = {
                    "min": round(float(np.min(arr)), 3),
                    "max": round(float(np.max(arr)), 3),
                    "mean": round(float(np.mean(arr)), 3),
                    "std": round(float(np.std(arr)), 3),
                }

        # Target action distribution
        target_counts: Dict[str, int] = {}
        for r in all_records:
            act = r.get("optimal_hvac_action", "UNKNOWN")
            target_counts[act] = target_counts.get(act, 0) + 1

        scenario_breakdown = {
            f"scenario_{sid}": len(recs)
            for sid, recs in scenario_records.items()
        }

        metadata = {
            "dataset_name": "HVEAC_V3_Thermal_Simulation_Dataset",
            "version": "3.0.0",
            "description": "Deterministic multi-zone synthetic thermal simulation dataset for training Thermal Intelligence models.",
            "synthetic_disclaimer": "CRITICAL: All computer workloads, occupancy counts, thermal dissipation, and optimal actions are SYNTHETICALLY generated. No real hardware was queried.",
            "total_records": len(all_records),
            "scenario_breakdown": scenario_breakdown,
            "num_columns": len(columns),
            "columns": columns,
            "column_types": column_types,
            "target_distribution": target_counts,
            "summary_statistics": summary_stats,
            "simulation_configuration": config_snapshot,
        }

        with open(filepath, mode="w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return filepath
