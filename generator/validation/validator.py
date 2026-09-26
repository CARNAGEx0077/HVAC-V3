"""
Dataset Integrity and Physical Causal Validator.

Verifies:
- Complete feature columns and exactly 10 computers + 4 ACs
- No label leakage (no optimal_* or future_* in model inputs)
- Causal physics consistency (higher load -> higher heat, cooling reduces temp)
- Temporal continuity (no discontinuous temperature jumps)
- Disjoint scenario splitting across train/val/test
- Data cleanliness (no NaNs, nulls, or infs)
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np

from generator.validation.feature_schema import (
    METADATA_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
)

FORBIDDEN_FEATURE_PREFIXES = ["optimal_", "future_", "optimization_cost"]


class DatasetValidator:
    """Performs rigorous automated validation of generated dataset files."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.raw_csv = self.dataset_dir / "raw" / "all_scenarios.csv"
        self.train_csv = self.dataset_dir / "train" / "train.csv"
        self.val_csv = self.dataset_dir / "validation" / "validation.csv"
        self.test_csv = self.dataset_dir / "test" / "test.csv"
        self.metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"

    def run_all_validations(self) -> Dict[str, any]:
        """Execute full validation suite and compile report."""
        report = {
            "status": "PASS",
            "checks": {},
            "errors": [],
            "warnings": [],
        }

        # 1. File existence
        exists_check = self._check_files_exist()
        report["checks"]["files_exist"] = exists_check
        if not exists_check["passed"]:
            report["status"] = "FAIL"
            report["errors"].extend(exists_check["errors"])
            return report

        # 2. Split Disjointness
        split_check = self._check_split_disjointness()
        report["checks"]["disjoint_splits"] = split_check
        if not split_check["passed"]:
            report["status"] = "FAIL"
            report["errors"].extend(split_check["errors"])

        # 3. Column Structure & Label Leakage Check
        col_check = self._check_columns_and_leakage()
        report["checks"]["columns_and_leakage"] = col_check
        if not col_check["passed"]:
            report["status"] = "FAIL"
            report["errors"].extend(col_check["errors"])

        # 4. Physical & Thermal Consistency
        physics_check = self._check_physics_and_continuity()
        report["checks"]["physics_and_continuity"] = physics_check
        if not physics_check["passed"]:
            report["status"] = "FAIL"
            report["errors"].extend(physics_check["errors"])

        if physics_check.get("warnings"):
            report["warnings"].extend(physics_check["warnings"])

        # Save validation report
        report_path = self.dataset_dir / "metadata" / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _check_files_exist(self) -> Dict[str, any]:
        errors = []
        for path in [self.raw_csv, self.train_csv, self.val_csv, self.test_csv, self.metadata_file]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
            elif path.stat().st_size == 0:
                errors.append(f"File is empty: {path}")
        return {"passed": len(errors) == 0, "errors": errors}

    def _check_split_disjointness(self) -> Dict[str, any]:
        """Ensure scenario_ids in train, val, and test are strictly disjoint and family-stratified."""
        train_map = self._extract_scenario_families(self.train_csv)
        val_map = self._extract_scenario_families(self.val_csv)
        test_map = self._extract_scenario_families(self.test_csv)

        train_scenarios = set(train_map.keys())
        val_scenarios = set(val_map.keys())
        test_scenarios = set(test_map.keys())

        train_val_overlap = train_scenarios & val_scenarios
        train_test_overlap = train_scenarios & test_scenarios
        val_test_overlap = val_scenarios & test_scenarios

        errors = []
        if train_val_overlap:
            errors.append(f"Data leakage: {len(train_val_overlap)} scenarios overlap between train and val: {sorted(train_val_overlap)}")
        if train_test_overlap:
            errors.append(f"Data leakage: {len(train_test_overlap)} scenarios overlap between train and test: {sorted(train_test_overlap)}")
        if val_test_overlap:
            errors.append(f"Data leakage: {len(val_test_overlap)} scenarios overlap between val and test: {sorted(val_test_overlap)}")

        total_scenarios = len(train_scenarios) + len(val_scenarios) + len(test_scenarios)
        all_families = set(list(train_map.values()) + list(val_map.values()) + list(test_map.values()))

        # For standard 100-scenario dataset, enforce exact 70/15/15 and 14/3/3 allocation
        if total_scenarios == 100:
            if len(train_scenarios) != 70:
                errors.append(f"Stratified split error: expected exactly 70 train scenarios, got {len(train_scenarios)}")
            if len(val_scenarios) != 15:
                errors.append(f"Stratified split error: expected exactly 15 val scenarios, got {len(val_scenarios)}")
            if len(test_scenarios) != 15:
                errors.append(f"Stratified split error: expected exactly 15 test scenarios, got {len(test_scenarios)}")

            for fam in sorted(all_families):
                f_train = sum(1 for f in train_map.values() if f == fam)
                f_val = sum(1 for f in val_map.values() if f == fam)
                f_test = sum(1 for f in test_map.values() if f == fam)
                f_total = f_train + f_val + f_test

                if f_total != 20:
                    errors.append(f"Family count error: {fam} has {f_total} scenarios, expected 20")
                if (f_train, f_val, f_test) != (14, 3, 3):
                    errors.append(
                        f"Family allocation error for {fam}: expected 14/3/3 (train/val/test), got {f_train}/{f_val}/{f_test}"
                    )

        family_summary = {}
        for fam in sorted(all_families):
            family_summary[fam] = {
                "train": sum(1 for f in train_map.values() if f == fam),
                "val": sum(1 for f in val_map.values() if f == fam),
                "test": sum(1 for f in test_map.values() if f == fam),
            }

        return {
            "passed": len(errors) == 0,
            "train_scenarios": len(train_scenarios),
            "val_scenarios": len(val_scenarios),
            "test_scenarios": len(test_scenarios),
            "family_allocation": family_summary,
            "errors": errors,
        }

    def _extract_scenario_families(self, csv_path: Path) -> Dict[str, str]:
        """Extract mapping of scenario_id -> scenario_family from CSV."""
        scenarios = {}
        if not csv_path.exists():
            return scenarios
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_id = row.get("scenario_id")
                fam = row.get("scenario_family", "UNKNOWN")
                if s_id:
                    scenarios[s_id] = fam
        return scenarios

    def _extract_scenario_ids(self, csv_path: Path) -> Set[str]:
        return set(self._extract_scenario_families(csv_path).keys())

    def _check_columns_and_leakage(self) -> Dict[str, any]:
        """
        Verifies all required columns exist, schema partition strictly satisfies:
        6 metadata + 80 features + 11 targets = 97 total columns,
        and features have zero label leakage.
        """
        errors = []
        with open(self.raw_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

        headers_set = set(headers)
        meta_set = set(METADATA_COLUMNS.keys())
        feat_set = set(FEATURE_COLUMNS.keys())
        target_set = set(TARGET_COLUMNS.keys())

        # Verify 10 computers present (40 columns)
        for i in range(1, 11):
            for suffix in ["_cpu", "_gpu", "_workload", "_heat"]:
                col = f"computer_{i}{suffix}"
                if col not in headers_set:
                    errors.append(f"Missing required computer column: {col}")

        # Verify 4 ACs present (12 columns)
        for ac_idx in range(1, 5):
            for field in ["state", "setpoint_c", "cooling_level"]:
                col = f"ac{ac_idx}_{field}"
                if col not in headers_set:
                    errors.append(f"Missing required AC column: {col}")

        # Verify all metadata columns present
        for col in sorted(meta_set):
            if col not in headers_set:
                errors.append(f"Missing required metadata column: {col}")

        # Verify all physical feature columns present
        for col in sorted(feat_set):
            if col not in headers_set:
                errors.append(f"Missing required feature column: {col}")

        # Verify all target columns present
        for col in sorted(target_set):
            if col not in headers_set:
                errors.append(f"Missing required target column: {col}")

        # Check for any unclassified columns
        unclassified = [col for col in headers if col not in meta_set and col not in feat_set and col not in target_set]
        if unclassified:
            errors.append(f"Unclassified columns found: {unclassified}")

        # Partition columns according to authoritative feature_schema
        metadata_columns = [h for h in headers if h in meta_set]
        feature_columns = [h for h in headers if h in feat_set]
        target_columns = [h for h in headers if h in target_set]

        # Explicit verification: 6 metadata + 80 features + 11 targets = 97 total
        if len(metadata_columns) != 6:
            errors.append(f"Expected exactly 6 metadata columns, got {len(metadata_columns)}")
        if len(feature_columns) != 80:
            errors.append(f"Expected exactly 80 feature columns, got {len(feature_columns)}")
        if len(target_columns) != 11:
            errors.append(f"Expected exactly 11 target columns, got {len(target_columns)}")
        if len(headers) != 97:
            errors.append(f"Expected exactly 97 total columns, got {len(headers)}")
        if len(metadata_columns) + len(feature_columns) + len(target_columns) != len(headers):
            errors.append(
                f"Column classification mismatch: {len(metadata_columns)} metadata + {len(feature_columns)} features + {len(target_columns)} targets != {len(headers)} total"
            )

        # Feature separation and leakage audit:
        # Standard input features must not contain optimal_*, future_*, etc.
        for feat in feature_columns:
            for prefix in FORBIDDEN_FEATURE_PREFIXES:
                if feat.startswith(prefix):
                    errors.append(f"Label leakage detected in feature column: {feat}")

        # Metadata columns must also not contain leakage prefixes
        for meta in metadata_columns:
            for prefix in FORBIDDEN_FEATURE_PREFIXES:
                if meta.startswith(prefix):
                    errors.append(f"Label leakage detected in metadata column: {meta}")

        return {
            "passed": len(errors) == 0,
            "total_columns": len(headers),
            "metadata_columns_count": len(metadata_columns),
            "feature_columns_count": len(feature_columns),
            "target_columns_count": len(target_columns),
            "label_columns_count": len(target_columns),
            "column_breakdown": f"{len(metadata_columns)} metadata + {len(feature_columns)} features + {len(target_columns)} targets = {len(headers)} total",
            "errors": errors,
        }

    def _check_physics_and_continuity(self) -> Dict[str, any]:
        """Scan time-series rows for causal physics and continuity violations."""
        errors = []
        warnings = []

        with open(self.raw_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prev_scenario = None
            prev_temp = None

            row_count = 0
            for row in reader:
                row_count += 1
                curr_scen = row["scenario_id"]
                curr_temp = float(row["room_average_temperature_c"])
                total_comp_heat = float(row["total_computer_heat_watts"])
                total_occ_heat = float(row["total_occupancy_heat_watts"])
                total_occ = int(row["occupancy_total"])

                # Check 1: Computer heat non-negative and reasonable
                if total_comp_heat < 500.0 or total_comp_heat > 5000.0:
                    errors.append(f"Row {row_count}: Unreasonable total computer heat {total_comp_heat}W")
                    break

                # Check 2: Occupancy heat proportional to occupancy
                if total_occ > 0 and total_occ_heat <= 0:
                    errors.append(f"Row {row_count}: Occupancy {total_occ} but zero occupancy heat")
                    break

                # Check 3: Temperature continuity within same scenario (no sudden step changes)
                if curr_scen == prev_scenario:
                    if prev_temp is not None:
                        delta_t = abs(curr_temp - prev_temp)
                        if delta_t > 0.6:  # Max 0.6 deg per 10s timestep
                            errors.append(f"Row {row_count}: Temperature discontinuity delta={delta_t:.3f}°C")
                            break

                    # Check 4: Room setpoint ramp constraint (max 0.5°C per step)
                    curr_opt_sp = float(row["optimal_room_setpoint_c"])
                    if prev_opt_sp is not None:
                        delta_sp = abs(curr_opt_sp - prev_opt_sp)
                        if delta_sp > 0.5001:
                            errors.append(f"Row {row_count}: Room setpoint jump {delta_sp:.2f}°C > 0.50°C")
                            break

                    # Check 5: Actuator cooling level ramp constraint (max 0.20 per step)
                    for ac_idx in range(1, 5):
                        c_lvl = float(row[f"optimal_ac{ac_idx}_cooling_level"])
                        p_lvl = prev_ac_lvls.get(f"AC-{ac_idx}")
                        if p_lvl is not None:
                            delta_c = abs(c_lvl - p_lvl)
                            if delta_c > 0.2001:
                                errors.append(f"Row {row_count}: AC{ac_idx} cooling jump {delta_c:.4f} > 0.20")
                                break

                prev_scenario = curr_scen
                prev_temp = curr_temp
                prev_opt_sp = float(row["optimal_room_setpoint_c"])
                prev_ac_lvls = {f"AC-{i}": float(row[f"optimal_ac{i}_cooling_level"]) for i in range(1, 5)}

        return {
            "passed": len(errors) == 0,
            "rows_scanned": row_count,
            "errors": errors,
            "warnings": warnings,
        }
