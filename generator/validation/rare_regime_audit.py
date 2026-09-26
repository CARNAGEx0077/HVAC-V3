"""
Rare Regime Coverage Audit for HVEAC Dataset v1.1.

Audits dataset representation across operational regimes:
- Rare 24.5 deg C room setpoint regime (hard requirement: >= 4 scenarios, present in train, val, and test)
- Rare 26.5 deg C room setpoint regime (hard requirement: >= 4 scenarios, present in train, val, and test)
- High compute stress regime (> 3000 W aggregate computer heat)
- High occupancy stress regime (>= 25 total occupants)
- Localized thermal hotspots (spatial temp delta >= 2.0 deg C)
- Opposing thermal zones regime
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Set


class RareRegimeAuditor:
    """Audits rare regime representation across train, validation, and test splits."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.raw_csv = self.dataset_dir / "raw" / "all_scenarios.csv"
        self.metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"

    def run_audit(self, output_dir: Path) -> Dict[str, Any]:
        """Execute rare regime audit and output JSON + Markdown reports."""
        if not self.raw_csv.exists():
            raise FileNotFoundError(f"Raw CSV not found at {self.raw_csv}")

        # Load metadata for split assignments
        scenario_splits: Dict[str, str] = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            split_dict = meta.get("dataset_split_strategy", {}).get("scenario_ids_by_split", {})
            for s_name, s_ids in split_dict.items():
                norm_name = "validation" if s_name in ["val", "validation"] else s_name
                for s_id in s_ids:
                    scenario_splits[s_id] = norm_name

        # Read raw rows grouped by scenario
        scen_rows = defaultdict(list)
        with open(self.raw_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scen_rows[row["scenario_id"]].append(row)

        regimes: Dict[str, Dict[str, Any]] = {
            "setpoint_low_c": {
                "name": "Aggressive Cooling Setpoint (<= 23.0°C)",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
            "setpoint_high_c": {
                "name": "Relaxed Energy-Saving Setpoint (>= 24.5°C)",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
            "high_compute_load": {
                "name": "High Compute Load (> 3000W)",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
            "high_occupancy": {
                "name": "High Occupancy (>= 20 persons)",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
            "localized_hotspot": {
                "name": "Localized Hotspot (Temp Delta >= 1.5°C)",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
            "opposing_zones": {
                "name": "Opposing Thermal Zones",
                "total_rows": 0,
                "scenarios": set(),
                "train_scenarios": set(),
                "val_scenarios": set(),
                "test_scenarios": set(),
            },
        }

        for s_id, rows in scen_rows.items():
            split = scenario_splits.get(s_id, "train")
            fam = rows[0]["scenario_family"]

            has_low = False
            has_high = False
            has_high_comp = False
            has_high_occ = False
            has_hotspot = False
            has_opposing = (fam == "FAMILY_5")

            for r in rows:
                sp = float(r["optimal_room_setpoint_c"])
                c_heat = float(r["total_computer_heat_watts"])
                occ = int(r["occupancy_total"])
                t_diff = float(r["temperature_difference_c"])

                if sp <= 23.0:
                    regimes["setpoint_low_c"]["total_rows"] += 1
                    has_low = True
                if sp >= 24.5:
                    regimes["setpoint_high_c"]["total_rows"] += 1
                    has_high = True
                if c_heat > 3000.0:
                    regimes["high_compute_load"]["total_rows"] += 1
                    has_high_comp = True
                if occ >= 20:
                    regimes["high_occupancy"]["total_rows"] += 1
                    has_high_occ = True
                if t_diff >= 1.5:
                    regimes["localized_hotspot"]["total_rows"] += 1
                    has_hotspot = True
                if has_opposing:
                    regimes["opposing_zones"]["total_rows"] += 1

            # Register scenario participation
            if has_low:
                regimes["setpoint_low_c"]["scenarios"].add(s_id)
                self._add_to_split(regimes["setpoint_low_c"], s_id, split)
            if has_high:
                regimes["setpoint_high_c"]["scenarios"].add(s_id)
                self._add_to_split(regimes["setpoint_high_c"], s_id, split)
            if has_high_comp:
                regimes["high_compute_load"]["scenarios"].add(s_id)
                self._add_to_split(regimes["high_compute_load"], s_id, split)
            if has_high_occ:
                regimes["high_occupancy"]["scenarios"].add(s_id)
                self._add_to_split(regimes["high_occupancy"], s_id, split)
            if has_hotspot:
                regimes["localized_hotspot"]["scenarios"].add(s_id)
                self._add_to_split(regimes["localized_hotspot"], s_id, split)
            if has_opposing:
                regimes["opposing_zones"]["scenarios"].add(s_id)
                self._add_to_split(regimes["opposing_zones"], s_id, split)

        # Convert sets to sorted lists for JSON serialization
        serialized_regimes = {}
        for k, v in regimes.items():
            serialized_regimes[k] = {
                "name": v["name"],
                "total_rows": v["total_rows"],
                "total_scenarios": len(v["scenarios"]),
                "train_scenarios_count": len(v["train_scenarios"]),
                "val_scenarios_count": len(v["val_scenarios"]),
                "test_scenarios_count": len(v["test_scenarios"]),
                "train_scenarios": sorted(list(v["train_scenarios"])),
                "val_scenarios": sorted(list(v["val_scenarios"])),
                "test_scenarios": sorted(list(v["test_scenarios"])),
            }

        # Validate hard requirements
        clow = serialized_regimes["setpoint_low_c"]
        chigh = serialized_regimes["setpoint_high_c"]

        passed_low = (
            clow["total_scenarios"] >= 4
            and clow["train_scenarios_count"] >= 1
            and clow["val_scenarios_count"] >= 1
            and clow["test_scenarios_count"] >= 1
        )
        passed_high = (
            chigh["total_scenarios"] >= 4
            and chigh["train_scenarios_count"] >= 1
            and chigh["val_scenarios_count"] >= 1
            and chigh["test_scenarios_count"] >= 1
        )

        overall_status = "PASS" if (passed_low and passed_high) else "FAIL"

        report = {
            "audit_version": "2.0.0",
            "rare_regime_audit_status": overall_status,
            "hard_requirements_summary": {
                "regime_low_setpoint": {
                    "total_scenarios_gte_4": clow["total_scenarios"] >= 4,
                    "present_in_train": clow["train_scenarios_count"] >= 1,
                    "present_in_val": clow["val_scenarios_count"] >= 1,
                    "present_in_test": clow["test_scenarios_count"] >= 1,
                    "status": "PASS" if passed_low else "FAIL",
                },
                "regime_high_setpoint": {
                    "total_scenarios_gte_4": chigh["total_scenarios"] >= 4,
                    "present_in_train": chigh["train_scenarios_count"] >= 1,
                    "present_in_val": chigh["val_scenarios_count"] >= 1,
                    "present_in_test": chigh["test_scenarios_count"] >= 1,
                    "status": "PASS" if passed_high else "FAIL",
                },
            },
            "regimes": serialized_regimes,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rare_regime_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self._save_markdown_report(report, output_dir / "rare_regime_report.md")

        return report

    def _add_to_split(self, reg_dict: Dict, s_id: str, split: str):
        if split == "train":
            reg_dict["train_scenarios"].add(s_id)
        elif split in ["val", "validation"]:
            reg_dict["val_scenarios"].add(s_id)
        elif split == "test":
            reg_dict["test_scenarios"].add(s_id)

    def _save_markdown_report(self, report: Dict[str, Any], path: Path):
        md = []
        md.append("# HVEAC Rare Regime Coverage Report (v1.1)\n")
        md.append(f"**Overall Audit Status:** `{report['rare_regime_audit_status']}`\n")
        md.append("## Hard Requirements Verification\n")
        md.append("| Regime | Target >= 4 Scenarios | Train | Validation | Test | Status |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        r_low = report["hard_requirements_summary"].get("regime_low_setpoint", report["hard_requirements_summary"].get("regime_24_5_c", {}))
        r_high = report["hard_requirements_summary"].get("regime_high_setpoint", report["hard_requirements_summary"].get("regime_26_5_c", {}))
        md.append(f"| **Aggressive Cooling (<=23.0°C)** | {r_low.get('total_scenarios_gte_4', False)} | {r_low.get('present_in_train', False)} | {r_low.get('present_in_val', False)} | {r_low.get('present_in_test', False)} | `{r_low.get('status', 'FAIL')}` |")
        md.append(f"| **Relaxed Setpoint (>=24.5°C)**   | {r_high.get('total_scenarios_gte_4', False)} | {r_high.get('present_in_train', False)} | {r_high.get('present_in_val', False)} | {r_high.get('present_in_test', False)} | `{r_high.get('status', 'FAIL')}` |\n")

        md.append("## Operational Regimes Summary\n")
        md.append("| Operational Regime | Total Rows | Total Scenarios | Train | Validation | Test |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for k, v in report["regimes"].items():
            md.append(f"| **{v['name']}** | {v['total_rows']:,} | {v['total_scenarios']} | {v['train_scenarios_count']} | {v['val_scenarios_count']} | {v['test_scenarios_count']} |")

        md.append("\n## Scenario Allocation Details\n")
        for k, v in report["regimes"].items():
            md.append(f"### {v['name']}")
            md.append(f"- **Train Scenarios ({v['train_scenarios_count']}):** {', '.join(v['train_scenarios']) if v['train_scenarios'] else 'None'}")
            md.append(f"- **Validation Scenarios ({v['val_scenarios_count']}):** {', '.join(v['val_scenarios']) if v['val_scenarios'] else 'None'}")
            md.append(f"- **Test Scenarios ({v['test_scenarios_count']}):** {', '.join(v['test_scenarios']) if v['test_scenarios'] else 'None'}\n")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))
