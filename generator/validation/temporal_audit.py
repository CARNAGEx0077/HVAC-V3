"""
Temporal Stability and Actuator Constraint Audit for HVEAC Dataset v1.1.

Audits scenario-by-scenario temporal control behavior:
- Room setpoint transitions, changes/hour, and dwell times (shortest, average, longest)
- AC cooling level transitions, average jump, and maximum single-step jump
- Strict compliance with actuator ramp rate constraints (<= 0.20 per 10s timestep)
- Strict compliance with room setpoint transition bounds (<= 0.5 deg C per timestep)
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


class TemporalAuditor:
    """Audits temporal stability and actuator constraints across all dataset scenarios."""

    def __init__(
        self,
        dataset_dir: Path,
        max_cooling_jump_thresh: float = 0.20,
        max_setpoint_jump_thresh: float = 0.50,
        timestep_seconds: float = 10.0,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_cooling_jump_thresh = max_cooling_jump_thresh
        self.max_setpoint_jump_thresh = max_setpoint_jump_thresh
        self.timestep_seconds = timestep_seconds
        self.raw_csv = self.dataset_dir / "raw" / "all_scenarios.csv"

    def run_audit(self, output_path: Path) -> Dict[str, Any]:
        """Perform comprehensive temporal stability audit across all scenarios."""
        if not self.raw_csv.exists():
            raise FileNotFoundError(f"Raw dataset CSV not found at {self.raw_csv}")

        scenario_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        with open(self.raw_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scenario_rows[row["scenario_id"]].append(row)

        scenario_audits: List[Dict[str, Any]] = []
        total_cooling_violations = 0
        total_setpoint_violations = 0
        dataset_max_cooling_jump = 0.0
        dataset_max_setpoint_jump = 0.0
        all_dwell_times: List[float] = []
        all_cooling_jumps: List[float] = []

        for scen_id, rows in sorted(scenario_rows.items()):
            fam = rows[0]["scenario_family"]
            setpoints = [float(r["optimal_room_setpoint_c"]) for r in rows]
            ac_levels = {
                f"AC-{i}": [float(r[f"optimal_ac{i}_cooling_level"]) for r in rows]
                for i in range(1, 5)
            }

            # 1. Setpoint dwell analysis
            dwells: List[float] = []
            curr_dwell = 1
            setpoint_changes = 0
            max_scen_sp_jump = 0.0

            for t in range(1, len(setpoints)):
                delta_sp = abs(setpoints[t] - setpoints[t - 1])
                max_scen_sp_jump = max(max_scen_sp_jump, delta_sp)
                if delta_sp > (self.max_setpoint_jump_thresh + 1e-4):
                    total_setpoint_violations += 1

                if setpoints[t] != setpoints[t - 1]:
                    setpoint_changes += 1
                    dwells.append(curr_dwell * self.timestep_seconds)
                    curr_dwell = 1
                else:
                    curr_dwell += 1
            dwells.append(curr_dwell * self.timestep_seconds)
            all_dwell_times.extend(dwells)
            dataset_max_setpoint_jump = max(dataset_max_setpoint_jump, max_scen_sp_jump)

            duration_hours = (len(rows) * self.timestep_seconds) / 3600.0
            changes_per_hour = round(setpoint_changes / max(0.1, duration_hours), 2)

            # 2. Cooling modulation transition analysis
            scen_cooling_jumps: List[float] = []
            scen_cooling_transitions = 0
            max_scen_cooling_jump = 0.0

            for ac_id, levels in ac_levels.items():
                for t in range(1, len(levels)):
                    c_jump = abs(levels[t] - levels[t - 1])
                    if c_jump > 1e-4:
                        scen_cooling_transitions += 1
                        scen_cooling_jumps.append(c_jump)
                        all_cooling_jumps.append(c_jump)
                        max_scen_cooling_jump = max(max_scen_cooling_jump, c_jump)
                        if c_jump > (self.max_cooling_jump_thresh + 1e-4):
                            total_cooling_violations += 1

            dataset_max_cooling_jump = max(dataset_max_cooling_jump, max_scen_cooling_jump)

            scen_audit = {
                "scenario_id": scen_id,
                "scenario_family": fam,
                "total_timesteps": len(rows),
                "setpoint_changes": setpoint_changes,
                "changes_per_hour": changes_per_hour,
                "shortest_dwell_seconds": round(float(np.min(dwells)), 1) if dwells else 0.0,
                "average_dwell_seconds": round(float(np.mean(dwells)), 1) if dwells else 0.0,
                "longest_dwell_seconds": round(float(np.max(dwells)), 1) if dwells else 0.0,
                "cooling_transitions_count": scen_cooling_transitions,
                "max_cooling_level_jump": round(float(max_scen_cooling_jump), 4),
                "average_cooling_level_jump": round(float(np.mean(scen_cooling_jumps)), 4) if scen_cooling_jumps else 0.0,
                "max_setpoint_jump_c": round(float(max_scen_sp_jump), 2),
                "actuator_violations": 0 if max_scen_cooling_jump <= (self.max_cooling_jump_thresh + 1e-4) else 1,
            }
            scenario_audits.append(scen_audit)

        status = "PASS" if (total_cooling_violations == 0 and total_setpoint_violations == 0) else "FAIL"

        report = {
            "audit_version": "1.1.0",
            "temporal_audit_status": status,
            "total_scenarios_audited": len(scenario_audits),
            "configured_limits": {
                "max_cooling_jump_per_step": self.max_cooling_jump_thresh,
                "max_setpoint_jump_per_step_c": self.max_setpoint_jump_thresh,
                "timestep_seconds": self.timestep_seconds,
            },
            "summary_metrics": {
                "dataset_max_cooling_jump": round(float(dataset_max_cooling_jump), 4),
                "dataset_max_setpoint_jump_c": round(float(dataset_max_setpoint_jump), 2),
                "total_actuator_cooling_violations": total_cooling_violations,
                "total_setpoint_ramp_violations": total_setpoint_violations,
                "mean_dwell_time_seconds": round(float(np.mean(all_dwell_times)), 1) if all_dwell_times else 0.0,
                "min_dwell_time_seconds": round(float(np.min(all_dwell_times)), 1) if all_dwell_times else 0.0,
                "max_dwell_time_seconds": round(float(np.max(all_dwell_times)), 1) if all_dwell_times else 0.0,
                "mean_cooling_level_jump": round(float(np.mean(all_cooling_jumps)), 4) if all_cooling_jumps else 0.0,
            },
            "scenario_temporal_details": scenario_audits,
        }

        output_path = Path(output_path)
        if output_path.is_dir() or not output_path.suffix:
            target_json = output_path / "temporal_audit_report.json"
        else:
            target_json = output_path

        target_json.parent.mkdir(parents=True, exist_ok=True)
        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report
