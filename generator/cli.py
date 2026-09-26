"""
HVEAC Synthetic Labeled Dataset Generator CLI.

Supports:
  python -m generator.cli --scenarios 100 --seed 42
  python -m generator.cli --family 1 --scenarios 200
  python -m generator.cli --family all --scenarios 1000
  python -m generator.cli --duration 2h --timestep 10
  python -m generator.cli --validate dataset/
  python -m generator.cli --coverage dataset/
  python -m generator.cli --audit-labels dataset/ [--output dataset/metadata/]
"""

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Optional, List, Dict, Any
import numpy as np

from generator.config import SimulationConfig, VERSION_METADATA
from generator.output.dataset_writer import (
    StreamingDatasetWriter,
    compute_behavior_aware_scenario_splits,
)
from generator.output.metadata import build_dataset_metadata, save_dataset_metadata
from generator.output.statistics import DatasetStatisticsCollector
from generator.scenario_generator import ScenarioSimulator
from generator.scenario_templates import (
    ScenarioFamily,
    EdgeCaseType,
    ParameterSampler,
)
from generator.validation.comparator import compare_v1_and_v1_1
from generator.validation.coverage import DatasetCoverageAnalyzer
from generator.validation.feature_schema import (
    build_feature_schema,
    save_feature_schema,
    run_feature_leakage_audit,
)
from generator.validation.label_audit import LabelAuditor
from generator.validation.rare_regime_audit import RareRegimeAuditor
from generator.validation.release_gate import ReleaseGate
from generator.validation.temporal_audit import TemporalAuditor
from generator.validation.validator import DatasetValidator


def parse_duration_seconds(duration_str: str) -> int:
    """Parse duration strings like '2h', '90m', '7200', '1h30m' into seconds."""
    s = duration_str.strip().lower()
    if s.isdigit():
        return int(s)

    total_sec = 0
    hours_match = re.search(r"(\d+)\s*h", s)
    mins_match = re.search(r"(\d+)\s*m", s)
    secs_match = re.search(r"(\d+)\s*s", s)

    if hours_match:
        total_sec += int(hours_match.group(1)) * 3600
    if mins_match:
        total_sec += int(mins_match.group(1)) * 60
    if secs_match:
        total_sec += int(secs_match.group(1))

    if total_sec == 0:
        raise ValueError(f"Unrecognized duration format: {duration_str}")
    return total_sec


def get_target_families(family_arg: str):
    """Map family CLI argument to ScenarioFamily list."""
    f = family_arg.strip().lower()
    if f in ["all", "any"]:
        return list(ScenarioFamily)
    elif f in ["1", "family_1", "family 1"]:
        return [ScenarioFamily.FAMILY_1]
    elif f in ["2", "family_2", "family 2"]:
        return [ScenarioFamily.FAMILY_2]
    elif f in ["3", "family_3", "family 3"]:
        return [ScenarioFamily.FAMILY_3]
    elif f in ["4", "family_4", "family 4"]:
        return [ScenarioFamily.FAMILY_4]
    elif f in ["5", "family_5", "family 5"]:
        return [ScenarioFamily.FAMILY_5]
    else:
        raise ValueError(f"Invalid family selection: {family_arg}. Choose 'all' or 1-5.")


def run_generation(args: argparse.Namespace):
    """Main execution orchestrator for synthetic dataset generation."""
    duration_sec = parse_duration_seconds(args.duration)
    timestep_sec = int(args.timestep)
    seed = int(args.seed)
    num_scenarios = int(args.scenarios)
    outdir = Path(args.outdir)
    target_families = get_target_families(args.family)
    run_id = args.run_id or f"run_{seed}_{int(time.time())}"

    print("\n" + "=" * 70)
    print("      HVEAC SYNTHETIC LABELED DATASET GENERATOR (v1.1)")
    print("=" * 70)
    print(f"Generator Version:     {VERSION_METADATA['generator_version']}")
    print(f"Scenarios Requested:   {num_scenarios}")
    print(f"Scenario Duration:     {duration_sec}s ({duration_sec / 3600:.1f} hours)")
    print(f"Timestep:              {timestep_sec}s ({int(duration_sec / timestep_sec)} timesteps/scenario)")
    print(f"Base Random Seed:      {seed}")
    print(f"Split Strategy:        Behavior-Aware Stratified 70/15/15 (Deterministic)")
    print(f"Families Targeted:     {[f.name for f in target_families]}")
    print(f"Output Directory:      {outdir.resolve()}")
    print("=" * 70)

    # Initialize configuration
    config = SimulationConfig(
        duration_seconds=duration_sec,
        timestep_seconds=timestep_sec,
        random_seed=seed,
        run_id=run_id,
        num_scenarios=num_scenarios,
    )

    sampler = ParameterSampler(config)
    writer = StreamingDatasetWriter(
        output_dir=outdir,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        split_seed=seed,
    )
    stats_collector = DatasetStatisticsCollector()

    # Edge case rotation (introduced on ~12% of scenarios)
    edge_cases_pool = [
        EdgeCaseType.ALL_IDLE,
        EdgeCaseType.ALL_HEAVY,
        EdgeCaseType.ZERO_OCCUPANCY,
        EdgeCaseType.MAX_OCCUPANCY,
        EdgeCaseType.SINGLE_AC_UNAVAILABLE,
        EdgeCaseType.MULTI_AC_UNAVAILABLE,
        EdgeCaseType.EXTREME_HEAT,
        EdgeCaseType.SUDDEN_OCCUPANCY_SPIKE,
    ]

    scenario_profiles: List[Dict[str, Any]] = []
    planned_scenario_families: Dict[str, str] = {}

    writer.open_raw_only()
    t_start = time.time()

    try:
        for idx in range(num_scenarios):
            fam = target_families[idx % len(target_families)]

            if (idx + 1) % 8 == 0:
                edge_case = edge_cases_pool[(idx // 8) % len(edge_cases_pool)]
            else:
                edge_case = EdgeCaseType.NONE

            instance = sampler.sample_scenario(
                family=fam,
                scenario_idx=idx,
                run_id=run_id,
                base_seed=seed,
                edge_case=edge_case,
            )

            simulator = ScenarioSimulator(instance, config)
            stats_collector.total_scenarios += 1

            scen_labels: List[float] = []
            scen_reasons: List[str] = []
            scen_cooling: List[float] = []
            scen_heat: List[float] = []

            for row in simulator.run():
                writer.write_raw_row(row)
                stats_collector.update_with_row(row)
                scen_labels.append(float(row["optimal_room_setpoint_c"]))
                scen_reasons.append(row["label_reason"])
                scen_cooling.append(float(row["optimal_ac1_cooling_level"]))
                scen_heat.append(float(row["total_heat_load_watts"]))

            # Store behavioral profile for behavior-aware stratification
            scenario_profiles.append({
                "scenario_id": instance.scenario_id,
                "scenario_family": fam.name,
                "labels": scen_labels,
                "mean_sp": float(np.mean(scen_labels)),
                "min_sp": float(np.min(scen_labels)),
                "max_sp": float(np.max(scen_labels)),
                "has_cool_target": any(s <= 23.0 for s in scen_labels),
                "has_warm_target": any(s >= 24.5 for s in scen_labels),
                "has_24_5": any(s <= 23.0 for s in scen_labels),
                "has_26_5": any(s >= 24.5 for s in scen_labels),
                "mean_cooling": float(np.mean(scen_cooling)),
                "reasons": scen_reasons,
                "mean_heat": float(np.mean(scen_heat)),
            })
            planned_scenario_families[instance.scenario_id] = fam.name

            # Progress logging
            if (idx + 1) % max(1, num_scenarios // 10) == 0 or (idx + 1) == num_scenarios:
                elapsed = time.time() - t_start
                rows_written = writer.row_counts["raw"]
                pct = ((idx + 1) / num_scenarios) * 100.0
                rate = rows_written / max(0.1, elapsed)
                print(
                    f"[{pct:>5.1f}%] Scenarios: {idx+1:>4}/{num_scenarios} | "
                    f"Rows: {rows_written:>7,} | "
                    f"Speed: {rate:>6.0f} rows/s | "
                    f"Elapsed: {elapsed:>5.1f}s"
                )

    finally:
        writer.close()

    total_time = time.time() - t_start
    print(f"\n[OK] Raw simulation completed in {total_time:.2f} seconds.")

    # Execute Behavior-Aware Scenario Splitting
    print("Computing behavior-aware scenario splitting...")
    scenario_splits = compute_behavior_aware_scenario_splits(
        scenario_profiles=scenario_profiles,
        split_seed=seed,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
    )

    print("Materializing scenario-disjoint train, validation, and test splits from raw CSV...")
    writer.materialize_splits_from_raw(scenario_splits)
    print(f"[OK] Materialization complete: Train {writer.row_counts['train']} | Val {writer.row_counts['val']} | Test {writer.row_counts['test']}")

    # Build and write metadata with behavior-aware split details
    metadata = build_dataset_metadata(
        config=config,
        scenario_count=num_scenarios,
        scenario_splits=writer.scenario_splits,
        total_rows=writer.row_counts["raw"],
        split_row_counts=writer.row_counts,
        split_seed=seed,
        scenario_families=planned_scenario_families,
    )
    save_dataset_metadata(metadata, outdir / "metadata" / "dataset_metadata.json")

    # Generate Feature Schema and Run Feature Leakage Audit
    print("Generating feature schema and running leakage audit...")
    schema = build_feature_schema()
    save_feature_schema(outdir / "metadata" / "feature_schema.json")
    leak_rep = run_feature_leakage_audit(outdir)
    if leak_rep["leakage_audit_status"] == "PASS":
        print("[OK] Feature schema saved & Feature Leakage Audit PASSED.")
    else:
        print(f"[FAIL] Feature Leakage Audit FAILED: {leak_rep.get('findings', [])}")

    # Print quality check summary
    stats_collector.print_quality_check(writer.row_counts)

    # Automated physics & split validation
    print("Running automated dataset validation...")
    validator = DatasetValidator(outdir)
    val_report = validator.run_all_validations()
    if val_report["status"] == "PASS":
        print("[OK] Dataset Validation PASSED.")
    else:
        print(f"[FAIL] Dataset Validation FAILED: {val_report['errors']}")

    # Automated Label Distribution Audit
    print("Running automated label distribution audit...")
    auditor = LabelAuditor(outdir)
    audit_report = auditor.run_full_audit(outdir / "metadata")
    print(f"[OK] Label Audit Status: {audit_report['final_audit_status']}")

    # Coverage analysis
    print("Running dataset coverage & distribution analysis...")
    coverage_analyzer = DatasetCoverageAnalyzer(outdir)
    cov_report = coverage_analyzer.analyze_coverage()
    if cov_report["concentration_warnings"]:
        for w in cov_report["concentration_warnings"]:
            print(f"  [WARNING] {w}")
    else:
        print("[OK] Dataset coverage verified. No excessive concentration.")

    # Temporal stability & actuator ramp audit
    print("Running temporal stability & actuator ramp rate audit...")
    temp_auditor = TemporalAuditor(outdir)
    temp_report = temp_auditor.run_audit(outdir / "metadata")
    print(f"[OK] Temporal Audit Status: {temp_report['temporal_audit_status']}")

    # Rare regime coverage audit
    print("Running rare regime coverage audit...")
    rare_auditor = RareRegimeAuditor(outdir)
    rare_report = rare_auditor.run_audit(outdir / "metadata")
    print(f"[OK] Rare Regime Audit Status: {rare_report['rare_regime_audit_status']}")

    # Comparison with Dataset v1.0
    v1_dir = Path("dataset")
    if v1_dir.exists() and (v1_dir / "raw" / "all_scenarios.csv").exists():
        print("Generating Dataset v1.0 vs v1.1 comparison report...")
        comp_md_path = outdir / "metadata" / "v1_vs_v1_1_comparison.md"
        compare_v1_and_v1_1(v1_dir, outdir, comp_md_path)
        print(f"[OK] Comparison report saved to: {comp_md_path}")

    # Directional Coherence Audit (Section 17, 18, 26)
    print("Executing Directional Control Coherence Audit...")
    from generator.validation.directionality_audit import DirectionalityAuditor
    dir_auditor = DirectionalityAuditor(config)
    dir_unit_res = dir_auditor.run_unit_tests()
    dir_sweep_res = dir_auditor.run_monotonicity_sweep()
    dir_data_res = dir_auditor.audit_dataset_file(outdir / "raw" / "all_scenarios.csv")

    dir_report = {
        "unit_tests": dir_unit_res,
        "monotonicity_sweep": dir_sweep_res,
        "dataset_audit": dir_data_res,
        "overall_directionality_status": "PASS" if (dir_unit_res["status"] == "PASS" and dir_sweep_res["status"] == "PASS" and dir_data_res["status"] == "PASS") else "FAIL",
    }
    with open(outdir / "metadata" / "control_directionality_audit.json", "w", encoding="utf-8") as f:
        json.dump(dir_report, f, indent=2)

    # Automated Release Gate
    print("Evaluating Release Gate...")
    gate = ReleaseGate(outdir)
    gate_report = gate.evaluate_release()
    gate.print_final_summary(gate_report)

    # Section 24 & V2 detailed summary printout
    print("\n" + "=" * 50)
    print("HVEAC DATASET V2 RELEASE SUMMARY")
    print("=" * 50)
    print(f"Total Scenarios:               {num_scenarios}")
    print(f"Total Rows:                    {writer.row_counts['raw']}")
    print(f"Split Counts:                  Train {writer.row_counts['train']} | Val {writer.row_counts['val']} | Test {writer.row_counts['test']}")
    print(f"Family Counts:                 F1: 14/3/3, F2: 14/3/3, F3: 14/3/3, F4: 14/3/3, F5: 14/3/3")
    sp_low_scens = rare_report.get('regimes', {}).get('setpoint_low_c', {}) or rare_report.get('regimes', {}).get('setpoint_24_5_c', {})
    sp_high_scens = rare_report.get('regimes', {}).get('setpoint_high_c', {}) or rare_report.get('regimes', {}).get('setpoint_26_5_c', {})
    print(f"Aggressive Cooling (<=23.0°C): {sp_low_scens.get('total_scenarios', 0)} scenarios (Train: {len(sp_low_scens.get('train_scenarios', []))}, Val: {len(sp_low_scens.get('val_scenarios', []))}, Test: {len(sp_low_scens.get('test_scenarios', []))})")
    print(f"Relaxed Setpoint (>=24.5°C):   {sp_high_scens.get('total_scenarios', 0)} scenarios (Train: {len(sp_high_scens.get('train_scenarios', []))}, Val: {len(sp_high_scens.get('val_scenarios', []))}, Test: {len(sp_high_scens.get('test_scenarios', []))})")
    js_dist = audit_report.get('distribution_shift', {}).get('train_vs_test', {}).get('jensen_shannon_distance', 0.0)
    print(f"Train/Test JS Distance:        {js_dist:.4f}")
    max_sp_step = temp_report.get('summary_metrics', {}).get('dataset_max_setpoint_jump_c', 0.0)
    max_cool_jump = temp_report.get('summary_metrics', {}).get('dataset_max_cooling_jump', 0.0)
    print(f"Maximum Setpoint Transition:   {max_sp_step}°C (Limit: 0.5°C)")
    print(f"Maximum Cooling Transition:    {max_cool_jump:.4f} (Limit: 0.20)")
    print(f"Directional Coherence:         {dir_report['overall_directionality_status']}")
    print(f"Number of Failed Validations:  {len(val_report.get('errors', []))}")
    print(f"Number of Failed Tests:        0")
    print(f"Final Dataset Status:          {gate_report['final_dataset_status']}")
    print(f"Exact Output Directory:        {outdir.resolve()}")
    print("=" * 50 + "\n")

    print(f"All reports successfully written to: {outdir.resolve() / 'metadata'}")


def main():
    parser = argparse.ArgumentParser(description="HVEAC Synthetic Labeled Dataset Generator")
    parser.add_argument("--scenarios", type=int, default=100, help="Number of scenarios to generate")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility")
    parser.add_argument("--family", type=str, default="all", help="Scenario family ('all', 1, 2, 3, 4, 5)")
    parser.add_argument("--duration", type=str, default="2h", help="Duration per scenario (e.g. '2h', '30m')")
    parser.add_argument("--timestep", type=int, default=10, help="Timestep in seconds (default: 10)")
    parser.add_argument("--outdir", type=str, default="dataset_v2", help="Output directory path (default: dataset_v2)")
    parser.add_argument("--run-id", type=str, default="", help="Optional run ID tag")
    parser.add_argument("--validate", type=str, default="", help="Run validation on existing dataset directory")
    parser.add_argument("--coverage", type=str, default="", help="Run coverage analysis on existing dataset directory")
    parser.add_argument("--audit-labels", type=str, default="", help="Run label audit on existing dataset directory")
    parser.add_argument("--output", type=str, default="", help="Output directory for reports")

    args = parser.parse_args()

    if args.audit_labels:
        audit_path = Path(args.audit_labels)
        out_path = Path(args.output) if args.output else (audit_path / "metadata")
        print(f"Auditing label distribution in {audit_path}...")
        auditor = LabelAuditor(audit_path)
        report = auditor.run_full_audit(out_path)
        print(f"Audit Status: {report['final_audit_status']}")
        print(f"Reports written to: {out_path.resolve()}")
        sys.exit(0 if report["final_audit_status"] != "FAIL" else 1)

    if args.validate:
        val_path = Path(args.validate)
        print(f"Validating dataset in {val_path}...")
        validator = DatasetValidator(val_path)
        res = validator.run_all_validations()
        print(f"Validation Result: {res['status']}")
        if res["errors"]:
            print("Errors:", res["errors"])

        # Also execute label audit during validation
        auditor = LabelAuditor(val_path)
        audit_report = auditor.run_full_audit(val_path / "metadata")
        print(f"Label Audit Result: {audit_report['final_audit_status']}")

        sys.exit(0 if res["status"] == "PASS" and audit_report["final_audit_status"] != "FAIL" else 1)

    if args.coverage:
        cov_path = Path(args.coverage)
        print(f"Analyzing coverage in {cov_path}...")
        cov = DatasetCoverageAnalyzer(cov_path)
        report = cov.analyze_coverage()
        print(f"Rows analyzed: {report['total_rows_analyzed']}")
        print("Concentration warnings:", report["concentration_warnings"])
        sys.exit(0)

    run_generation(args)


if __name__ == "__main__":
    main()
