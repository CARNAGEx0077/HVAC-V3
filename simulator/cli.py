"""
Command-Line Interface for HVEAC V3 Thermal Simulation Data Engine.

Usage:
    python -m simulator.cli --all
    python -m simulator.cli --scenario 1
    python -m simulator.cli --scenario 3 --runs 3 --duration 3600
"""

import argparse
import sys
import time
from typing import Any, Dict, List

from simulator.config import SimulationEngineConfig
from simulator.dataset_writer import DatasetWriter
from simulator.engine import ScenarioEngine
from simulator.scenarios import ALL_SCENARIOS


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HVEAC V3 Synthetic Thermal Simulation Data Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scenario_group = parser.add_mutually_exclusive_group(required=True)
    scenario_group.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Execute a specific scenario (1 to 5)",
    )
    scenario_group.add_argument(
        "--all",
        action="store_true",
        help="Execute all 5 scenarios and build combined dataset",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of variation runs per scenario (default: 1)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=7200.0,
        help="Duration of each simulation run in seconds (default: 7200s / 2 hours)",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=10.0,
        help="Simulation timestep in seconds (default: 10s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for reproducible stochastic generation (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_dataset",
        help="Output directory for generated dataset files (default: simulation_dataset)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv,jsonl",
        help="Comma-separated output formats: csv,jsonl (default: csv,jsonl)",
    )

    return parser.parse_args()


def build_config_snapshot(config: SimulationEngineConfig) -> Dict[str, Any]:
    """Serializes engine configuration for metadata."""
    return {
        "room_geometry": {
            "width_m": config.geometry.room_width_m,
            "length_m": config.geometry.room_length_m,
            "height_m": config.geometry.room_height_m,
            "total_area_m2": config.geometry.total_area_m2,
            "total_volume_m3": config.geometry.total_volume_m3,
            "num_zones": config.geometry.num_zones,
            "num_computers": config.geometry.num_computers,
            "num_ac_units": config.geometry.num_ac_units,
        },
        "physics": {
            "air_density_kg_m3": config.physics.air_density_kg_m3,
            "air_specific_heat_j_kg_k": config.physics.air_specific_heat_j_kg_k,
            "thermal_mass_multiplier": config.physics.thermal_mass_multiplier,
            "zone_heat_capacity_j_k": config.physics.zone_heat_capacity_j_k,
            "exterior_u_area_w_k": config.physics.exterior_u_area_w_k,
            "interzone_k_adjacent_w_k": config.physics.interzone_k_adjacent_w_k,
            "interzone_k_diagonal_w_k": config.physics.interzone_k_diagonal_w_k,
        },
        "computer_power_coefficients": {
            "base_idle_power_w": config.computer.base_idle_power_w,
            "cpu_max_power_w": config.computer.cpu_max_power_w,
            "gpu_max_power_w": config.computer.gpu_max_power_w,
        },
        "occupancy": {
            "sensible_heat_per_person_w": config.occupancy.sensible_heat_per_person_w,
        },
        "hvac": {
            "nominal_cooling_capacity_w": config.hvac.nominal_cooling_capacity_w,
            "default_setpoint_c": config.hvac.default_setpoint_c,
            "spatial_influence_weights": config.hvac.spatial_influence_weights,
        },
        "optimization": {
            "target_temperature_c": config.optimization.target_temperature_c,
            "comfort_lower_bound_c": config.optimization.comfort_lower_bound_c,
            "comfort_upper_bound_c": config.optimization.comfort_upper_bound_c,
            "safe_min_temperature_c": config.optimization.safe_min_temperature_c,
            "safe_max_temperature_c": config.optimization.safe_max_temperature_c,
            "w_comfort": config.optimization.w_comfort,
            "w_safety_penalty": config.optimization.w_safety_penalty,
            "w_zone_gradient": config.optimization.w_zone_gradient,
            "w_energy": config.optimization.w_energy,
        },
    }


def main():
    args = parse_arguments()

    print("=" * 60)
    print("  HVEAC V3 THERMAL SIMULATION DATA ENGINE")
    print("=" * 60)
    print(f"Master Seed       : {args.seed}")
    print(f"Duration          : {args.duration:.0f}s ({args.duration/3600:.1f} hours)")
    print(f"Timestep          : {args.timestep:.1f}s ({int(args.duration/args.timestep)} steps/run)")
    print(f"Runs per scenario : {args.runs}")
    print(f"Output Directory  : {args.output_dir}")
    print("=" * 60)

    config = SimulationEngineConfig(
        timestep_seconds=args.timestep,
        default_duration_seconds=args.duration,
        default_seed=args.seed,
        output_dir=args.output_dir,
    )
    engine = ScenarioEngine(config)
    writer = DatasetWriter(args.output_dir)

    scenario_ids = [1, 2, 3, 4, 5] if args.all else [args.scenario]
    all_combined_records: List[Dict[str, Any]] = []
    scenario_records_map: Dict[int, List[Dict[str, Any]]] = {}

    formats = [f.strip().lower() for f in args.format.split(",")]

    start_time = time.time()

    for sid in scenario_ids:
        scenario_cls = ALL_SCENARIOS[sid]
        scenario = scenario_cls(duration_seconds=args.duration, timestep_seconds=args.timestep)
        print(f"\n[RUNNING] Scenario {sid}: {scenario.name}")
        print(f"          Description: {scenario.description}")

        scenario_records: List[Dict[str, Any]] = []
        for r_idx in range(args.runs):
            run_records = engine.run_scenario(scenario, run_idx=r_idx, seed=args.seed)
            scenario_records.extend(run_records)
            print(f"  -> Run {r_idx + 1}/{args.runs} completed: {len(run_records)} timesteps generated.")

        scenario_records_map[sid] = scenario_records
        all_combined_records.extend(scenario_records)

        # Write individual scenario files
        if "csv" in formats:
            csv_path = writer.write_csv(scenario_records, f"scenario_{sid}.csv")
            print(f"  [SAVED] CSV  : {csv_path.name} ({len(scenario_records)} rows)")
        if "jsonl" in formats:
            jsonl_path = writer.write_jsonl(scenario_records, f"scenario_{sid}.jsonl")
            print(f"  [SAVED] JSONL: {jsonl_path.name}")

    # Write combined dataset and metadata if multiple scenarios were run or requested
    if args.all or len(scenario_ids) > 1:
        if "csv" in formats:
            comb_csv = writer.write_csv(all_combined_records, "combined_dataset.csv")
            print(f"\n[SAVED] Combined CSV  : {comb_csv.name} ({len(all_combined_records)} total rows)")
        if "jsonl" in formats:
            comb_jsonl = writer.write_jsonl(all_combined_records, "combined_dataset.jsonl")
            print(f"[SAVED] Combined JSONL: {comb_jsonl.name}")

    config_snapshot = build_config_snapshot(config)
    meta_path = writer.write_metadata(
        all_combined_records,
        scenario_records_map,
        config_snapshot,
        filename="dataset_metadata.json",
    )
    print(f"[SAVED] Metadata JSON : {meta_path.name}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"SIMULATION COMPLETE in {elapsed:.2f}s")
    print(f"Total Rows Generated : {len(all_combined_records)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
