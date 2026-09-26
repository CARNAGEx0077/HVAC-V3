"""
Simulator Metadata Generator.

Produces comprehensive dataset_metadata.json recording all physical constants,
geometry layouts, equipment specifications, comfort parameters, and split rules.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from generator.config import SimulationConfig, VERSION_METADATA
from generator.models.room import AC_DEFINITIONS, COMPUTER_DEFINITIONS, ZONES


def build_dataset_metadata(
    config: SimulationConfig,
    scenario_count: int,
    scenario_splits: Dict[str, str],
    total_rows: int,
    split_row_counts: Dict[str, int],
    split_seed: int = 42,
    scenario_families: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Assemble complete metadata payload adhering strictly to Section 33 and Part 1 Section 2."""
    scenario_families = scenario_families or {}

    # Group scenario IDs by split
    scenario_ids_by_split = {
        "train": sorted([s for s, sp in scenario_splits.items() if sp == "train"]),
        "validation": sorted([s for s, sp in scenario_splits.items() if sp == "val"]),
        "test": sorted([s for s, sp in scenario_splits.items() if sp == "test"]),
    }

    # Count family distribution per split
    family_counts_by_split = {"train": {}, "validation": {}, "test": {}}
    for split_name, s_ids in scenario_ids_by_split.items():
        counts = {}
        for s_id in s_ids:
            fam = scenario_families.get(s_id, "UNKNOWN")
            counts[fam] = counts.get(fam, 0) + 1
        family_counts_by_split[split_name] = counts

    metadata = {
        # Versions
        "generator_version": VERSION_METADATA["generator_version"],
        "thermal_model_version": VERSION_METADATA["thermal_model_version"],
        "comfort_model_version": VERSION_METADATA["comfort_model_version"],
        "hvac_model_version": VERSION_METADATA["hvac_model_version"],
        "optimizer_version": VERSION_METADATA["optimizer_version"],

        # Run Configuration
        "run_id": config.run_id,
        "random_seed": config.random_seed,
        "split_seed": split_seed,
        "scenario_count": scenario_count,
        "scenario_duration_seconds": config.duration_seconds,
        "timestep_seconds": config.timestep_seconds,
        "total_timesteps_per_scenario": int(config.duration_seconds / config.timestep_seconds),
        "total_dataset_rows": total_rows,

        # Partitioning Strategy (Part 1 requirements)
        "dataset_split_strategy": {
            "method": "family_stratified_scenario_disjoint_split",
            "split_seed": split_seed,
            "target_proportions": {
                "train": config.train_ratio,
                "validation": config.val_ratio,
                "test": config.test_ratio,
            },
            "actual_scenario_counts": {
                "train": len(scenario_ids_by_split["train"]),
                "validation": len(scenario_ids_by_split["validation"]),
                "test": len(scenario_ids_by_split["test"]),
            },
            "actual_row_counts": split_row_counts,
            "scenario_family_counts_by_split": family_counts_by_split,
            "scenario_ids_by_split": scenario_ids_by_split,
        },

        # Physical Geometry & Fixed Layouts
        "room_geometry": {
            "shape": "rectangular",
            "width_m": config.room.width_m,
            "length_m": config.room.length_m,
            "height_m": config.room.height_m,
            "total_volume_m3": config.room.total_volume_m3,
            "zones": ZONES,
            "zone_volume_m3": config.room.zone_volume_m3,
            "zone_thermal_capacitance_j_per_k": config.room.zone_thermal_capacitance_j_per_k,
            "adjacent_coupling_conductance_w_k": config.room.adjacent_zone_conductance_w_per_k,
            "diagonal_coupling_conductance_w_k": config.room.diagonal_zone_conductance_w_per_k,
        },

        # AC Equipment Layout
        "AC_positions": AC_DEFINITIONS,

        # Fixed Computer Placement
        "computer_positions": COMPUTER_DEFINITIONS,

        # Comfort Assumptions & Indian Prototype Baselines
        "comfort_assumptions": {
            "model_type": "ISO 7730 / ASHRAE 55 PMV-PPD",
            "air_speed_m_s": config.comfort.air_speed_m_s,
            "metabolic_rate_met": config.comfort.metabolic_rate_met,
            "clothing_insulation_clo": config.comfort.clothing_insulation_clo,
            "target_pmv": config.comfort.target_pmv,
            "acceptable_pmv_range": list(config.comfort.acceptable_pmv_range),
            "indian_baseline_references": {
                "bee_recommended_c": config.comfort.bee_reference_setpoint_c,
                "hyderabad_study_mean_c": config.comfort.hyderabad_study_reference_c,
                "chennai_study_mean_c": config.comfort.chennai_study_reference_c,
                "note": "Prototype references only. Not universal ground truth."
            },
            "lower_comfort_threshold_c": config.comfort.lower_comfort_threshold_c,
            "upper_comfort_threshold_c": config.comfort.upper_comfort_threshold_c,
        },

        # Equipment & Simulation Physical Constants
        "simulation_constants": {
            "computer_idle_watts": config.computer.base_idle_watts,
            "computer_max_cpu_watts": config.computer.max_cpu_watts,
            "computer_max_gpu_watts": config.computer.max_gpu_watts,
            "computer_theoretical_max_watts": getattr(config.computer, "theoretical_max_watts", 460.0),
            "computer_practical_max_watts": getattr(config.computer, "practical_max_watts", 450.97),
            "computer_model_formula": "P_comp = Base_Idle (60W) + Max_CPU (160W) * (CPU/100)^1.10 + Max_GPU (240W) * (GPU/100)^1.15",
            "computer_model_doc_note": "Authoritative v1.1 model; reaches ~451W in HEAVY workloads (theoretical max 460W). Obsolete ~320W documentation superseded.",
            "occupancy_sensible_watts_per_person": config.occupancy.sensible_heat_per_person_watts,
            "occupancy_latent_watts_per_person": config.occupancy.latent_heat_per_person_watts,
            "exterior_wall_conductance_w_k": config.room.exterior_wall_conductance_w_per_k,
            "hvac_unit_cooling_capacity_watts": config.hvac.cooling_capacity_watts,
            "hvac_ramp_rate_per_sec": config.hvac.ramp_rate_per_second,
            "hvac_max_cooling_ramp_per_step": getattr(config.hvac, "maximum_cooling_change_per_step", 0.20),
            "hvac_min_setpoint_dwell_seconds": getattr(config.hvac, "minimum_setpoint_dwell_seconds", 60.0),
            "hvac_setpoint_hysteresis_cost": getattr(config.hvac, "setpoint_hysteresis_cost", 0.03),
            "hvac_max_setpoint_change_per_step_c": getattr(config.hvac, "maximum_setpoint_change_per_step_c", 0.5),
            "hvac_min_on_seconds": config.hvac.min_on_seconds,
            "hvac_min_off_seconds": config.hvac.min_off_seconds,
        },

        # Optimization & Action Space
        "action_space": {
            "candidate_setpoints_c": list(config.hvac.candidate_setpoints),
            "step_size_c": 0.5,
            "cooling_level_range": [0.0, 1.0],
        },
        "optimization_weights": config.weights.as_dict(),
        "optimization_formulation": (
            "TOTAL_COST = comfort_weight * comfort_penalty + overheat_weight * overheating_penalty + "
            "overcool_weight * overcooling_penalty + directional_weight * directional_penalty + "
            "energy_weight * energy_penalty + switch_weight * switching_penalty"
        ),
    }
    return metadata


def save_dataset_metadata(metadata: Dict[str, Any], filepath: Path):
    """Write metadata dictionary to JSON file with indentation."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
