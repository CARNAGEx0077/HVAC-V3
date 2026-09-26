"""
Configuration module for HVEAC Synthetic Labeled Dataset Generator.

Contains all constants, physical assumptions, comfort parameters,
and optimization weights.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json


VERSION_METADATA = {
    "generator_version": "2.0.0",
    "thermal_model_version": "2.0.0",
    "comfort_model_version": "2.0.0-PMV-ISO7730",
    "hvac_model_version": "2.0.0",
    "optimizer_version": "2.0.0-directionally-coherent",
}


@dataclass
class RoomGeometryConfig:
    """Fixed room physical dimensions and thermal properties."""
    width_m: float = 10.0   # East-West
    length_m: float = 10.0  # North-South
    height_m: float = 3.0   # Floor to ceiling
    total_volume_m3: float = 300.0
    zone_volume_m3: float = 75.0  # 4 equal zones

    # Thermal capacitance parameters
    air_heat_capacity_j_per_kg_k: float = 1005.0
    air_density_kg_per_m3: float = 1.20
    # Internal thermal mass multiplier (furniture, walls, equipment casing)
    internal_thermal_mass_multiplier: float = 4.0

    @property
    def zone_thermal_capacitance_j_per_k(self) -> float:
        """Effective thermal capacitance per zone in J/K."""
        return (
            self.zone_volume_m3
            * self.air_density_kg_per_m3
            * self.air_heat_capacity_j_per_kg_k
            * self.internal_thermal_mass_multiplier
        )

    # Inter-zone thermal coupling (conductive and convective air exchange)
    adjacent_zone_conductance_w_per_k: float = 120.0
    diagonal_zone_conductance_w_per_k: float = 25.0

    # Building envelope thermal characteristics
    exterior_wall_conductance_w_per_k: float = 35.0
    solar_aperture_m2: float = 2.0  # Effective window/radiation aperture per zone


@dataclass
class ComputerHeatConfig:
    """Synthetic computer power and heat dissipation model.

    Power dissipation follows the nonlinear empirical relation:
        P_comp = Base_Idle + Max_CPU * (CPU/100)^1.10 + Max_GPU * (GPU/100)^1.15
    With Base_Idle = 60W, Max_CPU = 160W, Max_GPU = 240W:
    - Theoretical maximum power (at 100% CPU, 100% GPU): 60 + 160 + 240 = 460.0 W.
    - Practical maximum power in dataset (HEAVY workload at 98% CPU, 98% GPU): ~450.97 W (~451 W).
    """
    base_idle_watts: float = 60.0
    max_cpu_watts: float = 160.0
    max_gpu_watts: float = 240.0
    cpu_exponent: float = 1.1
    gpu_exponent: float = 1.15

    theoretical_max_watts: float = 460.0
    practical_max_watts: float = 450.97

    # Workload utilization bounds: (min_cpu, max_cpu, min_gpu, max_gpu)
    workload_ranges: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "IDLE": (3.0, 12.0, 0.0, 5.0),
            "LIGHT": (12.0, 30.0, 5.0, 20.0),
            "GENERAL": (30.0, 65.0, 15.0, 45.0),
            "CPU_INTENSIVE": (75.0, 98.0, 10.0, 30.0),
            "GPU_INTENSIVE": (25.0, 55.0, 80.0, 98.0),
            "HEAVY": (80.0, 98.0, 80.0, 98.0),
        }
    )


@dataclass
class OccupancyConfig:
    """Occupancy heat dissipation assumptions."""
    sensible_heat_per_person_watts: float = 95.0
    latent_heat_per_person_watts: float = 40.0
    max_zone_occupancy: int = 15
    max_total_occupancy: int = 40


@dataclass
class EnvironmentConfig:
    """Configurable outdoor environmental ranges (Warm/Humid Indian baseline prototype)."""
    min_outdoor_temp_c: float = 28.0
    max_outdoor_temp_c: float = 42.0
    default_outdoor_temp_c: float = 34.0

    min_humidity_percent: float = 45.0
    max_humidity_percent: float = 85.0
    default_humidity_percent: float = 65.0

    min_solar_flux_w_per_m2: float = 0.0
    max_solar_flux_w_per_m2: float = 800.0


@dataclass
class HVACConfig:
    """HVAC equipment control space and operational constraints (V2 Authoritative)."""
    candidate_setpoints: Tuple[float, ...] = (
        21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0, 25.5
    )
    cooling_capacity_watts: float = 4500.0  # ~1.28 TR per AC unit (x4 = ~5.1 TR total)
    max_cooling_level: float = 1.0
    min_cooling_level: float = 0.0
    ramp_rate_per_second: float = 0.02      # max change in cooling level per second (0.20 per 10s timestep)
    min_on_seconds: float = 120.0
    min_off_seconds: float = 120.0
    cop: float = 3.4                        # Coefficient of performance
    proportional_gain: float = 0.35         # Inverter response gain (level per deg C delta)

    # Temporal stability, dwell, hysteresis, and ramp rate constraints
    minimum_setpoint_dwell_seconds: float = 60.0
    setpoint_hysteresis_cost: float = 0.03
    setpoint_hysteresis_c: float = 0.5
    maximum_setpoint_change_per_step_c: float = 0.5
    minimum_cooling_dwell_seconds: float = 10.0
    maximum_cooling_change_per_step: float = 0.20  # 0.02 level/sec * 10s timestep


@dataclass
class ComfortConfig:
    """Comfort model assumptions based on ISO 7730 / ASHRAE 55 and Indian prototypes (V2)."""
    air_speed_m_s: float = 0.15
    metabolic_rate_met: float = 1.1         # Typing / seated office work
    clothing_insulation_clo: float = 0.5    # Lightweight summer indoor clothing
    target_pmv: float = 0.0
    acceptable_pmv_range: Tuple[float, float] = (-0.5, 0.5)

    # Indian prototype reference points
    bee_reference_setpoint_c: float = 24.0
    target_reference_setpoint_c: float = 22.5
    hyderabad_study_reference_c: float = 26.1
    chennai_study_reference_c: float = 27.0

    # V2 Operative comfort envelope bounds
    lower_comfort_threshold_c: float = 21.0  # Lower acceptable boundary
    upper_comfort_threshold_c: float = 24.5  # Upper acceptable boundary (overheating threshold)


@dataclass
class OptimizationWeights:
    """V2 Directionally Coherent Multi-Objective Optimization Weights.

    Priorities:
    1. Thermal safety & Overheating prevention (w_overheat=15.0, w_dir=20.0)
    2. Acceptable comfort (w_comfort=3.5, w_overcool=8.0)
    3. Stable control & Feasibility (w_switch=0.3)
    4. Energy efficiency (w_energy=0.35 - secondary to thermal correctness)
    """
    comfort_weight: float = 3.5
    overheat_weight: float = 15.0
    overcool_weight: float = 8.0
    directional_weight: float = 20.0
    energy_weight: float = 0.35
    switch_weight: float = 0.3

    # Lookahead horizon for candidate evaluation
    evaluation_horizon_seconds: float = 120.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "comfort_weight": self.comfort_weight,
            "overheat_weight": self.overheat_weight,
            "overcool_weight": self.overcool_weight,
            "directional_weight": self.directional_weight,
            "energy_weight": self.energy_weight,
            "switch_weight": self.switch_weight,
            "evaluation_horizon_seconds": self.evaluation_horizon_seconds,
        }


@dataclass
class SimulationConfig:
    """Global configuration for scenario simulation run."""
    duration_seconds: int = 7200   # 2 hours
    timestep_seconds: int = 10     # 10s timestep
    random_seed: int = 42
    run_id: str = "run_default"
    num_scenarios: int = 100

    # Dataset partition ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Models and sub-configs
    room: RoomGeometryConfig = field(default_factory=RoomGeometryConfig)
    computer: ComputerHeatConfig = field(default_factory=ComputerHeatConfig)
    occupancy: OccupancyConfig = field(default_factory=OccupancyConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    hvac: HVACConfig = field(default_factory=HVACConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    weights: OptimizationWeights = field(default_factory=OptimizationWeights)
