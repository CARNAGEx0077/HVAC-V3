"""
Configuration and Physical Parameters for HVEAC V3 Thermal Simulation.

Defines:
- Geometry constants (12m x 10m x 3m room, 4 zones)
- Thermodynamic constants (air density, specific heat, thermal mass multiplier)
- Computer heat model parameters (synthetic power coefficients)
- Occupancy heat model parameters (ASHRAE sensible heat per person)
- Envelope and inter-zone heat transfer coefficients
- HVAC capacities and spatial influence matrix
- Comfort and optimization objectives
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RoomGeometryConfig:
    """Fixed room spatial configuration."""
    room_width_m: float = 12.0     # X-axis (West to East)
    room_length_m: float = 10.0    # Y-axis (South to North)
    room_height_m: float = 3.0     # Ceiling height
    num_zones: int = 4
    num_computers: int = 10
    num_ac_units: int = 4

    @property
    def total_area_m2(self) -> float:
        return self.room_width_m * self.room_length_m

    @property
    def total_volume_m3(self) -> float:
        return self.total_area_m2 * self.room_height_m

    @property
    def zone_area_m2(self) -> float:
        return self.total_area_m2 / self.num_zones

    @property
    def zone_volume_m3(self) -> float:
        return self.total_volume_m3 / self.num_zones


@dataclass(frozen=True)
class ThermalPhysicsConfig:
    """Lumped-capacitance thermodynamic parameters."""
    air_density_kg_m3: float = 1.204              # Standard dry air density at 20 C
    air_specific_heat_j_kg_k: float = 1005.0       # Air specific heat capacity (J/(kg*K))
    thermal_mass_multiplier: float = 4.5          # Accounting for furniture, walls, partition thermal capacitance
    
    # Exterior envelope heat transmission (U * Area in W/K per zone)
    # Zone exterior perimeter: North/South 6m x 3m (18 m2) + East/West 5m x 3m (15 m2) = 33 m2
    # Standard insulated commercial envelope U-value ~ 0.65 W/(m2*K) -> ~ 21.5 W/K
    exterior_u_area_w_k: float = 21.5
    
    # Inter-zone convective/conductive heat transfer coefficients (W/K)
    # Interior boundary: 5m x 3m (15 m2) or 6m x 3m (18 m2) open office space
    interzone_k_adjacent_w_k: float = 110.0       # Direct adjacent zones (1-2, 2-3, 3-4, 4-1)
    interzone_k_diagonal_w_k: float = 15.0        # Cross-room diagonal diffusion (1-3, 2-4)

    @property
    def zone_heat_capacity_j_k(self) -> float:
        # C_zone = Volume * density * c_p * thermal_mass_multiplier
        # 90 m3 * 1.204 kg/m3 * 1005 J/kgK * 4.5 ~= 490,000 J/K
        return 90.0 * self.air_density_kg_m3 * self.air_specific_heat_j_kg_k * self.thermal_mass_multiplier


@dataclass(frozen=True)
class ComputerHeatConfig:
    """
    Synthetic computer heat dissipation model coefficients.
    Clearly marked as SIMULATED / SYNTHETIC parameters.
    """
    base_idle_power_w: float = 45.0               # Idle motherboard, storage, fan baseline
    cpu_max_power_w: float = 125.0                # Added dissipation at 100% CPU utilization
    gpu_max_power_w: float = 150.0                # Added dissipation at 100% GPU utilization
    
    # Workload category boundaries (CPU utilization percent)
    idle_threshold: float = 10.0
    light_threshold: float = 30.0
    general_threshold: float = 60.0
    cpu_intensive_threshold: float = 80.0


@dataclass(frozen=True)
class OccupancyHeatConfig:
    """Occupant sensible heat dissipation model."""
    sensible_heat_per_person_w: float = 85.0      # ASHRAE standard sensible heat for office activity


@dataclass(frozen=True)
class HvacConfig:
    """Air conditioning unit operational specifications."""
    nominal_cooling_capacity_w: float = 3500.0    # ~12,000 BTU/h (~3.5 kW) per AC unit
    min_cooling_level: float = 0.0                # 0.0 = OFF
    max_cooling_level: float = 1.0                # 1.0 = Maximum cooling output
    default_setpoint_c: float = 22.0              # Default target thermostat setpoint
    
    # Spatial influence matrix: AC k to Zone z (rows: AC-1..4, cols: Zone 1..4)
    # AC-1 (North): Zone 1: 0.45, Zone 2: 0.45, Zone 3: 0.05, Zone 4: 0.05
    # AC-2 (East):  Zone 1: 0.05, Zone 2: 0.45, Zone 3: 0.45, Zone 4: 0.05
    # AC-3 (South): Zone 1: 0.05, Zone 2: 0.05, Zone 3: 0.45, Zone 4: 0.45
    # AC-4 (West):  Zone 1: 0.45, Zone 2: 0.05, Zone 3: 0.05, Zone 4: 0.45
    spatial_influence_weights: Tuple[Tuple[float, ...], ...] = (
        (0.45, 0.45, 0.05, 0.05),  # AC-1 (North)
        (0.05, 0.45, 0.45, 0.05),  # AC-2 (East)
        (0.05, 0.05, 0.45, 0.45),  # AC-3 (South)
        (0.45, 0.05, 0.05, 0.45),  # AC-4 (West)
    )


@dataclass(frozen=True)
class OptimizationConfig:
    """Multi-objective HVAC target optimizer parameters."""
    target_temperature_c: float = 22.5
    comfort_lower_bound_c: float = 21.0
    comfort_upper_bound_c: float = 24.0
    safe_min_temperature_c: float = 20.0
    safe_max_temperature_c: float = 25.0
    
    # Objective weights in multi-objective cost function J(u)
    w_comfort: float = 1.0            # Tracking deviation from 22.5 C
    w_safety_penalty: float = 10.0    # Heavy quadratic penalty for exceeding safe bounds
    w_zone_gradient: float = 2.0      # Penalty for temperature disparity between zones
    w_energy: float = 0.3             # Penalty for high cooling level (energy conservation)


@dataclass(frozen=True)
class SimulationEngineConfig:
    """Master simulation runtime configuration."""
    timestep_seconds: float = 10.0                # 10s default timestep
    default_duration_seconds: float = 7200.0      # 2 hours (720 timesteps)
    default_seed: int = 42
    output_dir: str = "simulation_dataset"
    
    geometry: RoomGeometryConfig = field(default_factory=RoomGeometryConfig)
    physics: ThermalPhysicsConfig = field(default_factory=ThermalPhysicsConfig)
    computer: ComputerHeatConfig = field(default_factory=ComputerHeatConfig)
    occupancy: OccupancyHeatConfig = field(default_factory=OccupancyHeatConfig)
    hvac: HvacConfig = field(default_factory=HvacConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
