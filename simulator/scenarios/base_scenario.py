"""
Base Scenario Class and Dynamic Input Definitions for HVEAC V3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, FIXED_COMPUTERS, ZONE_IDS


@dataclass
class TimestepInput:
    """External drivers applied to the room at a given timestep."""
    computer_workloads: Dict[int, Tuple[float, float]]  # {computer_id: (cpu_util, gpu_util)}
    zone_occupancy: Dict[str, int]                      # {zone_id: occupant_count}
    outdoor_temp_c: float
    relative_humidity: float
    solar_irradiance_w_m2: float
    hvac_cooling_levels: Dict[str, float]               # {ac_id: level} (baseline cooling profile)
    hvac_setpoints: Dict[str, float]                    # {ac_id: setpoint_c}


class BaseScenario(ABC):
    """Abstract base for all simulation scenarios."""

    scenario_id: int = 0
    name: str = "Base Scenario"
    description: str = "Base scenario description"

    def __init__(self, duration_seconds: float = 7200.0, timestep_seconds: float = 10.0):
        self.duration_seconds = duration_seconds
        self.timestep_seconds = timestep_seconds

    @abstractmethod
    def get_initial_temperatures(self, run_idx: int = 0, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Return starting zone temperatures for this scenario/run."""
        pass

    @abstractmethod
    def get_timestep_input(
        self,
        t_seconds: float,
        current_temperatures: Dict[str, float],
        run_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> TimestepInput:
        """Calculate inputs for a specific simulation timestep."""
        pass

    @staticmethod
    def smooth_ramp(t: float, t_start: float, t_end: float, val_start: float, val_end: float) -> float:
        """Hermite cubic smooth interpolation between two values."""
        if t <= t_start:
            return val_start
        if t >= t_end:
            return val_end
        alpha = (t - t_start) / (t_end - t_start)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return val_start + smooth_alpha * (val_end - val_start)

    @staticmethod
    def default_hvac_levels() -> Dict[str, float]:
        """Default baseline cooling level of 0.25 (modest background cooling)."""
        return {ac_id: 0.25 for ac_id in AC_IDS}

    @staticmethod
    def default_hvac_setpoints() -> Dict[str, float]:
        return {ac_id: 22.0 for ac_id in AC_IDS}
