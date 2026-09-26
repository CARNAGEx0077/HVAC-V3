"""
Outdoor Environment Model.

Generates outdoor temperature, humidity, and solar thermal load.
Simulates smooth, gradual temporal variation without abrupt jumps,
calibrated to warm/humid Indian prototype baseline while fully configurable.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

from generator.config import EnvironmentConfig
from generator.models.room import ZONES


@dataclass
class EnvironmentState:
    outdoor_temperature_c: float
    humidity_percent: float
    solar_load_watts: float

    def zone_solar_loads(self) -> Dict[str, float]:
        """Distribute solar radiation by zone orientation.
        ZONE_1: North-West, ZONE_2: North-East, ZONE_3: South-East, ZONE_4: South-West.
        South and West receive highest thermal gain in hot climates.
        """
        return {
            "ZONE_1": round(self.solar_load_watts * 0.15, 2),
            "ZONE_2": round(self.solar_load_watts * 0.20, 2),
            "ZONE_3": round(self.solar_load_watts * 0.35, 2),
            "ZONE_4": round(self.solar_load_watts * 0.30, 2),
        }


class EnvironmentModel:
    """Simulates realistic, continuously evolving outdoor meteorological conditions."""

    def __init__(
        self,
        config: EnvironmentConfig = None,
        base_temp_c: float = 34.0,
        base_humidity_pct: float = 65.0,
        base_solar_watts: float = 500.0,
        time_of_day_hours: float = 13.0,  # 1:00 PM afternoon peak
    ):
        self.config = config or EnvironmentConfig()
        self.base_temp_c = base_temp_c
        self.base_humidity_pct = base_humidity_pct
        self.base_solar_watts = base_solar_watts
        self.time_of_day_hours = time_of_day_hours

        # Internal state
        self.current_temp_c = base_temp_c
        self.current_humidity_pct = base_humidity_pct
        self.current_solar_watts = base_solar_watts

        # Stochastic continuous drift variables
        self._temp_drift = 0.0
        self._humidity_drift = 0.0

    def step(self, elapsed_seconds: float, timestep_seconds: float, rng: np.random.Generator) -> EnvironmentState:
        """Advance outdoor conditions smoothly over timestep using diurnal variation and OU drift."""
        sim_time_hours = self.time_of_day_hours + (elapsed_seconds / 3600.0)

        # Smooth diurnal cycle (24 hour period)
        # Temperature peaks around 14:30 (2:30 PM), minimum around 05:00
        diurnal_rad = 2.0 * np.pi * (sim_time_hours - 14.5) / 24.0
        diurnal_temp = 3.5 * np.cos(diurnal_rad)
        # Humidity is inversely related to temperature
        diurnal_humidity = -5.0 * np.cos(diurnal_rad)

        # Solar load curve (zero at night, peak around 12:30 - 13:30)
        solar_rad = 2.0 * np.pi * (sim_time_hours - 13.0) / 24.0
        solar_factor = max(0.0, np.cos(solar_rad))

        # Ornstein-Uhlenbeck mean-reverting drift (very gentle, e.g. ~0.01 deg per 10s step)
        theta = 0.02
        sigma_temp = 0.03
        sigma_hum = 0.05

        self._temp_drift += -theta * self._temp_drift * timestep_seconds + sigma_temp * np.sqrt(timestep_seconds) * rng.normal()
        self._humidity_drift += -theta * self._humidity_drift * timestep_seconds + sigma_hum * np.sqrt(timestep_seconds) * rng.normal()

        # Combine deterministic diurnal + stochastic drift
        t_out = self.base_temp_c + diurnal_temp + self._temp_drift
        h_out = self.base_humidity_pct + diurnal_humidity + self._humidity_drift
        s_out = self.base_solar_watts * (solar_factor ** 1.3)

        # Clamp within configured bounds
        self.current_temp_c = float(np.clip(t_out, self.config.min_outdoor_temp_c, self.config.max_outdoor_temp_c))
        self.current_humidity_pct = float(np.clip(h_out, self.config.min_humidity_percent, self.config.max_humidity_percent))
        self.current_solar_watts = float(np.clip(s_out, self.config.min_solar_flux_w_per_m2, self.config.max_solar_flux_w_per_m2))

        return EnvironmentState(
            outdoor_temperature_c=round(self.current_temp_c, 2),
            humidity_percent=round(self.current_humidity_pct, 2),
            solar_load_watts=round(self.current_solar_watts, 2),
        )
