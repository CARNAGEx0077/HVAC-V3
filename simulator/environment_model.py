"""
Outdoor Meteorological and Envelope Heat Transfer Model for HVEAC V3.

Simulates:
- Outdoor dry-bulb temperature
- Relative humidity
- Solar irradiance load on building envelope
- Conductive heat exchange across external perimeter walls
"""

from dataclasses import dataclass
from typing import Dict, Optional
from simulator.config import ThermalPhysicsConfig
from simulator.room import ZONE_IDS


@dataclass
class EnvironmentState:
    """Atmospheric conditions and envelope thermal fluxes."""
    outdoor_temperature_c: float
    relative_humidity_percent: float
    solar_irradiance_w_m2: float
    zone_solar_heat_w: Dict[str, float]
    zone_transmission_heat_w: Dict[str, float]
    total_envelope_heat_w: float


class EnvironmentModel:
    """Calculates ambient meteorological conditions and exterior wall heat transfer."""

    def __init__(self, physics_config: Optional[ThermalPhysicsConfig] = None):
        self.physics = physics_config or ThermalPhysicsConfig()
        self.outdoor_temp_c = 28.0
        self.relative_humidity = 55.0
        self.solar_irradiance_w_m2 = 300.0

    def set_conditions(
        self,
        outdoor_temp_c: float,
        relative_humidity: float = 55.0,
        solar_irradiance_w_m2: float = 250.0,
    ):
        """Set current atmospheric state."""
        self.outdoor_temp_c = round(float(outdoor_temp_c), 2)
        self.relative_humidity = min(max(round(float(relative_humidity), 2), 0.0), 100.0)
        self.solar_irradiance_w_m2 = max(round(float(solar_irradiance_w_m2), 2), 0.0)

    def calculate_envelope_heat(self, zone_temperatures: Dict[str, float]) -> EnvironmentState:
        """
        Calculates heat transfer through building envelope into each zone:
        Q_env,z = (U*A)_ext * (T_outdoor - T_z) + Q_solar,z
        """
        # Distribute solar gain: south and east zones receive slightly higher solar exposure
        # Window / facade solar absorption: ~3.0 m2 equivalent window aperture per perimeter
        solar_weights = {
            "ZONE_1": 0.15,  # North-West
            "ZONE_2": 0.25,  # North-East
            "ZONE_3": 0.35,  # South-East (high afternoon solar)
            "ZONE_4": 0.25,  # South-West
        }

        transmission_heat = {}
        solar_heat = {}
        for z in ZONE_IDS:
            t_zone = zone_temperatures.get(z, 22.0)
            # Transmission through exterior wall: (U * Area) * (T_out - T_zone)
            q_trans = self.physics.exterior_u_area_w_k * (self.outdoor_temp_c - t_zone)
            # Solar gain: solar irradiance * solar aperture area factor
            q_sol = self.solar_irradiance_w_m2 * 2.5 * solar_weights[z]
            transmission_heat[z] = round(q_trans, 2)
            solar_heat[z] = round(q_sol, 2)

        total_envelope = round(
            sum(transmission_heat.values()) + sum(solar_heat.values()), 2
        )

        return EnvironmentState(
            outdoor_temperature_c=self.outdoor_temp_c,
            relative_humidity_percent=self.relative_humidity,
            solar_irradiance_w_m2=self.solar_irradiance_w_m2,
            zone_solar_heat_w=solar_heat,
            zone_transmission_heat_w=transmission_heat,
            total_envelope_heat_w=total_envelope,
        )
