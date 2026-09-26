"""
Zone-Based Lumped Thermal Network Model.

Simulates heat balances across 4 thermal zones with inter-zone conduction,
external envelope transmission, solar gains, internal loads (computers + occupants),
and distributed HVAC heat removal.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from generator.config import RoomGeometryConfig
from generator.models.room import ZONES, ZONE_ADJACENCIES


@dataclass
class ZoneThermalState:
    current_temperature_c: float
    occupancy: int
    computer_heat_watts: float
    occupancy_heat_watts: float
    environmental_heat_transfer_watts: float
    hvac_cooling_watts: float
    inter_zone_heat_watts: float
    net_heat_watts: float


@dataclass
class RoomThermalSummary:
    zone_1_temperature_c: float
    zone_2_temperature_c: float
    zone_3_temperature_c: float
    zone_4_temperature_c: float
    room_average_temperature_c: float
    minimum_temperature_c: float
    maximum_temperature_c: float
    temperature_difference_c: float


class ThermalModel:
    """Simulates multi-zone thermal dynamics with continuous physical evolution."""

    def __init__(self, config: RoomGeometryConfig = None, initial_temp_c: float = 25.0):
        self.config = config or RoomGeometryConfig()
        # Initialize zone temperatures
        self.zone_temperatures: Dict[str, float] = {
            zone: float(initial_temp_c) for zone in ZONES
        }
        self.last_zone_states: Dict[str, ZoneThermalState] = {}

    def set_zone_temperatures(self, temps: Dict[str, float]):
        """Explicitly set initial zone temperatures."""
        for z, t in temps.items():
            if z in self.zone_temperatures:
                self.zone_temperatures[z] = float(t)

    def calculate_inter_zone_heat(self, temps: Dict[str, float]) -> Dict[str, float]:
        """Compute conductive/convective heat flux between neighboring thermal zones.

        Q_ij = k * (T_j - T_i)
        Positive indicates net heat flowing into zone i.
        """
        k_adj = self.config.adjacent_zone_conductance_w_per_k
        k_diag = self.config.diagonal_zone_conductance_w_per_k
        flux: Dict[str, float] = {z: 0.0 for z in ZONES}

        # ZONE_1
        flux["ZONE_1"] += k_adj * (temps["ZONE_2"] - temps["ZONE_1"])
        flux["ZONE_1"] += k_adj * (temps["ZONE_4"] - temps["ZONE_1"])
        flux["ZONE_1"] += k_diag * (temps["ZONE_3"] - temps["ZONE_1"])

        # ZONE_2
        flux["ZONE_2"] += k_adj * (temps["ZONE_1"] - temps["ZONE_2"])
        flux["ZONE_2"] += k_adj * (temps["ZONE_3"] - temps["ZONE_2"])
        flux["ZONE_2"] += k_diag * (temps["ZONE_4"] - temps["ZONE_2"])

        # ZONE_3
        flux["ZONE_3"] += k_adj * (temps["ZONE_2"] - temps["ZONE_3"])
        flux["ZONE_3"] += k_adj * (temps["ZONE_4"] - temps["ZONE_3"])
        flux["ZONE_3"] += k_diag * (temps["ZONE_1"] - temps["ZONE_3"])

        # ZONE_4
        flux["ZONE_4"] += k_adj * (temps["ZONE_1"] - temps["ZONE_4"])
        flux["ZONE_4"] += k_adj * (temps["ZONE_3"] - temps["ZONE_4"])
        flux["ZONE_4"] += k_diag * (temps["ZONE_2"] - temps["ZONE_4"])

        return {z: round(v, 2) for z, v in flux.items()}

    def step(
        self,
        computer_heat: Dict[str, float],
        occupancy_heat: Dict[str, float],
        occupancy_counts: Dict[str, int],
        cooling_watts: Dict[str, float],
        outdoor_temp_c: float,
        solar_loads: Dict[str, float],
        timestep_seconds: float,
    ) -> RoomThermalSummary:
        """Advance thermal simulation by one timestep using 4-zone energy balance."""
        c_zone = self.config.zone_thermal_capacitance_j_per_k
        u_wall = self.config.exterior_wall_conductance_w_per_k

        inter_zone_heat = self.calculate_inter_zone_heat(self.zone_temperatures)
        new_temps = {}

        for zone in ZONES:
            t_curr = self.zone_temperatures[zone]

            # Environmental envelope transmission + solar gain
            q_env = u_wall * (outdoor_temp_c - t_curr) + solar_loads.get(zone, 0.0)

            q_comp = computer_heat.get(zone, 0.0)
            q_occ = occupancy_heat.get(zone, 0.0)
            q_inter = inter_zone_heat[zone]
            q_cool = cooling_watts.get(zone, 0.0)

            # Net thermal heat rate into zone (Watts)
            q_net = q_comp + q_occ + q_env + q_inter - q_cool

            # Explicit Euler thermal update (very stable at C ~ 3.6e5 J/K and dt = 10s)
            delta_t = (q_net * timestep_seconds) / c_zone
            t_new = t_curr + delta_t
            new_temps[zone] = float(t_new)

            # Record detailed state
            self.last_zone_states[zone] = ZoneThermalState(
                current_temperature_c=round(t_new, 4),
                occupancy=occupancy_counts.get(zone, 0),
                computer_heat_watts=round(q_comp, 2),
                occupancy_heat_watts=round(q_occ, 2),
                environmental_heat_transfer_watts=round(q_env, 2),
                hvac_cooling_watts=round(q_cool, 2),
                inter_zone_heat_watts=round(q_inter, 2),
                net_heat_watts=round(q_net, 2),
            )

        self.zone_temperatures = new_temps
        return self.get_summary()

    def get_summary(self) -> RoomThermalSummary:
        """Calculate and return aggregated room thermal metrics."""
        t1 = self.zone_temperatures["ZONE_1"]
        t2 = self.zone_temperatures["ZONE_2"]
        t3 = self.zone_temperatures["ZONE_3"]
        t4 = self.zone_temperatures["ZONE_4"]

        temps = [t1, t2, t3, t4]
        avg_temp = float(np.mean(temps))
        min_temp = float(np.min(temps))
        max_temp = float(np.max(temps))
        diff_temp = max_temp - min_temp

        return RoomThermalSummary(
            zone_1_temperature_c=round(t1, 3),
            zone_2_temperature_c=round(t2, 3),
            zone_3_temperature_c=round(t3, 3),
            zone_4_temperature_c=round(t4, 3),
            room_average_temperature_c=round(avg_temp, 3),
            minimum_temperature_c=round(min_temp, 3),
            maximum_temperature_c=round(max_temp, 3),
            temperature_difference_c=round(diff_temp, 3),
        )
