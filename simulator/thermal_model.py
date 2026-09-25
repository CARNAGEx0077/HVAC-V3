"""
4-Zone Lumped-Capacitance Thermal Dynamic Model for HVEAC V3.

Implements the fundamental energy balance per thermal zone:
    Q_net,z = Q_comp,z + Q_occ,z + Q_env,z + Q_interzone,z - Q_hvac,z
    dT_z/dt = Q_net,z / C_zone

Tracks:
- zone_1_temperature, zone_2_temperature, zone_3_temperature, zone_4_temperature
- room_average_temperature
- max_min_temperature_difference (thermal gradient)
- inter-zone heat transfer with exact energy conservation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from simulator.config import ThermalPhysicsConfig
from simulator.room import ZONE_IDS


@dataclass
class ThermalModelState:
    """Complete thermal state of all zones at a specific timestep."""
    zone_temperatures: Dict[str, float]
    room_average_temperature: float
    max_zone_temperature: float
    min_zone_temperature: float
    temperature_gradient: float
    
    # Net heat fluxes per zone (Watts)
    zone_comp_heat_w: Dict[str, float]
    zone_occ_heat_w: Dict[str, float]
    zone_env_heat_w: Dict[str, float]
    zone_interzone_heat_w: Dict[str, float]
    zone_hvac_cooling_w: Dict[str, float]
    zone_net_heat_w: Dict[str, float]
    
    # Cluster totals (Watts)
    total_comp_heat_w: float
    total_occ_heat_w: float
    total_env_heat_w: float
    total_hvac_cooling_w: float
    total_net_heat_w: float


class ThermalModel:
    """Simulates multi-zone thermal dynamics and heat exchange."""

    def __init__(
        self,
        physics_config: Optional[ThermalPhysicsConfig] = None,
        initial_temperatures: Optional[Dict[str, float]] = None,
    ):
        self.physics = physics_config or ThermalPhysicsConfig()
        default_t = 22.5
        self.temperatures: Dict[str, float] = {
            z: round(float(initial_temperatures.get(z, default_t)), 3)
            if initial_temperatures
            else default_t
            for z in ZONE_IDS
        }

    def reset(self, initial_temperatures: Optional[Dict[str, float]] = None):
        """Reset zone temperatures."""
        default_t = 22.5
        self.temperatures = {
            z: round(float(initial_temperatures.get(z, default_t)), 3)
            if initial_temperatures
            else default_t
            for z in ZONE_IDS
        }

    def calculate_interzone_heat_transfer(self) -> Dict[str, float]:
        """
        Computes heat exchange between zones.
        Adjacency:
        - ZONE_1 adjacent to ZONE_2 & ZONE_4 (k_adj), diagonal to ZONE_3 (k_diag)
        - ZONE_2 adjacent to ZONE_1 & ZONE_3 (k_adj), diagonal to ZONE_4 (k_diag)
        - ZONE_3 adjacent to ZONE_2 & ZONE_4 (k_adj), diagonal to ZONE_1 (k_diag)
        - ZONE_4 adjacent to ZONE_1 & ZONE_3 (k_adj), diagonal to ZONE_2 (k_diag)
        Guarantees exact sum(Q_interzone) == 0.
        """
        t = self.temperatures
        k_adj = self.physics.interzone_k_adjacent_w_k
        k_diag = self.physics.interzone_k_diagonal_w_k

        # Q_transfer = k * (T_neighbor - T_self)
        q1 = k_adj * (t["ZONE_2"] - t["ZONE_1"]) + k_adj * (t["ZONE_4"] - t["ZONE_1"]) + k_diag * (t["ZONE_3"] - t["ZONE_1"])
        q2 = k_adj * (t["ZONE_1"] - t["ZONE_2"]) + k_adj * (t["ZONE_3"] - t["ZONE_2"]) + k_diag * (t["ZONE_4"] - t["ZONE_2"])
        q3 = k_adj * (t["ZONE_2"] - t["ZONE_3"]) + k_adj * (t["ZONE_4"] - t["ZONE_3"]) + k_diag * (t["ZONE_1"] - t["ZONE_3"])
        q4 = k_adj * (t["ZONE_1"] - t["ZONE_4"]) + k_adj * (t["ZONE_3"] - t["ZONE_4"]) + k_diag * (t["ZONE_2"] - t["ZONE_4"])

        return {
            "ZONE_1": round(q1, 2),
            "ZONE_2": round(q2, 2),
            "ZONE_3": round(q3, 2),
            "ZONE_4": round(q4, 2),
        }

    def step(
        self,
        dt_seconds: float,
        comp_heat_by_zone: Dict[str, float],
        occ_heat_by_zone: Dict[str, float],
        env_heat_by_zone: Dict[str, float],
        hvac_cooling_by_zone: Dict[str, float],
    ) -> ThermalModelState:
        """
        Advance thermal simulation by dt_seconds.
        Updates zone temperatures and returns complete snapshot.
        """
        c_zone = self.physics.zone_heat_capacity_j_k
        interzone = self.calculate_interzone_heat_transfer()

        zone_net: Dict[str, float] = {}
        for z in ZONE_IDS:
            q_comp = comp_heat_by_zone.get(z, 0.0)
            q_occ = occ_heat_by_zone.get(z, 0.0)
            q_env = env_heat_by_zone.get(z, 0.0)
            q_inter = interzone.get(z, 0.0)
            q_cool = hvac_cooling_by_zone.get(z, 0.0)

            # Energy balance: Q_net = Q_comp + Q_occ + Q_env + Q_inter - Q_cooling
            net_heat = q_comp + q_occ + q_env + q_inter - q_cool
            zone_net[z] = round(net_heat, 2)

            # Temperature evolution: dT = (Q_net * dt) / C_zone
            delta_t = (net_heat * dt_seconds) / c_zone
            new_temp = self.temperatures[z] + delta_t
            self.temperatures[z] = round(new_temp, 3)

        temps_list = list(self.temperatures.values())
        avg_temp = round(sum(temps_list) / len(temps_list), 3)
        max_temp = round(max(temps_list), 3)
        min_temp = round(min(temps_list), 3)
        gradient = round(max_temp - min_temp, 3)

        total_comp = round(sum(comp_heat_by_zone.values()), 2)
        total_occ = round(sum(occ_heat_by_zone.values()), 2)
        total_env = round(sum(env_heat_by_zone.values()), 2)
        total_cooling = round(sum(hvac_cooling_by_zone.values()), 2)
        total_net = round(sum(zone_net.values()), 2)

        return ThermalModelState(
            zone_temperatures=dict(self.temperatures),
            room_average_temperature=avg_temp,
            max_zone_temperature=max_temp,
            min_zone_temperature=min_temp,
            temperature_gradient=gradient,
            zone_comp_heat_w=dict(comp_heat_by_zone),
            zone_occ_heat_w=dict(occ_heat_by_zone),
            zone_env_heat_w=dict(env_heat_by_zone),
            zone_interzone_heat_w=interzone,
            zone_hvac_cooling_w=dict(hvac_cooling_by_zone),
            zone_net_heat_w=zone_net,
            total_comp_heat_w=total_comp,
            total_occ_heat_w=total_occ,
            total_env_heat_w=total_env,
            total_hvac_cooling_w=total_cooling,
            total_net_heat_w=total_net,
        )
