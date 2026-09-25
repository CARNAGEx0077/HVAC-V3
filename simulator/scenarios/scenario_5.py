"""
Scenario 5: Opposing Thermal Zones.

One side of the room contains high-compute nodes (West side: ZONE_1 & ZONE_4).
The opposite side contains mostly low-compute nodes (East side: ZONE_2 & ZONE_3).
Occupancy is concentrated toward the opposite side (East side: ZONE_2 & ZONE_3).

Expected Result:
Two distinct thermal regions develop driven by fundamentally different heat sources:
computational heat in the West, and human occupancy heat in the East.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario, TimestepInput


class Scenario5OpposingZones(BaseScenario):
    """Scenario 5: Opposing thermal zones with compute vs occupancy heat sources."""

    scenario_id = 5
    name = "Opposing Thermal Zones"
    description = "West zones dominated by compute heat; East zones dominated by human occupancy"

    def get_initial_temperatures(self, run_idx: int = 0, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        noise = float(rng.uniform(-0.25, 0.25)) if rng else 0.0
        base_t = 22.0 + noise + (run_idx * 0.1)
        return {z: round(base_t, 3) for z in ZONE_IDS}

    def get_timestep_input(
        self,
        t_seconds: float,
        current_temperatures: Dict[str, float],
        run_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> TimestepInput:
        rand_val = lambda low, high: float(rng.uniform(low, high)) if rng else (low + high) / 2.0

        ramp = self.smooth_ramp(t_seconds, 240.0, 720.0, 0.0, 1.0)

        workloads: Dict[int, Tuple[float, float]] = {}
        # West side: Computers 1, 2, 3, 4 (Zone 1) and 9, 10 (Zone 4) -> Heavy Compute
        for cid in (1, 2, 3, 4, 9, 10):
            base_cpu = 30.0 + ramp * 55.0  # 30% -> 85%
            base_gpu = 20.0 + ramp * 65.0  # 20% -> 85%
            cpu = base_cpu + rand_val(-3.0, 3.0)
            gpu = base_gpu + rand_val(-3.0, 3.0)
            workloads[cid] = (round(min(max(cpu, 10.0), 98.0), 2), round(min(max(gpu, 0.0), 98.0), 2))

        # East side: Computers 5, 6 (Zone 2) and 7, 8 (Zone 3) -> Low Compute
        for cid in (5, 6, 7, 8):
            cpu = 15.0 + rand_val(-3.0, 4.0)
            gpu = 3.0 + rand_val(-2.0, 3.0)
            workloads[cid] = (round(min(max(cpu, 5.0), 25.0), 2), round(min(max(gpu, 0.0), 15.0), 2))

        # Occupancy: Concentrated in East side (ZONE_2 & ZONE_3)
        # West side (ZONE_1 & ZONE_4) has minimal occupants
        z2_occ = int(round(2 + ramp * 8 + rand_val(-0.4, 0.4)))   # 10 occupants
        z3_occ = int(round(2 + ramp * 9 + rand_val(-0.4, 0.4)))   # 11 occupants
        z1_occ = max(1, int(round(1 + rand_val(-0.3, 0.3))))     # 1 occupant
        z4_occ = max(1, int(round(1 + rand_val(-0.3, 0.3))))     # 1 occupant

        occupancy = {
            "ZONE_1": z1_occ,
            "ZONE_2": z2_occ,
            "ZONE_3": z3_occ,
            "ZONE_4": z4_occ,
        }

        # Outdoor conditions
        t_out = 27.5 + (t_seconds / self.duration_seconds) * 2.5 + (run_idx * 0.3)
        solar = 200.0 + (t_seconds / self.duration_seconds) * 120.0

        cooling = {ac_id: 0.25 for ac_id in AC_IDS}
        setpoints = {ac_id: 22.0 for ac_id in AC_IDS}

        return TimestepInput(
            computer_workloads=workloads,
            zone_occupancy=occupancy,
            outdoor_temp_c=round(t_out, 2),
            relative_humidity=54.0,
            solar_irradiance_w_m2=round(solar, 1),
            hvac_cooling_levels=cooling,
            hvac_setpoints=setpoints,
        )
