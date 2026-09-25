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

        # Ramp smoothly from t=0s to t=200s
        ramp = self.smooth_ramp(t_seconds, 0.0, 200.0, 0.0, 1.0)
        wave = lambda cid, phase=0.0: float(np.sin(2.0 * np.pi * t_seconds / 280.0 + cid + phase))

        workloads: Dict[int, Tuple[float, float]] = {}
        # West side: Computers 1, 2, 3, 4 (Zone 1) and 9, 10 (Zone 4) -> Heavy Compute
        west_specs = {
            1: (20.0, 92.0, 10.0, 83.0),
            2: (19.0, 88.5, 9.0, 80.0),
            3: (21.0, 90.5, 11.0, 81.5),
            4: (20.5, 89.0, 10.5, 82.0),
            9: (19.0, 88.0, 9.0, 80.0),
            10: (20.0, 91.0, 10.0, 82.5),
        }
        for cid, (c_init, c_target, g_init, g_target) in west_specs.items():
            cpu = c_init + ramp * (c_target - c_init) + 0.8 * wave(cid)
            gpu = g_init + ramp * (g_target - g_init) + 0.6 * wave(cid, phase=1.0)
            workloads[cid] = (round(min(max(cpu, 10.0), 99.0), 2), round(min(max(gpu, 0.0), 99.0), 2))

        # East side: Computers 5, 6 (Zone 2) and 7, 8 (Zone 3) -> Low Compute
        east_specs = {
            5: (16.0, 4.5),
            6: (17.5, 5.0),
            7: (15.5, 4.0),
            8: (18.0, 5.5),
        }
        for cid, (base_cpu, base_gpu) in east_specs.items():
            cpu = base_cpu + 0.6 * wave(cid)
            gpu = base_gpu + 0.4 * wave(cid, phase=0.5)
            workloads[cid] = (round(min(max(cpu, 5.0), 25.0), 2), round(min(max(gpu, 0.0), 15.0), 2))

        # Occupancy: Concentrated in East side (ZONE_2 & ZONE_3)
        # West side (ZONE_1 & ZONE_4) has minimal occupants
        z2_occ = int(round(2 + ramp * 8))   # 10 occupants
        z3_occ = int(round(2 + ramp * 9))   # 11 occupants
        z1_occ = 1
        z4_occ = 1

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
