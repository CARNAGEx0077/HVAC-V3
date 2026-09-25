"""
Scenario 2: Occupancy Concentration.

All 10 computers operate under similar moderate/light workloads.
Occupancy is strongly concentrated in one region of the room (ZONE_3).

Expected Result:
A localized occupancy-driven thermal hotspot forms in ZONE_3.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario, TimestepInput


class Scenario2OccupancyConcentration(BaseScenario):
    """Scenario 2: Dense Occupancy Concentration in Zone 3."""

    scenario_id = 2
    name = "Occupancy Concentration"
    description = "Crowded meeting concentrated in Zone 3 with uniform light computer workloads"

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

        # All 10 computers perform similar moderate/light workloads (18-28% CPU, 2-10% GPU)
        workloads: Dict[int, Tuple[float, float]] = {}
        for cid in range(1, 11):
            cpu = 22.0 + rand_val(-4.0, 5.0)
            gpu = 5.0 + rand_val(-3.0, 4.0)
            workloads[cid] = (round(min(max(cpu, 5.0), 40.0), 2), round(min(max(gpu, 0.0), 20.0), 2))

        # Occupancy gathering in Zone 3: ramps up from t=300s to t=900s
        ramp = self.smooth_ramp(t_seconds, 300.0, 900.0, 0.0, 1.0)
        
        # Zone 3 surges to 16 occupants (meeting/seminar)
        z3_occ = int(round(3 + ramp * 13))
        z1_occ = max(1, int(round(2 + rand_val(-0.5, 0.5))))
        z2_occ = max(1, int(round(2 + rand_val(-0.5, 0.5))))
        z4_occ = max(1, int(round(2 + rand_val(-0.5, 0.5))))

        occupancy = {
            "ZONE_1": z1_occ,
            "ZONE_2": z2_occ,
            "ZONE_3": z3_occ,
            "ZONE_4": z4_occ,
        }

        # Outdoor conditions
        t_out = 26.5 + (t_seconds / self.duration_seconds) * 2.5 + (run_idx * 0.3)
        solar = 180.0 + (t_seconds / self.duration_seconds) * 120.0

        cooling = {ac_id: 0.25 for ac_id in AC_IDS}
        setpoints = {ac_id: 22.0 for ac_id in AC_IDS}

        return TimestepInput(
            computer_workloads=workloads,
            zone_occupancy=occupancy,
            outdoor_temp_c=round(t_out, 2),
            relative_humidity=55.0,
            solar_irradiance_w_m2=round(solar, 1),
            hvac_cooling_levels=cooling,
            hvac_setpoints=setpoints,
        )
