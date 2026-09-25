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

        # Continuous smooth thermal wave (no discrete random jitter)
        wave = lambda cid, phase=0.0: float(np.sin(2.0 * np.pi * t_seconds / 260.0 + cid + phase))

        # All 10 computers perform similar moderate/light workloads (18-26% CPU, 4-10% GPU)
        comp_baselines = {
            1: (22.0, 6.0), 2: (20.5, 5.0), 3: (23.0, 7.5), 4: (21.5, 5.5),
            5: (19.0, 4.5), 6: (24.0, 8.0), 7: (22.5, 6.5), 8: (21.0, 5.0),
            9: (20.0, 4.8), 10: (23.5, 7.0),
        }
        workloads: Dict[int, Tuple[float, float]] = {}
        for cid, (base_cpu, base_gpu) in comp_baselines.items():
            cpu = base_cpu + 0.7 * wave(cid)
            gpu = base_gpu + 0.5 * wave(cid, phase=0.8)
            workloads[cid] = (round(min(max(cpu, 5.0), 40.0), 2), round(min(max(gpu, 0.0), 20.0), 2))

        # Occupancy gathering in Zone 3: ramps up smoothly from t=0s to t=200s
        ramp = self.smooth_ramp(t_seconds, 0.0, 200.0, 0.0, 1.0)
        
        # Zone 3 surges to 16 occupants (meeting/presentation)
        z3_occ = int(round(3 + ramp * 13))
        z1_occ = 2
        z2_occ = 2
        z4_occ = 2

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
