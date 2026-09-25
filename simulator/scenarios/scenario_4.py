"""
Scenario 4: High Occupancy / Low Computer Load.

All 10 computers perform low/simple workloads (5-15% CPU).
Occupancy is high and distributed throughout all zones (total 30-36 people).

Expected Result:
Occupancy is the dominant thermal contributor across all zones.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario, TimestepInput


class Scenario4HighOccupancyLowCompute(BaseScenario):
    """Scenario 4: Classroom/Auditorium high occupancy with low computing load."""

    scenario_id = 4
    name = "High Occupancy / Low Computer Load"
    description = "Crowded lecture room with minimal computer usage; metabolic heat dominates"

    def get_initial_temperatures(self, run_idx: int = 0, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        noise = float(rng.uniform(-0.2, 0.2)) if rng else 0.0
        base_t = 21.8 + noise + (run_idx * 0.1)
        return {z: round(base_t, 3) for z in ZONE_IDS}

    def get_timestep_input(
        self,
        t_seconds: float,
        current_temperatures: Dict[str, float],
        run_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> TimestepInput:
        rand_val = lambda low, high: float(rng.uniform(low, high)) if rng else (low + high) / 2.0

        # All 10 computers idle/simple (8-14% CPU, 1-3% GPU) with smooth continuous wave
        wave = lambda cid, phase=0.0: float(np.sin(2.0 * np.pi * t_seconds / 250.0 + cid + phase))
        idle_baselines = {
            1: (10.0, 2.0), 2: (9.0, 1.5), 3: (11.0, 2.5), 4: (9.5, 1.8),
            5: (8.5, 1.2), 6: (12.0, 3.0), 7: (10.5, 2.2), 8: (9.0, 1.6),
            9: (8.0, 1.0), 10: (11.5, 2.8),
        }
        workloads: Dict[int, Tuple[float, float]] = {}
        for cid, (base_cpu, base_gpu) in idle_baselines.items():
            cpu = base_cpu + 0.5 * wave(cid)
            gpu = base_gpu + 0.3 * wave(cid, phase=0.7)
            workloads[cid] = (round(min(max(cpu, 5.0), 20.0), 2), round(min(max(gpu, 0.0), 10.0), 2))

        # Occupancy fills the room: ramps up smoothly between t=0s and t=200s
        ramp = self.smooth_ramp(t_seconds, 0.0, 200.0, 0.0, 1.0)

        # 8-9 people per zone -> 34 people total (~2900 W metabolic heat vs ~600 W computer heat)
        z1_occ = int(round(3 + ramp * 5))
        z2_occ = int(round(3 + ramp * 6))
        z3_occ = int(round(3 + ramp * 6))
        z4_occ = int(round(3 + ramp * 5))

        occupancy = {
            "ZONE_1": z1_occ,
            "ZONE_2": z2_occ,
            "ZONE_3": z3_occ,
            "ZONE_4": z4_occ,
        }

        # Outdoor conditions
        t_out = 27.0 + (t_seconds / self.duration_seconds) * 2.0 + (run_idx * 0.3)
        solar = 160.0 + (t_seconds / self.duration_seconds) * 100.0

        cooling = {ac_id: 0.25 for ac_id in AC_IDS}
        setpoints = {ac_id: 22.0 for ac_id in AC_IDS}

        return TimestepInput(
            computer_workloads=workloads,
            zone_occupancy=occupancy,
            outdoor_temp_c=round(t_out, 2),
            relative_humidity=58.0,
            solar_irradiance_w_m2=round(solar, 1),
            hvac_cooling_levels=cooling,
            hvac_setpoints=setpoints,
        )
