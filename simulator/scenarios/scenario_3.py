"""
Scenario 3: Distributed Heavy Compute.

All 10 computers perform heavy workloads (80-95% CPU/GPU).
The computers are spatially distributed across all 4 zones.
Occupancy is approximately evenly distributed.

Expected Result:
Thermal load is broadly distributed throughout the entire room rather than
forming one dominant computer cluster hotspot.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario, TimestepInput


class Scenario3DistributedCompute(BaseScenario):
    """Scenario 3: Distributed Heavy Compute across all nodes."""

    scenario_id = 3
    name = "Distributed Heavy Compute"
    description = "Full distributed compute cluster load across all 10 nodes with even occupancy"

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

        # All 10 computers ramp into heavy compute smoothly from t=0s to t=200s
        ramp = self.smooth_ramp(t_seconds, 0.0, 200.0, 0.0, 1.0)
        wave = lambda cid, phase=0.0: float(np.sin(2.0 * np.pi * t_seconds / 270.0 + cid + phase))

        # Individual initial and target operating points per machine
        node_specs = {
            1: (20.0, 92.0, 10.0, 83.0),
            2: (19.0, 88.5, 9.0, 80.0),
            3: (21.0, 90.5, 11.0, 81.5),
            4: (20.5, 89.0, 10.5, 82.0),
            5: (18.5, 91.5, 8.5, 84.0),
            6: (22.0, 93.5, 12.0, 85.0),
            7: (19.5, 89.5, 9.5, 81.0),
            8: (21.5, 92.5, 11.5, 83.5),
            9: (18.0, 88.0, 8.0, 79.0),
            10: (20.0, 90.0, 10.0, 82.5),
        }
        workloads: Dict[int, Tuple[float, float]] = {}
        for cid, (c_init, c_target, g_init, g_target) in node_specs.items():
            cpu = c_init + ramp * (c_target - c_init) + 0.8 * wave(cid)
            gpu = g_init + ramp * (g_target - g_init) + 0.6 * wave(cid, phase=1.2)
            workloads[cid] = (round(min(max(cpu, 10.0), 99.0), 2), round(min(max(gpu, 0.0), 99.0), 2))

        # Occupancy: evenly distributed (3 to 4 people per zone)
        occupancy = {
            "ZONE_1": 3,
            "ZONE_2": 3,
            "ZONE_3": 4,
            "ZONE_4": 4,
        }

        # Outdoor conditions
        t_out = 28.0 + (t_seconds / self.duration_seconds) * 2.5 + (run_idx * 0.3)
        solar = 220.0 + (t_seconds / self.duration_seconds) * 100.0

        cooling = {ac_id: 0.30 for ac_id in AC_IDS}
        setpoints = {ac_id: 22.0 for ac_id in AC_IDS}

        return TimestepInput(
            computer_workloads=workloads,
            zone_occupancy=occupancy,
            outdoor_temp_c=round(t_out, 2),
            relative_humidity=50.0,
            solar_irradiance_w_m2=round(solar, 1),
            hvac_cooling_levels=cooling,
            hvac_setpoints=setpoints,
        )
