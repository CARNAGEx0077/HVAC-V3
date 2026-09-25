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

        # All 10 computers ramp into heavy compute between t=300s and t=900s
        ramp = self.smooth_ramp(t_seconds, 300.0, 900.0, 0.0, 1.0)

        workloads: Dict[int, Tuple[float, float]] = {}
        for cid in range(1, 11):
            base_cpu = 35.0 + ramp * 52.0  # 35% -> 87%
            base_gpu = 20.0 + ramp * 65.0  # 20% -> 85%
            cpu = base_cpu + rand_val(-3.5, 3.5)
            gpu = base_gpu + rand_val(-3.5, 3.5)
            workloads[cid] = (round(min(max(cpu, 10.0), 98.0), 2), round(min(max(gpu, 0.0), 98.0), 2))

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
