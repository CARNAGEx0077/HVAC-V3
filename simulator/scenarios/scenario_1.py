"""
Scenario 1: Localized Heavy Compute.

Four physically nearby computers form a cluster:
Computers 1, 2, 3, 4 (all physically located in ZONE_1).
These computers perform sustained heavy workloads (80-95% CPU/GPU).
Computers 5-10 perform simple background workloads (10-25% CPU).
Occupancy is distributed approximately evenly across all 4 zones.

Expected Result:
A localized computational thermal hotspot forms around computers 1-4 in ZONE_1.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from simulator.room import AC_IDS, ZONE_IDS
from simulator.scenarios.base_scenario import BaseScenario, TimestepInput


class Scenario1LocalizedCompute(BaseScenario):
    """Scenario 1: Localized Heavy Compute Cluster."""

    scenario_id = 1
    name = "Localized Heavy Compute"
    description = "Intense computational cluster in Zone 1 (Computers 1-4) with even occupancy"

    def get_initial_temperatures(self, run_idx: int = 0, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        noise = float(rng.uniform(-0.3, 0.3)) if rng else 0.0
        base_t = 22.2 + noise + (run_idx * 0.15)
        return {z: round(base_t, 3) for z in ZONE_IDS}

    def get_timestep_input(
        self,
        t_seconds: float,
        current_temperatures: Dict[str, float],
        run_idx: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> TimestepInput:
        rand_val = lambda low, high: float(rng.uniform(low, high)) if rng else (low + high) / 2.0

        # Computers 1-4 in ZONE_1 ramp smoothly from baseline into HEAVY workload (t=0s to t=200s)
        ramp = self.smooth_ramp(t_seconds, 0.0, 200.0, 0.0, 1.0)
        
        # Smooth continuous thermal wave (no discrete random jitter)
        wave = lambda cid, phase=0.0: float(np.sin(2.0 * np.pi * t_seconds / 280.0 + cid + phase))

        workloads: Dict[int, Tuple[float, float]] = {}
        # Workstation specs for Cluster (C1-C4) in ZONE_1: (init_cpu, target_cpu, init_gpu, target_gpu)
        cluster_specs = {
            1: (20.0, 92.4, 10.0, 84.1),
            2: (18.0, 88.5, 8.0, 79.2),
            3: (22.0, 91.0, 12.0, 82.5),
            4: (21.0, 89.8, 11.0, 81.0),
        }
        for cid, (c_init, c_target, g_init, g_target) in cluster_specs.items():
            cpu = c_init + ramp * (c_target - c_init) + 0.8 * wave(cid)
            gpu = g_init + ramp * (g_target - g_init) + 0.6 * wave(cid, phase=1.0)
            workloads[cid] = (round(min(max(cpu, 5.0), 99.0), 2), round(min(max(gpu, 0.0), 99.0), 2))

        # Computers 5-10: simple background workload (15-20% CPU, 3-7% GPU)
        bg_specs = {
            5: (16.5, 5.0),
            6: (18.0, 5.5),
            7: (15.0, 4.0),
            8: (19.0, 6.5),
            9: (17.5, 4.5),
            10: (16.0, 3.8),
        }
        for cid, (base_cpu, base_gpu) in bg_specs.items():
            cpu = base_cpu + 0.6 * wave(cid)
            gpu = base_gpu + 0.4 * wave(cid, phase=0.5)
            workloads[cid] = (round(min(max(cpu, 5.0), 30.0), 2), round(min(max(gpu, 0.0), 20.0), 2))

        # Occupancy: evenly distributed (3 to 4 people per zone)
        occupancy = {
            "ZONE_1": 3,
            "ZONE_2": 4,
            "ZONE_3": 3,
            "ZONE_4": 4,
        }

        # Outdoor temperature: warm day progressing from 27.5 C to 30.5 C
        t_out = 27.5 + (t_seconds / self.duration_seconds) * 3.0 + (run_idx * 0.4)
        solar = 200.0 + (t_seconds / self.duration_seconds) * 150.0

        # Baseline uniform HVAC cooling (25% on all units)
        cooling = {ac_id: 0.25 for ac_id in AC_IDS}
        setpoints = {ac_id: 22.0 for ac_id in AC_IDS}

        return TimestepInput(
            computer_workloads=workloads,
            zone_occupancy=occupancy,
            outdoor_temp_c=round(t_out, 2),
            relative_humidity=52.0,
            solar_irradiance_w_m2=round(solar, 1),
            hvac_cooling_levels=cooling,
            hvac_setpoints=setpoints,
        )
