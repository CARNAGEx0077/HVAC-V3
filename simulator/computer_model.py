"""
Synthetic Computer Computational Workload and Heat Generation Model.

NOTE: All computer metrics in this module are SYNTHETIC SIMULATION VARIABLES.
They do not represent real physical measurements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from simulator.config import ComputerHeatConfig
from simulator.room import ComputerLocation, FIXED_COMPUTERS


def classify_workload(cpu_utilization: float, config: ComputerHeatConfig) -> str:
    """
    Deterministic workload classifier matching HVEAC standard thresholds:
    0 <= CPU < 10%   -> IDLE
    10 <= CPU < 30%  -> LIGHT
    30 <= CPU < 60%  -> GENERAL
    60 <= CPU < 80%  -> CPU_INTENSIVE
    80 <= CPU <= 100% -> HEAVY_CPU_LOAD
    """
    cpu = min(max(float(cpu_utilization), 0.0), 100.0)
    if cpu < config.idle_threshold:
        return "IDLE"
    elif cpu < config.light_threshold:
        return "LIGHT"
    elif cpu < config.general_threshold:
        return "GENERAL"
    elif cpu < config.cpu_intensive_threshold:
        return "CPU_INTENSIVE"
    else:
        return "HEAVY_CPU_LOAD"


@dataclass
class ComputerState:
    """Represents the instantaneous synthetic state of a single computer."""
    computer_id: int
    name: str
    zone: str
    x: float
    y: float
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    workload_category: str
    synthetic_heat_w: float


class ComputerNodeModel:
    """Models a single synthetic computer node in the room."""

    def __init__(self, location: ComputerLocation, config: Optional[ComputerHeatConfig] = None):
        self.location = location
        self.config = config or ComputerHeatConfig()
        self.cpu_utilization = 5.0
        self.gpu_utilization = 0.0
        self.workload_category = "IDLE"
        self.synthetic_heat_w = self.calculate_heat(5.0, 0.0)

    def calculate_heat(self, cpu_util: float, gpu_util: float) -> float:
        """
        Computes synthetic computational heat dissipation in Watts:
        Q_comp = P_idle + alpha_cpu * (cpu_util / 100) + alpha_gpu * (gpu_util / 100)
        """
        cpu_ratio = min(max(cpu_util, 0.0), 100.0) / 100.0
        gpu_ratio = min(max(gpu_util, 0.0), 100.0) / 100.0
        heat = (
            self.config.base_idle_power_w
            + self.config.cpu_max_power_w * cpu_ratio
            + self.config.gpu_max_power_w * gpu_ratio
        )
        return round(heat, 2)

    def set_workload(self, cpu_util: float, gpu_util: float) -> ComputerState:
        """Set simulated workload and update heat dissipation."""
        self.cpu_utilization = min(max(round(float(cpu_util), 2), 0.0), 100.0)
        self.gpu_utilization = min(max(round(float(gpu_util), 2), 0.0), 100.0)
        self.workload_category = classify_workload(self.cpu_utilization, self.config)
        self.synthetic_heat_w = self.calculate_heat(self.cpu_utilization, self.gpu_utilization)
        return self.get_state()

    def get_state(self) -> ComputerState:
        return ComputerState(
            computer_id=self.location.computer_id,
            name=self.location.name,
            zone=self.location.zone,
            x=self.location.x,
            y=self.location.y,
            cpu_utilization_percent=self.cpu_utilization,
            gpu_utilization_percent=self.gpu_utilization,
            workload_category=self.workload_category,
            synthetic_heat_w=self.synthetic_heat_w,
        )


class ComputerClusterModel:
    """Manages all 10 fixed computers in the room."""

    def __init__(self, config: Optional[ComputerHeatConfig] = None):
        self.config = config or ComputerHeatConfig()
        self.computers: Dict[int, ComputerNodeModel] = {
            loc.computer_id: ComputerNodeModel(loc, self.config)
            for loc in FIXED_COMPUTERS
        }

    def set_workloads(self, workloads: Dict[int, Tuple[float, float]]):
        """Update multiple computers: {computer_id: (cpu_pct, gpu_pct)}."""
        for cid, (cpu, gpu) in workloads.items():
            if cid in self.computers:
                self.computers[cid].set_workload(cpu, gpu)

    def get_all_states(self) -> List[ComputerState]:
        return [c.get_state() for c in sorted(self.computers.values(), key=lambda x: x.location.computer_id)]

    def get_zone_heat_watts(self) -> Dict[str, float]:
        """Aggregate total computational heat by zone in Watts."""
        zone_heat = {"ZONE_1": 0.0, "ZONE_2": 0.0, "ZONE_3": 0.0, "ZONE_4": 0.0}
        for c in self.computers.values():
            zone_heat[c.location.zone] = round(zone_heat[c.location.zone] + c.synthetic_heat_w, 2)
        return zone_heat

    def get_total_heat_watts(self) -> float:
        """Total computational heat across all 10 computers."""
        return round(sum(c.synthetic_heat_w for c in self.computers.values()), 2)
