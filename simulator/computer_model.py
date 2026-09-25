"""
Synthetic Computer Computational Workload and Heat Generation Model.

NOTE: All computer metrics in this module are SYNTHETIC SIMULATION VARIABLES.
They do not represent real physical measurements.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from simulator.config import ComputerHeatConfig
from simulator.room import ComputerLocation, FIXED_COMPUTERS


def classify_workload(cpu_utilization: float, gpu_utilization: float = 0.0, config: Optional[ComputerHeatConfig] = None) -> str:
    """
    Deterministic workload classifier matching HVEAC standard thresholds:
    - HEAVY: CPU >= 75% AND GPU >= 60%
    - CPU_INTENSIVE: CPU >= 75%
    - GPU_INTENSIVE: GPU >= 60%
    - GENERAL: 30% <= CPU < 75% or 25% <= GPU < 60%
    - LIGHT: 10% <= CPU < 30% or 8% <= GPU < 25%
    - IDLE: CPU < 10% and GPU < 8%
    """
    cpu = min(max(float(cpu_utilization), 0.0), 100.0)
    gpu = min(max(float(gpu_utilization), 0.0), 100.0)
    
    if cpu >= 75.0 and gpu >= 60.0:
        return "HEAVY"
    elif cpu >= 75.0:
        return "CPU_INTENSIVE"
    elif gpu >= 60.0:
        return "GPU_INTENSIVE"
    elif cpu >= 30.0 or gpu >= 25.0:
        return "GENERAL"
    elif cpu >= 10.0 or gpu >= 8.0:
        return "LIGHT"
    else:
        return "IDLE"


def derive_thermal_contribution(heat_watts: float) -> str:
    """Derives simulated thermal contribution indicator from dissipated heat."""
    if heat_watts >= 240.0:
        return "VERY HIGH"
    elif heat_watts >= 170.0:
        return "HIGH"
    elif heat_watts >= 100.0:
        return "MODERATE"
    else:
        return "LOW"


def derive_simulated_status(workload_category: str, cpu_util: float = 0.0) -> str:
    """Derives simulated computer node operational status."""
    if workload_category in ("HEAVY", "CPU_INTENSIVE", "GPU_INTENSIVE") or cpu_util >= 75.0:
        return "HEAVY LOAD"
    elif workload_category == "IDLE" or cpu_util < 10.0:
        return "IDLE"
    else:
        return "RUNNING"


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
    thermal_contribution: str = "LOW"
    status: str = "RUNNING"


class ComputerNodeModel:
    """Models a single synthetic computer node in the room."""

    def __init__(self, location: ComputerLocation, config: Optional[ComputerHeatConfig] = None):
        self.location = location
        self.config = config or ComputerHeatConfig()
        self.cpu_utilization = 5.0
        self.gpu_utilization = 0.0
        self.workload_category = "IDLE"
        self.synthetic_heat_w = self.calculate_heat(5.0, 0.0)
        self.thermal_contribution = derive_thermal_contribution(self.synthetic_heat_w)
        self.status = derive_simulated_status(self.workload_category, self.cpu_utilization)

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
        self.workload_category = classify_workload(self.cpu_utilization, self.gpu_utilization, self.config)
        self.synthetic_heat_w = self.calculate_heat(self.cpu_utilization, self.gpu_utilization)
        self.thermal_contribution = derive_thermal_contribution(self.synthetic_heat_w)
        self.status = derive_simulated_status(self.workload_category, self.cpu_utilization)
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
            thermal_contribution=self.thermal_contribution,
            status=self.status,
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
