"""
HVEAC V3 - Node Telemetry Pydantic Data Contracts
Strict schema validation with full nullable sensor support
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SystemInfoSchema(BaseModel):
    hostname: str
    platform: str = "Windows"
    os_version: Optional[str] = None
    architecture: Optional[str] = None
    agent_version: Optional[str] = None
    uptime_seconds: Optional[int] = None


class HardwareProfileSchema(BaseModel):
    cpu_model: Optional[str] = None
    physical_cores: Optional[int] = None
    logical_processors: Optional[int] = None
    memory_total_gb: Optional[float] = None
    gpu_count: Optional[int] = 0


class CpuDataSchema(BaseModel):
    utilization_percent: float = Field(..., ge=0.0, le=100.0)
    temperature_c: Optional[float] = None
    frequency_mhz: Optional[float] = None
    power_watts: Optional[float] = None
    temperature_source: Optional[str] = None
    temperature_sensor_name: Optional[str] = None
    power_source: Optional[str] = None
    workload_category: Optional[str] = "IDLE"
    thermal_load_index: Optional[float] = None
    short_term_average_percent: Optional[float] = None
    sustained_load_seconds: Optional[int] = 0
    is_sustained_high_load: Optional[bool] = False
    thermal_signal_state: Optional[str] = "UNAVAILABLE"


class GpuDeviceSchema(BaseModel):
    id: int
    name: str
    utilization_percent: float = Field(0.0, ge=0.0, le=100.0)
    memory_utilization_percent: Optional[float] = None
    temperature_c: Optional[float] = None
    power_watts: Optional[float] = None
    power_limit_watts: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    core_clock_mhz: Optional[float] = None
    memory_clock_mhz: Optional[float] = None
    fan_percent: Optional[float] = None
    fan_rpm: Optional[float] = None


class GpuDataSchema(BaseModel):
    available: bool = False
    gpus: List[GpuDeviceSchema] = Field(default_factory=list)


class MemoryDataSchema(BaseModel):
    total_gb: float
    used_gb: float
    available_gb: float
    utilization_percent: float = Field(..., ge=0.0, le=100.0)


class DriveDataSchema(BaseModel):
    path: str
    total_gb: float
    used_gb: float
    free_gb: float
    utilization_percent: float = Field(..., ge=0.0, le=100.0)


class StorageDataSchema(BaseModel):
    system_drive: Optional[DriveDataSchema] = None


class DisplayDeviceSchema(BaseModel):
    display_id: str
    display_name: str
    resolution_width: Optional[int] = None
    resolution_height: Optional[int] = None
    resolution: Optional[str] = None
    refresh_rate_hz: Optional[float] = None
    brightness_percent: Optional[float] = None


class DisplayDataSchema(BaseModel):
    display_count: int = 0
    displays: List[DisplayDeviceSchema] = Field(default_factory=list)


class WorkloadDataSchema(BaseModel):
    foreground_application: Optional[str] = None
    top_cpu_processes: List[str] = Field(default_factory=list)
    category: str = "IDLE"
    heuristic_score: Optional[float] = None


class ThermalDataSchema(BaseModel):
    estimated_power_watts: Optional[float] = None
    estimated_heat_watts: Optional[float] = None
    node_thermal_load_index: Optional[float] = None
    signal_state: Optional[str] = "UNAVAILABLE"
    estimation_status: Optional[str] = "UNAVAILABLE"
    sources: List[str] = Field(default_factory=list)


class NodeTelemetryPayload(BaseModel):
    node_id: str
    timestamp: str
    sequence: int
    schema_version: str = "1.0"
    system: SystemInfoSchema
    hardware: HardwareProfileSchema
    cpu: CpuDataSchema
    gpu: GpuDataSchema
    memory: MemoryDataSchema
    storage: StorageDataSchema
    display: DisplayDataSchema
    workload: WorkloadDataSchema
    thermal: ThermalDataSchema
