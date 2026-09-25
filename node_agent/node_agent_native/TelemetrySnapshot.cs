using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace HveacNodeAgentNative
{
    public class TelemetryPayload
    {
        [JsonPropertyName("node_id")]
        public string NodeId { get; set; } = string.Empty;

        [JsonPropertyName("timestamp")]
        public string Timestamp { get; set; } = string.Empty;

        [JsonPropertyName("sequence")]
        public long Sequence { get; set; }

        [JsonPropertyName("schema_version")]
        public string SchemaVersion { get; set; } = "1.0";

        [JsonPropertyName("system")]
        public SystemInfo System { get; set; } = new();

        [JsonPropertyName("hardware")]
        public HardwareProfile Hardware { get; set; } = new();

        [JsonPropertyName("cpu")]
        public CpuData Cpu { get; set; } = new();

        [JsonPropertyName("gpu")]
        public GpuData Gpu { get; set; } = new();

        [JsonPropertyName("memory")]
        public MemoryData Memory { get; set; } = new();

        [JsonPropertyName("storage")]
        public StorageData Storage { get; set; } = new();

        [JsonPropertyName("display")]
        public DisplayData Display { get; set; } = new();

        [JsonPropertyName("workload")]
        public WorkloadData Workload { get; set; } = new();

        [JsonPropertyName("thermal")]
        public ThermalData Thermal { get; set; } = new();
    }

    public class SystemInfo
    {
        [JsonPropertyName("hostname")]
        public string Hostname { get; set; } = string.Empty;

        [JsonPropertyName("platform")]
        public string Platform { get; set; } = "Windows";

        [JsonPropertyName("os_version")]
        public string OsVersion { get; set; } = string.Empty;

        [JsonPropertyName("architecture")]
        public string Architecture { get; set; } = string.Empty;

        [JsonPropertyName("agent_version")]
        public string AgentVersion { get; set; } = string.Empty;

        [JsonPropertyName("uptime_seconds")]
        public long UptimeSeconds { get; set; }
    }

    public class HardwareProfile
    {
        [JsonPropertyName("cpu_model")]
        public string CpuModel { get; set; } = string.Empty;

        [JsonPropertyName("physical_cores")]
        public int PhysicalCores { get; set; }

        [JsonPropertyName("logical_processors")]
        public int LogicalProcessors { get; set; }

        [JsonPropertyName("memory_total_gb")]
        public double MemoryTotalGb { get; set; }

        [JsonPropertyName("gpu_count")]
        public int GpuCount { get; set; }
    }

    public class CpuData
    {
        [JsonPropertyName("utilization_percent")]
        public double UtilizationPercent { get; set; }

        [JsonPropertyName("temperature_c")]
        public double? TemperatureC { get; set; }

        [JsonPropertyName("frequency_mhz")]
        public double? FrequencyMhz { get; set; }

        [JsonPropertyName("power_watts")]
        public double? PowerWatts { get; set; }

        [JsonPropertyName("temperature_source")]
        public string? TemperatureSource { get; set; }

        [JsonPropertyName("temperature_sensor_name")]
        public string? TemperatureSensorName { get; set; }

        [JsonPropertyName("power_source")]
        public string? PowerSource { get; set; }

        [JsonPropertyName("workload_category")]
        public string WorkloadCategory { get; set; } = "IDLE";

        [JsonPropertyName("thermal_load_index")]
        public double ThermalLoadIndex { get; set; }

        [JsonPropertyName("short_term_average_percent")]
        public double ShortTermAveragePercent { get; set; }

        [JsonPropertyName("sustained_load_seconds")]
        public long SustainedLoadSeconds { get; set; }

        [JsonPropertyName("is_sustained_high_load")]
        public bool IsSustainedHighLoad { get; set; }

        [JsonPropertyName("thermal_signal_state")]
        public string ThermalSignalState { get; set; } = "UNAVAILABLE";
    }

    public class GpuData
    {
        [JsonPropertyName("available")]
        public bool Available { get; set; }

        [JsonPropertyName("gpus")]
        public List<GpuDeviceData> Gpus { get; set; } = new();
    }

    public class GpuDeviceData
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("utilization_percent")]
        public double UtilizationPercent { get; set; }

        [JsonPropertyName("memory_utilization_percent")]
        public double MemoryUtilizationPercent { get; set; }

        [JsonPropertyName("temperature_c")]
        public double? TemperatureC { get; set; }

        [JsonPropertyName("power_watts")]
        public double? PowerWatts { get; set; }

        [JsonPropertyName("power_limit_watts")]
        public double? PowerLimitWatts { get; set; }

        [JsonPropertyName("memory_used_mb")]
        public double MemoryUsedMb { get; set; }

        [JsonPropertyName("memory_total_mb")]
        public double MemoryTotalMb { get; set; }

        [JsonPropertyName("core_clock_mhz")]
        public double? CoreClockMhz { get; set; }

        [JsonPropertyName("memory_clock_mhz")]
        public double? MemoryClockMhz { get; set; }

        [JsonPropertyName("fan_percent")]
        public double? FanPercent { get; set; }

        [JsonPropertyName("fan_rpm")]
        public double? FanRpm { get; set; }
    }

    public class MemoryData
    {
        [JsonPropertyName("total_gb")]
        public double TotalGb { get; set; }

        [JsonPropertyName("used_gb")]
        public double UsedGb { get; set; }

        [JsonPropertyName("available_gb")]
        public double AvailableGb { get; set; }

        [JsonPropertyName("utilization_percent")]
        public double UtilizationPercent { get; set; }
    }

    public class StorageData
    {
        [JsonPropertyName("system_drive")]
        public DriveData? SystemDrive { get; set; }
    }

    public class DriveData
    {
        [JsonPropertyName("path")]
        public string Path { get; set; } = string.Empty;

        [JsonPropertyName("total_gb")]
        public double TotalGb { get; set; }

        [JsonPropertyName("used_gb")]
        public double UsedGb { get; set; }

        [JsonPropertyName("free_gb")]
        public double FreeGb { get; set; }

        [JsonPropertyName("utilization_percent")]
        public double UtilizationPercent { get; set; }
    }

    public class DisplayData
    {
        [JsonPropertyName("display_count")]
        public int DisplayCount { get; set; }

        [JsonPropertyName("displays")]
        public List<DisplayDeviceData> Displays { get; set; } = new();
    }

    public class DisplayDeviceData
    {
        [JsonPropertyName("display_id")]
        public string DisplayId { get; set; } = string.Empty;

        [JsonPropertyName("display_name")]
        public string DisplayName { get; set; } = string.Empty;

        [JsonPropertyName("resolution_width")]
        public int ResolutionWidth { get; set; }

        [JsonPropertyName("resolution_height")]
        public int ResolutionHeight { get; set; }

        [JsonPropertyName("resolution")]
        public string Resolution { get; set; } = string.Empty;

        [JsonPropertyName("refresh_rate_hz")]
        public double? RefreshRateHz { get; set; }

        [JsonPropertyName("brightness_percent")]
        public double? BrightnessPercent { get; set; }
    }

    public class WorkloadData
    {
        [JsonPropertyName("foreground_application")]
        public string? ForegroundApplication { get; set; }

        [JsonPropertyName("top_cpu_processes")]
        public List<string> TopCpuProcesses { get; set; } = new();

        [JsonPropertyName("category")]
        public string Category { get; set; } = "IDLE";

        [JsonPropertyName("heuristic_score")]
        public double HeuristicScore { get; set; }
    }

    public class ThermalData
    {
        [JsonPropertyName("estimated_power_watts")]
        public double? EstimatedPowerWatts { get; set; }

        [JsonPropertyName("estimated_heat_watts")]
        public double? EstimatedHeatWatts { get; set; }

        [JsonPropertyName("node_thermal_load_index")]
        public double? NodeThermalLoadIndex { get; set; }

        [JsonPropertyName("signal_state")]
        public string SignalState { get; set; } = "UNAVAILABLE";

        [JsonPropertyName("estimation_status")]
        public string EstimationStatus { get; set; } = "UNAVAILABLE";

        [JsonPropertyName("sources")]
        public List<string> Sources { get; set; } = new();
    }
}
