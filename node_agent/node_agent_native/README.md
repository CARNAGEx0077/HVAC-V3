# HVEAC Windows Endpoint Agent (Native .NET 10)

The **HVEAC Endpoint Agent** is a lightweight, high-performance Windows background agent written in C# on .NET 10. It executes on physical Windows computer nodes across an enterprise cluster, continuously collecting normalized hardware, computational, workload, display, and thermal dissipation metrics and securely transmitting them to the central HVEAC server.

---

## 1. System Architecture

```text
               +---------------------------------------+
               |        Physical Windows Node          |
               |                                       |
               |  [CPU] [GPU(s)] [RAM] [SSD] [Display] |
               +---------------------------------------+
                                  |
                                  v
                   HVEAC Windows Endpoint Agent
                (LibreHardwareMonitor + Win32 + WMI)
                                  |
       +--------------------------+--------------------------+
       |                          |                          |
TelemetryCollector          ProcessCollector          DisplayProvider
  (Hardware Profile,         (Top CPU Processes,        (Attached Screens,
   Clocks, Temps, Loads)      Deterministic Workload)    Resolution, Hz)
       |                          |                          |
       +--------------------------+--------------------------+
                                  |
                           ThermalEstimator
                   (Measured Joules / Heat Dissipation)
                                  |
                       TelemetryClient (FIFO)
                  (Bounded Queue, Exponential Backoff)
                                  |
                    HTTP POST /api/nodes/telemetry
                                  |
                                  v
                       HVEAC Backend (FastAPI)
                                  |
                                  v
                     Computer Nodes Dashboard
```

### Core Design Principles
1. **Zero Telemetry Fabrication**: Sensors that are physically absent, not supported, or require elevated ring-0 drivers report `null` / `N/A`. The agent **never** invents or simulates hardware state.
2. **Persistent Machine Identity**: Unique `NODE-XXXXXXXX` generated once and persisted in `C:\ProgramData\HVEAC\node_identity.json` (or `%LOCALAPPDATA%\HVEAC\node_identity.json`). Reused across reboots and restarts.
3. **Single Hardware Provider Lifetime**: Hardware monitoring drivers are initialized once at startup and maintained across cycles, avoiding driver reload overhead or resource leaks.
4. **Resilient Network Buffering**: If the HVEAC server becomes unreachable, telemetry is queued in a bounded FIFO memory buffer (default 100 packets). When the connection recovers, buffered telemetry is transmitted in order.

---

## 2. Requirements

- **Operating System**: Windows 10 / Windows 11 / Windows Server 2019+ (x64 / ARM64)
- **Runtime**: [.NET 10 SDK or Runtime (v10.0+)](https://dotnet.microsoft.com/download/dotnet/10.0)
- **Dependencies**:
  - `LibreHardwareMonitorLib` (v0.9.6)
  - `System.Management` (v10.0.12)
  - `System.Text.Json` (Built-in)

---

## 3. Build Instructions

Open PowerShell in the agent directory:

```powershell
# Navigate to agent directory
cd d:\HVAC-V2\node_agent\node_agent_native

# Restore dependencies
dotnet restore

# Build release configuration
dotnet build -c Release
```

The compiled binary will be placed at:
`bin\Release\net10.0\HveacNodeAgentNative.exe`

---

## 4. Configuration

The agent loads configuration in the following order of precedence:
1. Environment variables
2. Local `.env` file (searched in executable directory, working directory, and `C:\ProgramData\HVEAC\.env`)
3. Built-in defaults

### Supported Configuration Keys

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HVEAC_NODE_ID` | Auto-generated (`NODE-XXXXXXXX`) | Explicitly override the persistent node ID |
| `HVEAC_SERVER_URL` | `http://127.0.0.1:8000` | Target HVEAC FastAPI backend endpoint base URL |
| `TELEMETRY_INTERVAL_SECONDS`| `2` | Polling and reporting frequency in seconds |
| `MAX_BUFFERED_PACKETS` | `100` | Maximum number of packets buffered when offline |

### Example `.env` File

```ini
# HVEAC Node Agent Configuration
HVEAC_SERVER_URL=http://127.0.0.1:8000
TELEMETRY_INTERVAL_SECONDS=2
MAX_BUFFERED_PACKETS=100
```

---

## 5. Running the Agent

### Standard Run

```powershell
dotnet run
```

Or execute the published binary:

```powershell
.\bin\Release\net10.0\HveacNodeAgentNative.exe
```

Console Output Example:
```text
==================================================
HVEAC ENDPOINT AGENT
Version: 0.3.0
Node: NODE-7F3A91C2
Hostname: DESKTOP-M22II7F
Server: http://127.0.0.1:8000
==================================================

[INIT] Hardware provider...
[PASS] Hardware provider
[INIT] Display provider...
[PASS] Display provider
[INIT] Process & Thermal collectors...
[PASS] Process & Thermal collectors
[INIT] Telemetry engine & Network client...
[PASS] Telemetry engine
[PASS] Identity: NODE-7F3A91C2
================================================

[AGENT] Starting telemetry cycle (2s interval). Press Ctrl+C to stop.

[19:45:00] ONLINE | CPU: 24% (N/A, N/A) | GPU: 2% (55°C, 13W) | RAM: 57% | WORKLOAD: GENERAL | EST HEAT: 13W
[19:45:02] ONLINE | CPU: 18% (N/A, N/A) | GPU: 1% (55°C, 12W) | RAM: 57% | WORKLOAD: GENERAL | EST HEAT: 12W
```

### Hardware Diagnostic Mode

To inspect all recognized hardware components, sensors, and storage drives without starting the reporting loop:

```powershell
dotnet run -- --diagnose-hardware
```

Diagnostic Mode Output Example:
```text
==================================================
  HVEAC HARDWARE SENSOR DIAGNOSTIC MODE
==================================================
Machine: DESKTOP-M22II7F
Physical Cores: 6, Logical Processors: 12

[DEVICE] Type: Cpu | Name: AMD Ryzen 5 5600H with Radeon Graphics
  [Load] CPU Total: 23.7
  [Power] Package: 0.0
  [Temperature] Core (Tctl/Tdie): 0.0

[DEVICE] Type: GpuNvidia | Name: NVIDIA GeForce GTX 1650
  [Temperature] GPU Core: 55.0
  [Clock] GPU Core: 1380.0
  [Load] GPU Core: 0.0
  [Power] GPU Package: 12.6

==================================================
  STORAGE & SYSTEM DRIVES
==================================================
  Drive C:\ (Fixed): Total 325.98 GB, Free 65.5 GB, Format: NTFS
  Drive D:\ (Fixed): Total 150 GB, Free 21.88 GB, Format: NTFS
```

---

---

## 6. Computational & Thermal Load Proxy (Prototype)

### Problem Statement
Some CPUs and laptop systems (e.g., modern AMD Ryzen APUs, certain Intel mobile chipsets, or non-elevated user-mode execution contexts) do not expose hardware temperature registers (`Core Tctl/Tdie`) or electrical package power sensors through user-mode ring-3 drivers.

When physical sensors are unavailable, HVEAC still requires computational load signals because CPU and GPU workloads directly contribute to ambient room heat dissipation.

### Core Architecture & Guarantees
1. **Zero Fabrication**:
   - `cpu.temperature_c` remains `null` when no physical temperature sensor exists.
   - `cpu.power_watts` remains `null` when no physical electrical power sensor exists.
   - The agent **never** fabricates Celsius temperatures or Watts.
   - The agent **never** converts CPU utilization directly into °C or Watts.
2. **Three-Tier Semantic Separation**:
   - **Measured Telemetry**: Physical hardware sensor readings (`temperature_c`, `power_watts`, `utilization_percent`).
   - **Derived Workload Information**: Deterministic operational category (`IDLE`, `LIGHT`, `GENERAL`, `CPU_INTENSIVE`, `HEAVY_CPU_LOAD`).
   - **Derived Thermal-Load Proxy**: Normalized relative index (`0–100`) representing computational contribution to room heat.

```text
+-----------------------------------------------------------------------------------+
|                            HVEAC Telemetry Tiers                                  |
+-----------------------------------------------------------------------------------+
|  1. MEASURED TELEMETRY                                                            |
|     - CPU Utilization (%) [Real OS / Win32 counter]                              |
|     - CPU Temperature (°C) [Real sensor or null]                                 |
|     - CPU / GPU Power (W) [Real sensor or null]                                  |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|  2. DERIVED WORKLOAD INFORMATION (WorkloadAnalyzer.cs)                            |
|     - Workload Category: IDLE / LIGHT / GENERAL / CPU_INTENSIVE / HEAVY_CPU_LOAD  |
|     - Short-Term Average: 30-second bounded rolling mean                          |
|     - Sustained High-Load: True if short-term avg >= 60%, with elapsed seconds    |
+-----------------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------------+
|  3. DERIVED COMPUTATIONAL THERMAL LOAD PROXY                                      |
|     - CPU Thermal Load Index: 0–100 normalized computational proxy               |
|     - Node Thermal Load Index: 50% CPU + 50% GPU load combination (0–100)        |
|     - Thermal Signal State: MEASURED / PARTIAL / PROXY / UNAVAILABLE             |
|     - Estimated Heat (W): Real measured power sum only (null if no power sensors) |
+-----------------------------------------------------------------------------------+
```

### Data Semantics & Definitions
- **CPU utilization**: Real measured processor activity from the hardware telemetry provider (0.0–100.0%).
- **CPU workload category**: Deterministic classification derived from real CPU utilization:
  - `0 <= CPU < 10%` -> `IDLE`
  - `10 <= CPU < 30%` -> `LIGHT`
  - `30 <= CPU < 60%` -> `GENERAL`
  - `60 <= CPU < 80%` -> `CPU_INTENSIVE`
  - `80 <= CPU <= 100%` -> `HEAVY_CPU_LOAD`
- **CPU thermal load index**: Normalized 0–100 proxy score representing the relative computational load contribution of the CPU. **This is not temperature (°C) and is not electrical power (Watts).**
- **CPU temperature**: Physical sensor measurement in degrees Celsius when available; `null` if unavailable.
- **CPU power**: Physical/firmware sensor measurement in Watts when available; `null` if unavailable.
- **Short-term average percent**: 30-second bounded rolling average of CPU utilization to distinguish transient spikes from sustained workloads.
- **Sustained load seconds**: Number of seconds the CPU has experienced sustained high load (`short_term_average_percent >= 60%`), calculated strictly from telemetry timestamps.
- **Node thermal load index**: Normalized 0–100 computational thermal load proxy combining CPU and GPU workload signals (50% CPU + 50% GPU max). **Never converted to Watts.**
- **Estimated heat watts**: Sum of physical power sensors only (CPU power + GPU power). If both are unavailable, reports `null` and is never guessed.
- **Thermal signal state**:
  - `MEASURED`: Full physical electrical power telemetry is available across all active processors.
  - `PARTIAL`: Some physical power measurements exist (e.g. dedicated GPU power measured, but CPU package power unavailable).
  - `PROXY`: Physical power sensors are unavailable; utilization-based thermal load proxy is utilized.
  - `UNAVAILABLE`: Telemetry is insufficient to compute load or thermal state.

---

## 7. Telemetry Schema (JSON Contract)

The agent produces a single normalized JSON object transmitted via `POST /api/nodes/telemetry`:

```json
{
  "node_id": "NODE-61142012",
  "timestamp": "2026-09-25T17:15:30Z",
  "sequence": 42,
  "schema_version": "1.0",
  "system": {
    "hostname": "DESKTOP-M22II7F",
    "platform": "Windows",
    "os_version": "Microsoft Windows NT 10.0.26200.0",
    "architecture": "X64",
    "agent_version": "0.3.0",
    "uptime_seconds": 46268
  },
  "hardware": {
    "cpu_model": "AMD Ryzen 5 5600H with Radeon Graphics",
    "physical_cores": 6,
    "logical_processors": 12,
    "memory_total_gb": 15.34,
    "gpu_count": 2
  },
  "cpu": {
    "utilization_percent": 72.4,
    "temperature_c": null,
    "frequency_mhz": null,
    "power_watts": null,
    "temperature_source": null,
    "temperature_sensor_name": null,
    "power_source": null,
    "workload_category": "CPU_INTENSIVE",
    "thermal_load_index": 72.4,
    "short_term_average_percent": 68.7,
    "sustained_load_seconds": 24,
    "is_sustained_high_load": true,
    "thermal_signal_state": "PROXY"
  },
  "gpu": {
    "available": true,
    "gpus": [
      {
        "id": 0,
        "name": "AMD Radeon(TM) Graphics",
        "utilization_percent": 0.7,
        "memory_utilization_percent": 74.6,
        "temperature_c": null,
        "power_watts": null,
        "power_limit_watts": null,
        "memory_used_mb": 381.7,
        "memory_total_mb": 512.0,
        "core_clock_mhz": 400.0,
        "memory_clock_mhz": 1333.0,
        "fan_percent": null,
        "fan_rpm": null
      },
      {
        "id": 1,
        "name": "NVIDIA GeForce GTX 1650",
        "utilization_percent": 24.0,
        "memory_utilization_percent": 12.5,
        "temperature_c": 52.0,
        "power_watts": 14.2,
        "power_limit_watts": null,
        "memory_used_mb": 512.0,
        "memory_total_mb": 4096.0,
        "core_clock_mhz": 1380.0,
        "memory_clock_mhz": 5871.0,
        "fan_percent": null,
        "fan_rpm": null
      }
    ]
  },
  "memory": {
    "total_gb": 15.34,
    "used_gb": 11.35,
    "available_gb": 3.99,
    "utilization_percent": 74.0
  },
  "storage": {
    "system_drive": {
      "path": "C:",
      "total_gb": 325.98,
      "used_gb": 260.71,
      "free_gb": 65.27,
      "utilization_percent": 80.0
    }
  },
  "display": {
    "display_count": 1,
    "displays": [
      {
        "display_id": "\\\\.\\DISPLAY1",
        "display_name": "Generic PnP Monitor",
        "resolution_width": 1920,
        "resolution_height": 1080,
        "resolution": "1920x1080",
        "refresh_rate_hz": 144.0,
        "brightness_percent": 100.0
      }
    ]
  },
  "workload": {
    "foreground_application": null,
    "top_cpu_processes": [
      "language_server_windows_x64",
      "msedgewebview2",
      "node",
      "explorer"
    ],
    "category": "IDLE",
    "heuristic_score": 0.0
  },
  "thermal": {
    "estimated_power_watts": 14.2,
    "estimated_heat_watts": 14.2,
    "node_thermal_load_index": 48.2,
    "signal_state": "PARTIAL",
    "estimation_status": "PARTIAL",
    "sources": [
      "gpu_power",
      "cpu_load_proxy",
      "gpu_load_proxy"
    ]
  }
}
```

---

## 8. Backend Integration

The agent requires the following backend endpoints:

### `POST /api/nodes/telemetry`
- Receives the agent's telemetry snapshot.
- HTTP Status Codes:
  - `200 OK`: Telemetry accepted and stored in cluster runtime state.
  - `422 Unprocessable Entity`: Schema validation error.

### `GET /api/nodes`
- Returns active cluster inventory, online node count, `average_node_thermal_load_index`, `maximum_node_thermal_load_index`, `total_measured_power_watts`, and `cluster_thermal_mode`.

### `GET /api/nodes/{node_id}`
- Returns latest telemetry for a specific node.

---

## 9. Limitations & Prototype Boundaries

1. **Not a Temperature Measurement**: CPU utilization and the `thermal_load_index` proxy do **not** represent degrees Celsius (°C). They reflect relative computational load.
2. **Not an Electrical Power Measurement**: The load index must never be interpreted as Watts. Missing physical power sensors report `null`.
3. **Architecture Differences**: Different CPU architectures, lithography nodes (e.g. 5nm vs 14nm), and TDP targets dissipate significantly different amounts of heat at identical 50% utilization levels.
4. **Physical Influences**: Dynamic voltage/frequency scaling (DVFS), ambient room temperature, fan curve aggressive throttling, and chassis thermal dissipation capacity alter actual heat generation.
5. **Future Machine Learning Calibration**: The HVEAC Thermal Intelligence engine will ingest these raw and proxy signals alongside room occupancy and outdoor temperature to calibrate per-node thermal transfer functions over time.

---

## 10. Troubleshooting

1. **CPU Temperature or Power reports `null`**:
   - This is normal on systems without accessible Ring-3 power/temp sensors (such as non-elevated AMD Ryzen APUs). The agent activates `PROXY` mode automatically and reports `cpu.thermal_load_index` without fabricating data.
2. **Display Brightness reports `null`**:
   - Standard desktop external monitors connected via DisplayPort or HDMI typically do not expose Windows WMI brightness controllers. Only integrated laptop displays or DDC/CI compliant monitors report brightness. `null` is expected and normal.
3. **Backend shows `RECONNECTING` or `OFFLINE`**:
   - Verify that the HVEAC server is listening at the URL configured in `HVEAC_SERVER_URL`.
   - Verify firewall settings allow inbound HTTP traffic on the configured port.

