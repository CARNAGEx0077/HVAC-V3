using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Management;
using LibreHardwareMonitor.Hardware;

namespace HveacNodeAgentNative
{
    public class HardwareProvider : IDisposable
    {
        private readonly Computer _computer;
        private readonly UpdateVisitor _visitor;
        private int _physicalCores = 0;
        private int _logicalProcessors = 0;
        private bool _isOpen = false;

        public HardwareProvider()
        {
            _computer = new Computer
            {
                IsCpuEnabled = true,
                IsGpuEnabled = true,
                IsMemoryEnabled = true,
                IsMotherboardEnabled = true,
                IsControllerEnabled = true,
                IsNetworkEnabled = false,
                IsStorageEnabled = true
            };
            _visitor = new UpdateVisitor();
        }

        public void Initialize()
        {
            if (_isOpen) return;

            try
            {
                _computer.Open();
                _isOpen = true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] LibreHardwareMonitor Computer.Open() exception: {ex.Message}");
            }

            // Detect physical cores and logical processors once
            _logicalProcessors = Environment.ProcessorCount;
            _physicalCores = DetectPhysicalCores();
        }

        public void Update()
        {
            if (!_isOpen) return;

            try
            {
                _computer.Accept(_visitor);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARN] LibreHardwareMonitor update error: {ex.Message}");
            }
        }

        public void PopulateHardwareProfile(HardwareProfile profile)
        {
            profile.PhysicalCores = _physicalCores > 0 ? _physicalCores : _logicalProcessors;
            profile.LogicalProcessors = _logicalProcessors;

            // CPU Model
            var cpu = _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Cpu);
            if (cpu != null && !string.IsNullOrWhiteSpace(cpu.Name))
            {
                profile.CpuModel = cpu.Name.Trim();
            }
            else
            {
                profile.CpuModel = GetCpuModelWmi();
            }

            // Total Physical Memory
            var mem = _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Memory && h.Name == "Total Memory")
                   ?? _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Memory);
            if (mem != null)
            {
                var used = mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Used"))?.Value ?? 0;
                var avail = mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Available"))?.Value ?? 0;
                if (used + avail > 0)
                {
                    profile.MemoryTotalGb = Math.Round(used + avail, 2);
                }
            }

            if (profile.MemoryTotalGb <= 0)
            {
                profile.MemoryTotalGb = GetTotalMemoryWmi();
            }

            // GPUs
            var gpus = GetGpuHardwares();
            profile.GpuCount = gpus.Count;
        }

        public void PopulateCpuData(CpuData cpuData)
        {
            var cpu = _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Cpu);
            if (cpu == null) return;

            // Utilization %
            var utilSensor = cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load && s.Name == "CPU Total")
                          ?? cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load);
            if (utilSensor?.Value.HasValue == true)
            {
                cpuData.UtilizationPercent = Math.Round(utilSensor.Value.Value, 1);
            }

            // Temperature C
            var tempSensor = cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Temperature &&
                                (s.Name.Contains("Package") || s.Name.Contains("Tctl") || s.Name.Contains("Core Average") || s.Name.Contains("CPU Package")))
                          ?? cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Temperature && s.Name.Contains("Core"))
                          ?? cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Temperature);

            if (tempSensor?.Value.HasValue == true && tempSensor.Value.Value > 0)
            {
                cpuData.TemperatureC = Math.Round(tempSensor.Value.Value, 1);
                cpuData.TemperatureSource = "LibreHardwareMonitor";
                cpuData.TemperatureSensorName = tempSensor.Name;
            }
            else
            {
                // Fallback attempt: WMI ACPI ThermalZone (supported on some OEM laptops)
                var wmiTemp = QueryWmiThermalZoneTemp();
                if (wmiTemp.HasValue && wmiTemp.Value > 0)
                {
                    cpuData.TemperatureC = Math.Round(wmiTemp.Value, 1);
                    cpuData.TemperatureSource = "WMI (MSAcpi_ThermalZoneTemperature)";
                    cpuData.TemperatureSensorName = "ThermalZone";
                }
                else
                {
                    cpuData.TemperatureC = null;
                    cpuData.TemperatureSource = null;
                    cpuData.TemperatureSensorName = null;
                }
            }

            // Frequency MHz
            var freqSensor = cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Clock && (s.Name.Contains("Core #1") || s.Name.Contains("Core #0") || s.Name.Contains("CPU Core")))
                          ?? cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Clock);
            if (freqSensor?.Value.HasValue == true && freqSensor.Value.Value > 0)
            {
                cpuData.FrequencyMhz = Math.Round(freqSensor.Value.Value, 0);
            }
            else
            {
                cpuData.FrequencyMhz = null;
            }

            // Power Watts
            var powerSensor = cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Power && (s.Name.Contains("Package") || s.Name.Contains("PPT") || s.Name.Contains("CPU Package")))
                           ?? cpu.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Power);
            if (powerSensor?.Value.HasValue == true && powerSensor.Value.Value > 0)
            {
                cpuData.PowerWatts = Math.Round(powerSensor.Value.Value, 1);
                cpuData.PowerSource = "LibreHardwareMonitor";
            }
            else
            {
                cpuData.PowerWatts = null;
                cpuData.PowerSource = null;
            }
        }

        public void PopulateGpuData(GpuData gpuData)
        {
            var gpus = GetGpuHardwares();
            if (gpus.Count == 0)
            {
                gpuData.Available = false;
                gpuData.Gpus = new List<GpuDeviceData>();
                return;
            }

            gpuData.Available = true;
            gpuData.Gpus = new List<GpuDeviceData>();

            for (int i = 0; i < gpus.Count; i++)
            {
                var g = gpus[i];
                var device = new GpuDeviceData
                {
                    Id = i,
                    Name = g.Name
                };

                // Utilization
                var loadSensor = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load && (s.Name == "GPU Core" || s.Name == "D3D 3D"))
                              ?? g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load);
                if (loadSensor?.Value.HasValue == true)
                {
                    device.UtilizationPercent = Math.Round(loadSensor.Value.Value, 1);
                }

                // Memory Utilization
                var memLoadSensor = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load && (s.Name.Contains("Memory") || s.Name.Contains("D3D Video Memory")));
                if (memLoadSensor?.Value.HasValue == true)
                {
                    device.MemoryUtilizationPercent = Math.Round(memLoadSensor.Value.Value, 1);
                }

                // Temperature
                var tempSensor = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Temperature && (s.Name.Contains("Core") || s.Name.Contains("GPU")))
                              ?? g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Temperature);
                if (tempSensor?.Value.HasValue == true && tempSensor.Value.Value > 0)
                {
                    device.TemperatureC = Math.Round(tempSensor.Value.Value, 1);
                }

                // Power Watts
                var powerSensor = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Power && (s.Name.Contains("Package") || s.Name.Contains("GPU") || s.Name.Contains("Board") || s.Name.Contains("Total")))
                               ?? g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Power);
                if (powerSensor?.Value.HasValue == true && powerSensor.Value.Value > 0)
                {
                    device.PowerWatts = Math.Round(powerSensor.Value.Value, 1);
                }

                // Power Limit Watts
                var pwrLimitSensor = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Power && s.Name.Contains("Limit"));
                if (pwrLimitSensor?.Value.HasValue == true && pwrLimitSensor.Value.Value > 0)
                {
                    device.PowerLimitWatts = Math.Round(pwrLimitSensor.Value.Value, 1);
                }

                // Clocks
                var coreClock = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Clock && (s.Name == "GPU Core" || s.Name.Contains("Core")));
                if (coreClock?.Value.HasValue == true && coreClock.Value.Value > 0)
                {
                    device.CoreClockMhz = Math.Round(coreClock.Value.Value, 0);
                }

                var memClock = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Clock && (s.Name == "GPU Memory" || s.Name.Contains("Memory")));
                if (memClock?.Value.HasValue == true && memClock.Value.Value > 0)
                {
                    device.MemoryClockMhz = Math.Round(memClock.Value.Value, 0);
                }

                // Fan
                var fanPercent = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Control && s.Name.Contains("Fan"));
                if (fanPercent?.Value.HasValue == true) device.FanPercent = Math.Round(fanPercent.Value.Value, 1);

                var fanRpm = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Fan);
                if (fanRpm?.Value.HasValue == true && fanRpm.Value.Value > 0) device.FanRpm = Math.Round(fanRpm.Value.Value, 0);

                // VRAM
                var vramUsedSmall = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.SmallData && (s.Name == "GPU Memory Used" || s.Name == "D3D Dedicated Memory Used"));
                var vramTotalSmall = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.SmallData && (s.Name == "GPU Memory Total" || s.Name == "D3D Dedicated Memory Total"));

                if (vramUsedSmall?.Value.HasValue == true)
                {
                    device.MemoryUsedMb = Math.Round(vramUsedSmall.Value.Value, 1);
                }
                else
                {
                    var vramUsedData = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Used"));
                    if (vramUsedData?.Value.HasValue == true)
                    {
                        device.MemoryUsedMb = Math.Round(vramUsedData.Value.Value * 1024.0, 1);
                    }
                }

                if (vramTotalSmall?.Value.HasValue == true)
                {
                    device.MemoryTotalMb = Math.Round(vramTotalSmall.Value.Value, 1);
                }
                else
                {
                    var vramTotalData = g.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Total"));
                    if (vramTotalData?.Value.HasValue == true)
                    {
                        device.MemoryTotalMb = Math.Round(vramTotalData.Value.Value * 1024.0, 1);
                    }
                }

                if (device.MemoryUtilizationPercent <= 0 && device.MemoryTotalMb > 0 && device.MemoryUsedMb > 0)
                {
                    device.MemoryUtilizationPercent = Math.Round((device.MemoryUsedMb / device.MemoryTotalMb) * 100.0, 1);
                }

                gpuData.Gpus.Add(device);
            }
        }

        public void PopulateMemoryData(MemoryData memoryData)
        {
            var mem = _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Memory && h.Name == "Total Memory")
                   ?? _computer.Hardware.FirstOrDefault(h => h.HardwareType == HardwareType.Memory);
            if (mem != null)
            {
                var usedSensor = mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Used"));
                var availSensor = mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Data && s.Name.Contains("Available"));
                var loadSensor = mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load && s.Name == "Memory")
                              ?? mem.Sensors.FirstOrDefault(s => s.SensorType == SensorType.Load);

                if (usedSensor?.Value.HasValue == true && availSensor?.Value.HasValue == true)
                {
                    double used = usedSensor.Value.Value;
                    double avail = availSensor.Value.Value;
                    memoryData.UsedGb = Math.Round(used, 2);
                    memoryData.AvailableGb = Math.Round(avail, 2);
                    memoryData.TotalGb = Math.Round(used + avail, 2);
                    if (memoryData.TotalGb > 0)
                    {
                        memoryData.UtilizationPercent = loadSensor?.Value.HasValue == true
                            ? Math.Round(loadSensor.Value.Value, 1)
                            : Math.Round((used / memoryData.TotalGb) * 100.0, 1);
                    }
                    return;
                }
            }

            // Fallback: WMI
            PopulateMemoryWmi(memoryData);
        }

        public void PopulateStorageData(StorageData storageData)
        {
            try
            {
                string root = Path.GetPathRoot(Environment.SystemDirectory) ?? "C:\\";
                var driveInfo = new DriveInfo(root);

                if (driveInfo.IsReady)
                {
                    double totalGb = Math.Round((double)driveInfo.TotalSize / (1024.0 * 1024.0 * 1024.0), 2);
                    double freeGb = Math.Round((double)driveInfo.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0), 2);
                    double usedGb = Math.Round(totalGb - freeGb, 2);
                    double util = totalGb > 0 ? Math.Round((usedGb / totalGb) * 100.0, 1) : 0;

                    storageData.SystemDrive = new DriveData
                    {
                        Path = driveInfo.Name.TrimEnd('\\'),
                        TotalGb = totalGb,
                        UsedGb = usedGb,
                        FreeGb = freeGb,
                        UtilizationPercent = util
                    };
                    return;
                }
            }
            catch { }

            storageData.SystemDrive = null;
        }

        public void RunDiagnostics()
        {
            Initialize();
            Update();

            Console.WriteLine("==================================================");
            Console.WriteLine("  HVEAC HARDWARE SENSOR DIAGNOSTIC MODE");
            Console.WriteLine("==================================================");
            Console.WriteLine($"Machine: {Environment.MachineName}");
            Console.WriteLine($"Physical Cores: {_physicalCores}, Logical Processors: {_logicalProcessors}\n");

            foreach (var hardware in _computer.Hardware)
            {
                Console.WriteLine($"[DEVICE] Type: {hardware.HardwareType} | Name: {hardware.Name}");
                if (hardware.Sensors.Length == 0)
                {
                    Console.WriteLine("  (No direct sensors detected)");
                }
                foreach (var sensor in hardware.Sensors)
                {
                    string val = sensor.Value.HasValue ? sensor.Value.Value.ToString("F1") : "null";
                    Console.WriteLine($"  [{sensor.SensorType}] {sensor.Name}: {val}");
                }

                foreach (var sub in hardware.SubHardware)
                {
                    Console.WriteLine($"  [SUB-DEVICE] Type: {sub.HardwareType} | Name: {sub.Name}");
                    foreach (var sensor in sub.Sensors)
                    {
                        string val = sensor.Value.HasValue ? sensor.Value.Value.ToString("F1") : "null";
                        Console.WriteLine($"    [{sensor.SensorType}] {sensor.Name}: {val}");
                    }
                }
                Console.WriteLine();
            }

            Console.WriteLine("==================================================");
            Console.WriteLine("  STORAGE & SYSTEM DRIVES");
            Console.WriteLine("==================================================");
            foreach (var d in DriveInfo.GetDrives().Where(d => d.IsReady))
            {
                double total = Math.Round((double)d.TotalSize / (1024.0 * 1024.0 * 1024.0), 2);
                double free = Math.Round((double)d.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0), 2);
                Console.WriteLine($"  Drive {d.Name} ({d.DriveType}): Total {total} GB, Free {free} GB, Format: {d.DriveFormat}");
            }
            Console.WriteLine();
        }

        private List<IHardware> GetGpuHardwares()
        {
            return _computer.Hardware.Where(h =>
                h.HardwareType == HardwareType.GpuNvidia ||
                h.HardwareType == HardwareType.GpuAmd ||
                h.HardwareType == HardwareType.GpuIntel
            ).ToList();
        }

        private int DetectPhysicalCores()
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("SELECT NumberOfCores FROM Win32_Processor");
                int cores = 0;
                foreach (ManagementObject item in searcher.Get())
                {
                    cores += Convert.ToInt32(item["NumberOfCores"]);
                }
                if (cores > 0) return cores;
            }
            catch { }

            return Environment.ProcessorCount;
        }

        private string GetCpuModelWmi()
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_Processor");
                foreach (ManagementObject item in searcher.Get())
                {
                    var name = item["Name"]?.ToString();
                    if (!string.IsNullOrWhiteSpace(name)) return name.Trim();
                }
            }
            catch { }

            return "Unknown CPU";
        }

        private double GetTotalMemoryWmi()
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("SELECT TotalPhysicalMemory FROM Win32_ComputerSystem");
                foreach (ManagementObject item in searcher.Get())
                {
                    var bytes = Convert.ToDouble(item["TotalPhysicalMemory"]);
                    return Math.Round(bytes / (1024.0 * 1024.0 * 1024.0), 2);
                }
            }
            catch { }

            return 0;
        }

        private void PopulateMemoryWmi(MemoryData memoryData)
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("SELECT TotalVisibleMemorySize, FreePhysicalMemory FROM Win32_OperatingSystem");
                foreach (ManagementObject item in searcher.Get())
                {
                    double totalKb = Convert.ToDouble(item["TotalVisibleMemorySize"]);
                    double freeKb = Convert.ToDouble(item["FreePhysicalMemory"]);
                    double totalGb = Math.Round(totalKb / (1024.0 * 1024.0), 2);
                    double freeGb = Math.Round(freeKb / (1024.0 * 1024.0), 2);
                    double usedGb = Math.Round(totalGb - freeGb, 2);
                    double util = totalGb > 0 ? Math.Round((usedGb / totalGb) * 100.0, 1) : 0;

                    memoryData.TotalGb = totalGb;
                    memoryData.AvailableGb = freeGb;
                    memoryData.UsedGb = usedGb;
                    memoryData.UtilizationPercent = util;
                    return;
                }
            }
            catch { }
        }

        private static double? QueryWmiThermalZoneTemp()
        {
            try
            {
                using var searcher = new ManagementObjectSearcher("root\\WMI", "SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature");
                foreach (ManagementObject item in searcher.Get())
                {
                    if (item["CurrentTemperature"] != null)
                    {
                        // Stored in tenths of Kelvin: (Kelvin - 273.2)
                        double decikelvin = Convert.ToDouble(item["CurrentTemperature"]);
                        double celsius = (decikelvin - 2732.0) / 10.0;
                        if (celsius > 0 && celsius < 125)
                        {
                            return celsius;
                        }
                    }
                }
            }
            catch { }

            return null;
        }

        public void Dispose()
        {
            if (_isOpen)
            {
                try
                {
                    _computer.Close();
                }
                catch { }
                _isOpen = false;
            }
        }
    }

    public class UpdateVisitor : IVisitor
    {
        public void VisitComputer(IComputer computer)
        {
            computer.Traverse(this);
        }

        public void VisitHardware(IHardware hardware)
        {
            hardware.Update();
            foreach (IHardware sub in hardware.SubHardware)
            {
                sub.Accept(this);
            }
        }

        public void VisitSensor(ISensor sensor) { }
        public void VisitParameter(IParameter parameter) { }
    }
}
