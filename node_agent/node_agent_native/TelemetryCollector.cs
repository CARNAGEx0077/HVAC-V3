using System;

namespace HveacNodeAgentNative
{
    public class TelemetryCollector
    {
        private readonly AgentIdentity _identity;
        private readonly HardwareProvider _hardwareProvider;
        private readonly DisplayProvider _displayProvider;
        private readonly ProcessCollector _processCollector;
        private readonly WorkloadAnalyzer _workloadAnalyzer;
        private readonly ThermalEstimator _thermalEstimator;
        private readonly string _schemaVersion;
        private long _sequence = 0;

        public TelemetryCollector(
            AgentIdentity identity,
            HardwareProvider hardwareProvider,
            DisplayProvider displayProvider,
            ProcessCollector processCollector,
            WorkloadAnalyzer workloadAnalyzer,
            ThermalEstimator thermalEstimator,
            string schemaVersion = "1.0")
        {
            _identity = identity;
            _hardwareProvider = hardwareProvider;
            _displayProvider = displayProvider;
            _processCollector = processCollector;
            _workloadAnalyzer = workloadAnalyzer;
            _thermalEstimator = thermalEstimator;
            _schemaVersion = schemaVersion;
        }

        public TelemetryPayload CollectSnapshot()
        {
            _sequence++;
            var nowUtc = DateTime.UtcNow;

            var payload = new TelemetryPayload
            {
                NodeId = _identity.NodeId,
                Timestamp = nowUtc.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                Sequence = _sequence,
                SchemaVersion = _schemaVersion,
                System = new SystemInfo
                {
                    Hostname = _identity.Hostname,
                    Platform = _identity.Platform,
                    OsVersion = _identity.OsVersion,
                    Architecture = _identity.Architecture,
                    AgentVersion = _identity.AgentVersion,
                    UptimeSeconds = _identity.UptimeSeconds
                }
            };

            // 1. Update hardware driver readings
            _hardwareProvider.Update();

            // 2. Hardware Profile & Base CPU Telemetry
            _hardwareProvider.PopulateHardwareProfile(payload.Hardware);
            _hardwareProvider.PopulateCpuData(payload.Cpu);

            // 3. Workload Analysis: classify workload, rolling load average & thermal proxy index
            _workloadAnalyzer.Analyze(payload.Cpu, nowUtc);

            // 4. GPU, Memory, Storage & Display Telemetry
            _hardwareProvider.PopulateGpuData(payload.Gpu);
            _hardwareProvider.PopulateMemoryData(payload.Memory);
            _hardwareProvider.PopulateStorageData(payload.Storage);
            _displayProvider.PopulateDisplayData(payload.Display);

            // 5. Process Workload Heuristic
            _processCollector.PopulateWorkloadData(payload.Workload, payload.Cpu, payload.Gpu, payload.Memory);

            // 6. Node-Level Thermal Estimation (Measured Watts vs Proxy Index)
            _thermalEstimator.Estimate(payload.Thermal, payload.Cpu, payload.Gpu);

            return payload;
        }
    }
}
