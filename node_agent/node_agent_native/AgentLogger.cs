using System;

namespace HveacNodeAgentNative
{
    public class AgentLogger
    {
        public void PrintBanner(string agentVersion, string nodeId, string hostname, string serverUrl)
        {
            Console.WriteLine("==================================================");
            Console.WriteLine("HVEAC ENDPOINT AGENT");
            Console.WriteLine($"Version: {agentVersion}");
            Console.WriteLine($"Node: {nodeId}");
            Console.WriteLine($"Hostname: {hostname}");
            Console.WriteLine($"Server: {serverUrl}");
            Console.WriteLine("==================================================\n");
        }

        public void LogInit(string step, bool success, string? detail = null)
        {
            if (success)
            {
                Console.WriteLine($"[PASS] {step}{(detail != null ? $" - {detail}" : "")}");
            }
            else
            {
                Console.WriteLine($"[FAIL] {step}{(detail != null ? $" - {detail}" : "")}");
            }
        }

        public void PrintDetailedSummary(TelemetryPayload payload)
        {
            Console.WriteLine("==================================================");
            Console.WriteLine("HVEAC ENDPOINT AGENT - HARDWARE & PROXY STATUS");
            Console.WriteLine($"NODE: {payload.NodeId}");
            Console.WriteLine("==================================================");

            string cpuTemp = payload.Cpu.TemperatureC.HasValue ? $"{payload.Cpu.TemperatureC.Value:F0}°C" : "N/A";
            string cpuPower = payload.Cpu.PowerWatts.HasValue ? $"{payload.Cpu.PowerWatts.Value:F1} W" : "N/A";

            Console.WriteLine($"CPU LOAD        : {payload.Cpu.UtilizationPercent:F0}%");
            Console.WriteLine($"CPU WORKLOAD    : {payload.Cpu.WorkloadCategory}");
            Console.WriteLine($"CPU TEMP        : {cpuTemp}");
            Console.WriteLine($"CPU POWER       : {cpuPower}");
            Console.WriteLine($"CPU THERMAL     : {payload.Cpu.ThermalLoadIndex:F0}/100");
            Console.WriteLine($"CPU SIGNAL      : {payload.Cpu.ThermalSignalState}");

            if (payload.Gpu.Available && payload.Gpu.Gpus.Count > 0)
            {
                var g = payload.Gpu.Gpus[0];
                string gpuTemp = g.TemperatureC.HasValue ? $"{g.TemperatureC.Value:F0}°C" : "N/A";
                string gpuPower = g.PowerWatts.HasValue ? $"{g.PowerWatts.Value:F1} W" : "N/A";
                Console.WriteLine($"\nGPU LOAD        : {g.UtilizationPercent:F0}%");
                Console.WriteLine($"GPU TEMP        : {gpuTemp}");
                Console.WriteLine($"GPU POWER       : {gpuPower}");
            }
            else
            {
                Console.WriteLine("\nGPU LOAD        : N/A");
                Console.WriteLine("GPU TEMP        : N/A");
                Console.WriteLine("GPU POWER       : N/A");
            }

            string nodeThermal = payload.Thermal.NodeThermalLoadIndex.HasValue
                ? $"{payload.Thermal.NodeThermalLoadIndex.Value:F0}/100"
                : "N/A";
            Console.WriteLine($"\nNODE THERMAL    : {nodeThermal}");
            Console.WriteLine($"THERMAL SIGNAL  : {payload.Thermal.SignalState}");
            if (payload.Thermal.EstimatedHeatWatts.HasValue)
            {
                Console.WriteLine($"MEASURED POWER  : {payload.Thermal.EstimatedHeatWatts.Value:F1} W");
            }
            Console.WriteLine("==================================================\n");
        }

        public void LogStateTransition(ConnectionState oldState, ConnectionState newState, int bufferedCount)
        {
            string time = DateTime.Now.ToString("HH:mm:ss");
            switch (newState)
            {
                case ConnectionState.ONLINE:
                    Console.WriteLine($"[{time}] ONLINE: Successfully connected to HVEAC backend. Buffer drained.");
                    break;
                case ConnectionState.RECONNECTING:
                    Console.WriteLine($"[{time}] RECONNECTING: Backend unreachable. Buffering telemetry ({bufferedCount} packets queued).");
                    break;
                case ConnectionState.OFFLINE:
                    Console.WriteLine($"[{time}] OFFLINE: Backend persistently unavailable. Local buffering active ({bufferedCount} packets queued).");
                    break;
            }
        }

        public void LogTelemetryCycle(TelemetryPayload payload, ConnectionState state, int bufferedCount)
        {
            string time = DateTime.Now.ToString("HH:mm:ss");

            // CPU summary distinguishing measured vs proxy
            string cpuTemp = payload.Cpu.TemperatureC.HasValue ? $"{payload.Cpu.TemperatureC.Value:F0}°C" : "N/A";
            string cpuPower = payload.Cpu.PowerWatts.HasValue ? $"{payload.Cpu.PowerWatts.Value:F0}W" : "N/A";
            string cpuSummary = $"CPU: {payload.Cpu.UtilizationPercent:F0}% ({payload.Cpu.ThermalSignalState}, {payload.Cpu.ThermalLoadIndex:F0}/100, {payload.Cpu.WorkloadCategory})";

            // GPU summary
            string gpuSummary = "GPU: N/A";
            if (payload.Gpu.Available && payload.Gpu.Gpus.Count > 0)
            {
                var g = payload.Gpu.Gpus[0];
                string gpuTemp = g.TemperatureC.HasValue ? $"{g.TemperatureC.Value:F0}°C" : "N/A";
                string gpuPower = g.PowerWatts.HasValue ? $"{g.PowerWatts.Value:F0}W" : "N/A";
                gpuSummary = $"GPU: {g.UtilizationPercent:F0}% ({gpuTemp}, {gpuPower})";
            }

            // RAM summary
            string ramSummary = $"RAM: {payload.Memory.UtilizationPercent:F0}%";

            // Node Thermal Index vs Measured Heat
            string nodeThermal = payload.Thermal.NodeThermalLoadIndex.HasValue
                ? $"NODE THERMAL: {payload.Thermal.NodeThermalLoadIndex.Value:F0}/100 ({payload.Thermal.SignalState})"
                : $"NODE THERMAL: N/A ({payload.Thermal.SignalState})";

            string heatSummary = payload.Thermal.EstimatedHeatWatts.HasValue
                ? $"EST HEAT: {payload.Thermal.EstimatedHeatWatts.Value:F1}W"
                : "EST HEAT: N/A";

            string stateTag = state == ConnectionState.ONLINE ? "ONLINE" : $"{state} ({bufferedCount} buf)";

            Console.WriteLine($"[{time}] {stateTag} | {cpuSummary} | {gpuSummary} | {ramSummary} | {nodeThermal} | {heatSummary}");
        }
    }
}
