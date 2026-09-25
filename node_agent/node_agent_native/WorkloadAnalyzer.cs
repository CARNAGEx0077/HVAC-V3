using System;
using System.Collections.Generic;
using System.Linq;

namespace HveacNodeAgentNative
{
    /// <summary>
    /// Analyzes CPU utilization to derive deterministic workload classification,
    /// a normalized computational thermal-load proxy index (0-100), a 30-second rolling
    /// load average, and sustained high-load duration tracking.
    ///
    /// CRITICAL SEMANTICS:
    /// - thermal_load_index is a computational proxy representing relative thermal stress.
    /// - It is NOT physical temperature in degrees Celsius.
    /// - It is NOT electrical power in Watts.
    /// - Temperature and power are only reported when real hardware sensors exist.
    /// </summary>
    public class WorkloadAnalyzer
    {
        private readonly double _windowSeconds;
        private readonly double _highLoadThreshold;
        private readonly List<(DateTime Timestamp, double CpuPercent)> _history = new();
        private DateTime? _sustainedStartTime = null;

        public WorkloadAnalyzer(double windowSeconds = 30.0, double highLoadThreshold = 60.0)
        {
            _windowSeconds = Math.Max(5.0, windowSeconds);
            _highLoadThreshold = highLoadThreshold;
        }

        /// <summary>
        /// Analyzes the CPU readings and populates all derived workload and proxy fields.
        /// </summary>
        public void Analyze(CpuData cpu, DateTime timestamp)
        {
            // 1. Clamp CPU utilization to valid 0-100 range
            double rawUtil = cpu.UtilizationPercent;
            double clampedUtil = Math.Clamp(rawUtil, 0.0, 100.0);
            cpu.UtilizationPercent = Math.Round(clampedUtil, 1);

            // 2. Classify CPU Workload Category
            cpu.WorkloadCategory = ClassifyCategory(cpu.UtilizationPercent);

            // 3. Derived CPU Thermal Load Index (0 - 100 proxy)
            // For prototype: relative computational load index equals CPU utilization
            cpu.ThermalLoadIndex = Math.Round(clampedUtil, 1);

            // 4. Update rolling history (bounded to 30 seconds)
            _history.Add((timestamp, clampedUtil));
            DateTime cutoff = timestamp.AddSeconds(-_windowSeconds);
            _history.RemoveAll(entry => entry.Timestamp < cutoff);

            // 5. Short-term load average percent
            double avg = _history.Count > 0 ? _history.Average(e => e.CpuPercent) : clampedUtil;
            cpu.ShortTermAveragePercent = Math.Round(avg, 1);

            // 6. Sustained High Load Tracking
            if (cpu.ShortTermAveragePercent >= _highLoadThreshold)
            {
                cpu.IsSustainedHighLoad = true;
                _sustainedStartTime ??= timestamp;
                cpu.SustainedLoadSeconds = (long)Math.Max(0, (timestamp - _sustainedStartTime.Value).TotalSeconds);
            }
            else
            {
                cpu.IsSustainedHighLoad = false;
                _sustainedStartTime = null;
                cpu.SustainedLoadSeconds = 0;
            }

            // 7. CPU Thermal Signal State
            if (cpu.TemperatureC.HasValue && cpu.PowerWatts.HasValue)
            {
                cpu.ThermalSignalState = "MEASURED";
            }
            else if (!double.IsNaN(cpu.UtilizationPercent))
            {
                // Temperature or power unavailable -> utilization acts as thermal proxy
                cpu.ThermalSignalState = "PROXY";
            }
            else
            {
                cpu.ThermalSignalState = "UNAVAILABLE";
            }
        }

        /// <summary>
        /// Deterministic workload category thresholds:
        /// 0 <= CPU < 10   -> IDLE
        /// 10 <= CPU < 30  -> LIGHT
        /// 30 <= CPU < 60  -> GENERAL
        /// 60 <= CPU < 80  -> CPU_INTENSIVE
        /// 80 <= CPU <= 100 -> HEAVY_CPU_LOAD
        /// </summary>
        public static string ClassifyCategory(double cpuPercent)
        {
            if (cpuPercent < 10.0) return "IDLE";
            if (cpuPercent < 30.0) return "LIGHT";
            if (cpuPercent < 60.0) return "GENERAL";
            if (cpuPercent < 80.0) return "CPU_INTENSIVE";
            return "HEAVY_CPU_LOAD";
        }

        /// <summary>
        /// Resets the rolling history.
        /// </summary>
        public void Reset()
        {
            _history.Clear();
            _sustainedStartTime = null;
        }
    }
}
