using System;
using System.Collections.Generic;
using System.Linq;

namespace HveacNodeAgentNative
{
    /// <summary>
    /// Computes node-level thermal signals, strictly separating:
    /// 1. Measured Electrical Power (in Watts) - derived ONLY from physical sensors.
    /// 2. Computational Thermal-Load Proxy Index (0-100) - derived from CPU/GPU utilization.
    ///
    /// CRITICAL SEMANTICS:
    /// - estimated_power_watts: Electrical power converted to heat. Only populated if real
    ///   wattage sensors exist. Never fabricated or guessed.
    /// - node_thermal_load_index: Normalized 0-100 proxy representing aggregate computational
    ///   thermal stress. Never converted to Watts or Celsius.
    /// - signal_state: MEASURED, PARTIAL, PROXY, or UNAVAILABLE.
    /// </summary>
    public class ThermalEstimator
    {
        public void Estimate(ThermalData thermal, CpuData cpu, GpuData gpu)
        {
            thermal.Sources.Clear();

            // -------------------------------------------------------------
            // 1. Measured Electrical Power
            // -------------------------------------------------------------
            double totalMeasuredPower = 0.0;
            bool cpuPowerMeasured = false;
            bool gpuPowerMeasured = false;

            if (cpu.PowerWatts.HasValue && cpu.PowerWatts.Value > 0)
            {
                totalMeasuredPower += cpu.PowerWatts.Value;
                thermal.Sources.Add("cpu_power");
                cpuPowerMeasured = true;
            }

            if (gpu.Available && gpu.Gpus.Count > 0)
            {
                double totalGpuPower = 0.0;
                bool anyGpuPower = false;
                foreach (var g in gpu.Gpus)
                {
                    if (g.PowerWatts.HasValue && g.PowerWatts.Value > 0)
                    {
                        totalGpuPower += g.PowerWatts.Value;
                        anyGpuPower = true;
                    }
                }

                if (anyGpuPower)
                {
                    totalMeasuredPower += totalGpuPower;
                    thermal.Sources.Add("gpu_power");
                    gpuPowerMeasured = true;
                }
            }

            // -------------------------------------------------------------
            // 2. Node Computational Thermal Load Proxy (0 - 100)
            // -------------------------------------------------------------
            double? cpuProxy = cpu.ThermalLoadIndex;
            double? gpuProxy = null;

            if (gpu.Available && gpu.Gpus.Count > 0)
            {
                gpuProxy = Math.Clamp(gpu.Gpus.Max(g => g.UtilizationPercent), 0.0, 100.0);
            }

            double? nodeThermalIndex = null;
            if (cpuProxy.HasValue && gpuProxy.HasValue)
            {
                // Weighted 50% CPU, 50% GPU
                nodeThermalIndex = Math.Round((0.5 * cpuProxy.Value) + (0.5 * gpuProxy.Value), 1);
                thermal.Sources.Add("cpu_load_proxy");
                thermal.Sources.Add("gpu_load_proxy");
            }
            else if (cpuProxy.HasValue)
            {
                nodeThermalIndex = Math.Round(cpuProxy.Value, 1);
                thermal.Sources.Add("cpu_load_proxy");
            }
            else if (gpuProxy.HasValue)
            {
                nodeThermalIndex = Math.Round(gpuProxy.Value, 1);
                thermal.Sources.Add("gpu_load_proxy");
            }

            thermal.NodeThermalLoadIndex = nodeThermalIndex;

            // -------------------------------------------------------------
            // 3. Assign Power and Signal State
            // -------------------------------------------------------------
            if (cpuPowerMeasured && (gpuPowerMeasured || !gpu.Available))
            {
                // Full measured power across all active computational components
                thermal.EstimatedPowerWatts = Math.Round(totalMeasuredPower, 1);
                thermal.EstimatedHeatWatts = Math.Round(totalMeasuredPower, 1);
                thermal.SignalState = "MEASURED";
                thermal.EstimationStatus = "MEASURED";
            }
            else if (cpuPowerMeasured || gpuPowerMeasured)
            {
                // Some physical wattage measured (e.g. dedicated GPU power measured, CPU power unavailable)
                // Report what was physically measured, but do NOT invent missing components
                thermal.EstimatedPowerWatts = Math.Round(totalMeasuredPower, 1);
                thermal.EstimatedHeatWatts = Math.Round(totalMeasuredPower, 1);
                thermal.SignalState = "PARTIAL";
                thermal.EstimationStatus = "PARTIAL";
            }
            else
            {
                // Zero physical wattage measured -> Fallback to PROXY
                thermal.EstimatedPowerWatts = null;
                thermal.EstimatedHeatWatts = null;

                if (nodeThermalIndex.HasValue)
                {
                    thermal.SignalState = "PROXY";
                    thermal.EstimationStatus = "PROXY";
                }
                else
                {
                    thermal.SignalState = "UNAVAILABLE";
                    thermal.EstimationStatus = "UNAVAILABLE";
                }
            }
        }
    }
}
