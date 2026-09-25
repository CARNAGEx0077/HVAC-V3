using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;

namespace HveacNodeAgentNative
{
    public class ProcessCollector
    {
        [DllImport("user32.dll")]
        private static extern IntPtr GetForegroundWindow();

        [DllImport("user32.dll")]
        private static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

        private readonly Dictionary<int, (string Name, TimeSpan CpuTime, DateTime SampleTime)> _previousCpuTimes = new();

        public void PopulateWorkloadData(WorkloadData workload, CpuData cpu, GpuData gpu, MemoryData mem)
        {
            workload.ForegroundApplication = GetForegroundApp();
            workload.TopCpuProcesses = GetTopCpuProcesses();

            // Deterministic classification
            double cpuScore = cpu.UtilizationPercent;
            double gpuScore = gpu.Available && gpu.Gpus.Count > 0 ? gpu.Gpus.Max(g => g.UtilizationPercent) : 0;
            double memScore = mem.UtilizationPercent;

            if (gpuScore > 50.0)
            {
                workload.Category = "GPU_INTENSIVE";
                workload.HeuristicScore = Math.Round(gpuScore, 1);
            }
            else if (cpuScore > 60.0)
            {
                workload.Category = "CPU_INTENSIVE";
                workload.HeuristicScore = Math.Round(cpuScore, 1);
            }
            else if (memScore > 85.0)
            {
                workload.Category = "MEMORY_INTENSIVE";
                workload.HeuristicScore = Math.Round(memScore, 1);
            }
            else if (cpuScore > 10.0 || gpuScore > 10.0)
            {
                workload.Category = "GENERAL";
                workload.HeuristicScore = Math.Round(Math.Max(cpuScore, gpuScore), 1);
            }
            else
            {
                workload.Category = "IDLE";
                workload.HeuristicScore = 0.0;
            }
        }

        private static string? GetForegroundApp()
        {
            try
            {
                IntPtr hwnd = GetForegroundWindow();
                if (hwnd != IntPtr.Zero)
                {
                    GetWindowThreadProcessId(hwnd, out uint pid);
                    if (pid > 0)
                    {
                        using var proc = Process.GetProcessById((int)pid);
                        return proc.ProcessName;
                    }
                }
            }
            catch { }
            return null;
        }

        private List<string> GetTopCpuProcesses()
        {
            var now = DateTime.UtcNow;
            var currentSnapshot = new Dictionary<int, (string Name, TimeSpan CpuTime, DateTime SampleTime)>();
            var deltas = new List<(string Name, double CpuPercent, long MemoryBytes)>();

            Process[] processes;
            try
            {
                processes = Process.GetProcesses();
            }
            catch
            {
                return new List<string>();
            }

            int coreCount = Math.Max(1, Environment.ProcessorCount);

            foreach (var p in processes)
            {
                try
                {
                    if (p.Id <= 4 || p.HasExited) continue; // Skip System and Idle

                    string name = p.ProcessName;
                    TimeSpan cpuTime = p.TotalProcessorTime;
                    long workingSet = p.WorkingSet64;

                    currentSnapshot[p.Id] = (name, cpuTime, now);

                    if (_previousCpuTimes.TryGetValue(p.Id, out var prev))
                    {
                        var elapsed = (now - prev.SampleTime).TotalMilliseconds;
                        var cpuUsed = (cpuTime - prev.CpuTime).TotalMilliseconds;

                        if (elapsed > 200)
                        {
                            double percent = (cpuUsed / (elapsed * coreCount)) * 100.0;
                            deltas.Add((name, percent, workingSet));
                        }
                    }
                    else
                    {
                        // Fallback on memory weight if first cycle
                        deltas.Add((name, 0.0, workingSet));
                    }
                }
                catch
                {
                    // Access denied on privileged/system processes is normal
                }
                finally
                {
                    p.Dispose();
                }
            }

            _previousCpuTimes.Clear();
            foreach (var kv in currentSnapshot)
            {
                _previousCpuTimes[kv.Key] = kv.Value;
            }

            // Order by actual CPU delta if available, otherwise by working set
            var top = deltas
                .GroupBy(d => d.Name)
                .Select(g => new
                {
                    Name = g.Key,
                    TotalCpu = g.Sum(x => x.CpuPercent),
                    TotalMem = g.Sum(x => x.MemoryBytes)
                })
                .OrderByDescending(x => x.TotalCpu > 0 ? x.TotalCpu : (x.TotalMem / (1024.0 * 1024.0)))
                .Take(5)
                .Select(x => x.Name)
                .ToList();

            return top;
        }
    }
}
