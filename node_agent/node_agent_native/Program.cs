using System;
using System.Threading;
using System.Threading.Tasks;

namespace HveacNodeAgentNative
{
    class Program
    {
        static async Task Main(string[] args)
        {
            // 1. Diagnostic Mode & Automated Tests
            if (args.Length > 0 && args[0] == "--diagnose-hardware")
            {
                using var diagHardware = new HardwareProvider();
                diagHardware.RunDiagnostics();
                return;
            }

            if (args.Length > 0 && args[0] == "--test")
            {
                RunAutomatedTests();
                return;
            }

            // 2. Load Configuration
            var config = Config.Load();
            var logger = new AgentLogger();

            // 3. Persistent Identity
            var identity = new AgentIdentity(config.AgentVersion, config.NodeIdOverride);

            // 4. Banner
            logger.PrintBanner(config.AgentVersion, identity.NodeId, identity.Hostname, config.ServerUrl);

            // 5. Initialize Hardware Provider ONCE
            Console.WriteLine("[INIT] Hardware provider...");
            using var hardwareProvider = new HardwareProvider();
            try
            {
                hardwareProvider.Initialize();
                hardwareProvider.Update();
                logger.LogInit("Hardware provider", true);
            }
            catch (Exception ex)
            {
                logger.LogInit("Hardware provider", false, ex.Message);
            }

            // 6. Initialize Display Provider
            Console.WriteLine("[INIT] Display provider...");
            var displayProvider = new DisplayProvider();
            logger.LogInit("Display provider", true);

            // 7. Initialize Process, Workload & Thermal Analyzers
            Console.WriteLine("[INIT] Process & Workload analyzers...");
            var processCollector = new ProcessCollector();
            var workloadAnalyzer = new WorkloadAnalyzer(windowSeconds: 30.0, highLoadThreshold: 60.0);
            var thermalEstimator = new ThermalEstimator();
            logger.LogInit("Process & Workload analyzers", true);

            // 8. Initialize Telemetry Collector & Network Client
            Console.WriteLine("[INIT] Telemetry engine & Network client...");
            var collector = new TelemetryCollector(
                identity,
                hardwareProvider,
                displayProvider,
                processCollector,
                workloadAnalyzer,
                thermalEstimator,
                config.SchemaVersion
            );

            using var client = new TelemetryClient(config.ServerUrl, config.MaxBufferedPackets);
            client.StateChanged += (oldState, newState) =>
            {
                logger.LogStateTransition(oldState, newState, client.BufferedCount);
            };

            logger.LogInit("Telemetry engine", true);
            logger.LogInit($"Identity: {identity.NodeId}", true);
            Console.WriteLine("================================================\n");

            // Setup Graceful Shutdown
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (sender, eventArgs) =>
            {
                eventArgs.Cancel = true;
                Console.WriteLine("\n[AGENT] Shutdown signal received. Exiting gracefully...");
                cts.Cancel();
            };

            Console.WriteLine($"[AGENT] Starting telemetry cycle ({config.TelemetryIntervalSeconds}s interval). Press Ctrl+C to stop.\n");

            bool firstSummaryPrinted = false;

            // 9. Main Telemetry Loop
            while (!cts.IsCancellationRequested)
            {
                var cycleStart = DateTime.UtcNow;

                try
                {
                    // Collect telemetry snapshot
                    var snapshot = collector.CollectSnapshot();

                    // Print detailed summary once on startup
                    if (!firstSummaryPrinted)
                    {
                        logger.PrintDetailedSummary(snapshot);
                        firstSummaryPrinted = true;
                    }

                    // Send asynchronously to backend
                    await client.SendTelemetryAsync(snapshot, cts.Token);

                    // Log telemetry status
                    logger.LogTelemetryCycle(snapshot, client.State, client.BufferedCount);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [ERROR] Unexpected telemetry error: {ex.Message}");
                }

                // Compute sleep duration with backoff if disconnected
                var cycleElapsed = (DateTime.UtcNow - cycleStart).TotalSeconds;
                int targetInterval = client.GetBackoffSeconds(config.TelemetryIntervalSeconds);
                double remainingSeconds = targetInterval - cycleElapsed;

                if (remainingSeconds > 0)
                {
                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(remainingSeconds), cts.Token);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                }
            }

            Console.WriteLine("[AGENT] Node Agent stopped.");
        }

        private static void RunAutomatedTests()
        {
            Console.WriteLine("==================================================");
            Console.WriteLine("  HVEAC ENDPOINT AGENT - AUTOMATED TEST SUITE");
            Console.WriteLine("==================================================\n");

            int passed = 0;
            int total = 10;

            // TEST 1: CPU utilization < 10% (IDLE), 10-30% (LIGHT)
            bool t1 = WorkloadAnalyzer.ClassifyCategory(5.0) == "IDLE" &&
                      WorkloadAnalyzer.ClassifyCategory(20.0) == "LIGHT";
            PrintTestResult(1, "CPU utilization low -> IDLE / LIGHT", t1);
            if (t1) passed++;

            // TEST 2: CPU utilization = 45% (GENERAL)
            bool t2 = WorkloadAnalyzer.ClassifyCategory(45.0) == "GENERAL";
            PrintTestResult(2, "CPU utilization 45% -> GENERAL", t2);
            if (t2) passed++;

            // TEST 3: CPU utilization = 70% (CPU_INTENSIVE)
            bool t3 = WorkloadAnalyzer.ClassifyCategory(70.0) == "CPU_INTENSIVE";
            PrintTestResult(3, "CPU utilization 70% -> CPU_INTENSIVE", t3);
            if (t3) passed++;

            // TEST 4: CPU utilization = 90% (HEAVY_CPU_LOAD)
            bool t4 = WorkloadAnalyzer.ClassifyCategory(90.0) == "HEAVY_CPU_LOAD";
            PrintTestResult(4, "CPU utilization 90% -> HEAVY_CPU_LOAD", t4);
            if (t4) passed++;

            // TEST 5: CPU temp unavailable, CPU power unavailable, CPU util available
            var analyzer = new WorkloadAnalyzer(30.0, 60.0);
            var cpu5 = new CpuData { UtilizationPercent = 45.0, TemperatureC = null, PowerWatts = null };
            analyzer.Analyze(cpu5, DateTime.UtcNow);
            bool t5 = cpu5.TemperatureC == null &&
                      cpu5.PowerWatts == null &&
                      cpu5.ThermalSignalState == "PROXY" &&
                      cpu5.ThermalLoadIndex == 45.0;
            PrintTestResult(5, "CPU temp/power null, util available -> PROXY signal & valid index", t5);
            if (t5) passed++;

            // TEST 6: CPU power unavailable, GPU power available -> PARTIAL
            var estimator = new ThermalEstimator();
            var thermal6 = new ThermalData();
            var cpu6 = new CpuData { UtilizationPercent = 50.0, PowerWatts = null };
            var gpu6 = new GpuData
            {
                Available = true,
                Gpus = new System.Collections.Generic.List<GpuDeviceData>
                {
                    new GpuDeviceData { Id = 0, Name = "Test GPU", PowerWatts = 15.0, UtilizationPercent = 20.0 }
                }
            };
            analyzer.Analyze(cpu6, DateTime.UtcNow);
            estimator.Estimate(thermal6, cpu6, gpu6);
            bool t6 = cpu6.PowerWatts == null &&
                      thermal6.EstimatedPowerWatts == 15.0 &&
                      thermal6.SignalState == "PARTIAL";
            PrintTestResult(6, "CPU power null, GPU power 15W -> PARTIAL power (15W)", t6);
            if (t6) passed++;

            // TEST 7: CPU/GPU power unavailable, CPU/GPU util available -> PROXY & no fake watts
            var thermal7 = new ThermalData();
            var cpu7 = new CpuData { UtilizationPercent = 80.0, PowerWatts = null };
            var gpu7 = new GpuData
            {
                Available = true,
                Gpus = new System.Collections.Generic.List<GpuDeviceData>
                {
                    new GpuDeviceData { Id = 0, Name = "Test GPU", PowerWatts = null, UtilizationPercent = 40.0 }
                }
            };
            analyzer.Analyze(cpu7, DateTime.UtcNow);
            estimator.Estimate(thermal7, cpu7, gpu7);
            bool t7 = thermal7.EstimatedPowerWatts == null &&
                      thermal7.EstimatedHeatWatts == null &&
                      thermal7.SignalState == "PROXY" &&
                      thermal7.NodeThermalLoadIndex == 60.0; // (0.5 * 80) + (0.5 * 40) = 60
            PrintTestResult(7, "No measured wattage -> estimated_power=null, node_proxy=60/100, PROXY", t7);
            if (t7) passed++;

            // TEST 8: Rolling average over window changes correctly
            var analyzer8 = new WorkloadAnalyzer(30.0, 60.0);
            var now = DateTime.UtcNow;
            var cpu8a = new CpuData { UtilizationPercent = 80.0 };
            analyzer8.Analyze(cpu8a, now);
            var cpu8b = new CpuData { UtilizationPercent = 60.0 };
            analyzer8.Analyze(cpu8b, now.AddSeconds(2));
            bool t8 = cpu8b.ShortTermAveragePercent == 70.0;
            PrintTestResult(8, "Rolling average updates across window (80% + 60% = 70%)", t8);
            if (t8) passed++;

            // TEST 9: Sustained high load tracking
            var analyzer9 = new WorkloadAnalyzer(30.0, 60.0);
            var t0 = DateTime.UtcNow;
            var cpu9a = new CpuData { UtilizationPercent = 80.0 };
            analyzer9.Analyze(cpu9a, t0);
            var cpu9b = new CpuData { UtilizationPercent = 85.0 };
            analyzer9.Analyze(cpu9b, t0.AddSeconds(15));
            bool t9 = cpu9b.IsSustainedHighLoad && cpu9b.SustainedLoadSeconds == 15;
            PrintTestResult(9, "Sustained high load duration correctly tracks 15 seconds", t9);
            if (t9) passed++;

            // TEST 10: Reset resets rolling history
            analyzer9.Reset();
            var cpu10 = new CpuData { UtilizationPercent = 30.0 };
            analyzer9.Analyze(cpu10, DateTime.UtcNow);
            bool t10 = cpu10.ShortTermAveragePercent == 30.0 && !cpu10.IsSustainedHighLoad && cpu10.SustainedLoadSeconds == 0;
            PrintTestResult(10, "Reset clears historical state cleanly without residual data", t10);
            if (t10) passed++;

            Console.WriteLine($"\nRESULTS: {passed}/{total} tests passed.");
            if (passed != total)
            {
                Environment.Exit(1);
            }
        }

        private static void PrintTestResult(int testNum, string description, bool passed)
        {
            string status = passed ? "[PASS]" : "[FAIL]";
            Console.WriteLine($"{status} Test {testNum}: {description}");
        }
    }
}
