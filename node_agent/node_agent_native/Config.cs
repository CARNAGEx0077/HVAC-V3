using System;
using System.IO;

namespace HveacNodeAgentNative
{
    public class Config
    {
        public string? NodeIdOverride { get; set; } = null;
        public string ServerUrl { get; set; } = "http://127.0.0.1:8000";
        public int TelemetryIntervalSeconds { get; set; } = 2;
        public string AgentVersion { get; set; } = "0.3.0";
        public string SchemaVersion { get; set; } = "1.0";
        public int MaxBufferedPackets { get; set; } = 100;

        public static Config Load()
        {
            var cfg = new Config();

            // Check potential .env paths
            string[] candidatePaths = new[]
            {
                Path.Combine(AppContext.BaseDirectory, ".env"),
                Path.Combine(Directory.GetCurrentDirectory(), ".env"),
                Path.Combine(Directory.GetCurrentDirectory(), "../.env"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData), "HVEAC", ".env")
            };

            foreach (var path in candidatePaths)
            {
                try
                {
                    if (File.Exists(path))
                    {
                        ParseEnvFile(path, cfg);
                        break;
                    }
                }
                catch { }
            }

            // Environment variables override .env
            var envNodeId = Environment.GetEnvironmentVariable("HVEAC_NODE_ID");
            if (!string.IsNullOrWhiteSpace(envNodeId)) cfg.NodeIdOverride = envNodeId.Trim();

            var envServerUrl = Environment.GetEnvironmentVariable("HVEAC_SERVER_URL");
            if (!string.IsNullOrWhiteSpace(envServerUrl)) cfg.ServerUrl = envServerUrl.Trim();

            var envInterval = Environment.GetEnvironmentVariable("TELEMETRY_INTERVAL_SECONDS");
            if (int.TryParse(envInterval, out var interval) && interval > 0) cfg.TelemetryIntervalSeconds = interval;

            var envMaxBuffer = Environment.GetEnvironmentVariable("MAX_BUFFERED_PACKETS");
            if (int.TryParse(envMaxBuffer, out var maxBuf) && maxBuf > 0) cfg.MaxBufferedPackets = maxBuf;

            return cfg;
        }

        private static void ParseEnvFile(string path, Config cfg)
        {
            foreach (var line in File.ReadAllLines(path))
            {
                var trimmed = line.Trim();
                if (string.IsNullOrEmpty(trimmed) || trimmed.StartsWith("#")) continue;

                var parts = trimmed.Split('=', 2);
                if (parts.Length != 2) continue;

                var k = parts[0].Trim();
                var v = parts[1].Trim().Trim('"', '\'');

                if (k == "HVEAC_NODE_ID" && !string.IsNullOrWhiteSpace(v)) cfg.NodeIdOverride = v;
                else if (k == "HVEAC_SERVER_URL" && !string.IsNullOrWhiteSpace(v)) cfg.ServerUrl = v;
                else if (k == "TELEMETRY_INTERVAL_SECONDS" && int.TryParse(v, out var ti) && ti > 0) cfg.TelemetryIntervalSeconds = ti;
                else if (k == "MAX_BUFFERED_PACKETS" && int.TryParse(v, out var mb) && mb > 0) cfg.MaxBufferedPackets = mb;
            }
        }
    }
}
