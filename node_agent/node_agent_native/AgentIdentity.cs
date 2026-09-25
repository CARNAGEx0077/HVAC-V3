using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace HveacNodeAgentNative
{
    public class IdentityData
    {
        public string NodeId { get; set; } = string.Empty;
        public string CreatedAt { get; set; } = string.Empty;
        public string Hostname { get; set; } = string.Empty;
    }

    public class AgentIdentity
    {
        public string NodeId { get; private set; } = string.Empty;
        public string Hostname => Environment.MachineName;
        public string Platform => "Windows";
        public string OsVersion => Environment.OSVersion.VersionString;
        public string Architecture => RuntimeInformation.OSArchitecture.ToString();
        public string AgentVersion { get; }
        public long UptimeSeconds => (long)TimeSpan.FromMilliseconds(Environment.TickCount64).TotalSeconds;

        private readonly string _identityPath;

        public AgentIdentity(string agentVersion, string? overrideNodeId = null)
        {
            AgentVersion = agentVersion;

            string programData = Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData);
            string hveacDir = Path.Combine(programData, "HVEAC");
            _identityPath = Path.Combine(hveacDir, "node_identity.json");

            // Priority 1: Explicit Override from environment / config
            if (!string.IsNullOrWhiteSpace(overrideNodeId))
            {
                NodeId = overrideNodeId.Trim();
                TryPersistIdentity(NodeId);
                return;
            }

            // Priority 2: Load existing persisted identity
            var existingId = TryLoadPersistedIdentity();
            if (!string.IsNullOrWhiteSpace(existingId))
            {
                NodeId = existingId;
                return;
            }

            // Priority 3: Generate new unique identity & persist
            NodeId = GenerateUniqueNodeId();
            TryPersistIdentity(NodeId);
        }

        private static string GenerateUniqueNodeId()
        {
            // Format: NODE-XXXXXXXX (8 uppercase hex chars)
            string hex = Guid.NewGuid().ToString("N")[..8].ToUpperInvariant();
            return $"NODE-{hex}";
        }

        private string? TryLoadPersistedIdentity()
        {
            try
            {
                if (File.Exists(_identityPath))
                {
                    string json = File.ReadAllText(_identityPath);
                    var data = JsonSerializer.Deserialize<IdentityData>(json);
                    if (!string.IsNullOrWhiteSpace(data?.NodeId))
                    {
                        return data.NodeId.Trim();
                    }
                }
            }
            catch
            {
                // Fallback check in user LocalAppData if CommonApplicationData had issues
                string fallbackPath = GetFallbackIdentityPath();
                try
                {
                    if (File.Exists(fallbackPath))
                    {
                        string json = File.ReadAllText(fallbackPath);
                        var data = JsonSerializer.Deserialize<IdentityData>(json);
                        if (!string.IsNullOrWhiteSpace(data?.NodeId))
                        {
                            return data.NodeId.Trim();
                        }
                    }
                }
                catch { }
            }

            return null;
        }

        private void TryPersistIdentity(string nodeId)
        {
            var data = new IdentityData
            {
                NodeId = nodeId,
                CreatedAt = DateTime.UtcNow.ToString("o"),
                Hostname = Hostname
            };

            string json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });

            // Try standard machine-local ProgramData directory
            try
            {
                string? dir = Path.GetDirectoryName(_identityPath);
                if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                File.WriteAllText(_identityPath, json);
                return;
            }
            catch
            {
                // If ProgramData requires elevated permissions, fallback to LocalAppData
                try
                {
                    string fallbackPath = GetFallbackIdentityPath();
                    string? fallbackDir = Path.GetDirectoryName(fallbackPath);
                    if (!string.IsNullOrEmpty(fallbackDir) && !Directory.Exists(fallbackDir))
                    {
                        Directory.CreateDirectory(fallbackDir);
                    }
                    File.WriteAllText(fallbackPath, json);
                }
                catch { }
            }
        }

        private static string GetFallbackIdentityPath()
        {
            string localApp = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            return Path.Combine(localApp, "HVEAC", "node_identity.json");
        }
    }
}
