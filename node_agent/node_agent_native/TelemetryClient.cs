using System;
using System.Collections.Concurrent;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace HveacNodeAgentNative
{
    public enum ConnectionState
    {
        STARTING,
        ONLINE,
        RECONNECTING,
        OFFLINE
    }

    public class TelemetryClient : IDisposable
    {
        private readonly HttpClient _http;
        private readonly string _serverUrl;
        private readonly int _maxBuffer;
        private readonly ConcurrentQueue<TelemetryPayload> _buffer;
        private readonly JsonSerializerOptions _jsonOptions;
        private readonly SemaphoreSlim _sendLock = new(1, 1);

        public ConnectionState State { get; private set; } = ConnectionState.STARTING;
        public int BufferedCount => _buffer.Count;
        public int ConsecutiveFailures { get; private set; } = 0;

        public event Action<ConnectionState, ConnectionState>? StateChanged;

        public TelemetryClient(string serverUrl, int maxBuffer = 100)
        {
            _serverUrl = serverUrl.TrimEnd('/');
            _maxBuffer = Math.Max(10, maxBuffer);
            _buffer = new ConcurrentQueue<TelemetryPayload>();

            _http = new HttpClient
            {
                Timeout = TimeSpan.FromSeconds(5)
            };

            _jsonOptions = new JsonSerializerOptions
            {
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                WriteIndented = false
            };
        }

        public async Task<bool> SendTelemetryAsync(TelemetryPayload payload, CancellationToken cancellationToken = default)
        {
            // 1. Enqueue payload into FIFO buffer
            _buffer.Enqueue(payload);

            // 2. Enforce bounded buffer limit (drop oldest if capacity exceeded)
            while (_buffer.Count > _maxBuffer)
            {
                _buffer.TryDequeue(out _);
            }

            // 3. Attempt to transmit buffered payloads in FIFO order
            await _sendLock.WaitAsync(cancellationToken);
            try
            {
                return await FlushQueueAsync(cancellationToken);
            }
            finally
            {
                _sendLock.Release();
            }
        }

        private async Task<bool> FlushQueueAsync(CancellationToken cancellationToken)
        {
            string url = $"{_serverUrl}/api/nodes/telemetry";

            while (!_buffer.IsEmpty && !cancellationToken.IsCancellationRequested)
            {
                // Peek next payload to send
                if (!_buffer.TryPeek(out var nextPayload))
                {
                    break;
                }

                bool success = false;
                try
                {
                    string json = JsonSerializer.Serialize(nextPayload, _jsonOptions);
                    using var content = new StringContent(json, Encoding.UTF8, "application/json");

                    var response = await _http.PostAsync(url, content, cancellationToken);
                    if (response.IsSuccessStatusCode)
                    {
                        success = true;
                        // Successfully delivered; dequeue it now
                        _buffer.TryDequeue(out _);
                    }
                    else
                    {
                        // HTTP error (e.g. 500 or 400)
                        success = false;
                    }
                }
                catch
                {
                    success = false;
                }

                if (success)
                {
                    ConsecutiveFailures = 0;
                    if (State != ConnectionState.ONLINE)
                    {
                        TransitionTo(ConnectionState.ONLINE);
                    }
                }
                else
                {
                    ConsecutiveFailures++;
                    if (State == ConnectionState.ONLINE || State == ConnectionState.STARTING)
                    {
                        TransitionTo(ConnectionState.RECONNECTING);
                    }
                    else if (State == ConnectionState.RECONNECTING && ConsecutiveFailures >= 5)
                    {
                        TransitionTo(ConnectionState.OFFLINE);
                    }

                    // Stop draining queue on first network failure to preserve order
                    return false;
                }
            }

            return true;
        }

        private void TransitionTo(ConnectionState newState)
        {
            var oldState = State;
            State = newState;
            StateChanged?.Invoke(oldState, newState);
        }

        public int GetBackoffSeconds(int baseIntervalSeconds)
        {
            if (State == ConnectionState.ONLINE || ConsecutiveFailures == 0)
            {
                return baseIntervalSeconds;
            }

            // Exponential backoff: 2s -> 4s -> 8s -> 16s (max 30s)
            int exp = Math.Min(ConsecutiveFailures, 5);
            int backoff = (int)Math.Pow(2, exp);
            return Math.Min(Math.Max(baseIntervalSeconds, backoff), 30);
        }

        public void Dispose()
        {
            _sendLock.Dispose();
            _http.Dispose();
        }
    }
}
