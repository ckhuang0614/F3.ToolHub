using System.Buffers.Binary;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Monitoring;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class ModbusTcpClient : IModbusClient, IModbusClientMetrics, IAsyncDisposable
{
    private readonly IOptionsMonitor<ModbusOptions> _options;
    private readonly ILogger<ModbusTcpClient> _logger;
    private readonly IPlcAlertService _alertService;
    private readonly ConcurrentBag<PooledConnection> _pool = new();
    private readonly SemaphoreSlim _poolLimiter;
    private readonly int _poolSize;
    private int _activeConnections;
    private long _successCount;
    private long _failureCount;
    private readonly object _stateGate = new();
    private int _consecutiveFailures;
    private DateTimeOffset _circuitOpenUntil = DateTimeOffset.MinValue;
    private DateTimeOffset? _lastSuccess;
    private DateTimeOffset? _lastFailure;

    public ModbusTcpClient(IOptionsMonitor<ModbusOptions> options, ILogger<ModbusTcpClient> logger, IPlcAlertService alertService)
    {
        _options = options;
        _logger = logger;
        _alertService = alertService;
        var resilience = options.CurrentValue.Resilience ?? new ModbusResilienceOptions();
        _poolSize = Math.Max(1, resilience.ConnectionPoolSize);
        _poolLimiter = new SemaphoreSlim(_poolSize, _poolSize);
    }

    public async Task<ushort[]> ReadHoldingRegistersAsync(ushort startAddress, ushort numberOfPoints, CancellationToken cancellationToken)
    {
        if (numberOfPoints == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfPoints));
        }

        return await ExecuteWithResilienceAsync(startAddress, numberOfPoints, cancellationToken).ConfigureAwait(false);
    }

    public async ValueTask DisposeAsync()
    {
        while (_pool.TryTake(out var connection))
        {
            await connection.DisposeAsync().ConfigureAwait(false);
        }

        _poolLimiter.Dispose();
    }

    public ModbusClientHealthSnapshot Capture()
    {
        lock (_stateGate)
        {
            var circuitOpen = _circuitOpenUntil > DateTimeOffset.UtcNow;
            return new ModbusClientHealthSnapshot(
                _lastSuccess,
                _lastFailure,
                _consecutiveFailures,
                circuitOpen,
                Volatile.Read(ref _activeConnections),
                _poolLimiter.CurrentCount,
                _successCount,
                _failureCount);
        }
    }

    private async Task<ushort[]> ExecuteWithResilienceAsync(ushort startAddress, ushort numberOfPoints, CancellationToken cancellationToken)
    {
        var resilience = _options.CurrentValue.Resilience ?? new ModbusResilienceOptions();
        var maxAttempts = Math.Max(1, resilience.MaxRetryAttempts);
        Exception? lastError = null;

        for (var attempt = 1; attempt <= maxAttempts; attempt++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (TryGetCircuitOpenDelay(out var remainingDelay))
            {
                throw new InvalidOperationException($"Modbus circuit breaker is open. Retry after {remainingDelay.TotalSeconds:F1}s.");
            }

            PooledConnection? connection = null;
            try
            {
                var settings = _options.CurrentValue;
                connection = await AcquireConnectionAsync(settings, cancellationToken).ConfigureAwait(false);
                var registers = await PerformReadAsync(connection, settings, startAddress, numberOfPoints, cancellationToken).ConfigureAwait(false);
                await ReleaseConnectionAsync(connection, success: true).ConfigureAwait(false);
                MarkSuccess();
                return registers;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                lastError = ex;
                if (connection is not null)
                {
                    await ReleaseConnectionAsync(connection, success: false).ConfigureAwait(false);
                }

                _logger.LogWarning(ex, "Modbus read attempt {Attempt}/{Total} failed.", attempt, maxAttempts);
                var circuitOpened = RecordFailure(resilience);
                if (circuitOpened)
                {
                    throw new InvalidOperationException("Modbus circuit breaker opened due to repeated failures.", lastError);
                }

                if (attempt < maxAttempts)
                {
                    var delay = CalculateBackoffDelay(resilience, attempt);
                    _logger.LogInformation("Retrying Modbus read in {Delay}.", delay);
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }
        }

        throw new InvalidOperationException("Modbus read failed after exhausting all retries.", lastError);
    }

    private async Task<PooledConnection> AcquireConnectionAsync(ModbusOptions settings, CancellationToken cancellationToken)
    {
        await _poolLimiter.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            while (_pool.TryTake(out var pooled))
            {
                if (pooled.IsHealthy)
                {
                    Interlocked.Increment(ref _activeConnections);
                    return pooled;
                }

                await pooled.DisposeAsync().ConfigureAwait(false);
            }

            var created = await CreateConnectionAsync(settings, cancellationToken).ConfigureAwait(false);
            Interlocked.Increment(ref _activeConnections);
            return created;
        }
        catch
        {
            _poolLimiter.Release();
            throw;
        }
    }

    private async ValueTask ReleaseConnectionAsync(PooledConnection connection, bool success)
    {
        try
        {
            if (success && connection.IsHealthy)
            {
                connection.Touch();
                _pool.Add(connection);
            }
            else
            {
                await connection.DisposeAsync().ConfigureAwait(false);
            }
        }
        finally
        {
            Interlocked.Decrement(ref _activeConnections);
            _poolLimiter.Release();
        }
    }

    private async Task<PooledConnection> CreateConnectionAsync(ModbusOptions settings, CancellationToken cancellationToken)
    {
        var client = new TcpClient
        {
            NoDelay = true
        };
        client.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);

        using (var connectCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken))
        {
            connectCts.CancelAfter(TimeSpan.FromSeconds(Math.Max(1, settings.ConnectTimeoutSeconds)));
            await client.ConnectAsync(settings.Host, settings.Port, connectCts.Token).ConfigureAwait(false);
        }

        _logger.LogInformation("Established Modbus TCP connection to {Host}:{Port}.", settings.Host, settings.Port);
        return new PooledConnection(client);
    }

    private async Task<ushort[]> PerformReadAsync(PooledConnection connection, ModbusOptions settings, ushort startAddress, ushort numberOfPoints, CancellationToken cancellationToken)
    {
        var timeout = TimeSpan.FromSeconds(Math.Max(1, settings.ResponseTimeoutSeconds));
        using var responseCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        responseCts.CancelAfter(timeout);

        var request = BuildRequestFrame(settings.SlaveId, startAddress, numberOfPoints);
        await connection.Stream.WriteAsync(request, responseCts.Token).ConfigureAwait(false);
        await connection.Stream.FlushAsync(responseCts.Token).ConfigureAwait(false);

        var header = new byte[7];
        await connection.Stream.ReadExactlyAsync(header, responseCts.Token).ConfigureAwait(false);
        var length = BinaryPrimitives.ReadUInt16BigEndian(header.AsSpan(4, 2));
        if (length <= 0)
        {
            throw new InvalidOperationException("Invalid Modbus response length.");
        }

        var payload = new byte[length];
        payload[0] = header[6];
        if (length > 1)
        {
            await connection.Stream.ReadExactlyAsync(payload.AsMemory(1), responseCts.Token).ConfigureAwait(false);
        }

        ValidateResponse(settings.SlaveId, numberOfPoints, payload);
        return ExtractRegisters(numberOfPoints, payload);
    }

    private bool TryGetCircuitOpenDelay(out TimeSpan delay)
    {
        lock (_stateGate)
        {
            if (_circuitOpenUntil <= DateTimeOffset.UtcNow)
            {
                delay = TimeSpan.Zero;
                return false;
            }

            delay = _circuitOpenUntil - DateTimeOffset.UtcNow;
            return true;
        }
    }

    private void MarkSuccess()
    {
        lock (_stateGate)
        {
            _successCount++;
            _lastSuccess = DateTimeOffset.UtcNow;
            _consecutiveFailures = 0;
            _circuitOpenUntil = DateTimeOffset.MinValue;
        }
    }

    private bool RecordFailure(ModbusResilienceOptions resilience)
    {
        lock (_stateGate)
        {
            _failureCount++;
            _lastFailure = DateTimeOffset.UtcNow;
            _consecutiveFailures++;
            if (_consecutiveFailures < Math.Max(1, resilience.CircuitBreakerFailureThreshold))
            {
                return false;
            }

            _consecutiveFailures = 0;
            _circuitOpenUntil = DateTimeOffset.UtcNow.AddSeconds(Math.Max(1, resilience.CircuitBreakerDurationSeconds));
            _logger.LogError("Modbus circuit breaker opened for {Duration}s due to repeated failures.", resilience.CircuitBreakerDurationSeconds);
            _alertService.Raise(new PlcAlert("modbus.client", "Circuit breaker opened", "High", DateTimeOffset.UtcNow));
            return true;
        }
    }

    private static TimeSpan CalculateBackoffDelay(ModbusResilienceOptions resilience, int attempt)
    {
        var baseMs = Math.Max(50, resilience.BaseDelayMilliseconds);
        var maxMs = Math.Max(baseMs, resilience.MaxDelayMilliseconds);
        var exponential = baseMs * Math.Pow(2, attempt - 1);
        var clamped = Math.Min(maxMs, exponential);
        return TimeSpan.FromMilliseconds(clamped);
    }

    private static byte[] BuildRequestFrame(byte slaveId, ushort startAddress, ushort numberOfPoints)
    {
        var frame = new byte[12];
        var transactionId = (ushort)Random.Shared.Next(ushort.MaxValue);
        BinaryPrimitives.WriteUInt16BigEndian(frame.AsSpan(0, 2), transactionId);
        BinaryPrimitives.WriteUInt16BigEndian(frame.AsSpan(2, 2), 0);
        BinaryPrimitives.WriteUInt16BigEndian(frame.AsSpan(4, 2), 6);
        frame[6] = slaveId;
        frame[7] = 0x03;
        BinaryPrimitives.WriteUInt16BigEndian(frame.AsSpan(8, 2), startAddress);
        BinaryPrimitives.WriteUInt16BigEndian(frame.AsSpan(10, 2), numberOfPoints);
        return frame;
    }

    private void ValidateResponse(byte expectedSlave, ushort expectedPoints, ReadOnlySpan<byte> payload)
    {
        if (payload.Length < 3)
        {
            throw new InvalidOperationException("Modbus response is too short.");
        }

        var unitId = payload[0];
        var functionCode = payload[1];

        if (unitId != expectedSlave)
        {
            throw new InvalidOperationException($"Unexpected UnitId {unitId}.");
        }

        if ((functionCode & 0x80) == 0x80)
        {
            var exceptionCode = payload.Length > 2 ? payload[2] : (byte)0;
            throw new InvalidOperationException($"PLC reported Modbus exception {exceptionCode}.");
        }

        if (functionCode != 0x03)
        {
            throw new InvalidOperationException($"Unexpected function code {functionCode}.");
        }

        var byteCount = payload[2];
        if (byteCount != expectedPoints * 2)
        {
            _logger.LogWarning("Expected {Expected} bytes but received {Actual}", expectedPoints * 2, byteCount);
        }

        if (payload.Length < byteCount + 3)
        {
            throw new InvalidOperationException("Incomplete Modbus payload.");
        }
    }

    private static ushort[] ExtractRegisters(ushort numberOfPoints, ReadOnlySpan<byte> payload)
    {
        var data = new ushort[numberOfPoints];
        for (var i = 0; i < numberOfPoints; i++)
        {
            var offset = 3 + (i * 2);
            data[i] = BinaryPrimitives.ReadUInt16BigEndian(payload.Slice(offset, 2));
        }

        return data;
    }

    private sealed class PooledConnection : IAsyncDisposable
    {
        public PooledConnection(TcpClient client)
        {
            Client = client;
            Stream = client.GetStream();
            LastUsed = DateTimeOffset.UtcNow;
        }

        public TcpClient Client { get; }

        public NetworkStream Stream { get; }

        public DateTimeOffset LastUsed { get; private set; }

        public bool IsHealthy => Client.Connected && Stream.CanRead && Stream.CanWrite;

        public void Touch()
        {
            LastUsed = DateTimeOffset.UtcNow;
        }

        public ValueTask DisposeAsync()
        {
            Stream.Dispose();
            Client.Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
