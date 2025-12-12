using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;

namespace F3.ToolHub.IntegrationTests.Infrastructure;

public sealed class ModbusSimulator : IAsyncDisposable
{
    private readonly TcpListener _listener;
    private readonly CancellationTokenSource _cts = new();
    private readonly List<Task> _clientHandlers = new();
    private Task? _acceptLoop;
    private readonly ushort[] _registers;

    public ModbusSimulator(int port = 0, ushort[]? registers = null)
    {
        _listener = new TcpListener(IPAddress.Loopback, port);
        _registers = registers ?? new ushort[] { 10, 20, 30, 40 };
    }

    public int Port { get; private set; }

    public Task StartAsync()
    {
        _listener.Start();
        Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
        _acceptLoop = Task.Run(AcceptLoopAsync, _cts.Token);
        return Task.CompletedTask;
    }

    public async Task StopAsync()
    {
        _cts.Cancel();
        _listener.Stop();

        if (_acceptLoop is not null)
        {
            try
            {
                await _acceptLoop.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // ignored
            }
        }

        Task[] clientTasks;
        lock (_clientHandlers)
        {
            clientTasks = _clientHandlers.ToArray();
        }

        if (clientTasks.Length > 0)
        {
            try
            {
                await Task.WhenAll(clientTasks).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // ignored
            }
        }
    }

    private async Task AcceptLoopAsync()
    {
        try
        {
            while (!_cts.IsCancellationRequested)
            {
                TcpClient client;
                try
                {
                    client = await _listener.AcceptTcpClientAsync(_cts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    break;
                }

                var handlerTask = Task.Run(() => HandleClientAsync(client, _cts.Token), _cts.Token);
                TrackClient(handlerTask);
            }
        }
        catch (ObjectDisposedException)
        {
            // listener stopped
        }
    }

    private void TrackClient(Task task)
    {
        lock (_clientHandlers)
        {
            _clientHandlers.Add(task);
        }

        task.ContinueWith(
            _ =>
            {
                lock (_clientHandlers)
                {
                    _clientHandlers.Remove(task);
                }
            },
            CancellationToken.None,
            TaskContinuationOptions.ExecuteSynchronously,
            TaskScheduler.Default);
    }

    private async Task HandleClientAsync(TcpClient client, CancellationToken token)
    {
        using var tcpClient = client;
        using var stream = tcpClient.GetStream();
        var frame = new byte[12];

        while (!token.IsCancellationRequested)
        {
            try
            {
                await stream.ReadExactlyAsync(frame, token).ConfigureAwait(false);
            }
            catch
            {
                break;
            }

            var transactionId = BinaryPrimitives.ReadUInt16BigEndian(frame.AsSpan(0, 2));
            var protocolId = BinaryPrimitives.ReadUInt16BigEndian(frame.AsSpan(2, 2));
            var unitId = frame[6];
            var functionCode = frame[7];
            var startAddress = BinaryPrimitives.ReadUInt16BigEndian(frame.AsSpan(8, 2));
            var numberOfPoints = BinaryPrimitives.ReadUInt16BigEndian(frame.AsSpan(10, 2));

            if (functionCode != 0x03 || numberOfPoints == 0)
            {
                continue;
            }

            var byteCount = numberOfPoints * 2;
            var response = new byte[9 + byteCount];
            BinaryPrimitives.WriteUInt16BigEndian(response.AsSpan(0, 2), transactionId);
            BinaryPrimitives.WriteUInt16BigEndian(response.AsSpan(2, 2), protocolId);
            BinaryPrimitives.WriteUInt16BigEndian(response.AsSpan(4, 2), (ushort)(3 + byteCount));
            response[6] = unitId;
            response[7] = 0x03;
            response[8] = (byte)byteCount;

            for (var i = 0; i < numberOfPoints; i++)
            {
                var registerIndex = (startAddress + i) % _registers.Length;
                BinaryPrimitives.WriteUInt16BigEndian(response.AsSpan(9 + (i * 2), 2), _registers[registerIndex]);
            }

            await stream.WriteAsync(response, token).ConfigureAwait(false);
            await stream.FlushAsync(token).ConfigureAwait(false);
        }
    }

    public async ValueTask DisposeAsync()
    {
        await StopAsync().ConfigureAwait(false);
        _cts.Dispose();
    }
}
