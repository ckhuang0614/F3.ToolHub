using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Telemetry;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using MQTTnet;
using MQTTnet.Client;
using MQTTnet.Protocol;

namespace F3.ToolHub.Infrastructure.Modbus.Sinks;

public sealed class MqttPlcDataSink : IPlcDataSink, IAsyncDisposable
{
    private readonly IOptionsMonitor<ModbusOptions> _options;
    private readonly ILogger<MqttPlcDataSink> _logger;
    private readonly SemaphoreSlim _connectionLock = new(1, 1);
    private IMqttClient? _client;

    public MqttPlcDataSink(IOptionsMonitor<ModbusOptions> options, ILogger<MqttPlcDataSink> logger)
    {
        _options = options;
        _logger = logger;
    }

    public async Task PublishAsync(PlcDataSnapshot snapshot, CancellationToken cancellationToken)
    {
        var settings = _options.CurrentValue.Output?.Mqtt;
        if (settings is null || !settings.Enabled)
        {
            return;
        }

        var client = await EnsureClientAsync(settings, cancellationToken).ConfigureAwait(false);
        if (client is null)
        {
            return;
        }

        try
        {
            var payload = JsonSerializer.SerializeToUtf8Bytes(snapshot);
            var message = new MqttApplicationMessageBuilder()
                .WithTopic(settings.Topic)
                .WithPayload(payload)
                .WithQualityOfServiceLevel(MQTTnet.Protocol.MqttQualityOfServiceLevel.AtLeastOnce)
                .Build();

            await client.PublishAsync(message, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to publish PLC snapshot to MQTT topic {Topic}", settings.Topic);
        }
    }

    private async Task<IMqttClient?> EnsureClientAsync(MqttOutputOptions settings, CancellationToken cancellationToken)
    {
        if (_client is { IsConnected: true })
        {
            return _client;
        }

        await _connectionLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_client is { IsConnected: true })
            {
                return _client;
            }

            _client?.Dispose();
            var factory = new MqttFactory();
            _client = factory.CreateMqttClient();

            var options = new MqttClientOptionsBuilder()
                .WithClientId(settings.ClientId ?? $"f3-toolhub-{Environment.MachineName}-{Guid.NewGuid():N}")
                .WithTcpServer(settings.Host, settings.Port)
                .Build();

            await _client.ConnectAsync(options, cancellationToken).ConfigureAwait(false);
            _logger.LogInformation("Connected to MQTT broker {Host}:{Port}", settings.Host, settings.Port);
            return _client;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to connect to MQTT broker {Host}:{Port}", settings.Host, settings.Port);
            return null;
        }
        finally
        {
            _connectionLock.Release();
        }
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_client is not null)
            {
                await _client.DisconnectAsync();
                _client.Dispose();
            }
        }
        catch
        {
            // ignore during shutdown
        }
        finally
        {
            _connectionLock.Dispose();
        }
    }
}
