using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Telemetry;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Modbus.Sinks;

public sealed class FilePlcDataSink : IPlcDataSink
{
    private readonly IOptionsMonitor<ModbusOptions> _options;
    private readonly ILogger<FilePlcDataSink> _logger;

    public FilePlcDataSink(IOptionsMonitor<ModbusOptions> options, ILogger<FilePlcDataSink> logger)
    {
        _options = options;
        _logger = logger;
    }

    public async Task PublishAsync(PlcDataSnapshot snapshot, CancellationToken cancellationToken)
    {
        var settings = _options.CurrentValue.Output?.File;
        if (settings is null || !settings.Enabled || string.IsNullOrWhiteSpace(settings.Path))
        {
            return;
        }

        try
        {
            var directory = Path.GetDirectoryName(settings.Path);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await using var stream = new FileStream(settings.Path, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            await JsonSerializer.SerializeAsync(stream, snapshot, cancellationToken: cancellationToken).ConfigureAwait(false);
            await stream.WriteAsync(new[] { (byte)'\n' }, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to write PLC snapshot to file sink {Path}", settings.Path);
        }
    }
}
