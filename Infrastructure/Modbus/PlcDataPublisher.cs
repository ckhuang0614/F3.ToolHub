using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Telemetry;
using Microsoft.Extensions.Logging;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class PlcDataPublisher : IPlcDataPublisher
{
    private readonly IEnumerable<IPlcDataSink> _sinks;
    private readonly ILogger<PlcDataPublisher> _logger;

    public PlcDataPublisher(IEnumerable<IPlcDataSink> sinks, ILogger<PlcDataPublisher> logger)
    {
        _sinks = sinks;
        _logger = logger;
    }

    public async Task PublishAsync(PlcDataSnapshot snapshot, CancellationToken cancellationToken)
    {
        foreach (var sink in _sinks)
        {
            try
            {
                await sink.PublishAsync(snapshot, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                _logger.LogError(ex, "PlcDataSink {Sink} failed to export snapshot {Action}", sink.GetType().Name, snapshot.ActionName);
            }
        }
    }
}
