using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Monitoring;
using Microsoft.Extensions.Logging;

namespace F3.ToolHub.Infrastructure.Monitoring;

public sealed class PlcAlertService : IPlcAlertService
{
    private const int MaxAlerts = 200;
    private readonly ConcurrentQueue<PlcAlert> _alerts = new();
    private readonly IEnumerable<IPlcAlertSink> _sinks;
    private readonly ILogger<PlcAlertService> _logger;

    public PlcAlertService(IEnumerable<IPlcAlertSink> sinks, ILogger<PlcAlertService> logger)
    {
        _sinks = sinks;
        _logger = logger;
    }

    public void Raise(PlcAlert alert)
    {
        _alerts.Enqueue(alert);
        while (_alerts.Count > MaxAlerts && _alerts.TryDequeue(out _))
        {
        }

        _ = DispatchAsync(alert);
    }

    public IReadOnlyCollection<PlcAlert> GetRecent(int? take = null)
    {
        var result = _alerts.OrderByDescending(alert => alert.RaisedAt).ToList();
        if (take is { } limit && limit > 0)
        {
            return result.Take(limit).ToArray();
        }

        return result;
    }

    private Task DispatchAsync(PlcAlert alert)
    {
        if (_sinks is null)
        {
            return Task.CompletedTask;
        }

        var publishTasks = _sinks.Select(sink => PublishAsync(sink, alert));
        return Task.WhenAll(publishTasks);
    }

    private async Task PublishAsync(IPlcAlertSink sink, PlcAlert alert)
    {
        try
        {
            await sink.PublishAsync(alert, CancellationToken.None).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Alert sink {Sink} failed, source={Source}", sink.GetType().Name, alert.Source);
        }
    }
}
