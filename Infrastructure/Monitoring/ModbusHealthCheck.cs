using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Infrastructure.Modbus;
using Microsoft.Extensions.Diagnostics.HealthChecks;

namespace F3.ToolHub.Infrastructure.Monitoring;

public sealed class ModbusHealthCheck : IHealthCheck
{
    private readonly IModbusClientMetrics _metrics;
    private readonly IPlcAlertService _alertService;

    public ModbusHealthCheck(IModbusClientMetrics metrics, IPlcAlertService alertService)
    {
        _metrics = metrics;
        _alertService = alertService;
    }

    public Task<HealthCheckResult> CheckHealthAsync(HealthCheckContext context, CancellationToken cancellationToken = default)
    {
        var snapshot = _metrics.Capture();
        var alerts = _alertService.GetRecent(5);
        var unhealthy = snapshot.CircuitBreakerOpen || snapshot.ConsecutiveFailures > 0;
        if (!unhealthy)
        {
            unhealthy = alerts.Any(alert => string.Equals(alert.Severity, "High", StringComparison.OrdinalIgnoreCase));
        }

        var data = new Dictionary<string, object>
        {
            ["lastSuccess"] = snapshot.LastSuccess?.ToString("O") ?? "n/a",
            ["lastFailure"] = snapshot.LastFailure?.ToString("O") ?? "n/a",
            ["consecutiveFailures"] = snapshot.ConsecutiveFailures,
            ["activeConnections"] = snapshot.ActiveConnections,
            ["availableConnections"] = snapshot.AvailableConnections,
            ["alerts"] = alerts
        };

        if (unhealthy)
        {
            return Task.FromResult(HealthCheckResult.Unhealthy("Modbus client unhealthy", data: data));
        }

        return Task.FromResult(HealthCheckResult.Healthy("Modbus client healthy", data));
    }
}
