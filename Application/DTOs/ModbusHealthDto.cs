using System;
using System.Collections.Generic;
using F3.ToolHub.Domain.Monitoring;
using F3.ToolHub.Infrastructure.Modbus;

namespace F3.ToolHub.Application.DTOs;

public sealed record ModbusHealthDto(
    DateTimeOffset? LastSuccess,
    DateTimeOffset? LastFailure,
    int ConsecutiveFailures,
    bool CircuitBreakerOpen,
    int ActiveConnections,
    int AvailableConnections,
    long TotalSuccess,
    long TotalFailures,
    IReadOnlyCollection<PlcAlert> RecentAlerts)
{
    public static ModbusHealthDto FromSnapshot(ModbusClientHealthSnapshot snapshot, IReadOnlyCollection<PlcAlert> alerts)
        => new(
            snapshot.LastSuccess,
            snapshot.LastFailure,
            snapshot.ConsecutiveFailures,
            snapshot.CircuitBreakerOpen,
            snapshot.ActiveConnections,
            snapshot.AvailableConnections,
            snapshot.TotalSuccess,
            snapshot.TotalFailures,
            alerts);
}
