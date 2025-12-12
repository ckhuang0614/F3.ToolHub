using System;

namespace F3.ToolHub.Infrastructure.Modbus;

public interface IModbusClientMetrics
{
    ModbusClientHealthSnapshot Capture();
}

public sealed record ModbusClientHealthSnapshot(
    DateTimeOffset? LastSuccess,
    DateTimeOffset? LastFailure,
    int ConsecutiveFailures,
    bool CircuitBreakerOpen,
    int ActiveConnections,
    int AvailableConnections,
    long TotalSuccess,
    long TotalFailures);
