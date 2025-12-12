namespace F3.ToolHub.Domain.Telemetry;

public sealed record PlcDataSnapshot(
    string ActionName,
    DateTimeOffset RetrievedAt,
    IReadOnlyList<ushort> Registers,
    IReadOnlyList<PlcDataPoint> Points);
