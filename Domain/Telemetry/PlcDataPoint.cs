namespace F3.ToolHub.Domain.Telemetry;

public sealed record PlcDataPoint(
    string Name,
    object? Value,
    object? RawValue,
    string DataType,
    double Scale,
    string? Unit,
    string Quality);
