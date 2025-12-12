namespace F3.ToolHub.Domain.Monitoring;

public sealed record PlcAlert(
    string Source,
    string Message,
    string Severity,
    DateTimeOffset RaisedAt,
    string? ActionName = null,
    string? PointName = null,
    string? Quality = null);
