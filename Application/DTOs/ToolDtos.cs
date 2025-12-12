using System.Linq;
using F3.ToolHub.Domain.Entities;
using F3.ToolHub.Domain.Execution;
using F3.ToolHub.Domain.Telemetry;

namespace F3.ToolHub.Application.DTOs;

public sealed record ToolDto(string Id, string Name, string Description, IReadOnlyCollection<ToolActionDto> Actions);

public sealed record ToolActionDto(string Name, string Description, IReadOnlyDictionary<string, string> Metadata);

public sealed record ExecutionResultDto(bool Success, string? Message, object? Data);

public sealed record PlcDataPointDto(string Name, object? Value, object? RawValue, string DataType, double Scale, string? Unit, string Quality);

public sealed record PlcDataSnapshotDto(string ActionName, DateTimeOffset RetrievedAt, IReadOnlyList<ushort> Registers, IReadOnlyList<PlcDataPointDto> Points);

public static class ToolMappings
{
    public static ToolDto ToDto(this Tool tool)
        => new(
            tool.Id,
            tool.Name,
            tool.Description,
            tool.Actions.Select(action => new ToolActionDto(action.Name, action.Description, action.Metadata)).ToArray());

    public static ExecutionResultDto ToDto(this ExecutionResult result)
        => new(result.Success, result.Message, result.Data);

    public static PlcDataSnapshotDto ToDto(this PlcDataSnapshot snapshot)
        => new(
            snapshot.ActionName,
            snapshot.RetrievedAt,
            snapshot.Registers,
            snapshot.Points.Select(point => point.ToDto()).ToArray());

    public static PlcDataPointDto ToDto(this PlcDataPoint point)
        => new(point.Name, point.Value, point.RawValue, point.DataType, point.Scale, point.Unit, point.Quality);
}
