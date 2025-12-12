using System.Collections.ObjectModel;
using System.Collections.Generic;

namespace F3.ToolHub.Domain.Execution;

public sealed record ExecutionContext(
    string RequestedBy,
    DateTimeOffset RequestedAt,
    IReadOnlyDictionary<string, string> Parameters)
{
    private static readonly IReadOnlyDictionary<string, string> EmptyParameters =
        new ReadOnlyDictionary<string, string>(new Dictionary<string, string>());

    public static ExecutionContext System(string? source = null)
        => new(source ?? "system", DateTimeOffset.UtcNow, EmptyParameters);

    public static ExecutionContext FromRequest(string? requestedBy, IReadOnlyDictionary<string, string>? parameters)
        => new(
            string.IsNullOrWhiteSpace(requestedBy) ? "anonymous" : requestedBy,
            DateTimeOffset.UtcNow,
            parameters ?? EmptyParameters);
}
