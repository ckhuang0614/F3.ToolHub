namespace F3.ToolHub.Domain.Entities;

public sealed record ToolAction(
    string Name,
    string Description,
    IReadOnlyDictionary<string, string> Metadata);
