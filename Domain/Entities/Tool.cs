namespace F3.ToolHub.Domain.Entities;

public sealed record Tool(
    string Id,
    string Name,
    string Description,
    IReadOnlyCollection<ToolAction> Actions);
