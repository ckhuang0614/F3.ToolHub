using F3.ToolHub.Domain.Entities;

namespace F3.ToolHub.Domain.Contracts;

public interface IToolRegistry
{
    IReadOnlyCollection<Tool> GetTools();

    IToolProvider? FindProvider(string toolId);
}
