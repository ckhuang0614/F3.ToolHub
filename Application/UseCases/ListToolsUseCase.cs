using System.Linq;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Domain.Contracts;

namespace F3.ToolHub.Application.UseCases;

public sealed class ListToolsUseCase
{
    private readonly IToolRegistry _registry;

    public ListToolsUseCase(IToolRegistry registry)
    {
        _registry = registry;
    }

    public Task<IReadOnlyCollection<ToolDto>> HandleAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var tools = _registry.GetTools()
            .Select(tool => tool.ToDto())
            .ToArray();

        return Task.FromResult<IReadOnlyCollection<ToolDto>>(tools);
    }
}
