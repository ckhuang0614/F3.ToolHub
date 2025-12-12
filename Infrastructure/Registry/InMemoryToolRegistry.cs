using System.Collections.Concurrent;
using System.Linq;
using F3.ToolHub.Domain.Contracts;
using F3.ToolHub.Domain.Entities;

namespace F3.ToolHub.Infrastructure.Registry;

public sealed class InMemoryToolRegistry : IToolRegistry
{
    private readonly ConcurrentDictionary<string, IToolProvider> _providers;

    public InMemoryToolRegistry(IEnumerable<IToolProvider> providers)
    {
        _providers = new ConcurrentDictionary<string, IToolProvider>(
            providers.ToDictionary(provider => provider.Tool.Id, provider => provider, StringComparer.OrdinalIgnoreCase));
    }

    public IReadOnlyCollection<Tool> GetTools()
        => _providers.Values.Select(provider => provider.Tool).ToArray();

    public IToolProvider? FindProvider(string toolId)
        => _providers.TryGetValue(toolId, out var provider) ? provider : null;
}
