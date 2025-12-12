using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Telemetry;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class PlcDataCache : IPlcDataCache
{
    private readonly ConcurrentDictionary<string, PlcDataSnapshot> _cache = new(StringComparer.OrdinalIgnoreCase);

    public void Store(PlcDataSnapshot snapshot)
    {
        _cache[snapshot.ActionName] = snapshot;
    }

    public bool TryGet(string actionName, [MaybeNullWhen(false)] out PlcDataSnapshot snapshot)
    {
        return _cache.TryGetValue(actionName, out snapshot);
    }

    public IReadOnlyCollection<PlcDataSnapshot> ListSnapshots()
    {
        return _cache.Values.ToArray();
    }
}
