using System.Diagnostics.CodeAnalysis;
using F3.ToolHub.Domain.Telemetry;

namespace F3.ToolHub.Application.Contracts;

public interface IPlcDataCache
{
    void Store(PlcDataSnapshot snapshot);

    bool TryGet(string actionName, [MaybeNullWhen(false)] out PlcDataSnapshot snapshot);

    IReadOnlyCollection<PlcDataSnapshot> ListSnapshots();
}
