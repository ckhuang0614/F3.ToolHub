using System.Collections.Generic;

namespace F3.ToolHub.Infrastructure.Modbus;

public interface IModbusMappingStore
{
    IReadOnlyCollection<ModbusActionOptions> ListActions();

    ModbusActionOptions? GetAction(string actionName);

    void Upsert(ModbusActionOptions action);

    bool Delete(string actionName);
}
