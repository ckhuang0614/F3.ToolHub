namespace F3.ToolHub.Infrastructure.Modbus;

public interface IModbusClient
{
    Task<ushort[]> ReadHoldingRegistersAsync(ushort startAddress, ushort numberOfPoints, CancellationToken cancellationToken);
}
