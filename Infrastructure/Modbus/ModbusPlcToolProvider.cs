using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Contracts;
using F3.ToolHub.Domain.Entities;
using F3.ToolHub.Domain.Execution;
using F3.ToolHub.Domain.Monitoring;
using F3.ToolHub.Domain.Telemetry;
using Microsoft.Extensions.Logging;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class ModbusPlcToolProvider : IToolProvider
{
    public const string ToolId = "plc.modbus";

    private readonly IModbusClient _client;
    private readonly IPlcDataCache _cache;
    private readonly IModbusMappingStore _mappingStore;
    private readonly IPlcDataPublisher _publisher;
    private readonly IPlcAlertService _alertService;
    private readonly ILogger<ModbusPlcToolProvider> _logger;

    public ModbusPlcToolProvider(
        IModbusClient client,
        IPlcDataCache cache,
        IModbusMappingStore mappingStore,
        IPlcDataPublisher publisher,
        IPlcAlertService alertService,
        ILogger<ModbusPlcToolProvider> logger)
    {
        _client = client;
        _cache = cache;
        _mappingStore = mappingStore;
        _publisher = publisher;
        _alertService = alertService;
        _logger = logger;
    }

    public Tool Tool => BuildTool(_mappingStore.ListActions());

    public async Task<ExecutionResult> ExecuteAsync(string actionName, ExecutionContext context, CancellationToken cancellationToken)
    {
        var action = _mappingStore.GetAction(actionName);
        if (action is null)
        {
            return ExecutionResult.Fail($"Action '{actionName}' is not configured for the Modbus tool.");
        }

        try
        {
            var registers = await _client.ReadHoldingRegistersAsync(action.StartAddress, action.NumberOfPoints, cancellationToken).ConfigureAwait(false);
            var points = MapDataPoints(action, registers);
            var snapshot = new PlcDataSnapshot(action.Name, DateTimeOffset.UtcNow, registers, points);
            _cache.Store(snapshot);
            await _publisher.PublishAsync(snapshot, cancellationToken).ConfigureAwait(false);
            RaiseQualityAlerts(snapshot);

            var payload = new
            {
                action.Name,
                snapshot.RetrievedAt,
                Registers = registers,
                Points = points,
                RequestedBy = context.RequestedBy
            };

            return ExecutionResult.Ok(payload, $"Read {registers.Length} registers starting at {action.StartAddress}.");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to execute Modbus action {Action}", actionName);
            _alertService.Raise(new PlcAlert("modbus.plc", ex.Message, "High", DateTimeOffset.UtcNow, actionName));
            return ExecutionResult.Fail($"Modbus read failed: {ex.Message}");
        }
    }

    private void RaiseQualityAlerts(PlcDataSnapshot snapshot)
    {
        foreach (var point in snapshot.Points)
        {
            if (!string.Equals(point.Quality, "Good", StringComparison.OrdinalIgnoreCase))
            {
                var severity = string.Equals(point.Quality, "Error", StringComparison.OrdinalIgnoreCase) ? "High" : "Medium";
                _alertService.Raise(new PlcAlert("modbus.plc", $"Point {point.Name} quality={point.Quality}", severity, snapshot.RetrievedAt, snapshot.ActionName, point.Name, point.Quality));
            }
        }
    }

    private IReadOnlyList<PlcDataPoint> MapDataPoints(ModbusActionOptions action, IReadOnlyList<ushort> registers)
    {
        if (action.Registers is null || action.Registers.Count == 0)
        {
            return Array.Empty<PlcDataPoint>();
        }

        var points = new List<PlcDataPoint>(action.Registers.Count);
        foreach (var map in action.Registers)
        {
            try
            {
                var rawValue = ExtractRawValue(map, registers);
                var scaledValue = map.DataType == ModbusRegisterDataType.Boolean
                    ? rawValue
                    : ApplyScale(rawValue, map.Scale);
                var quality = EvaluateQuality(map, scaledValue);

                points.Add(new PlcDataPoint(map.Name, scaledValue, rawValue, map.DataType.ToString(), map.Scale, map.Unit, quality));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to map register {Register} for action {Action}.", map.Name, action.Name);
                points.Add(new PlcDataPoint(map.Name, null, null, map.DataType.ToString(), map.Scale, map.Unit, "Error"));
            }
        }

        return points;
    }

    private static string EvaluateQuality(ModbusRegisterMapOptions map, object? value)
    {
        if (value is null)
        {
            return "Error";
        }

        if (map.MinValue is null && map.MaxValue is null)
        {
            return "Good";
        }

        if (value is bool)
        {
            return "Good";
        }

        if (!double.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Any, CultureInfo.InvariantCulture, out var numeric))
        {
            return "Good";
        }

        if (map.MinValue is not null && numeric < map.MinValue)
        {
            return "Low";
        }

        if (map.MaxValue is not null && numeric > map.MaxValue)
        {
            return "High";
        }

        return "Good";
    }

    private static object ExtractRawValue(ModbusRegisterMapOptions map, IReadOnlyList<ushort> registers)
    {
        var offset = map.Offset;
        return map.DataType switch
        {
            ModbusRegisterDataType.UInt16 => GetRegister(registers, offset),
            ModbusRegisterDataType.Int16 => unchecked((short)GetRegister(registers, offset)),
            ModbusRegisterDataType.UInt32 => ReadUInt32(registers, offset),
            ModbusRegisterDataType.Int32 => unchecked((int)ReadUInt32(registers, offset)),
            ModbusRegisterDataType.Float32 => BitConverter.Int32BitsToSingle(unchecked((int)ReadUInt32(registers, offset))),
            ModbusRegisterDataType.Boolean => ReadBoolean(registers, offset, map.BitIndex),
            _ => GetRegister(registers, offset)
        };
    }

    private static double ApplyScale(object rawValue, double scale)
    {
        if (Math.Abs(scale - 1d) < double.Epsilon)
        {
            return Convert.ToDouble(rawValue);
        }

        var numeric = Convert.ToDouble(rawValue);
        return numeric * scale;
    }

    private static ushort GetRegister(IReadOnlyList<ushort> registers, int offset)
    {
        if (offset < 0 || offset >= registers.Count)
        {
            throw new InvalidOperationException($"Register offset {offset} is outside of the captured range.");
        }

        return registers[offset];
    }

    private static uint ReadUInt32(IReadOnlyList<ushort> registers, int offset)
    {
        if (offset < 0 || offset + 1 >= registers.Count)
        {
            throw new InvalidOperationException("Not enough registers to read a 32-bit value.");
        }

        return ((uint)registers[offset] << 16) | registers[offset + 1];
    }

    private static bool ReadBoolean(IReadOnlyList<ushort> registers, int offset, int? bitIndex)
    {
        var registerValue = GetRegister(registers, offset);
        if (bitIndex is null)
        {
            return registerValue != 0;
        }

        if (bitIndex < 0 || bitIndex > 15)
        {
            throw new InvalidOperationException($"Bit index {bitIndex} is invalid for a Modbus register.");
        }

        return ((registerValue >> bitIndex.Value) & 1) == 1;
    }

    private static Tool BuildTool(IReadOnlyCollection<ModbusActionOptions> actions)
    {
        var mappedActions = (actions ?? Array.Empty<ModbusActionOptions>()).Select(action =>
            new ToolAction(
                action.Name,
                action.Description,
                new Dictionary<string, string>
                {
                    ["startAddress"] = action.StartAddress.ToString(),
                    ["numberOfPoints"] = action.NumberOfPoints.ToString(),
                    ["mappedRegisters"] = (action.Registers?.Count ?? 0).ToString()
                })).ToArray();

        return new Tool(
            ToolId,
            "Modbus PLC",
            "Reads data from PLC registers using Modbus TCP.",
            mappedActions);
    }
}
