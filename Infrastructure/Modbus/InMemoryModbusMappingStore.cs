using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class InMemoryModbusMappingStore : IModbusMappingStore
{
    private readonly ConcurrentDictionary<string, ModbusActionOptions> _actions = new(StringComparer.OrdinalIgnoreCase);
    private readonly ILogger<InMemoryModbusMappingStore> _logger;

    public InMemoryModbusMappingStore(IOptionsMonitor<ModbusOptions> options, ILogger<InMemoryModbusMappingStore> logger)
    {
        _logger = logger;
        LoadFromOptions(options.CurrentValue);
        options.OnChange(LoadFromOptions);
    }

    public IReadOnlyCollection<ModbusActionOptions> ListActions()
    {
        return _actions.Values.Select(CloneAction).OrderBy(action => action.Name, StringComparer.OrdinalIgnoreCase).ToArray();
    }

    public ModbusActionOptions? GetAction(string actionName)
    {
        if (_actions.TryGetValue(actionName, out var action))
        {
            return CloneAction(action);
        }

        return null;
    }

    public void Upsert(ModbusActionOptions action)
    {
        ValidateAction(action);
        _actions.AddOrUpdate(action.Name, _ => CloneAction(action), (_, __) => CloneAction(action));
    }

    public bool Delete(string actionName)
    {
        return _actions.TryRemove(actionName, out _);
    }

    private void LoadFromOptions(ModbusOptions options)
    {
        if (options.Actions is null)
        {
            return;
        }

        foreach (var action in options.Actions)
        {
            try
            {
                ValidateAction(action);
                _actions[action.Name] = CloneAction(action);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Skipping invalid Modbus action {Action}", action.Name);
            }
        }
    }

    private static void ValidateAction(ModbusActionOptions action)
    {
        if (string.IsNullOrWhiteSpace(action.Name))
        {
            throw new ArgumentException("Action name is required.");
        }

        if (action.NumberOfPoints == 0)
        {
            throw new ArgumentException("NumberOfPoints must be greater than zero.");
        }

        if (action.Registers is null)
        {
            action.Registers = new List<ModbusRegisterMapOptions>();
        }
    }

    private static ModbusActionOptions CloneAction(ModbusActionOptions action)
    {
        return new ModbusActionOptions
        {
            Name = action.Name,
            Description = action.Description,
            StartAddress = action.StartAddress,
            NumberOfPoints = action.NumberOfPoints,
            Registers = action.Registers?.Select(CloneRegister).ToList() ?? new List<ModbusRegisterMapOptions>()
        };
    }

    private static ModbusRegisterMapOptions CloneRegister(ModbusRegisterMapOptions register)
    {
        return new ModbusRegisterMapOptions
        {
            Name = register.Name,
            Offset = register.Offset,
            DataType = register.DataType,
            Scale = register.Scale,
            Unit = register.Unit,
            BitIndex = register.BitIndex,
            MinValue = register.MinValue,
            MaxValue = register.MaxValue
        };
    }
}
