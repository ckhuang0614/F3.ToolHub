using System;
using System.Collections.Generic;
using System.Linq;
using F3.ToolHub.Infrastructure.Modbus;

namespace F3.ToolHub.Application.DTOs;

public sealed record ModbusRegisterMappingDto(
    string Name,
    ushort Offset,
    ModbusRegisterDataType DataType,
    double Scale,
    string? Unit,
    int? BitIndex,
    double? MinValue,
    double? MaxValue);

public sealed record ModbusActionMappingDto(
    string Name,
    string Description,
    ushort StartAddress,
    ushort NumberOfPoints,
    IReadOnlyCollection<ModbusRegisterMappingDto> Registers);

public static class ModbusMappingDtoExtensions
{
    public static ModbusActionMappingDto ToDto(this ModbusActionOptions action)
        => new(
            action.Name,
            action.Description,
            action.StartAddress,
            action.NumberOfPoints,
            action.Registers?.Select(r => r.ToDto()).ToArray() ?? Array.Empty<ModbusRegisterMappingDto>());

    public static ModbusRegisterMappingDto ToDto(this ModbusRegisterMapOptions register)
        => new(
            register.Name,
            register.Offset,
            register.DataType,
            register.Scale,
            register.Unit,
            register.BitIndex,
            register.MinValue,
            register.MaxValue);

    public static ModbusActionOptions ToOptions(this ModbusActionMappingDto dto)
        => new()
        {
            Name = dto.Name,
            Description = dto.Description,
            StartAddress = dto.StartAddress,
            NumberOfPoints = dto.NumberOfPoints,
            Registers = (dto.Registers ?? Array.Empty<ModbusRegisterMappingDto>()).Select(r => r.ToOptions()).ToList()
        };

    public static ModbusRegisterMapOptions ToOptions(this ModbusRegisterMappingDto dto)
        => new()
        {
            Name = dto.Name,
            Offset = dto.Offset,
            DataType = dto.DataType,
            Scale = dto.Scale,
            Unit = dto.Unit,
            BitIndex = dto.BitIndex,
            MinValue = dto.MinValue,
            MaxValue = dto.MaxValue
        };
}
