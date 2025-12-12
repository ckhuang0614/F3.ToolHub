using System.Collections.Generic;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class ModbusOptions
{
    public const string SectionName = "Modbus";

    public string Host { get; set; } = "127.0.0.1";

    public int Port { get; set; } = 502;

    public byte SlaveId { get; set; } = 1;

    public int ConnectTimeoutSeconds { get; set; } = 3;

    public int ResponseTimeoutSeconds { get; set; } = 3;

    public List<ModbusActionOptions> Actions { get; set; } = new();

    public ModbusPollingOptions Polling { get; set; } = new();

    public ModbusResilienceOptions Resilience { get; set; } = new();

    public ModbusOutputOptions Output { get; set; } = new();

    public ModbusMonitoringOptions Monitoring { get; set; } = new();
}

public sealed class ModbusActionOptions
{
    public string Name { get; set; } = string.Empty;

    public string Description { get; set; } = string.Empty;

    public ushort StartAddress { get; set; }

    public ushort NumberOfPoints { get; set; } = 1;

    public List<ModbusRegisterMapOptions> Registers { get; set; } = new();
}

public sealed class ModbusRegisterMapOptions
{
    public string Name { get; set; } = string.Empty;

    public ushort Offset { get; set; }

    public ModbusRegisterDataType DataType { get; set; } = ModbusRegisterDataType.UInt16;

    public double Scale { get; set; } = 1.0d;

    public string? Unit { get; set; }

    public int? BitIndex { get; set; }

    public double? MinValue { get; set; }

    public double? MaxValue { get; set; }
}

public enum ModbusRegisterDataType
{
    UInt16,
    Int16,
    UInt32,
    Int32,
    Float32,
    Boolean
}

public sealed class ModbusPollingOptions
{
    public bool Enabled { get; set; }

    public string ActionName { get; set; } = string.Empty;

    public int IntervalSeconds { get; set; } = 5;
}

public sealed class ModbusResilienceOptions
{
    public int MaxRetryAttempts { get; set; } = 3;

    public int BaseDelayMilliseconds { get; set; } = 500;

    public int MaxDelayMilliseconds { get; set; } = 5000;

    public int CircuitBreakerFailureThreshold { get; set; } = 5;

    public int CircuitBreakerDurationSeconds { get; set; } = 30;

    public int ConnectionPoolSize { get; set; } = 4;
}

public sealed class ModbusOutputOptions
{
    public FileOutputOptions File { get; set; } = new();

    public MqttOutputOptions Mqtt { get; set; } = new();
}

public sealed class FileOutputOptions
{
    public bool Enabled { get; set; } = true;

    public string Path { get; set; } = "plc-history.jsonl";
}

public sealed class MqttOutputOptions
{
    public bool Enabled { get; set; }

    public string Host { get; set; } = "localhost";

    public int Port { get; set; } = 1883;

    public string Topic { get; set; } = "plc/data";

    public string? ClientId { get; set; }
}

public sealed class ModbusMonitoringOptions
{
    public List<AlertWebhookOptions> AlertWebhooks { get; set; } = new();
}

public sealed class AlertWebhookOptions
{
    public bool Enabled { get; set; }

    public string? Name { get; set; }

    public string Url { get; set; } = string.Empty;

    public string? Secret { get; set; }

    public string? SecretHeaderName { get; set; }

    public Dictionary<string, string>? Headers { get; set; }
}
