using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Execution;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Infrastructure.Modbus;

public sealed class ModbusPollingBackgroundService : BackgroundService
{
    private readonly IServiceScopeFactory _scopeFactory;
    private readonly IOptionsMonitor<ModbusOptions> _options;
    private readonly ILogger<ModbusPollingBackgroundService> _logger;

    public ModbusPollingBackgroundService(
        IServiceScopeFactory scopeFactory,
        IOptionsMonitor<ModbusOptions> options,
        ILogger<ModbusPollingBackgroundService> logger)
    {
        _scopeFactory = scopeFactory;
        _options = options;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Modbus polling service started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            var polling = _options.CurrentValue.Polling;
            if (polling is null || !polling.Enabled || string.IsNullOrWhiteSpace(polling.ActionName))
            {
                await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken);
                continue;
            }

            await RunActionAsync(polling.ActionName, stoppingToken);

            var interval = TimeSpan.FromSeconds(Math.Max(1, polling.IntervalSeconds));
            await Task.Delay(interval, stoppingToken);
        }
    }

    private async Task RunActionAsync(string actionName, CancellationToken cancellationToken)
    {
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var executor = scope.ServiceProvider.GetRequiredService<IToolExecutor>();
            var context = ExecutionContext.System("modbus-poller");
            await executor.ExecuteAsync(ModbusPlcToolProvider.ToolId, actionName, context, cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            // shutdown requested
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to run scheduled Modbus action {Action}", actionName);
        }
    }
}
