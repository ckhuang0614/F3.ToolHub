using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Contracts;
using F3.ToolHub.Domain.Execution;
using Microsoft.Extensions.Logging;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Infrastructure.Execution;

public sealed class ToolExecutor : IToolExecutor
{
    private readonly IToolRegistry _registry;
    private readonly ILogger<ToolExecutor> _logger;

    public ToolExecutor(IToolRegistry registry, ILogger<ToolExecutor> logger)
    {
        _registry = registry;
        _logger = logger;
    }

    public async Task<ExecutionResult> ExecuteAsync(string toolId, string actionName, ExecutionContext context, CancellationToken cancellationToken)
    {
        var provider = _registry.FindProvider(toolId);
        if (provider is null)
        {
            _logger.LogWarning("Tool {ToolId} was not found", toolId);
            return ExecutionResult.Fail($"Tool '{toolId}' is not registered.");
        }

        try
        {
            return await provider.ExecuteAsync(actionName, context, cancellationToken);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing tool {ToolId} action {Action}", toolId, actionName);
            return ExecutionResult.Fail($"Execution failed: {ex.Message}");
        }
    }
}
