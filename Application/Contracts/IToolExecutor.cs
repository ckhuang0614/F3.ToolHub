using F3.ToolHub.Domain.Execution;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Application.Contracts;

public interface IToolExecutor
{
    Task<ExecutionResult> ExecuteAsync(string toolId, string actionName, ExecutionContext context, CancellationToken cancellationToken);
}
