using F3.ToolHub.Domain.Entities;
using F3.ToolHub.Domain.Execution;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Domain.Contracts;

public interface IToolProvider
{
    Tool Tool { get; }

    Task<ExecutionResult> ExecuteAsync(string actionName, ExecutionContext context, CancellationToken cancellationToken);
}
