using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Application.Requests;
using F3.ToolHub.Domain.Execution;
using ExecutionContext = F3.ToolHub.Domain.Execution.ExecutionContext;

namespace F3.ToolHub.Application.UseCases;

public sealed class ExecuteToolActionUseCase
{
    private readonly IToolExecutor _executor;

    public ExecuteToolActionUseCase(IToolExecutor executor)
    {
        _executor = executor;
    }

    public async Task<ExecutionResultDto> HandleAsync(
        string toolId,
        string actionName,
        ExecuteActionRequest? request,
        CancellationToken cancellationToken)
    {
        request ??= new ExecuteActionRequest();
        var context = ExecutionContext.FromRequest(request.RequestedBy, request.Parameters);
        var result = await _executor.ExecuteAsync(toolId, actionName, context, cancellationToken);
        return result.ToDto();
    }
}
