namespace F3.ToolHub.Domain.Execution;

public sealed record ExecutionResult(bool Success, string? Message, object? Data)
{
    public static ExecutionResult Ok(object? data = null, string? message = null)
        => new(true, message, data);

    public static ExecutionResult Fail(string message)
        => new(false, message, null);
}
