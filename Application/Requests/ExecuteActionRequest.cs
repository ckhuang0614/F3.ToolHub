namespace F3.ToolHub.Application.Requests;

public sealed class ExecuteActionRequest
{
    public string RequestedBy { get; set; } = "api";

    public Dictionary<string, string> Parameters { get; set; } = new(StringComparer.OrdinalIgnoreCase);
}
