using System.Collections.Generic;

namespace F3.ToolHub.Infrastructure.Security;

public static class ApiSecurityConstants
{
    public const string SchemeName = "ApiKey";
    public const string HeaderName = "X-Api-Key";
}

public sealed class ApiSecurityOptions
{
    public const string SectionName = "ApiSecurity";

    public string HeaderName { get; set; } = ApiSecurityConstants.HeaderName;

    public List<ApiKeyOption> ApiKeys { get; set; } = new();
}

public sealed class ApiKeyOption
{
    public string Name { get; set; } = string.Empty;

    public string Key { get; set; } = string.Empty;

    public List<string> Roles { get; set; } = new();
}
