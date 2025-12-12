using System.Collections.Generic;
using System.Linq;
using System.Security.Claims;
using System.Text.Encodings.Web;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Security;

public sealed class ApiKeyAuthenticationHandler : AuthenticationHandler<AuthenticationSchemeOptions>
{
    private readonly IOptionsMonitor<ApiSecurityOptions> _options;

    public ApiKeyAuthenticationHandler(
        IOptionsMonitor<ApiSecurityOptions> options,
        IOptionsMonitor<AuthenticationSchemeOptions> schemeOptions,
        ILoggerFactory logger,
        UrlEncoder encoder)
        : base(schemeOptions, logger, encoder)
    {
        _options = options;
    }

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        var settings = _options.CurrentValue;
        var headerName = string.IsNullOrWhiteSpace(settings.HeaderName)
            ? ApiSecurityConstants.HeaderName
            : settings.HeaderName;

        if (!Request.Headers.TryGetValue(headerName, out var providedValue))
        {
            return Task.FromResult(AuthenticateResult.Fail($"Missing {headerName} header."));
        }

        var apiKey = providedValue.ToString();
        if (string.IsNullOrWhiteSpace(apiKey))
        {
            return Task.FromResult(AuthenticateResult.Fail("API key header is empty."));
        }

        var match = settings.ApiKeys.FirstOrDefault(key => string.Equals(key.Key, apiKey, StringComparison.Ordinal));
        if (match is null)
        {
            return Task.FromResult(AuthenticateResult.Fail("API key is invalid."));
        }

        var claims = new List<Claim>
        {
            new(ClaimTypes.NameIdentifier, match.Name),
            new(ClaimTypes.Name, match.Name)
        };

        foreach (var role in match.Roles)
        {
            if (!string.IsNullOrWhiteSpace(role))
            {
                claims.Add(new Claim(ClaimTypes.Role, role));
            }
        }

        var identity = new ClaimsIdentity(claims, Scheme.Name);
        var principal = new ClaimsPrincipal(identity);
        var ticket = new AuthenticationTicket(principal, Scheme.Name);

        return Task.FromResult(AuthenticateResult.Success(ticket));
    }

    protected override Task HandleChallengeAsync(AuthenticationProperties properties)
    {
        Response.StatusCode = StatusCodes.Status401Unauthorized;
        var headerName = _options.CurrentValue.HeaderName ?? ApiSecurityConstants.HeaderName;
        Response.Headers["WWW-Authenticate"] = $"ApiKey header={headerName}";
        return Task.CompletedTask;
    }

    protected override Task HandleForbiddenAsync(AuthenticationProperties properties)
    {
        Response.StatusCode = StatusCodes.Status403Forbidden;
        return Task.CompletedTask;
    }
}
