using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;

namespace F3.ToolHub.IntegrationTests.Infrastructure;

public sealed class ToolHubApiFactory : IAsyncDisposable
{
    private readonly IReadOnlyDictionary<string, string?> _overrides;
    private readonly string _environment;
    private WebApplication? _app;

    public ToolHubApiFactory(IReadOnlyDictionary<string, string?> overrides, string environment = "Simulator")
    {
        _overrides = overrides;
        _environment = environment;
    }

    public async Task InitializeAsync()
    {
        _app = Program.BuildApplication(
            configureBuilder: builder =>
            {
                builder.WebHost.UseTestServer();
                if (_overrides.Count > 0)
                {
                    builder.Configuration.AddInMemoryCollection(_overrides);
                }
            },
            environmentName: _environment);

        await _app.StartAsync();
    }

    public HttpClient CreateClient()
    {
        if (_app is null)
        {
            throw new InvalidOperationException("Factory not initialized.");
        }

        return _app.GetTestClient();
    }

    public async ValueTask DisposeAsync()
    {
        if (_app is not null)
        {
            await _app.StopAsync();
            await _app.DisposeAsync();
        }
    }
}
