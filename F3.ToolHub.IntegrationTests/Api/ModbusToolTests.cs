using System.Globalization;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Application.Requests;
using F3.ToolHub.Infrastructure.Modbus;
using F3.ToolHub.Infrastructure.Security;
using F3.ToolHub.IntegrationTests.Infrastructure;

namespace F3.ToolHub.IntegrationTests.Api;

public sealed class ModbusToolTests : IAsyncLifetime
{
    private ModbusSimulator? _simulator;
    private ToolHubApiFactory? _factory;
    private HttpClient? _client;
    private const string TestApiKey = "integration-test-key";

    private ToolHubApiFactory Factory => _factory ?? throw new InvalidOperationException("API factory is not initialized.");

    private HttpClient Client => _client ?? throw new InvalidOperationException("HTTP client is not initialized.");

    public async Task InitializeAsync()
    {
        _simulator = new ModbusSimulator();
        await _simulator.StartAsync();

        var overrides = new Dictionary<string, string?>
        {
            [$"{ModbusOptions.SectionName}:Host"] = "127.0.0.1",
            [$"{ModbusOptions.SectionName}:Port"] = _simulator.Port.ToString(CultureInfo.InvariantCulture),
            [$"{ApiSecurityOptions.SectionName}:HeaderName"] = ApiSecurityConstants.HeaderName,
            [$"{ApiSecurityOptions.SectionName}:ApiKeys:0:Name"] = "integration-test",
            [$"{ApiSecurityOptions.SectionName}:ApiKeys:0:Key"] = TestApiKey,
            [$"{ApiSecurityOptions.SectionName}:ApiKeys:0:Roles:0"] = "tools.read",
            [$"{ApiSecurityOptions.SectionName}:ApiKeys:0:Roles:1"] = "tools.execute"
        };

        _factory = new ToolHubApiFactory(overrides);
        await _factory.InitializeAsync();
        _client = _factory.CreateClient();
        _client.DefaultRequestHeaders.Add(ApiSecurityConstants.HeaderName, TestApiKey);
    }

    public async Task DisposeAsync()
    {
        _client?.Dispose();
        if (_factory is not null)
        {
            await _factory.DisposeAsync();
        }

        if (_simulator is not null)
        {
            await _simulator.DisposeAsync();
        }
    }

    [Fact]
    public async Task MissingApiKey_ShouldReturn401()
    {
        using var anonymousClient = Factory.CreateClient();
        var response = await anonymousClient.GetAsync("/api/tools");
        Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
    }

    [Fact]
    public async Task ListTools_ShouldExposeModbusTool()
    {
        var response = await Client.GetAsync("/api/tools");
        response.EnsureSuccessStatusCode();

        var tools = await response.Content.ReadFromJsonAsync<List<ToolDto>>();
        Assert.NotNull(tools);
        var toolList = tools!;
        Assert.Contains(toolList, tool => tool.Id == ModbusPlcToolProvider.ToolId);
    }

    [Fact]
    public async Task ExecuteModbusAction_ShouldReturnSnapshot()
    {
        var payload = new ExecuteActionRequest
        {
            RequestedBy = "integration-tests"
        };

        var response = await Client.PostAsJsonAsync($"/api/tools/{ModbusPlcToolProvider.ToolId}/actions/demo-holding-registers", payload);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<ExecutionResultDto>();
        Assert.NotNull(result);
        Assert.True(result!.Success);

        var snapshotsResponse = await Client.GetAsync("/api/tools/plc/data");
        snapshotsResponse.EnsureSuccessStatusCode();
        var snapshots = await snapshotsResponse.Content.ReadFromJsonAsync<List<PlcDataSnapshotDto>>();
        Assert.NotNull(snapshots);
        var snapshotList = snapshots!;
        Assert.NotEmpty(snapshotList);
    }
}
