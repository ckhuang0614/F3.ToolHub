using System;
using System.Linq;
using System.Threading.RateLimiting;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Application.Requests;
using F3.ToolHub.Application.UseCases;
using F3.ToolHub.Domain.Contracts;
using F3.ToolHub.Infrastructure.Execution;
using F3.ToolHub.Infrastructure.Modbus;
using F3.ToolHub.Infrastructure.Modbus.Sinks;
using F3.ToolHub.Infrastructure.Monitoring;
using F3.ToolHub.Infrastructure.Registry;
using F3.ToolHub.Infrastructure.Security;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.RateLimiting;
using Serilog;

var app = Program.BuildApplication(args);
app.Run();

public partial class Program
{
    internal const string ToolsPolicy = "ToolsAccess";
    internal const string ToolsRatePolicy = "tools";

    public static WebApplication BuildApplication(
        string[]? args = null,
        Action<WebApplicationBuilder>? configureBuilder = null,
        Action<WebApplication>? configureApp = null,
        string? environmentName = null)
    {
        WebApplicationBuilder builder;
        if (environmentName is null)
        {
            builder = WebApplication.CreateBuilder(args ?? Array.Empty<string>());
        }
        else
        {
            builder = WebApplication.CreateBuilder(new WebApplicationOptions
            {
                Args = args ?? Array.Empty<string>(),
                EnvironmentName = environmentName
            });
        }

        builder.Configuration.AddUserSecrets<Program>(optional: true, reloadOnChange: true);
        builder.Host.UseSerilog((context, services, loggerConfiguration) =>
        {
            loggerConfiguration
                .ReadFrom.Configuration(context.Configuration)
                .ReadFrom.Services(services)
                .Enrich.FromLogContext()
                .Enrich.WithProperty("Application", "F3.ToolHub");
        });

        configureBuilder?.Invoke(builder);
        ConfigureServices(builder);

        var app = builder.Build();
        ConfigurePipeline(app);
        configureApp?.Invoke(app);

        return app;
    }

    private static void ConfigureServices(WebApplicationBuilder builder)
    {
        builder.Services.AddProblemDetails();
        builder.Services.AddOpenApi();

        builder.Services.Configure<ModbusOptions>(builder.Configuration.GetSection(ModbusOptions.SectionName));
        builder.Services.Configure<ApiSecurityOptions>(builder.Configuration.GetSection(ApiSecurityOptions.SectionName));

        builder.Services.AddAuthentication(options =>
            {
                options.DefaultScheme = ApiSecurityConstants.SchemeName;
                options.DefaultAuthenticateScheme = ApiSecurityConstants.SchemeName;
                options.DefaultChallengeScheme = ApiSecurityConstants.SchemeName;
            })
            .AddScheme<AuthenticationSchemeOptions, ApiKeyAuthenticationHandler>(ApiSecurityConstants.SchemeName, static options =>
            {
                options.TimeProvider ??= TimeProvider.System;
            });

        // 修正：使用 AddAuthorizationBuilder 來註冊授權服務與建立政策
        builder.Services.AddAuthorizationBuilder()
            .AddPolicy(ToolsPolicy, policy => policy.RequireAuthenticatedUser());

        builder.Services.AddRateLimiter(options =>
        {
            options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
            options.AddPolicy(ToolsRatePolicy, httpContext =>
            {
                var partitionKey = httpContext.User?.Identity?.Name ?? httpContext.Connection.RemoteIpAddress?.ToString() ?? "anonymous";
                return RateLimitPartition.GetFixedWindowLimiter(partitionKey, _ => new FixedWindowRateLimiterOptions
                {
                    PermitLimit = 60,
                    Window = TimeSpan.FromMinutes(1),
                    QueueLimit = 0
                });
            });
        });

        builder.Services.AddSingleton<IPlcDataCache, PlcDataCache>();
        builder.Services.AddSingleton<IPlcAlertService, PlcAlertService>();
        builder.Services.AddHttpClient<WebhookPlcAlertSink>();
        builder.Services.AddTransient<IPlcAlertSink>(static sp => sp.GetRequiredService<WebhookPlcAlertSink>());
        builder.Services.AddSingleton<IPlcDataPublisher, PlcDataPublisher>();
        builder.Services.AddSingleton<IPlcDataSink, FilePlcDataSink>();
        builder.Services.AddSingleton<IPlcDataSink, MqttPlcDataSink>();
        builder.Services.AddSingleton<IModbusMappingStore, InMemoryModbusMappingStore>();
        builder.Services.AddSingleton<IModbusClient, ModbusTcpClient>();
        builder.Services.AddSingleton<IModbusClientMetrics>(sp => (ModbusTcpClient)sp.GetRequiredService<IModbusClient>());
        builder.Services.AddSingleton<IToolProvider, ModbusPlcToolProvider>();
        builder.Services.AddSingleton<IToolRegistry, InMemoryToolRegistry>();
        builder.Services.AddSingleton<IToolExecutor, ToolExecutor>();
        builder.Services.AddHostedService<ModbusPollingBackgroundService>();
        builder.Services.AddHealthChecks().AddCheck<ModbusHealthCheck>("modbus");

        builder.Services.AddScoped<ListToolsUseCase>();
        builder.Services.AddScoped<ExecuteToolActionUseCase>();
    }

    private static void ConfigurePipeline(WebApplication app)
    {
        if (app.Environment.IsDevelopment())
        {
            app.MapOpenApi();
        }

        app.MapHealthChecks("/healthz");

        app.UseHttpsRedirection();
        app.UseRateLimiter();
        app.UseAuthentication();
        app.UseAuthorization();

        var toolsApi = app.MapGroup("/api/tools")
            .RequireAuthorization(ToolsPolicy)
            .RequireRateLimiting(ToolsRatePolicy);

        toolsApi.MapGet("/", async (ListToolsUseCase useCase, CancellationToken token) =>
        {
            var tools = await useCase.HandleAsync(token);
            return Results.Ok(tools);
        });

        toolsApi.MapPost("/{toolId}/actions/{actionName}", async (
            string toolId,
            string actionName,
            ExecuteToolActionUseCase useCase,
            ExecuteActionRequest? request,
            CancellationToken token) =>
        {
            var result = await useCase.HandleAsync(toolId, actionName, request, token);
            return result.Success ? Results.Ok(result) : Results.BadRequest(result);
        });

        var plcGroup = toolsApi.MapGroup("/plc");

        plcGroup.MapGet("/mappings", (IModbusMappingStore store) =>
        {
            var mappings = store.ListActions().Select(action => action.ToDto());
            return Results.Ok(mappings);
        });

        plcGroup.MapGet("/mappings/{actionName}", (string actionName, IModbusMappingStore store) =>
        {
            var action = store.GetAction(actionName);
            return action is not null ? Results.Ok(action.ToDto()) : Results.NotFound();
        });

        plcGroup.MapPut("/mappings/{actionName}", (string actionName, ModbusActionMappingDto dto, IModbusMappingStore store) =>
        {
            if (!string.Equals(actionName, dto.Name, StringComparison.OrdinalIgnoreCase))
            {
                return Results.BadRequest("Action name in route does not match payload.");
            }

            store.Upsert(dto.ToOptions());
            return Results.Ok(dto);
        });

        plcGroup.MapDelete("/mappings/{actionName}", (string actionName, IModbusMappingStore store) =>
        {
            return store.Delete(actionName) ? Results.NoContent() : Results.NotFound();
        });

        plcGroup.MapGet("/actions/{actionName}/data", (string actionName, IPlcDataCache cache) =>
        {
            return cache.TryGet(actionName, out var snapshot)
                ? Results.Ok(snapshot.ToDto())
                : Results.NotFound();
        });

        plcGroup.MapGet("/data", (IPlcDataCache cache) =>
        {
            var snapshots = cache.ListSnapshots().Select(snapshot => snapshot.ToDto());
            return Results.Ok(snapshots);
        });

        plcGroup.MapGet("/alerts", (IPlcAlertService alertService) => Results.Ok(alertService.GetRecent()));

        plcGroup.MapGet("/health", (IModbusClientMetrics metrics, IPlcAlertService alertService) =>
        {
            var snapshot = metrics.Capture();
            var alerts = alertService.GetRecent(10);
            return Results.Ok(ModbusHealthDto.FromSnapshot(snapshot, alerts));
        });
    }
}
