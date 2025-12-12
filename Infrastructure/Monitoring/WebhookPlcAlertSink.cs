using System;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Domain.Monitoring;
using F3.ToolHub.Infrastructure.Modbus;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Monitoring;

public sealed class WebhookPlcAlertSink : IPlcAlertSink
{
    private static readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web);

    private readonly HttpClient _httpClient;
    private readonly IOptionsMonitor<ModbusOptions> _options;
    private readonly ILogger<WebhookPlcAlertSink> _logger;

    public WebhookPlcAlertSink(HttpClient httpClient, IOptionsMonitor<ModbusOptions> options, ILogger<WebhookPlcAlertSink> logger)
    {
        _httpClient = httpClient;
        _options = options;
        _logger = logger;
    }

    public async Task PublishAsync(PlcAlert alert, CancellationToken cancellationToken)
    {
        var monitoring = _options.CurrentValue.Monitoring;
        var webhooks = monitoring?.AlertWebhooks?.Where(hook => hook.Enabled && !string.IsNullOrWhiteSpace(hook.Url)).ToArray();
        if (webhooks is null || webhooks.Length == 0)
        {
            return;
        }

        var payload = JsonSerializer.Serialize(alert, SerializerOptions);
        foreach (var hook in webhooks)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Post, hook.Url)
                {
                    Content = new StringContent(payload, Encoding.UTF8, "application/json")
                };

                if (!string.IsNullOrWhiteSpace(hook.Secret))
                {
                    request.Headers.TryAddWithoutValidation(hook.SecretHeaderName ?? "X-Alert-Secret", hook.Secret);
                }

                if (hook.Headers is not null)
                {
                    foreach (var header in hook.Headers)
                    {
                        request.Headers.TryAddWithoutValidation(header.Key, header.Value);
                    }
                }

                var response = await _httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);
                if (!response.IsSuccessStatusCode)
                {
                    _logger.LogWarning("Webhook {Name} responded with {StatusCode}", hook.Name ?? hook.Url, (int)response.StatusCode);
                }
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                _logger.LogError(ex, "Failed to post PLC alert to webhook {Name}", hook.Name ?? hook.Url);
            }
        }
    }
}
