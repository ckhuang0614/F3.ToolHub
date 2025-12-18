using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.Options;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Rag;

internal sealed class AzureOpenAiEmbeddingClient : IRagEmbeddingClient
{
    private readonly HttpClient _httpClient;
    private readonly RagOptions _options;
    private readonly ILogger<AzureOpenAiEmbeddingClient> _logger;
    private readonly string _deployment;
    private readonly string _apiVersion;

    public AzureOpenAiEmbeddingClient(HttpClient httpClient, IOptions<RagOptions> options, ILogger<AzureOpenAiEmbeddingClient> logger)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
        ArgumentNullException.ThrowIfNull(options);
        _options = options.Value ?? throw new InvalidOperationException("Rag options must be configured.");
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));

        var embedding = _options.Embedding ?? throw new InvalidOperationException("Rag:Embedding configuration must be provided.");
        if (string.IsNullOrWhiteSpace(embedding.Endpoint))
        {
            throw new InvalidOperationException("Rag:Embedding.Endpoint must be configured.");
        }

        if (_httpClient.BaseAddress is null)
        {
            _httpClient.BaseAddress = new Uri(embedding.Endpoint, UriKind.Absolute);
        }

        if (string.IsNullOrWhiteSpace(embedding.ApiKey))
        {
            throw new InvalidOperationException("Rag:Embedding.ApiKey must be configured.");
        }

        if (!_httpClient.DefaultRequestHeaders.Contains("api-key"))
        {
            _httpClient.DefaultRequestHeaders.Add("api-key", embedding.ApiKey);
        }

        _deployment = !string.IsNullOrWhiteSpace(embedding.Deployment)
            ? embedding.Deployment
            : throw new InvalidOperationException("Rag:Embedding.Deployment must be configured.");

        _apiVersion = string.IsNullOrWhiteSpace(embedding.ApiVersion)
            ? "2024-08-01-preview"
            : embedding.ApiVersion;
    }

    public async Task<IReadOnlyList<float>> GenerateEmbeddingAsync(string input, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            throw new ArgumentException("Embedding input cannot be empty.", nameof(input));
        }

        var request = new EmbeddingRequest(input, _options.EmbeddingModel);
        using var response = await _httpClient
            .PostAsJsonAsync(GetRequestPath(), request, cancellationToken)
            .ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var error = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            _logger.LogError(
                "Azure OpenAI embedding request failed with status {Status}. Body: {Body}",
                response.StatusCode,
                error);
            response.EnsureSuccessStatusCode();
        }

        var payload = await response.Content
            .ReadFromJsonAsync<EmbeddingResponse>(cancellationToken: cancellationToken)
            .ConfigureAwait(false);

        var vector = payload?.Data?.FirstOrDefault()?.Embedding;
        if (vector is null || vector.Count == 0)
        {
            throw new InvalidOperationException("Azure OpenAI did not return an embedding vector.");
        }

        return vector;
    }

    private string GetRequestPath() => $"openai/deployments/{_deployment}/embeddings?api-version={_apiVersion}";

    private sealed record EmbeddingRequest(
        [property: JsonPropertyName("input")] string Input,
        [property: JsonPropertyName("model")] string Model);

    private sealed record EmbeddingResponse(
        [property: JsonPropertyName("data")] IReadOnlyList<EmbeddingData> Data);

    private sealed record EmbeddingData(
        [property: JsonPropertyName("embedding")] IReadOnlyList<float> Embedding);
}
