using System;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.Options;
using F3.ToolHub.Application.UseCases;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Infrastructure.Rag;

public static class RagServiceCollectionExtensions
{
    public static IServiceCollection AddRagServices(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<RagOptions>(configuration.GetSection(RagOptions.SectionName));

        services.AddHttpClient<AzureOpenAiEmbeddingClient>((sp, client) =>
        {
            var embedding = sp.GetRequiredService<IOptions<RagOptions>>().Value.Embedding;
            if (embedding is null || string.IsNullOrWhiteSpace(embedding.Endpoint))
            {
                throw new InvalidOperationException("Rag:Embedding.Endpoint must be configured.");
            }

            client.BaseAddress = new Uri(embedding.Endpoint, UriKind.Absolute);
            client.DefaultRequestHeaders.Remove("api-key");
            if (string.IsNullOrWhiteSpace(embedding.ApiKey))
            {
                throw new InvalidOperationException("Rag:Embedding.ApiKey must be configured.");
            }

            client.DefaultRequestHeaders.Add("api-key", embedding.ApiKey);
        });
        services.AddSingleton<IRagEmbeddingClient>(sp => sp.GetRequiredService<AzureOpenAiEmbeddingClient>());

        services.AddSingleton<TextChunker>();
        services.AddSingleton<IRagVectorStore>(sp =>
        {
            var options = sp.GetRequiredService<IOptions<RagOptions>>();
            var kind = options.Value.VectorStore.Kind ?? string.Empty;
            return string.Equals(kind, "AzureSearch", StringComparison.OrdinalIgnoreCase)
                ? ActivatorUtilities.CreateInstance<AzureSearchVectorStore>(sp)
                : new InMemoryRagVectorStore();
        });
        services.AddSingleton<IRagRetriever, DefaultRagRetriever>();
        services.AddSingleton<IRagGenerationClient, TemplateRagGenerationClient>();
        services.AddSingleton<IRagMetrics, RagMetrics>();
        services.AddSingleton<IRagAuditSink, SerilogRagAuditSink>();
        services.AddScoped<IRagDocumentIngestor, DefaultRagDocumentIngestor>();

        services.AddScoped<QueryRagUseCase>();
        services.AddScoped<IngestRagDocumentUseCase>();
        services.AddScoped<RebuildRagIndexUseCase>();

        return services;
    }
}
