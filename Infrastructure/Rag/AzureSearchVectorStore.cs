using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.Options;
using F3.ToolHub.Domain.Rag;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Text.Json.Serialization;

namespace F3.ToolHub.Infrastructure.Rag;

internal sealed class AzureSearchVectorStore : IRagVectorStore
{
    private readonly SearchIndexClient _indexClient;
    private readonly SearchClient _searchClient;
    private readonly ILogger<AzureSearchVectorStore> _logger;
    private readonly IRagEmbeddingClient _embeddingClient;
    private readonly string _indexName;
    private readonly SemaphoreSlim _indexInitLock = new(1, 1);
    private readonly int _embeddingDimensions;
    private const string EmbeddingFieldName = "embedding";
    private const string VectorProfileName = "rag-vector-profile";
    private const string VectorAlgorithmConfigurationName = "rag-hnsw-config";
    private bool _indexReady;

    public AzureSearchVectorStore(IOptions<RagOptions> options, IRagEmbeddingClient embeddingClient, ILogger<AzureSearchVectorStore> logger)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(embeddingClient);
        ArgumentNullException.ThrowIfNull(logger);

        var vectorOptions = options.Value.VectorStore;
        if (string.IsNullOrWhiteSpace(vectorOptions.Endpoint))
        {
            throw new InvalidOperationException("Rag:VectorStore.Endpoint must be configured for AzureSearch vector store.");
        }

        if (string.IsNullOrWhiteSpace(vectorOptions.Index))
        {
            throw new InvalidOperationException("Rag:VectorStore.Index must be configured for AzureSearch vector store.");
        }

        if (string.IsNullOrWhiteSpace(vectorOptions.ApiKey))
        {
            throw new InvalidOperationException("Rag:VectorStore.ApiKey must be configured for AzureSearch vector store.");
        }

        _embeddingClient = embeddingClient;
        _embeddingDimensions = Math.Max(1, options.Value.EmbeddingDimensions);
        _indexName = vectorOptions.Index;
        var credential = new AzureKeyCredential(vectorOptions.ApiKey);
        _indexClient = new SearchIndexClient(new Uri(vectorOptions.Endpoint), credential);
        _searchClient = _indexClient.GetSearchClient(_indexName);
        _logger = logger;
    }

    public async Task UpsertAsync(IReadOnlyCollection<RagChunk> chunks, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (chunks.Count == 0)
        {
            return;
        }

        await EnsureIndexAsync(cancellationToken).ConfigureAwait(false);

        var documents = chunks.Select(chunk =>
        {
            var embedding = chunk.Embedding ?? throw new InvalidOperationException($"Chunk {chunk.ChunkId} is missing an embedding vector.");
            return new AzureSearchChunkDocument
            {
                ChunkId = chunk.ChunkId,
                DocumentId = chunk.DocumentId.ToString(),
                Order = chunk.Order,
                Content = chunk.Content ?? string.Empty,
                MetadataJson = SerializeMetadata(chunk.Metadata),
                Tags = BuildTagCollection(chunk.Metadata),
                Embedding = embedding.ToArray()
            };
        }).ToArray();

        try
        {
            var batch = IndexDocumentsBatch.MergeOrUpload(documents);
            await _searchClient.IndexDocumentsAsync(batch, cancellationToken: cancellationToken).ConfigureAwait(false);
        }
        catch (RequestFailedException ex)
        {
            _logger.LogError(ex, "Failed to upsert RAG chunks to Azure Search index {Index}", _indexName);
            throw;
        }
    }

    public async Task<IReadOnlyCollection<RagChunk>> SearchAsync(RagQuery query, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        await EnsureIndexAsync(cancellationToken).ConfigureAwait(false);

        var searchVector = await CreateQueryVectorAsync(query, cancellationToken).ConfigureAwait(false);
        if (searchVector is null)
        {
            throw new InvalidOperationException("Vector search requires a generated embedding.");
        }

        var options = new SearchOptions
        {
            Size = Math.Max(query.ContextSize * 4, query.ContextSize)
        };
        options.Select.Add("chunkId");
        options.Select.Add("documentId");
        options.Select.Add("order");
        options.Select.Add("content");
        options.Select.Add("metadataJson");
        options.Select.Add("tags");

        options.VectorSearch ??= new VectorSearchOptions();
        var vectorQuery = new VectorizedQuery(searchVector.ToArray())
        {
            KNearestNeighborsCount = options.Size
        };
        vectorQuery.Fields.Add(EmbeddingFieldName);
        options.VectorSearch.Queries.Add(vectorQuery);

        var results = new List<RagChunk>();
        try
        {
            var searchText = "*"; // keep logging query.Query but don't use it as a lexical filter
            var response = await _searchClient
                .SearchAsync<AzureSearchChunkDocument>(searchText, options, cancellationToken)
                .ConfigureAwait(false);

            if (_logger.IsEnabled(LogLevel.Debug))
            {
                var rawJson = response.GetRawResponse().Content?.ToString();
                _logger.LogDebug("Raw Azure Search response: {RawJson}", string.IsNullOrWhiteSpace(rawJson) ? "<empty>" : rawJson);
            }

            var searchResults = new List<SearchResult<AzureSearchChunkDocument>>();
            await foreach (var result in response.Value.GetResultsAsync().WithCancellation(cancellationToken))
            {
                searchResults.Add(result);
            }

            if (_logger.IsEnabled(LogLevel.Debug))
            {
                _logger.LogDebug("Yielded {ResultCount} search results", searchResults.Count);
                foreach (var r in searchResults.Take(5))
                {
                    var docDebug = r.Document;
                    _logger.LogDebug(
                        "Doc chunkId={ChunkId}, documentId={DocumentId}, order={Order}, hasContent={HasContent}, metadataJson={MetadataJson}",
                        docDebug?.ChunkId,
                        docDebug?.DocumentId,
                        docDebug?.Order,
                        string.IsNullOrWhiteSpace(docDebug?.Content) ? "empty" : "filled",
                        docDebug?.MetadataJson);
                }
            }

            foreach (var result in searchResults)
            {
                var doc = result.Document;
                if (doc is null)
                {
                    _logger.LogDebug("Skip result because Document is null");
                    continue;
                }

                if (!Guid.TryParse(doc.DocumentId, out var documentId))
                {
                    _logger.LogDebug("Skip result because DocumentId cannot parse: {DocumentId}", doc.DocumentId);
                    continue;
                }

                var metadata = DeserializeMetadata(doc.MetadataJson);
                results.Add(new RagChunk(documentId, doc.ChunkId, doc.Order, doc.Content, metadata));
            }
        }
        catch (RequestFailedException ex)
        {
            _logger.LogError(ex, "Failed to search Azure Search index {Index}", _indexName);
            throw;
        }

        if (_logger.IsEnabled(LogLevel.Debug) && results.Count > 0)
        {
            var previewCount = Math.Min(results.Count, Math.Max(1, query.ContextSize * 2));
            var preview = results
                .Take(previewCount)
                .Select(DescribeChunk)
                .ToArray();

            _logger.LogDebug(
                "Vector search candidates for {Query} (retrieved {Total}) with tags {TagKeys} and values {@TagValues}: {@Preview}",
                query.Query,
                results.Count,
                query.Tags,
                query.TagValues,
                preview);
        }

        _logger.LogInformation("Azure Search returned {ResultCount} candidates for query {Query}", results.Count, query.Query);

        var filteredResults = FilterByTags(results, query)
            .Take(query.ContextSize)
            .ToArray();
        if (filteredResults.Length != results.Count)
        {
            _logger.LogInformation("Tag filtering reduced candidates from {OriginalCount} to {FilteredCount} for query {Query}", results.Count, filteredResults.Length, query.Query);
        }

        return filteredResults;
    }

    public async Task RebuildAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        await _indexInitLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var index = CreateIndexDefinition();
            await _indexClient.CreateOrUpdateIndexAsync(index, cancellationToken: cancellationToken).ConfigureAwait(false);
            _indexReady = true;

            _logger.LogInformation(
                "Rebuilt/refresh Azure Search index {Index} with vector profile {Profile} and algorithm {Algorithm}",
                _indexName,
                VectorProfileName,
                VectorAlgorithmConfigurationName);
        }
        catch (RequestFailedException ex)
        {
            _logger.LogError(ex, "Failed to rebuild Azure Search index {Index}", _indexName);
            throw;
        }
        finally
        {
            _indexInitLock.Release();
        }
    }

    private static string SerializeMetadata(IReadOnlyDictionary<string, string> metadata)
    {
        return JsonSerializer.Serialize(metadata ?? new Dictionary<string, string>());
    }

    private static IReadOnlyDictionary<string, string> DeserializeMetadata(string? metadataJson)
    {
        if (string.IsNullOrWhiteSpace(metadataJson))
        {
            return new Dictionary<string, string>();
        }

        return JsonSerializer.Deserialize<Dictionary<string, string>>(metadataJson) ?? new Dictionary<string, string>();
    }

    private static IReadOnlyCollection<string>? BuildTagCollection(IReadOnlyDictionary<string, string> metadata)
    {
        if (metadata is null || metadata.Count == 0)
        {
            return Array.Empty<string>();
        }

        var tags = metadata
            .Where(pair => pair.Key.StartsWith("tag:", StringComparison.OrdinalIgnoreCase))
            .SelectMany(pair =>
            {
                var key = pair.Key[4..];
                var value = pair.Value;
                return new[] { key, string.IsNullOrWhiteSpace(value) ? key : $"{key}:{value}" };
            })
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        return tags.Length == 0 ? Array.Empty<string>() : tags;
    }

    private async Task EnsureIndexAsync(CancellationToken cancellationToken)
    {
        if (_indexReady)
        {
            return;
        }

        await _indexInitLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_indexReady)
            {
                return;
            }

            try
            {
                var response = await _indexClient.GetIndexAsync(_indexName, cancellationToken).ConfigureAwait(false);
                var index = response.Value;
                if (EnsureIndexDefinition(index))
                {
                    await _indexClient.CreateOrUpdateIndexAsync(index, cancellationToken: cancellationToken).ConfigureAwait(false);
                }
            }
            catch (RequestFailedException ex) when (ex.Status == 404)
            {
                var index = CreateIndexDefinition();
                await _indexClient.CreateIndexAsync(index, cancellationToken).ConfigureAwait(false);
            }

            _indexReady = true;
        }
        finally
        {
            _indexInitLock.Release();
        }
    }

    private SearchIndex CreateIndexDefinition()
    {
        var index = new SearchIndex(_indexName)
        {
            Fields =
            {
                new SimpleField("chunkId", SearchFieldDataType.String) { IsKey = true, IsFilterable = true },
                new SimpleField("documentId", SearchFieldDataType.String) { IsFilterable = true },
                new SimpleField("order", SearchFieldDataType.Int32) { IsFilterable = true },
                new SearchableField("content") { AnalyzerName = LexicalAnalyzerName.ZhHantLucene },
                new SearchField("tags", SearchFieldDataType.Collection(SearchFieldDataType.String)) { IsFilterable = true },
                new SearchableField("metadataJson"),
                CreateEmbeddingField()
            },
            VectorSearch = CreateVectorSearchDefinition()
        };

        return index;
    }

    private bool EnsureIndexDefinition(SearchIndex index)
    {
        var updated = false;
        if (!index.Fields.Any(field => string.Equals(field.Name, EmbeddingFieldName, StringComparison.OrdinalIgnoreCase)))
        {
            index.Fields.Add(CreateEmbeddingField());
            updated = true;
        }

        if (index.VectorSearch is null)
        {
            index.VectorSearch = CreateVectorSearchDefinition();
            updated = true;
        }
        else
        {
            updated |= EnsureVectorSearchProfiles(index.VectorSearch);
        }

        return updated;
    }

    private SearchField CreateEmbeddingField()
    {
        return new SearchField(EmbeddingFieldName, SearchFieldDataType.Collection(SearchFieldDataType.Single))
        {
            VectorSearchDimensions = _embeddingDimensions,
            VectorSearchProfileName = VectorProfileName,
            IsSearchable = true
        };
    }

    private static VectorSearch CreateVectorSearchDefinition()
    {
        var vectorSearch = new VectorSearch();
        vectorSearch.Algorithms.Add(new HnswAlgorithmConfiguration(VectorAlgorithmConfigurationName));
        vectorSearch.Profiles.Add(new VectorSearchProfile(VectorProfileName, VectorAlgorithmConfigurationName));
        return vectorSearch;
    }

    private static bool EnsureVectorSearchProfiles(VectorSearch vectorSearch)
    {
        var updated = false;

        if (!vectorSearch.Algorithms.Any(config => string.Equals(config.Name, VectorAlgorithmConfigurationName, StringComparison.Ordinal)))
        {
            vectorSearch.Algorithms.Add(new HnswAlgorithmConfiguration(VectorAlgorithmConfigurationName));
            updated = true;
        }

        if (!vectorSearch.Profiles.Any(profile => string.Equals(profile.Name, VectorProfileName, StringComparison.Ordinal)))
        {
            vectorSearch.Profiles.Add(new VectorSearchProfile(VectorProfileName, VectorAlgorithmConfigurationName));
            updated = true;
        }

        return updated;
    }

    private async Task<IReadOnlyList<float>?> CreateQueryVectorAsync(RagQuery query, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(query.Query))
        {
            throw new ArgumentException("Query text cannot be empty when using vector search.", nameof(query));
        }

        try
        {
            var embeddingInput = BuildQueryEmbeddingInput(query);
            return await _embeddingClient.GenerateEmbeddingAsync(embeddingInput, cancellationToken).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to generate embedding for query {Query}.", query.Query);
            throw new InvalidOperationException("Unable to generate embedding for vector search.", ex);
        }
    }

    private static string BuildQueryEmbeddingInput(RagQuery query)
    {
        var builder = new StringBuilder(query.Query.Trim());

        var tagKeys = query.Tags?
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .ToArray();

        if (tagKeys is not null && tagKeys.Length > 0)
        {
            builder.AppendLine().AppendLine("Relevant tags:");
            foreach (var key in tagKeys)
            {
                builder.Append("tag:").Append(key);
                if (query.TagValues.TryGetValue(key, out var value) && !string.IsNullOrWhiteSpace(value))
                {
                    builder.Append(": ").Append(value.Trim());
                }

                builder.AppendLine();
            }
        }

        return builder.ToString().Trim();
    }

    private static IReadOnlyCollection<RagChunk> FilterByTags(IReadOnlyCollection<RagChunk> chunks, RagQuery query)
    {
        if (query.Tags is null || query.Tags.Count == 0)
        {
            return chunks;
        }

        var normalizedTags = query.Tags
            .Where(tag => !string.IsNullOrWhiteSpace(tag))
            .Select(tag => tag.Trim())
            .ToArray();

        if (normalizedTags.Length == 0)
        {
            return chunks;
        }

        return chunks
            .Where(chunk => MatchesTags(chunk, normalizedTags, query.TagValues))
            .ToArray();
    }

    private static bool MatchesTags(
        RagChunk chunk,
        IReadOnlyCollection<string> requiredTags,
        IReadOnlyDictionary<string, string> requiredValues)
    {
        if (chunk.Metadata is null || chunk.Metadata.Count == 0)
        {
            return false;
        }

        foreach (var tag in requiredTags)
        {
            if (string.IsNullOrWhiteSpace(tag))
            {
                continue;
            }

            if (!chunk.Metadata.TryGetValue($"tag:{tag}", out var chunkValue))
            {
                return false;
            }

            if (requiredValues.TryGetValue(tag, out var expectedValue) &&
                !string.IsNullOrWhiteSpace(expectedValue) &&
                !string.Equals(chunkValue, expectedValue, StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }
        }

        return true;
    }

    private static string DescribeChunk(RagChunk chunk)
    {
        var metadata = chunk.Metadata ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var documentName = metadata.TryGetValue("documentName", out var name)
            ? name
            : chunk.DocumentId.ToString();

        var tagPairs = metadata
            .Where(pair => pair.Key.StartsWith("tag:", StringComparison.OrdinalIgnoreCase))
            .Select(pair => $"{pair.Key[4..]}={pair.Value}")
            .ToArray();

        return $"{documentName}/{chunk.ChunkId} [{string.Join(", ", tagPairs)}]";
    }

    private sealed class AzureSearchChunkDocument
    {
        [JsonPropertyName("chunkId")]
        public string ChunkId { get; set; } = string.Empty;

        [JsonPropertyName("documentId")]
        public string DocumentId { get; set; } = string.Empty;

        [JsonPropertyName("order")]
        public int Order { get; set; }

        [JsonPropertyName("content")]
        public string Content { get; set; } = string.Empty;

        [JsonPropertyName("metadataJson")]
        public string MetadataJson { get; set; } = string.Empty;

        [JsonPropertyName("tags")]
        public IReadOnlyCollection<string>? Tags { get; set; }

        [JsonPropertyName(EmbeddingFieldName)]
        public IReadOnlyList<float> Embedding { get; set; } = Array.Empty<float>();
    }
}
