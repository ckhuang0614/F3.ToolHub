using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.Options;
using F3.ToolHub.Domain.Rag;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Serilog;

namespace F3.ToolHub.Infrastructure.Rag;

internal sealed class DefaultRagDocumentIngestor : IRagDocumentIngestor
{
    private readonly TextChunker _chunker;
    private readonly IRagVectorStore _vectorStore;
    private readonly IRagEmbeddingClient _embeddingClient;
    private readonly IRagMetrics _metrics;
    private readonly IRagAuditSink _auditSink;
    private readonly ILogger<DefaultRagDocumentIngestor> _logger;

    public DefaultRagDocumentIngestor(
        TextChunker chunker,
        IRagVectorStore vectorStore,
        IRagEmbeddingClient embeddingClient,
        IRagMetrics metrics,
        IRagAuditSink auditSink,
        ILogger<DefaultRagDocumentIngestor> logger)
    {
        _chunker = chunker;
        _vectorStore = vectorStore;
        _embeddingClient = embeddingClient;
        _metrics = metrics;
        _auditSink = auditSink;
        _logger = logger;
    }

    public async Task<RagDocument> IngestAsync(RagDocument document, string content, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var stopwatch = Stopwatch.StartNew();
        var chunks = _chunker.Chunk(document, content).ToArray();
        if (chunks.Length == 0)
        {
            stopwatch.Stop();
            var emptyDocument = document with { ChunkCount = 0 };
            _metrics.TrackIngestion(stopwatch.Elapsed, 0);
            await _auditSink.WriteIngestionAsync(emptyDocument, cancellationToken);
            _logger.LogInformation("Ingest skipped for document {Document}; no content generated chunks.", document.Name);
            return emptyDocument;
        }

        var enrichedChunks = new List<RagChunk>(chunks.Length);
        foreach (var chunk in chunks)
        {
            var embeddingInput = BuildEmbeddingInput(chunk);
            var embedding = await _embeddingClient.GenerateEmbeddingAsync(embeddingInput, cancellationToken).ConfigureAwait(false);
            enrichedChunks.Add(chunk with { Embedding = embedding });
        }

        await _vectorStore.UpsertAsync(enrichedChunks, cancellationToken).ConfigureAwait(false);
        stopwatch.Stop();

        var updated = document with { ChunkCount = enrichedChunks.Count };
        _metrics.TrackIngestion(stopwatch.Elapsed, enrichedChunks.Count);
        await _auditSink.WriteIngestionAsync(updated, cancellationToken);
        _logger.LogInformation("Ingested RAG document {Document} with {ChunkCount} chunks", updated.Name, updated.ChunkCount);

        return updated;
    }

    private static string BuildEmbeddingInput(RagChunk chunk)
    {
        var builder = new StringBuilder();
        if (!string.IsNullOrWhiteSpace(chunk.Content))
        {
            builder.AppendLine(chunk.Content.Trim());
        }

        if (chunk.Metadata is not null && chunk.Metadata.Count > 0)
        {
            foreach (var pair in chunk.Metadata)
            {
                if (string.IsNullOrWhiteSpace(pair.Key) || string.IsNullOrWhiteSpace(pair.Value))
                {
                    continue;
                }

                builder.Append(pair.Key).Append(':').Append(' ').AppendLine(pair.Value.Trim());
            }
        }

        var text = builder.ToString().Trim();
        return string.IsNullOrWhiteSpace(text)
            ? string.Empty
            : text;
    }
}

internal sealed class TextChunker
{
    private readonly IOptions<RagOptions> _options;

    public TextChunker(IOptions<RagOptions> options)
    {
        _options = options;
    }

    public IReadOnlyCollection<RagChunk> Chunk(RagDocument document, string? content)
    {
        if (string.IsNullOrWhiteSpace(content))
        {
            return Array.Empty<RagChunk>();
        }

        var chunkSize = Math.Max(120, _options.Value.Chunk.Size);
        var overlap = Math.Clamp(_options.Value.Chunk.Overlap, 0, chunkSize - 1);
        var result = new List<RagChunk>();
        var metadataTemplate = BuildMetadata(document);

        var index = 0;
        var order = 0;
        while (index < content.Length)
        {
            var length = Math.Min(chunkSize, content.Length - index);
            var chunkContent = content.Substring(index, length).Trim();
            if (!string.IsNullOrWhiteSpace(chunkContent))
            {
                var metadata = new Dictionary<string, string>(metadataTemplate)
                {
                    ["chunkOrder"] = order.ToString()
                };

                var chunk = new RagChunk(
                    document.Id,
                    $"{document.Id:N}-{order}",
                    order,
                    chunkContent,
                    metadata);
                result.Add(chunk);
            }

            if (chunkSize == overlap)
            {
                break;
            }

            index += Math.Max(1, chunkSize - overlap);
            order++;
        }

        return result;
    }

    private static Dictionary<string, string> BuildMetadata(RagDocument document)
    {
        var metadata = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["documentName"] = document.Name,
            ["source"] = document.Source,
            ["version"] = document.Version
        };

        foreach (var tag in document.Tags)
        {
            if (!string.IsNullOrWhiteSpace(tag.Key) && !string.IsNullOrWhiteSpace(tag.Value))
            {
                metadata[$"tag:{tag.Key}"] = tag.Value;
            }
        }

        return metadata;
    }
}

internal sealed class InMemoryRagVectorStore : IRagVectorStore
{
    private readonly List<RagChunk> _chunks = new();
    private readonly object _lock = new();

    public Task UpsertAsync(IReadOnlyCollection<RagChunk> chunks, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        if (chunks.Count == 0)
        {
            return Task.CompletedTask;
        }

        lock (_lock)
        {
            foreach (var chunk in chunks)
            {
                _chunks.RemoveAll(existing => existing.DocumentId == chunk.DocumentId && existing.ChunkId == chunk.ChunkId);
                _chunks.Add(chunk);
            }
        }

        return Task.CompletedTask;
    }

    public Task<IReadOnlyCollection<RagChunk>> SearchAsync(RagQuery query, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        RagChunk[] matches;
        lock (_lock)
        {
            matches = _chunks
                .Select(chunk => (Chunk: chunk, Score: Score(chunk, query)))
                .Where(tuple => tuple.Score > 0)
                .OrderByDescending(tuple => tuple.Score)
                .Take(query.ContextSize)
                .Select(tuple => tuple.Chunk)
                .ToArray();
        }

        return Task.FromResult<IReadOnlyCollection<RagChunk>>(matches);
    }

    public Task RebuildAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return Task.CompletedTask;
    }

    private static double Score(RagChunk chunk, RagQuery query)
    {
        var text = chunk.Content ?? string.Empty;
        if (string.IsNullOrWhiteSpace(query.Query) || string.IsNullOrWhiteSpace(text))
        {
            return 0;
        }

        var score = 0.0;
        var terms = query.Query.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        foreach (var term in terms)
        {
            if (text.Contains(term, StringComparison.OrdinalIgnoreCase))
            {
                score += 1;
            }
        }

        if (query.Tags.Count > 0)
        {
            var matchesAllTags = true;
            foreach (var tag in query.Tags)
            {
                if (string.IsNullOrWhiteSpace(tag))
                {
                    continue;
                }

                if (!chunk.Metadata.TryGetValue($"tag:{tag}", out var chunkValue))
                {
                    matchesAllTags = false;
                    break;
                }

                if (query.TagValues.TryGetValue(tag, out var expectedValue) &&
                    !string.IsNullOrWhiteSpace(expectedValue) &&
                    !string.Equals(chunkValue, expectedValue, StringComparison.OrdinalIgnoreCase))
                {
                    matchesAllTags = false;
                    break;
                }
            }

            if (!matchesAllTags)
            {
                score *= 0.5;
            }
        }

        return score;
    }
}

internal sealed class DefaultRagRetriever : IRagRetriever
{
    private readonly IRagVectorStore _vectorStore;

    public DefaultRagRetriever(IRagVectorStore vectorStore)
    {
        _vectorStore = vectorStore;
    }

    public Task<IReadOnlyCollection<RagChunk>> RetrieveAsync(RagQuery query, CancellationToken cancellationToken)
    {
        return _vectorStore.SearchAsync(query, cancellationToken);
    }
}

internal sealed class TemplateRagGenerationClient : IRagGenerationClient
{
    public Task<RagAnswer> GenerateAsync(RagQuery query, IReadOnlyCollection<RagChunk> context, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        var builder = new StringBuilder();
        builder.AppendLine($"查詢：{query.Query}");

        if (context.Count == 0)
        {
            builder.AppendLine("尚未找到相關文件，請新增資料或調整查詢條件。");
            return Task.FromResult(new RagAnswer(builder.ToString().Trim(), Array.Empty<RagCitation>(), 0.2));
        }

        builder.AppendLine("摘要：");
        foreach (var chunk in context.Take(3))
        {
            var snippet = Trim(chunk.Content, 160);
            builder.AppendLine($"- {snippet}");
        }

        var citations = context
            .Select((chunk, index) =>
            {
                var documentName = chunk.Metadata.TryGetValue("documentName", out var name)
                    ? name
                    : chunk.DocumentId.ToString();
                var score = Math.Max(0.35, 0.85 - index * 0.05);
                var snippet = Trim(chunk.Content, 200);
                return new RagCitation(chunk.DocumentId, documentName, chunk.ChunkId, score, snippet);
            })
            .ToArray();

        var confidence = Math.Clamp(0.55 + context.Count * 0.05, 0.0, 0.95);
        return Task.FromResult(new RagAnswer(builder.ToString().Trim(), citations, confidence));
    }

    private static string Trim(string? value, int maxLength)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return string.Empty;
        }

        return value.Length <= maxLength ? value : value[..maxLength] + "...";
    }
}

internal sealed class RagMetrics : IRagMetrics, IDisposable
{
    private readonly Meter _meter = new("F3.ToolHub.Rag");
    private readonly Histogram<double> _queryLatency;
    private readonly Histogram<int> _contextSize;
    private readonly Histogram<double> _ingestLatency;
    private readonly Histogram<int> _chunkCount;

    public RagMetrics()
    {
        _queryLatency = _meter.CreateHistogram<double>("rag_query_latency_ms");
        _contextSize = _meter.CreateHistogram<int>("rag_query_context_size");
        _ingestLatency = _meter.CreateHistogram<double>("rag_ingest_latency_ms");
        _chunkCount = _meter.CreateHistogram<int>("rag_chunk_count");
    }

    public void TrackQuery(TimeSpan duration, int contextCount)
    {
        _queryLatency.Record(duration.TotalMilliseconds);
        _contextSize.Record(contextCount);
    }

    public void TrackIngestion(TimeSpan duration, int chunkCount)
    {
        _ingestLatency.Record(duration.TotalMilliseconds);
        _chunkCount.Record(chunkCount);
    }

    public void Dispose()
    {
        _meter.Dispose();
    }
}

internal sealed class SerilogRagAuditSink : IRagAuditSink
{
    private readonly Serilog.ILogger _logger;

    public SerilogRagAuditSink(Serilog.ILogger logger)
    {
        _logger = logger.ForContext("SourceContext", "Rag.Audit");
    }

    public Task WriteQueryAsync(RagQuery query, RagAnswer answer, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _logger.Information(
            "RAG query {@Query} produced answer with confidence {Confidence}",
            query,
            answer.Confidence);
        return Task.CompletedTask;
    }

    public Task WriteIngestionAsync(RagDocument document, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _logger.Information(
            "RAG document {DocumentName} ({DocumentId}) ingested with {ChunkCount} chunks",
            document.Name,
            document.Id,
            document.ChunkCount);
        return Task.CompletedTask;
    }
}
