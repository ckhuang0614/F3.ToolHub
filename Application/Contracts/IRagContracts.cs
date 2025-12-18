using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Domain.Rag;

namespace F3.ToolHub.Application.Contracts;

public interface IRagDocumentIngestor
{
    Task<RagDocument> IngestAsync(RagDocument document, string content, CancellationToken cancellationToken);
}

public interface IRagVectorStore
{
    Task UpsertAsync(IReadOnlyCollection<RagChunk> chunks, CancellationToken cancellationToken);

    Task<IReadOnlyCollection<RagChunk>> SearchAsync(RagQuery query, CancellationToken cancellationToken);

    Task RebuildAsync(CancellationToken cancellationToken);
}

public interface IRagRetriever
{
    Task<IReadOnlyCollection<RagChunk>> RetrieveAsync(RagQuery query, CancellationToken cancellationToken);
}

public interface IRagGenerationClient
{
    Task<RagAnswer> GenerateAsync(RagQuery query, IReadOnlyCollection<RagChunk> context, CancellationToken cancellationToken);
}

public interface IRagEmbeddingClient
{
    Task<IReadOnlyList<float>> GenerateEmbeddingAsync(string input, CancellationToken cancellationToken);
}

public interface IRagMetrics
{
    void TrackQuery(TimeSpan duration, int contextCount);

    void TrackIngestion(TimeSpan duration, int chunkCount);
}

public interface IRagAuditSink
{
    Task WriteQueryAsync(RagQuery query, RagAnswer answer, CancellationToken cancellationToken);

    Task WriteIngestionAsync(RagDocument document, CancellationToken cancellationToken);
}
