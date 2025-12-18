using System;
using System.Collections.Generic;

namespace F3.ToolHub.Domain.Rag;

public sealed record RagDocument(
    Guid Id,
    string Name,
    string ContentType,
    string Source,
    string Version,
    long Size,
    IReadOnlyDictionary<string, string> Tags,
    DateTimeOffset UploadedAt,
    int ChunkCount);

public sealed record RagChunk(
    Guid DocumentId,
    string ChunkId,
    int Order,
    string Content,
    IReadOnlyDictionary<string, string> Metadata,
    IReadOnlyList<float>? Embedding = null);

public sealed record RagQuery(
    string Query,
    string? Language,
    string? Tool,
    IReadOnlyCollection<string> Tags,
    IReadOnlyDictionary<string, string> TagValues,
    int ContextSize);

public sealed record RagAnswer(
    string Content,
    IReadOnlyCollection<RagCitation> Citations,
    double Confidence);

public sealed record RagCitation(
    Guid DocumentId,
    string DocumentName,
    string ChunkId,
    double Score,
    string Snippet);
