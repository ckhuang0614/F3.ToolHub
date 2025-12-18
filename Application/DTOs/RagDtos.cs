using System;
using System.Collections.Generic;
using System.Linq;
using F3.ToolHub.Domain.Rag;

namespace F3.ToolHub.Application.DTOs;

public sealed record RagDocumentDto(
    Guid Id,
    string Name,
    string ContentType,
    string Source,
    string Version,
    long Size,
    int ChunkCount,
    DateTimeOffset UploadedAt,
    IReadOnlyDictionary<string, string> Tags);

public sealed record RagAnswerDto(
    string Content,
    double Confidence,
    IReadOnlyCollection<RagCitationDto> Citations);

public sealed record RagCitationDto(
    Guid DocumentId,
    string DocumentName,
    string ChunkId,
    double Score,
    string Snippet);

public static class RagDtoMapper
{
    public static RagDocumentDto ToDto(this RagDocument document)
    {
        return new RagDocumentDto(
            document.Id,
            document.Name,
            document.ContentType,
            document.Source,
            document.Version,
            document.Size,
            document.ChunkCount,
            document.UploadedAt,
            document.Tags);
    }

    public static RagAnswerDto ToDto(this RagAnswer answer)
    {
        var citations = answer.Citations
            .Select(static citation => new RagCitationDto(
                citation.DocumentId,
                citation.DocumentName,
                citation.ChunkId,
                citation.Score,
                citation.Snippet))
            .ToArray();

        return new RagAnswerDto(answer.Content, answer.Confidence, citations);
    }
}
