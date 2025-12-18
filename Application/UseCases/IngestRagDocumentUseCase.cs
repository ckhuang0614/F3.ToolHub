using System;
using System.Collections.Generic;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Application.Requests;
using F3.ToolHub.Domain.Rag;

namespace F3.ToolHub.Application.UseCases;

public sealed class IngestRagDocumentUseCase
{
    private readonly IRagDocumentIngestor _ingestor;

    public IngestRagDocumentUseCase(IRagDocumentIngestor ingestor)
    {
        _ingestor = ingestor;
    }

    public async Task<RagDocumentDto> HandleAsync(RagDocumentUploadRequest request, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();

        ArgumentNullException.ThrowIfNull(request);

        var tags = request.Tags ?? new Dictionary<string, string>();
        var document = new RagDocument(
            Guid.NewGuid(),
            request.Name,
            request.ContentType,
            request.Source,
            request.Version,
            request.Content?.Length ?? 0,
            tags,
            DateTimeOffset.UtcNow,
            0);

        var ingested = await _ingestor.IngestAsync(document, request.Content ?? string.Empty, cancellationToken);
        return ingested.ToDto();
    }
}
