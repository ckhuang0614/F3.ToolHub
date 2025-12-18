using System.Diagnostics;
using F3.ToolHub.Application.Contracts;
using F3.ToolHub.Application.DTOs;
using F3.ToolHub.Application.Options;
using F3.ToolHub.Application.Requests;
using Microsoft.Extensions.Options;

namespace F3.ToolHub.Application.UseCases;

public sealed class QueryRagUseCase
{
    private readonly IRagRetriever _retriever;
    private readonly IRagGenerationClient _generationClient;
    private readonly IRagMetrics _metrics;
    private readonly IRagAuditSink _auditSink;
    private readonly IOptions<RagOptions> _options;

    public QueryRagUseCase(
        IRagRetriever retriever,
        IRagGenerationClient generationClient,
        IRagMetrics metrics,
        IRagAuditSink auditSink,
        IOptions<RagOptions> options)
    {
        _retriever = retriever;
        _generationClient = generationClient;
        _metrics = metrics;
        _auditSink = auditSink;
        _options = options;
    }

    public async Task<RagAnswerDto> HandleAsync(RagQueryRequest request, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentNullException.ThrowIfNull(request);

        var options = _options.Value;
        var query = request.ToQuery(options.DefaultContextSize);

        var stopwatch = Stopwatch.StartNew();
        var context = await _retriever.RetrieveAsync(query, cancellationToken);
        var answer = await _generationClient.GenerateAsync(query, context, cancellationToken);
        stopwatch.Stop();

        _metrics.TrackQuery(stopwatch.Elapsed, context.Count);
        await _auditSink.WriteQueryAsync(query, answer, cancellationToken);

        return answer.ToDto();
    }
}
