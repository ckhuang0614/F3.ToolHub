using F3.ToolHub.Application.Contracts;

namespace F3.ToolHub.Application.UseCases;

public sealed class RebuildRagIndexUseCase
{
    private readonly IRagVectorStore _vectorStore;

    public RebuildRagIndexUseCase(IRagVectorStore vectorStore)
    {
        _vectorStore = vectorStore;
    }

    public Task HandleAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return _vectorStore.RebuildAsync(cancellationToken);
    }
}
