using System.Collections.Generic;

namespace F3.ToolHub.Application.Options;

public sealed class RagOptions
{
    public const string SectionName = "Rag";

    public string EmbeddingProvider { get; set; } = "AzureOpenAI";

    public string CompletionProvider { get; set; } = "AzureOpenAI";

    public string EmbeddingModel { get; set; } = "text-embedding-3-large";

    public string CompletionModel { get; set; } = "gpt-4o-mini";

    public AzureOpenAiOptions Embedding { get; set; } = new();

    public AzureOpenAiOptions Completion { get; set; } = new();

    public VectorStoreOptions VectorStore { get; set; } = new();

    public ChunkingOptions Chunk { get; set; } = new();

    public SecurityOptions Security { get; set; } = new();

    public int DefaultContextSize { get; set; } = 6;

    public int EmbeddingDimensions { get; set; } = 1536;

    public sealed class VectorStoreOptions
    {
        public string Kind { get; set; } = "AzureSearch";

        public string Endpoint { get; set; } = string.Empty;

        public string Index { get; set; } = string.Empty;

        public string ApiKey { get; set; } = string.Empty;
    }

    public sealed class ChunkingOptions
    {
        public int Size { get; set; } = 750;

        public int Overlap { get; set; } = 100;
    }

    public sealed class SecurityOptions
    {
        public List<string> AllowedRoles { get; set; } = new() { RagRoles.Admin, RagRoles.Reader };
    }

    public sealed class AzureOpenAiOptions
    {
        public string Endpoint { get; set; } = string.Empty;

        public string Deployment { get; set; } = string.Empty;

        public string ApiKey { get; set; } = string.Empty;

        public string ApiVersion { get; set; } = "2024-08-01-preview";
    }
}

public static class RagRoles
{
    public const string Admin = "RagAdmin";

    public const string Reader = "RagReader";
}
