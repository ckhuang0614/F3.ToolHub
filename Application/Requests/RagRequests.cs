using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using F3.ToolHub.Domain.Rag;

namespace F3.ToolHub.Application.Requests;

public sealed class RagDocumentUploadRequest
{
    [Required]
    public string Name { get; set; } = string.Empty;

    [Required]
    public string Content { get; set; } = string.Empty;

    public string ContentType { get; set; } = "text/plain";

    public string Source { get; set; } = "manual";

    public string Version { get; set; } = "v1";

    public Dictionary<string, string> Tags { get; set; } = new();
}

public sealed class RagQueryRequest
{
    [Required]
    public string Query { get; set; } = string.Empty;

    public string? Language { get; set; }

    public string? Tool { get; set; }

    public List<string> Tags { get; set; } = new();

    public Dictionary<string, string> TagValues { get; set; } = new();

    public int? ContextSize { get; set; }
}

public static class RagRequestExtensions
{
    public static RagQuery ToQuery(this RagQueryRequest request, int defaultContextSize)
    {
        var sanitizedTagValues = request.TagValues?
            .Where(pair => !string.IsNullOrWhiteSpace(pair.Key) && !string.IsNullOrWhiteSpace(pair.Value))
            .ToDictionary(
                pair => pair.Key.Trim(),
                pair => pair.Value.Trim(),
                StringComparer.OrdinalIgnoreCase)
            ?? new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        var tagKeys = new List<string>();
        var tagKeySet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        if (request.Tags is not null)
        {
            foreach (var rawTag in request.Tags)
            {
                var trimmed = rawTag?.Trim();
                if (string.IsNullOrWhiteSpace(trimmed))
                {
                    continue;
                }

                var separatorIndex = trimmed.IndexOf(':');
                var key = separatorIndex >= 0 ? trimmed[..separatorIndex].Trim() : trimmed;
                var value = separatorIndex >= 0 ? trimmed[(separatorIndex + 1)..].Trim() : null;

                if (string.IsNullOrWhiteSpace(key))
                {
                    continue;
                }

                if (tagKeySet.Add(key))
                {
                    tagKeys.Add(key);
                }

                if (!string.IsNullOrWhiteSpace(value) && !sanitizedTagValues.ContainsKey(key))
                {
                    sanitizedTagValues[key] = value;
                }
            }
        }

        foreach (var kvp in sanitizedTagValues.Keys.ToArray())
        {
            if (tagKeySet.Add(kvp))
            {
                tagKeys.Add(kvp);
            }
        }

        var size = request.ContextSize.HasValue && request.ContextSize.Value > 0
            ? request.ContextSize.Value
            : defaultContextSize;

        return new RagQuery(
            request.Query,
            request.Language,
            request.Tool,
            tagKeys,
            sanitizedTagValues,
            size);
    }
}
