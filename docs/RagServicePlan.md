# RAG 服務規劃

## 目標
- 提供針對 PLC/工具操作手冊、故障排除紀錄與安全規範的檢索式問答能力。
- 以 API 方式供現有 Minimal API 及未來 UI 使用，輸出可追蹤來源的回答。
- 可線上/離線更新知識庫並維持審批與稽核紀錄，符合內網資安政策。

## 系統架構概覽
1. **資料蒐集層**：檔案上傳 (PDF/Word/純文字)、Webhook 匯入、排程爬取現有維運筆記；使用 `IRagDocumentIngestor` 抽象行為。
2. **前處理/嵌入層**：拆段 (chunking) + 清洗；呼叫 Azure OpenAI Embedding API (可替換為本地 SentenceTransformer)。
3. **向量儲存層**：優先採用 Azure AI Search 或 PostgreSQL+pgvector。介面 `IRagVectorStore`，以背景作業寫入，並保留 `Metadata` (來源、版次、敏感等級)。
4. **檢索層**：`IRagRetriever` 根據查詢做 hybrid search (keyword + vector)，支援過濾器 (工具、線別、語系、敏感標籤)。
5. **生成層**：`RagOrchestrator` 將檢索結果轉為 prompt，呼叫 Azure OpenAI/GPT-4o-mini，在回答中列出引用片段與信心分數。
6. **服務端點**：Minimal API 下新增 `/api/rag/documents`、`/api/rag/queries`，沿用 `X-Api-Key`，並於 `ApiSecurityOptions` 新增 `Roles:RagAdmin/Reader` 控制權限。
7. **觀測性**：透過 Serilog + `IModbusClientMetrics` 類似模式，新增 `IRagMetrics` (查詢延遲、命中數)、`IRagAuditSink` (紀錄查詢與回答摘要)。

## 主要模組
- `Domain/Rag`：`RagDocument`, `RagChunk`, `RagQuery`, `RagAnswer`。
- `Application/Contracts`：`IRagService`, `IRagDocumentIngestor`, `IRagVectorStore`, `IRagRetriever`, `IRagGenerationClient`。
- `Application/UseCases`：`IngestRagDocumentUseCase`, `QueryRagUseCase`, `RebuildRagIndexUseCase`。
- `Infrastructure/Rag`：
  - `AzureOpenAiEmbeddingClient`、`AzureOpenAiCompletionClient`。
  - `PgVectorStore` or `AzureSearchVectorStore`。
  - `BlobDocumentStore` (原始檔案) 與 `BackgroundChunkingService`。
  - `SerilogRagAuditSink`。

## 可選地端方案
- **模型/推論**：於地端 GPU 伺服器部署 `Llama-3.1 70B` 或 `Mistral Large` 等商用授權模型，透過 `vLLM`/`Text Generation Inference` 暴露 REST API，`IRagGenerationClient` 以環境變數切換端點。
- **嵌入**：使用 `nomic-embed-text` 或 `bge-m3` 等開源模型，以 `ONNX Runtime` 或 `SentenceTransformers` 服務提供 Embedding API。
- **向量/原始檔儲存**：採用本地 `PostgreSQL + pgvector` 搭配 `MinIO` 儲存原始檔案，全部運行在內網 VLAN。若需更高吞吐，可改用 `Milvus` 或 `Qdrant`。
- **部署拓撲**：以 Kubernetes/Openshift 為主，提供 `rag-embedding`, `rag-vectorstore`, `rag-llm`, `rag-api` 多個 Deployment 與內部 Service，並設 `NetworkPolicy` 限制僅允許 ToolHub API Pod 存取。
- **同步策略**：制定離線匯入流程 (USB/隔離網段)，透過 `IngestRagDocumentUseCase` 的 batch CLI 將文件同步至地端環境，並在審批後才進入 chunk/嵌入。
- **管理/監控**：使用 Prometheus/Grafana 監控 GPU/CPU、查詢延遲與 Queue 深度，審計資料寫入 Elasticsearch 或 Splunk，滿足 OT 資安稽核需求。

## appsettings 設定需求
- `appsettings.json`：定義預設 `ApiSecurity`（需新增 `RagAdmin`/`RagReader` 金鑰）、`Rag` 區塊（提供 embedding/completion provider、VectorStore、Chunk 參數）以及任何全域監控設定。
- `appsettings.Development.json`：覆寫本機端點，例如 `Rag:EmbeddingProvider=Local`、`Rag:VectorStore:Endpoint=http://localhost:9200`、`ApiSecurity.ApiKeys` 的測試金鑰等，方便開發人員在 Debug profile 下運作。
- `appsettings.Simulator.json`：供模擬器模式使用，除了沿用 `Modbus` 模擬參數，也應覆寫 `Rag` 區塊以指向本機或內網的 on-prem 模型/pgvector。
- 可視需要再新增 `appsettings.<Environment>.json`（例如 `Production`），將實際 Azure/OpenAI/pgvector 服務的連線字串、私密金鑰或 VNet DNS 寫入，並於部署 profile 指定對應 `ASPNETCORE_ENVIRONMENT`。

## 資料流程
1. **上傳/匯入**：`POST /api/rag/documents` -> 保存原檔 (Blob) -> 觸發背景佇列。
2. **前處理**：Background service 解析檔案、切片、寫入 `RagChunk` 表。
3. **嵌入**：批次呼叫嵌入 API，將向量與中繼資料寫入 Vector Store。
4. **查詢**：`POST /api/rag/queries` -> `QueryRagUseCase` 驗證權限 -> 構建檢索請求 -> 取得最相關片段 -> 呼叫生成 API -> 回傳答案 + 引用。
5. **監控/審核**：查詢紀錄寫入 `IRagAuditSink`，異常 (低信心/毒性) 觸發 Alert Webhook。

## 可設定項目 (appsettings.json)
```jsonc
  "Rag": {
    "EmbeddingProvider": "AzureOpenAI",
    "EmbeddingModel": "text-embedding-3-small",
    "Embedding": {
      "Endpoint": "https://openai-0503.openai.azure.com/",
      "Deployment": "text-embedding-3-small",
      "ApiKey": "<OpenAI-Key>"
    },
    "CompletionProvider": "AzureOpenAI",
    "CompletionModel": "gpt-4.1-mini",
    "Completion": {
      "Endpoint": "https://openai-0503.openai.azure.com/",
      "Deployment": "gpt-4.1-mini",
      "ApiKey": "<OpenAI-Key>"
    },
    "VectorStore": {
      "Kind": "AzureSearch",
      "Endpoint": "https://bd2-search.search.windows.net",
      "Index": "toolhub-rag-index",
      "ApiKey": "<Search-Admin-Key>"
    },
    "Chunk": {
      "Size": 750,
      "Overlap": 100
    },
    "DefaultContextSize": 6,
    "Security": { "AllowedRoles": [ "RagAdmin", "RagReader" ] }
  }
```
> 地端方案可透過 `Rag:EmbeddingProvider=Local`、`Rag:CompletionProvider=Local` 與 `Rag:VectorStore:Kind=PgVector` 切換，並於 `appsettings.Development.json` 指向內網 URL。

## 與現有系統整合
- **DI 註冊**：在 `Program.cs` 內新增 `AddRagServices()` 擴充方法，統一註冊上述介面與背景服務。
- **API Key 授權**：沿用 `ApiKeyAuthenticationHandler`，於 handler 中讀取 `X-Api-Key` 對應角色，再決定是否允許 `Rag` 端點。
- **Logging/Metrics**：延伸現有 Serilog sink，為 RAG 指定 `SourceContext="Rag"`，並在 `Metrics` 使用 Prometheus counter。
- **Dev/CI**：整合 docker-compose 以啟動本地 pgvector + Azure OpenAI mock (如 `azurite` 或 `local-ai`) 方便測試。

## 風險與緩解
- **嵌入/API 成本**：支援批次寫入與快取，並可切換本地模型。
- **資料外洩**：所有文件需標記敏感等級，RAG 查詢需附帶角色資訊過濾；對雲端模型使用自簽約專用網路。
- **回答 hallucination**：生成 prompt 附 `must cite` 規則與 `response schema`，低信心 (<0.6) 時返回 `needs_manual_review`。
- **延遲**：使用 hybrid search + 前 6 段輸入模型，並以 streaming 回傳答案。

## MVP 交付範圍
1. REST API + DI 介面雛型。
2. Azure OpenAI + Azure Search 實作。
3. 基本文件上傳 (單檔) + chunk/向量寫入背景服務。
4. 單元測試：`QueryRagUseCaseTests` (mock retriever/generator)、`IngestRagDocumentUseCaseTests`。
5. 整合測試：模擬查詢流程，驗證回答包含引用。

## 後續擴充
- UI 對話視覺化、上下文保持與使用者反饋 (thumbs up/down)。
- 排程同步 Confluence / SharePoint。
- 多語言版本：以語言標籤 + translation pipeline。
- 結合 Modbus 實時資料，將 PLC 異常截圖/紀錄自動寫入知識庫。
- 引入自動化安全掃描，對上傳文件進行惡意掃描與 DLP。

## 使用範例
1. **上傳文件**
   ```bash
   curl -X POST https://toolhub.local/api/rag/documents \
     -H "X-Api-Key: <RagAdmin-Key>" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "plc-maintenance-guide",
       "contentType": "text/plain",
       "source": "manual",
       "version": "2024.07",
       "tags": { "line": "L1", "sensitivity": "internal" },
       "content": "1. 斷電 → 2. 拔除模組 ..."
     }'
   ```

   ```bash
    curl.exe -X POST "http://127.0.0.1:5281/api/rag/documents" `
      -H "X-Api-Key: rag-admin-f3" `
      -H "Content-Type: application/json" `
      -d @'
    {
      "name": "plc-maintenance-guide",
      "contentType": "text/plain",
      "source": "manual",
      "version": "2024.07",
      "tags": { "line": "L1", "sensitivity": "internal" },
      "content": "1. 斷電 → 2. 拔除模組 ..."
    }
    '@
   ```

   回傳 `RagDocumentDto`，可記錄 `id` 供稽核使用。

2. **執行查詢**
   ```bash
   curl -X POST https://127.0.0.1:5281/api/rag/queries \
     -H "X-Api-Key: <RagReader-Key>" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "L1 線 PLC 無法啟動要怎麼復位?",
       "language": "zh-TW",
       "tags": ["line:L1"],
       "contextSize": 6
     }'
   ```
   ```bash
    curl.exe -X POST "http://127.0.0.1:5281/api/rag/queries" `
       -H "X-Api-Key: rag-reader-f3" `
       -H "Content-Type: application/json" `
       -d @'
    {
      "query": "L1 線 PLC 無法啟動要怎麼復位?",
      "language": "zh-TW",
      "tags": ["line"],
      "tagValues":{"line":"L1"},
      "contextSize": 6
    }
    '@
   ```

   回傳 `RagAnswerDto`，其中 `citations` 會列出引用 chunk 與信心分數，可顯示在 UI。

3. **重建索引 (批次)**
   ```bash
   curl -X POST https://toolhub.local/api/rag/indexes/rebuild \
     -H "X-Api-Key: <RagAdmin-Key>"
   ```
   觸發 `RebuildRagIndexUseCase` 重新掃描 Vector Store，回傳 `202 Accepted`。

## 向量索引設定（Azure AI Search）
- Index 名稱：`toolhub-rag-index`（`Rag:VectorStore:Index`）。
- Fields：`chunkId`(key/filter)、`documentId`(filter)、`order`(filter)、`content`(searchable, ZhHant analyzer)、`tags`(filter)、`metadataJson`(searchable)、`embedding`(vector)。
- Vector 設定：`embedding` 維度 = `Rag:EmbeddingDimensions`（預設 1536），`VectorSearchProfile` = `rag-vector-profile`，`Algorithm` = `rag-hnsw-config`。
- 索引建立/更新：由 `AzureSearchVectorStore.EnsureIndexAsync` 自動建立；若缺少向量欄位或 profiles 會自動補齊。
- Upsert：`MergeOrUpload`，`chunkId` 唯一；`Tags` 由 `metadata` 中 `tag:` 前綴生成，支援 `line`、`sensitivity` 等。

## 查詢流程（Retrieval）
1. `QueryRagUseCase` 檢查 `X-Api-Key`/角色，驗證 `RagQuery`（query/tags/contextSize）。
2. `AzureSearchVectorStore.SearchAsync`：
   - 以 `query` + tags 組合成 embedding 輸入，呼叫 `IRagEmbeddingClient` 產生查詢向量。
   - SearchOptions：`Size = max(ContextSize*4, ContextSize)`；Vector KNN 於 `embedding` 欄位，不使用 lexical filter（searchText = `*`）。
   - 取回結果後依 tags 做二次過濾，最後截取 `ContextSize`。
3. `RagOrchestrator` 將片段組合 prompt，呼叫 `IRagGenerationClient` 生成回答，附 citations。

## 地端方案（PgVector / 本地模型）
- 切換設定：
  - `Rag:VectorStore:Kind = PgVector`
  - `Rag:EmbeddingProvider = Local`（如 bge-m3 / nomic-embed-text 服務）
  - `Rag:CompletionProvider = Local`（vLLM / TGI / Ollama 等 REST 端點）
- 儲存：PostgreSQL + pgvector；Blob 改用 MinIO/S3 相容儲存。
- 部署：K8s/Openshift 以 `rag-embedding`、`rag-vectorstore`、`rag-llm`、`rag-api` 多個 Deployment，搭配 NetworkPolicy 僅允許 API Pod 存取。
- 離線同步：提供 batch CLI 使用 USB/隔離網段匯入，審批後才進入 chunk/嵌入。

## API 範例（現行 Minimal API）
- 上傳文件（需要 `RagAdmin`）：
  ```bash
  curl -X POST "http://127.0.0.1:5281/api/rag/documents" \
    -H "X-Api-Key: <RagAdmin-Key>" \
    -H "Content-Type: application/json" \
    -d '{
      "name":"plc-maintenance-guide",
      "contentType":"text/plain",
      "source":"manual",
      "version":"2024.07",
      "tags":{"line":"L1","sensitivity":"internal"},
      "content":"1. 斷電 → 2. 拔除模組 ..."
    }'
  ```
- 查詢（需要 `RagReader`）：
  ```bash
  curl -X POST "http://127.0.0.1:5281/api/rag/queries" \
    -H "X-Api-Key: <RagReader-Key>" \
    -H "Content-Type: application/json" \
    -d '{
      "query":"L1 線 PLC 無法啟動要怎麼復位?",
      "language":"zh-TW",
      "tags":["line"],
      "tagValues":{"line":"L1"},
      "contextSize":6
    }'
  ```
- 重建索引（目前 Azure Search 不需重建，預留 API 回傳 202）：
  ```bash
  curl -X POST "http://127.0.0.1:5281/api/rag/indexes/rebuild" \
    -H "X-Api-Key: <RagAdmin-Key>"
  ```

## 注意事項
- 角色與金鑰：`ApiSecurity.ApiKeys` 必須配置 `RagAdmin`、`RagReader`。
- 語言與 Analyzer：Azure Search content 欄位使用 `zh-Hant Lucene`，若多語系可額外提供 `language` 標籤或自動檢測再選 analyzer。
- 向量維度需與 Embedding 模型一致（預設 text-embedding-3-small 1536 維）。

## Azure Portal 驗證（Embedding 與查詢）
- 驗證 Embedding 生成（Azure OpenAI）：
  ```bash
  curl -X POST "https://openai-0503.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-08-01-preview" \
       -H "api-key: <Your-AOAI-Key>" \
       -H "Content-Type: application/json" \
       -d @'
  {
    "input": "L1 線 PLC 無法啟動要怎麼復位?",
    "model": "text-embedding-3-small"
  }
  '@
  ```
  回傳的 `data[0].embedding` 為向量，可直接貼到 Azure Portal/Search REST 進行驗證。

- 在 Azure Portal / Search REST 測試 vector query（取剛才的 embedding 向量）：
  ```json
  {
    "search": "*",
    "count": true,
    "vectorQueries": [
      {
        "kind": "vector",
        "vector": [
          -0.012147922,
          ...,
          0.02362407
        ],
        "fields": "embedding"
      }
    ],
    "top": 5
  }
  ```
  - `vector`：貼上上一步的 embedding。
  - `fields`：必須對應索引中的向量欄位 `embedding`。
  - `top`：可調整回傳筆數；可搭配 `count=true` 確認命中數。

## Azure Search 索引加入向量化工具（Vectorizer）
- 目的：在 Azure AI Search 端直接配置向量化工具，便於 Portal/Index 內建執行 embedding（可與程式側 embedding 互斥或做備援）。
- 步驟（Portal）：
  1. 至 Azure AI Search -> Indexes -> 選取/建立 `toolhub-rag-index`。
  2. 新增 `Vectorizer`：選擇 Azure OpenAI，填入 `Endpoint` / `ApiKey`，`Deployment` = `text-embedding-3-small`，模型名稱同 `model`。
  3. 在 `embedding` 欄位設定 `VectorSearchProfile` 為 `rag-vector-profile`，並關聯剛新增的 Vectorizer。
  4. 儲存後可於 `Search explorer` 使用 `vectorQueries` 測試，或由程式端 `SearchAsync` 直接使用已填入的 embedding 欄位。
- 程式側仍保留自產 embedding（透過 `IRagEmbeddingClient`），若要改用索引內建向量化，可在寫入前不帶向量，改由 Azure Search ingestion 時自動產生；但需確保欄位與 profile 名稱一致。
