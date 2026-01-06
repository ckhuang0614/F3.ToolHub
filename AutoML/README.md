# AutoML 目錄說明

此文件說明專案中 `AutoML/` 目錄的用途、結構與常用操作。

## 目錄結構（摘要）

- `build_script.ps1` — Windows PowerShell 打包／建置腳本。
- `docker-compose.yml` — 啟動多服務（gateway、trainers、ClearML 等）的整合設定。
- `dataset/`
  - `demo.csv` — 範例資料集。
- `shared_lib/`
  - `__init__.py`, `grouping.py`, `run_request.py`, `metrics.py` — 共用的前處理、請求封裝與評估工具。
- `infra/`
  - `clearml/clearml.conf` — ClearML 連線與設定檔。
  - `clearml-agent/Dockerfile` — ClearML agent 映像建置設定。
- `trainers/`
  - `autogluon/` — AutoGluon 訓練腳本與 `Dockerfile`。
  - `flaml/` — FLAML 訓練腳本、`requirements.txt`、`requirements_v113.txt`、`Dockerfile`。
  - `ultralytics/` — YOLO 訓練器（`train.py`, `Dockerfile`, `requirements.txt`, `payload_example.json`）。
- `gateway/`
  - `app.py`, `requirements.txt`, `Dockerfile` — 提供 HTTP/API 入口的服務。

## 快速開始

1. 確認已安裝 Docker 與 Docker Compose（或 Docker Desktop）。
2. 若要使用 ClearML，請檢查並更新 `infra/clearml/clearml.conf` 中的設定。

### 使用 docker-compose 啟動（建議）

在 `AutoML/` 根目錄下執行：

```bash
docker compose up --build
```

這會建立並啟動 `gateway` 與訓練相關服務（依 `docker-compose.yml` 配置）。

## `build_script.ps1` 常用命令與範例

`build_script.ps1` 中包含可直接執行或參考的範例 payload 與常用命令，以下為重點整理，方便貼回 PowerShell 或 README 中使用。

+ Autogluon 範例 payload（可用於 POST `/runs` API）：
  
```json
{
  "trainer": "autogluon",
  "schema_version": 2,
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "group_key": ["user_id"],
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "group_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "demo-run"
}
```

- 具有額外 `extras` 的 Autogluon 範例：

```json
{
  "trainer": "autogluon",
  "schema_version": 2,
  "dataset": {"type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label"},
  "time_budget_s": 300,
  "group_key": ["user_id"],
  "metric": "accuracy",
  "task_type": "classification",
  "split": {"method": "row_shuffle", "test_size": 0.2, "random_seed": 42},
  "run_name": "tabular-run",
  "extras": {
    "autogluon": {
      "mode": "tabular",
      "fit_args": {
        "presets": "medium_quality_faster_train",
        "num_bag_folds": 5,
        "num_stack_levels": 1,
        "hyperparameters": "default"
      },
      "analysis": {
        "summary": true,
        "corr": true,
        "mutual_info": false,
        "target_corr": true,
        "shap": false
      },
      "leaderboard": true,
      "feature_importance": true,
      "feature_importance_args": {},
      "fit_summary": true
    }
  }
}

{
  "trainer": "autogluon",
  "schema_version": 2,
  "dataset": { "type": "tabular", "uri": "s3://datasets/mm_demo.csv", "label": "label" },
  "group_key": [],
  "time_budget_s": 600,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "multimodal-run",
  "extras": {
    "autogluon": {
      "mode": "multimodal",
      "fit_args": {
        "presets": "medium_quality",
        "column_types": {
          "image": ["image_path"],
          "text": ["title", "description"],
          "numerical": ["price"],
          "categorical": ["category"]
        }
      },
      "leaderboard": true,
      "feature_importance": false,
      "fit_summary": true
    }
  }
}

{
  "trainer": "autogluon",
  "dataset": {"type": "tabular", "uri": "s3://datasets/ts_demo.csv", "label": "target"},
  "time_budget_s": 600,
  "metric": "MASE",
  "task_type": "regression",
  "group_key": [],
  "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "timeseries-run",
  "extras": {
    "autogluon": {
      "mode": "timeseries",
      "fit_args": {
        "presets": "medium_quality"
      },
      "timeseries": {
        "prediction_length": 24,
        "item_id": "series_id",
        "timestamp": "date",
        "target": "target",
        "predictor_args": {
          "freq": "D"
        },
        "allow_unsafe_torch_load": false
      },
      "leaderboard": true,
      "feature_importance": true,
      "fit_summary": true
    }
  }
}

```

- 發送 Autogluon payload 至本地 gateway 的範例（PowerShell / bash）：

```bash
# 使用 curl 上傳 payload.json
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d @trainers/autogluon/payload_example.json
```

或直接在 PowerShell 中以變數傳入：

```powershell
$payload = Get-Content -Raw -Path "trainers/autogluon/payload_example.json"
curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload
```

+ FLAML 範例 payload（含 extras）：

```json
{
  "trainer": "flaml",
  "schema_version": 2,
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "group_key": ["user_id"],
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "group_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "demo-run",
  "extras": {
    "flaml": {
      "summary": true,
      "feature_importance": true,
      "fit_args": {
        "estimator_list": ["lgbm", "xgboost"],
        "log_file_name": "flaml.log"
      }
    }
  }
}
```

- 發送 FLAML payload 至本地 gateway 的範例（PowerShell / bash）：

```bash
# 使用 curl 上傳 payload.json
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d @trainers/flaml/payload_example.json
```

或直接在 PowerShell 中以變數傳入：

```powershell
$payload = Get-Content -Raw -Path "trainers/flaml/payload_example.json"
curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload
```

+ 新增 YOLO 範例 payload 與訓練支援：

- YOLO 範例 payload（可使用 gateway 提交）：

```json
{
  "trainer": "ultralytics",
  "schema_version": 2,
  "dataset": { "type": "yolo", "uri": "s3://datasets/yolo_dataset.zip", "yaml_path": "labels.yaml" },
  "time_budget_s": 3600,
  "metric": "mAP50",
  "task_type": "detection",
  "run_name": "yolo-demo-run",
  "extras": {
    "yolo": {
      "imgsz": 640,
      "batch": 16,
      "epochs": 50,
      "weights": "yolov8n.pt"
    }
  }
}
```

- 發送 YOLO payload 至本地 gateway 的範例（PowerShell / bash）：

```bash
# 使用 curl 上傳 payload.json
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d @trainers/ultralytics/payload_example.json
```

或直接在 PowerShell 中以變數傳入：

```powershell
$payload = Get-Content -Raw -Path "trainers/ultralytics/payload_example.json"
curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload
```

- Docker / docker-compose 常用命令（`clearml-1.13` 範例）：

```bash
# 只 build 映像
docker compose --profile build-only build
# 啟動背景模式
docker compose up -d
# 停止並移除容器
docker compose down
# 單獨建置 trainer 映像
docker compose build autogluon-trainer
docker compose build flaml-trainer
# 啟動 clearml-agent
docker compose up -d clearml-agent
```

- Docker Compose for ClearML 2.3（若有 `docker-compose-clearml-2.3.yml`）：

```bash
docker compose -f docker-compose-clearml-2.3.yml --profile build-only build
docker compose -f docker-compose-clearml-2.3.yml up -d
docker compose -f docker-compose-clearml-2.3.yml down
docker compose -f docker-compose-clearml-2.3.yml build autogluon-trainer
docker compose -f docker-compose-clearml-2.3.yml build flaml-trainer
docker compose -f docker-compose-clearml-2.3.yml build ultralytics-trainer
docker compose -f docker-compose-clearml-2.3.yml build gateway
docker compose -f docker-compose-clearml-2.3.yml up -d --force-recreate gateway

```

- 使用 amazon/aws-cli 與 MinIO 上傳／檢視 S3 物件（在 `automl_default` network 下執行）：

```bash
# 列出 datasets bucket 內容
docker run --rm --network automl_default `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 ls s3://datasets

# 上傳本地 demo.csv 至 MinIO 的 datasets bucket
docker run --rm --network automl_default -v ${PWD}/dataset:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/demo.csv s3://datasets/demo.csv
```

> 注意：以上命令有些在 PowerShell 中需使用反引號換行（如範例），或在 Linux/macOS bash 中使用反斜線 `\` 換行；請依使用的 shell 調整換行符號與環境變數語法。

## ClearML 整合

- 若使用 ClearML 追蹤或分派任務，請確認 `infra/clearml/clearml.conf` 已設定正確的 server、api、web 與 credentials。
- 可使用 `infra/clearml-agent/Dockerfile` 建立 agent 映像以在容器中執行任務。
- ClearML 設定檔統一使用同一份來源：`infra/clearml/clearml.conf` 與 `infra/clearml/storage_credentials.conf` 會掛載到 gateway/agent/trainers 的 `/etc/clearml/`，並由 `CLEARML_CONFIG_FILE` / `TRAINS_CONFIG_FILE` 指向該路徑。
- clearml-agent 啟動的 task 容器也會以 bind mount 掛載同一份設定；若你的專案路徑不同，請在啟動前設定 `CLEARML_CONFIG_HOST_DIR` 指向主機上的 `infra/clearml` 目錄：

```powershell
$env:CLEARML_CONFIG_HOST_DIR="D:\Project\F3.ToolHub\AutoML\infra\clearml"
docker compose -f docker-compose-clearml-2.3.yml up -d
```

- Windows 瀏覽器無法解析 `http://fileserver:8081` 時，可在本機 hosts 加一行讓舊資料可載入：
  - 檔案：`C:\Windows\System32\drivers\etc\hosts`
  - 內容：`127.0.0.1 fileserver`
  - 更新後執行：`ipconfig /flushdns`，再重新整理 UI

### ClearML Queues

- 送任務時可在 payload 指定 `queue`，或用環境變數指定 trainer 對應 queue：`CLEARML_QUEUE_AUTOGLOUON` / `CLEARML_QUEUE_FLAML` / `CLEARML_QUEUE_ULTRALYTICS`。
- `docker-compose-clearml-2.3.yml` 已新增多個 agent 服務：`clearml-agent-cpu`（queue: `cpu`）、`clearml-agent-gpu`（queue: `gpu`，預設 `--gpus all` / `nvidia` runtime）、`clearml-agent-services`（queue: `services`）。可用 `CLEARML_CPU_QUEUE` / `CLEARML_GPU_QUEUE` / `CLEARML_SERVICES_QUEUE` 覆蓋。
- gateway 提供 `GET /queues` 方便查詢目前 queue 列表與 default queue。

範例：使用「trainer → queue」預設對應（payload 不填 `queue`）

```powershell
$env:CLEARML_QUEUE_AUTOGLOUON="cpu"
$env:CLEARML_QUEUE_FLAML="services"
$env:CLEARML_QUEUE_ULTRALYTICS="gpu"
```

```json
{
  "trainer": "autogluon",
  "schema_version": 2,
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "run_name": "ag-default-queue"
}
```

```json
{
  "trainer": "flaml",
  "schema_version": 2,
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "run_name": "flaml-default-queue"
}
```

```json
{
  "trainer": "ultralytics",
  "schema_version": 2,
  "dataset": { "type": "yolo", "uri": "s3://datasets/yolo_dataset.zip", "yaml_path": "labels.yaml" },
  "time_budget_s": 3600,
  "metric": "mAP50",
  "task_type": "detection",
  "run_name": "yolo-default-queue",
  "extras": { "yolo": { "weights": "yolo11n.pt" } }
}
```

查詢 queue 狀態（gateway）：

```powershell
curl.exe http://localhost:8000/queues
```

UI 監控位置：
- ClearML Web UI → **Queues & Workers**：可查看 queue 的 pending/running 狀態與 agent 在線狀態。
- 若要告警（例如 queue 無 worker 或 pending 過久），可定期輪詢 `/queues` 或使用 ClearML API 做外部監控。

## ClearML Datasets 版本化

可在 payload 的 `dataset` 內使用 `clearml` 參照資料集版本，系統會在訓練容器中下載並使用該版本。

- `dataset.clearml.id`：直接使用 ClearML Dataset ID
- 或 `dataset.clearml.name` + `dataset.clearml.project` + `dataset.clearml.version`
- Tabular 需提供 `dataset.label`，若 ClearML Dataset 內有多個 CSV，請加上 `dataset.path`
- YOLO 可用 `dataset.yaml_path` 指向 dataset 內的 yaml
- datasets bucket 使用 `infra/clearml/storage_credentials.conf` 定義（預設 bucket `datasets`，ACL 為 private），可依 MinIO/S3 設定調整。

Dataset API（gateway）：
- `GET /datasets`：列出 datasets（支援 `project/name/version/tags`）
- `GET /datasets/lookup`：依 name+version+tags 查詢單一 dataset
- `GET /datasets/{id}`：依 id 查詢 dataset
- `GET /datasets/{id}/versions`：列出同名版本
- `GET /datasets/{id}/lineage`：列出 dataset lineage（含 parents）

Tabular 範例：

```json
{
  "trainer": "autogluon",
  "schema_version": 2,
  "dataset": {
    "type": "tabular",
    "label": "label",
    "path": "demo.csv",
    "clearml": {
      "name": "demo-dataset",
      "project": "AutoML",
      "version": "1.0.0"
    }
  },
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "clearml-tabular-v1"
}
```

YOLO 範例：

```json
{
  "trainer": "ultralytics",
  "schema_version": 2,
  "dataset": {
    "type": "yolo",
    "yaml_path": "data.yaml",
    "clearml": {
      "id": "YOUR_DATASET_ID"
    }
  },
  "time_budget_s": 3600,
  "metric": "mAP50",
  "task_type": "detection",
  "run_name": "clearml-yolo-v1",
  "extras": {
    "yolo": {
      "imgsz": 640,
      "batch": 16,
      "epochs": 50,
      "weights": "yolov8n.pt"
    }
  }
}
```

## ClearML Model Endpoints

> Model Endpoints 需要 clearml-serving inference 服務；已在 `docker-compose-clearml-2.3.yml` 內新增 `clearml-serving`（profile: `serving`）。

啟動流程（建議）：

1) 取得 Serving Service（control plane）ID  
   - 方式 A：直接呼叫 gateway `/endpoints`（若未提供 `service_id` 且環境變數也未設定，gateway 會自動建立並回傳 `service_id`）。  
   - 方式 B：使用 `clearml-serving create --name "AutoML Serving"` 產生 ID。  

2) 將 `service_id` 設定到環境變數 `CLEARML_SERVING_TASK_ID`，再啟動 clearml-serving：  

```powershell
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
docker compose -f docker-compose-clearml-2.3.yml --profile serving up -d clearml-serving
```

> 使用自建 image（預設 `f3.clearml-serving:latest`）；可設定 `CLEARML_SERVING_IMAGE` 覆蓋。
> 若還需要額外套件，可用 `CLEARML_EXTRA_PYTHON_PACKAGES` 補充安裝。

模型狀態流（candidate → production）：

- 訓練容器支援以下環境變數：
  - `CLEARML_MODEL_TAGS=candidate`：自動在模型上加候選標籤
  - `CLEARML_MODEL_PUBLISH=true`：自動 publish
- 若透過 clearml-agent 執行，請將上述 env 透過 agent 的環境變數或 `CLEARML_AGENT_EXTRA_DOCKER_ARGS` 傳入 task 容器。
- 審核後可在 UI 修改 tags 或直接 publish 進 production。

### Gateway /endpoints API

建立 / 更新 endpoint：

```powershell
$payload = @'
{
  "endpoint": "automl-tabular",
  "engine": "sklearn",
  "model_id": "YOUR_MODEL_ID",
  "version": "1"
}
'@

curl.exe -X POST http://localhost:8000/endpoints `
  -H "Content-Type: application/json" `
  -d $payload
```

推論入口（clearml-serving）：`http://localhost:8082/serve/<endpoint>/<version>`（不帶 version 則為預設版本）。

或用 name/project + tags 選模型：

```json
{
  "endpoint": "automl-tabular",
  "engine": "sklearn",
  "model_name": "flaml-demo-run",
  "model_project": "AutoML-Models",
  "model_tags": ["candidate"],
  "model_published": false,
  "version": "1"
}
```

灰度 / 權重路由（canary）：

```powershell
$payload = @'
{
  "endpoint": "automl-tabular-canary",
  "weights": [0.1, 0.9],
  "input_endpoints": ["automl-tabular/2", "automl-tabular/1"]
}
'@

curl.exe -X POST http://localhost:8000/endpoints/canary `
  -H "Content-Type: application/json" `
  -d $payload
```

回滾（移除指定版本）：

```powershell
$payload = @'
{
  "endpoint": "automl-tabular",
  "version": "2"
}
'@

curl.exe -X POST http://localhost:8000/endpoints/rollback `
  -H "Content-Type: application/json" `
  -d $payload
```

> `engine` 可用：`sklearn` / `xgboost` / `lightgbm` / `triton` / `custom` / `custom_async`。若需自訂前後處理，可在 payload 加上 `preprocess_code`（gateway 內部可見的本地路徑）。

Autogluon（custom）範例：

> 需確保 clearml-serving image 有安裝 autogluon（可用 `CLEARML_EXTRA_PYTHON_PACKAGES`，或自行在 `infra/clearml-serving/Dockerfile` 內加 `autogluon[all]==1.4.0`）。

```json
{
  "endpoint": "automl-tabular",
  "engine": "custom",
  "model_id": "YOUR_MODEL_ID",
  "version": "1",
  "preprocess_code": "/app/serving/autogluon_preprocess.py"
}
```

Autogluon 推論請求格式（擇一）：

```json
{ "records": [ { "f1": 1, "f2": 2 } ], "return_proba": true }
```

```json
{ "columns": ["f1","f2"], "data": [[1,2]] }
```

PowerShell 範例（以交易資料欄位為例）：

```powershell
$payload = @'
{
  "records":[{
    "transaction_id": 1,
    "amount": 100.5,
    "transaction_hour": 13,
    "merchant_category": "groceries",
    "foreign_transaction": 0,
    "location_mismatch": 0,
    "device_trust_score": 0.82,
    "velocity_last_24h": 3,
    "cardholder_age": 35
  }],
  "return_proba": true
}
'@

curl.exe -X POST http://localhost:8082/serve/automl-tabular/1 `
  -H "Content-Type: application/json" `
  -d $payload
```

### Model Endpoints 監控（metrics logging + Prometheus/Grafana）

ClearML 的 Model Endpoints 會顯示 instance/requests/latency，但前提是 clearml-serving 有把推論 metrics 上報。
完整流程如下（可先用現有 Kafka/Prometheus/Grafana）：

1) 準備 Kafka（metrics queue）  
   - compose 已加上 `kafka`（profile: `monitoring`），預設可用 `kafka:9092`。  
   - 若使用外部 Kafka，請設定 `CLEARML_DEFAULT_KAFKA_SERVE_URL` 指向你的 broker。

2) 啟動 serving + statistics 服務  
   - serving：已用 `clearml-serving` container 提供推論 API。  
   - statistics：負責消費 Kafka metrics，寫回 ClearML UI，並暴露 Prometheus `/metrics`。

   方式 A（臨時執行）：

```powershell
docker compose -f docker-compose-clearml-2.3.yml run --rm `
  -e CLEARML_DEFAULT_KAFKA_SERVE_URL=kafka:9092 `
  -e CLEARML_SERVING_TASK_ID=$env:CLEARML_SERVING_TASK_ID `
  --entrypoint "python -m clearml_serving.statistics.main" `
  clearml-serving
```

   方式 B（加一個常駐服務，需自行加到 compose）：

```yaml
clearml-serving-stats:
  image: ${CLEARML_SERVING_IMAGE:-f3.clearml-serving:latest}
  environment:
    CLEARML_DEFAULT_KAFKA_SERVE_URL: ${CLEARML_DEFAULT_KAFKA_SERVE_URL:-kafka:9092}
    CLEARML_SERVING_TASK_ID: ${CLEARML_SERVING_TASK_ID:-}
  command: ["python", "-m", "clearml_serving.statistics.main"]
  ports:
    - "9999:9999"
```

> 若 UI 仍空白，請確認 control-plane task（DevOps / AutoML Serving）內的 `metric_logging` 設定沒有被關閉。

3) 發送推論請求（即觸發 metrics）  
   - 任何 `POST http://<serving-host>/serve/<endpoint>/<version>` 都算一次推論請求。
   - 沒有流量時，Model Endpoints UI 會保持空白。

4) Prometheus 抓取 metrics  
   - statistics 服務預設在 `:9999/metrics` 暴露 Prometheus 指標。
   - `prometheus.yml` 範例：

```yaml
scrape_configs:
  - job_name: "clearml-serving"
    static_configs:
      - targets: ["clearml-serving-stats:9999"]
```

5) Grafana  
   - 新增 Prometheus Data Source（指向 Prometheus URL）。  
   - 建議用 dashboard 顯示：requests/min、p50/p95 latency、error rate、instances。

同步頻率說明：
- serving 端會以 `CLEARML_SERVING_POLL_FREQ`（預設 1 秒）同步 endpoint 設定。
- metrics 更新頻率由 statistics 服務批次上報決定，建議看服務 log 以確認實際刷新節奏。

### 單一 serve instance + stats 正常回寫（重啟後標準流程）

在執行 `docker compose -f docker-compose-clearml-2.3.yml down` 之後，建議用以下流程啟動，避免產生多個 serve instance：

```powershell
# 0) 設定 control-plane 與 Kafka URL
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
$env:CLEARML_DEFAULT_KAFKA_SERVE_URL="kafka:9092"

# 1) 啟動 ClearML 核心服務
docker compose -f docker-compose-clearml-2.3.yml up -d `
  apiserver webserver fileserver redis mongo elasticsearch minio

# 2) 重新 build clearml-serving（確保含 kafka/lz4）
docker compose -f docker-compose-clearml-2.3.yml build clearml-serving

# 3) 啟動 Kafka + serving + stats + Prometheus/Grafana
docker compose -f docker-compose-clearml-2.3.yml --profile serving --profile monitoring up -d `
  zookeeper kafka clearml-serving clearml-serving-stats prometheus grafana

# 4) 啟動 gateway
docker compose -f docker-compose-clearml-2.3.yml up -d gateway

# 5) 啟動 clearml-agent
docker compose -f docker-compose-clearml-2.3.yml up -d clearml-agent clearml-agent-cpu clearml-agent-gpu clearml-agent-services


# 6) 啟動 autogluon 訓練任務
$payload = Get-Content -Raw -Path "trainers/autogluon/payload_example.json"
curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload
{"task_id":"f1402cfd6fa246d08fc0946c63458942","queue":"cpu","docker_image":"f3.autogluon-trainer:latest"}

# 7) 用 gateway 自動建立 control‑plane
$payload = @'
{
  "endpoint": "automl-tabular",
  "engine": "custom",
  "model_id": "{MODEL_ID}",
  "version": "1",
  "preprocess_code": "/app/serving/autogluon_preprocess.py"
}
'@

curl.exe -X POST http://localhost:8000/endpoints `
  -H "Content-Type: application/json" `
  -d $payload

```

驗證（可選）：

```powershell
docker compose -f docker-compose-clearml-2.3.yml logs -f clearml-serving-stats
```

```powershell
curl.exe -X POST http://localhost:8082/serve/automl-tabular/1 `
  -H "Content-Type: application/json" `
  -d $payload
```

> 避免使用 `docker compose run clearml-serving`，會額外產生新的 serve instance task。

#### 如何查 control-plane task id

方式 A（UI）：
- ClearML Web UI → Project `DevOps` → Tasks
- 篩選 tag `SERVING-CONTROL-PLANE` 或名稱 `AutoML Serving`
- 進入任務後，右上角顯示的 `ID` 即為 `CLEARML_SERVING_TASK_ID`

方式 B（Gateway 回傳）：
- 呼叫 `POST /endpoints` 建立或更新 endpoint，回應的 `service_id` 即 control-plane task id

#### 清掉舊的 serve instance

- ClearML Web UI → Project `DevOps` → Tasks
- 篩選 tag `SERVICE` 或名稱 `AutoML Serving - serve instance`
- 選擇舊任務後點 **Archive**（或 **Abort** 停止正在跑的 instance）

## ClearML Pipelines

- Pipeline 骨架：`pipelines/automl_pipeline.py`
- 範例設定：`pipelines/pipeline_example.json`
- 執行範例：

```bash
python pipelines/automl_pipeline.py --config pipelines/pipeline_example.json
```

> Pipeline 會依設定建立 ClearML task 並送入 queue；訓練映像名稱可用 `AUTOGLOUON_IMAGE` / `FLAML_IMAGE` / `ULTRALYTICS_IMAGE` 覆蓋。

Pipeline 也支援 dataset step（先產生 dataset，再給訓練使用）：

```json
{
  "name": "AutoML Pipeline with Dataset",
  "project": "AutoML-Tabular",
  "queue": "default",
  "steps": [
    {
      "name": "build-dataset",
      "type": "dataset",
      "payload": {
        "project": "AutoML",
        "name": "demo-dataset",
        "version": "1.0.0",
        "files": ["/data/demo.csv"],
        "upload": true,
        "finalize": true
      }
    },
    {
      "name": "autogluon-train",
      "parents": ["build-dataset"],
      "payload": {
        "trainer": "autogluon",
        "schema_version": 2,
        "dataset_ref": "build-dataset",
        "dataset": { "type": "tabular", "label": "label" },
        "time_budget_s": 300,
        "metric": "accuracy",
        "task_type": "classification",
        "run_name": "pipeline-ag"
      }
    }
  ]
}
```

> 若 dataset step 有 `parents`，會自動串到 `parent_datasets` 形成 lineage。

## Projects Dashboard metadata 標準

為了讓 ClearML UI 的比較與篩選一致，任務會統一寫入以下 metadata：

- tags：自動加上 `automl`、`trainer:<name>`、`schema:v<ver>`，並合併 payload 內的 `tags`。
- parameters：`Run/*` 與 `Dataset/*` 欄位（包含 dataset uri/label/clearml ref 等）。
- 模型關聯：訓練完成後會寫入 `Model/id`、`Model/name`、`Model/project`、`Model/version`（若有註冊）。

示範 payload（含 project / tags）：

```json
{
  "trainer": "autogluon",
  "schema_version": 2,
  "project": "AutoML-Tabular",
  "tags": ["baseline", "v1"],
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "ag-baseline-v1"
}
```

## Gateway 建立 ClearML Dataset

gateway 提供 `POST /datasets` 以建立資料集版本。若使用本機檔案，請先掛載到 gateway 容器內（例如 `./dataset:/data:ro`）。

本機檔案範例：

```powershell
$payload = @'
{
  "project": "AutoML",
  "name": "demo-dataset",
  "version": "1.0.0",
  "files": ["/data/demo.csv"],
  "upload": true,
  "finalize": true
}
'@

curl.exe -X POST http://localhost:8000/datasets `
  -H "Content-Type: application/json" `
  -d $payload
```

S3/MinIO URI 範例（預設以 external 方式註冊，不下載）：

```powershell
$payload = @'
{
  "project": "AutoML",
  "name": "demo-dataset",
  "version": "1.0.1",
  "uris": ["s3://datasets/demo.csv"],
  "external": true,
  "upload": true,
  "finalize": true
}
'@

curl.exe -X POST http://localhost:8000/datasets `
  -H "Content-Type: application/json" `
  -d $payload
```

> 若需將 URI 內容下載並上傳到 ClearML Fileserver，可將 `external` 設為 `false`；此時 gateway 需要具備對應的 S3/MinIO 認證與 endpoint 環境變數。

## Gateway 建立 ClearML Pipeline

gateway 提供 `POST /pipelines` 建立 Pipeline。此 API 僅接受 inline payload，不支援 payload 檔案路徑。
gateway 會在本機進程建立 pipeline controller，不會把 controller 丟到 queue（避免中斷 gateway 服務）。`wait=false` 會改成背景執行並立即回應。

```powershell
$payload = @'
{
  "name": "AutoML Pipeline via Gateway",
  "project": "AutoML-Tabular",
  "queue": "default",
  "wait": false,
  "steps": [
    {
      "name": "autogluon-train",
      "payload": {
        "trainer": "autogluon",
        "schema_version": 2,
        "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
        "time_budget_s": 300,
        "metric": "accuracy",
        "task_type": "classification",
        "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
        "run_name": "pipeline-ag"
      }
    },
    {
      "name": "flaml-train",
      "parents": ["autogluon-train"],
      "payload": {
        "trainer": "flaml",
        "schema_version": 2,
        "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
        "time_budget_s": 300,
        "metric": "accuracy",
        "task_type": "classification",
        "split": { "method": "row_shuffle", "test_size": 0.2, "random_seed": 42 },
        "run_name": "pipeline-flaml"
      }
    }
  ]
}
'@

curl.exe -X POST http://localhost:8000/pipelines `
  -H "Content-Type: application/json" `
  -d $payload
```

## 建置訓練映像

每個訓練器底下的 `Dockerfile` 可用於建置對應映像：

```bash
# 範例：建置 FLAML 訓練映像
docker build -t automl-flaml:local -f trainers/flaml/Dockerfile .
```

## 常見問題

- 若遇到套件相依問題，請確認使用的 Python 版本與 `requirements_*.txt` 相容。
- 若 Docker 映像無法啟動，檢查 `docker compose logs` 或 `docker logs <container>` 以取得錯誤資訊。

## 貢獻

對 `AutoML/` 的改動請採 PR 流程，並在變更中更新本 README 或相關文件以利團隊了解。

---
