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

## ClearML Datasets 版本化

可在 payload 的 `dataset` 內使用 `clearml` 參照資料集版本，系統會在訓練容器中下載並使用該版本。

- `dataset.clearml.id`：直接使用 ClearML Dataset ID
- 或 `dataset.clearml.name` + `dataset.clearml.project` + `dataset.clearml.version`
- Tabular 需提供 `dataset.label`，若 ClearML Dataset 內有多個 CSV，請加上 `dataset.path`
- YOLO 可用 `dataset.yaml_path` 指向 dataset 內的 yaml

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

## ClearML Pipelines

- Pipeline 骨架：`pipelines/automl_pipeline.py`
- 範例設定：`pipelines/pipeline_example.json`
- 執行範例：

```bash
python pipelines/automl_pipeline.py --config pipelines/pipeline_example.json
```

> Pipeline 會依設定建立 ClearML task 並送入 queue；訓練映像名稱可用 `AUTOGLOUON_IMAGE` / `FLAML_IMAGE` / `ULTRALYTICS_IMAGE` 覆蓋。

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
