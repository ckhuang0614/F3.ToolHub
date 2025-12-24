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
  - `yolo/` — YOLO 訓練器（`train.py`, `Dockerfile`, `requirements.txt`, `payload_example.json`）。
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

- 範例 payload（可用於 POST `/runs` API）：

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
  "dataset": { "type": "tabular", "uri": "s3://datasets/demo.csv", "label": "label" },
  "group_key": ["user_id"],
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "group_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "demo-run",
  "extras": {
    "autogluon": {
      "leaderboard": true,
      "feature_importance": true,
      "fit_summary": true,
      "fit_args": {
        "presets": "best_quality",
        "num_bag_folds": 5,
        "num_stack_levels": 1,
        "hyperparameters": "default"
      }
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

- FLAML 範例 payload（含 extras）：

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
  "trainer": "yolo",
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
  -d @trainers/yolo/payload_example.json
```

或直接在 PowerShell 中以變數傳入：

```powershell
$payload = Get-Content -Raw -Path "trainers/yolo/payload_example.json"
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
docker compose -f docker-compose-clearml-2.3.yml build yolo-trainer
```

- 使用 amazon/aws-cli 與 MinIO 上傳／檢視 S3 物件（在 `automl_default` network 下執行）：

```bash
# 列出 datasets bucket 內容
docker run --rm --network automl_default ``
  -e AWS_ACCESS_KEY_ID=minioadmin ``
  -e AWS_SECRET_ACCESS_KEY=minioadmin ``
  -e AWS_EC2_METADATA_DISABLED=true ``
  amazon/aws-cli --endpoint-url http://minio:9000 s3 ls s3://datasets

# 上傳本地 demo.csv 至 MinIO 的 datasets bucket
docker run --rm --network automl_default -v ${PWD}/dataset:/data ``
  -e AWS_ACCESS_KEY_ID=minioadmin ``
  -e AWS_SECRET_ACCESS_KEY=minioadmin ``
  -e AWS_EC2_METADATA_DISABLED=true ``
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/demo.csv s3://datasets/demo.csv
```

> 注意：以上命令有些在 PowerShell 中需使用反引號換行（如範例），或在 Linux/macOS bash 中使用反斜線 `\` 換行；請依使用的 shell 調整換行符號與環境變數語法。

## ClearML 整合

- 若使用 ClearML 追蹤或分派任務，請確認 `infra/clearml/clearml.conf` 已設定正確的 server、api、web 與 credentials。
- 可使用 `infra/clearml-agent/Dockerfile` 建立 agent 映像以在容器中執行任務。

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
