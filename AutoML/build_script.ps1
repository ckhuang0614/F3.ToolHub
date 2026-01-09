
# clearml-2.3
docker compose -f docker-compose-clearml-2.3.yml --profile build-only build
docker compose -f docker-compose-clearml-2.3.yml up -d
docker compose -f docker-compose-clearml-2.3.yml down
docker compose -f docker-compose-clearml-2.3.yml build autogluon-trainer
docker compose -f docker-compose-clearml-2.3.yml build flaml-trainer
docker compose -f docker-compose-clearml-2.3.yml build ultralytics-trainer
docker compose -f docker-compose-clearml-2.3.yml up -d clearml-agent
docker compose -f docker-compose-clearml-2.3.yml build gateway
docker compose -f docker-compose-clearml-2.3.yml up -d --force-recreate gateway

# clearml-serving (Model Endpoints)
docker compose -f docker-compose-clearml-2.3.yml build clearml-serving
docker compose -f docker-compose-clearml-2.3.yml run --rm --entrypoint clearml-serving gateway create --name "AutoML Serving"
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
docker compose -f docker-compose-clearml-2.3.yml --profile serving up -d clearml-serving

# clearml-serving (safe start, avoid extra instances)
# Do NOT use "docker compose run" to start serving; it creates extra instance tasks.
# Use the same control-plane id each time (from /endpoints response).
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
docker compose -f docker-compose-clearml-2.3.yml --profile serving --profile monitoring up -d `
  clearml-serving clearml-serving-stats

# clean stop before reboot (prevents extra completed instances)
docker compose -f docker-compose-clearml-2.3.yml stop clearml-serving clearml-serving-stats

# clearml-serving (reuse instance id + cleanup old running instances)
# Requires CLEARML_SERVING_TASK_ID to be set.
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
$instanceId = @'
import os
from clearml import Task

project = os.getenv("CLEARML_SERVING_PROJECT", "DevOps")
name = os.getenv("CLEARML_SERVING_NAME", "AutoML Serving")
task_name = f"{name} - serve instance"

tasks = Task.get_tasks(project_name=project, task_name=task_name, allow_archived=True)
def _updated(t):
    try:
        return t._get_last_update()
    except Exception:
        return getattr(t, "created", 0) or 0
tasks.sort(key=_updated, reverse=True)

print(tasks[0].id if tasks else "")
'@ | docker compose -f docker-compose-clearml-2.3.yml run --rm --entrypoint python gateway -
$env:CLEARML_INFERENCE_TASK_ID = $instanceId.Trim()

@'
import os
from clearml import Task

project = os.getenv("CLEARML_SERVING_PROJECT", "DevOps")
name = os.getenv("CLEARML_SERVING_NAME", "AutoML Serving")

def cleanup(task_name):
    tasks = Task.get_tasks(project_name=project, task_name=task_name, allow_archived=True)
    def _updated(t):
        try:
            return t._get_last_update()
        except Exception:
            return getattr(t, "created", 0) or 0
    tasks.sort(key=_updated, reverse=True)
    for t in tasks[1:]:
        try:
            t.mark_stopped()
        except Exception:
            pass

cleanup(f"{name} - serve instance")
cleanup(f"{name} - statistics controller")
'@ | docker compose -f docker-compose-clearml-2.3.yml run --rm --entrypoint python gateway -

docker compose -f docker-compose-clearml-2.3.yml --profile serving --profile monitoring up -d `
  clearml-serving clearml-serving-stats

# clearml-serving (archive extra instances)
$env:CLEARML_SERVING_TASK_ID="YOUR_SERVING_TASK_ID"
@'
import os
from clearml import Task

project = os.getenv("CLEARML_SERVING_PROJECT", "DevOps")
name = os.getenv("CLEARML_SERVING_NAME", "AutoML Serving")
keep = int(os.getenv("CLEARML_SERVING_KEEP", "1"))

def updated(t):
    try:
        return t._get_last_update()
    except Exception:
        return getattr(t, "created", 0) or 0

for task_name in (f"{name} - serve instance", f"{name} - statistics controller"):
    tasks = Task.get_tasks(project_name=project, task_name=task_name, allow_archived=True)
    tasks.sort(key=updated, reverse=True)
    for t in tasks[keep:]:
        try:
            t.mark_stopped()
        except Exception:
            pass
        t.set_archived(True)
'@ | docker compose -f docker-compose-clearml-2.3.yml run --rm --entrypoint python gateway -

#Powershell by BuildKit
$env:DOCKER_BUILDKIT=1; docker compose -f docker-compose-clearml-2.3.yml build autogluon-trainer flaml-trainer ultralytics-trainer
$env:DOCKER_BUILDKIT=1; docker compose -f docker-compose-clearml-2.3.yml build autogluon-trainer
$env:DOCKER_BUILDKIT=1; docker compose -f docker-compose-clearml-2.3.yml build flaml-trainer
$env:DOCKER_BUILDKIT=1; docker compose -f docker-compose-clearml-2.3.yml build ultralytics-trainer


docker run --rm --network automl_default `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 ls s3://datasets


# ml-trainer
$payload = @'
{
  "trainer": "autogluon",
  "dataset": { "csv_uri": "s3://datasets/demo.csv", "target": "label" },
  "group_key": ["user_id"],
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "group_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "demo-run"
}
'@


$payload = @'
{
  "trainer": "autogluon",
  "dataset": { "csv_uri": "s3://datasets/demo.csv", "target": "label" },
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
'@


$payload = @'
{
  "trainer": "flaml",
  "dataset": { "csv_uri": "s3://datasets/demo.csv", "target": "label" },
  "group_key": ["user_id"],
  "time_budget_s": 300,
  "metric": "accuracy",
  "task_type": "classification",
  "split": { "method": "group_shuffle", "test_size": 0.2, "random_seed": 42 },
  "run_name": "demo-run"
}
'@


$payload = @'
{
  "trainer": "flaml",
  "dataset": { "csv_uri": "s3://datasets/demo.csv", "target": "label" },
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
'@

docker run --rm --network automl_default -v ${PWD}/dataset:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/demo.csv s3://datasets/demo.csv

curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload



# yolo-trainer
cd dataset
Compress-Archive -Path images,labels,labels.yaml -DestinationPath yolo_dataset.zip -Force

docker run --rm --network automl_default -v ${PWD}/dataset:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/yolo_dataset.zip s3://datasets/yolo_dataset.zip


docker run --rm --network automl_default -v ${PWD}/trainers/ultralytics/pretrained:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/yolo11n.pt s3://models/pretrained/yolo11n.pt

docker run --rm --network automl_default -v ${PWD}/dataset:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/bus.jpg s3://datasets/bus.jpg

curl -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d @trainers/ultralytics/payload_example.json
