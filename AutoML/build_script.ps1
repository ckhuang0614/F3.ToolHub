# clearml-1.13
docker compose --profile build-only build
docker compose up -d
docker compose down
docker compose build autogluon-trainer
docker compose build flaml-trainer
docker compose up -d clearml-agent

# clearml-2.3
docker compose -f docker-compose-clearml-2.3.yml --profile build-only build
docker compose -f docker-compose-clearml-2.3.yml up -d
docker compose -f docker-compose-clearml-2.3.yml down
docker compose -f docker-compose-clearml-2.3.yml build autogluon-trainer
docker compose -f docker-compose-clearml-2.3.yml build flaml-trainer
docker compose -f docker-compose-clearml-2.3.yml build ultralytics-trainer
docker compose -f docker-compose-clearml-2.3.yml up -d clearml-agent
docker compose -f docker-compose-clearml-2.3.yml build gateway
docker compose -f docker-compose-clearml-2.3.yml up -d gateway

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


curl -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d @trainers/ultralytics/payload_example.json
