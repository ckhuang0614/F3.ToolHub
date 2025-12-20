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


curl.exe -X POST http://localhost:8000/runs -H "Content-Type: application/json" -d $payload

docker compose down

docker compose --profile build-only build

docker compose build autogluon-trainer
docker compose build flaml-trainer

docker compose up -d clearml-agent


docker run --rm --network automl_default `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 ls s3://datasets


docker run --rm --network automl_default -v ${PWD}:/data `
  -e AWS_ACCESS_KEY_ID=minioadmin `
  -e AWS_SECRET_ACCESS_KEY=minioadmin `
  -e AWS_EC2_METADATA_DISABLED=true `
  amazon/aws-cli --endpoint-url http://minio:9000 s3 cp /data/demo.csv s3://datasets/demo.csv


