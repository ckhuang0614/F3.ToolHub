# AutoGluon 1.4.0 Metrics by Predictor

## TabularPredictor

| problem_type | metrics (eval_metric names) |
|---|---|
| binary | accuracy, acc, balanced_accuracy, mcc, log_loss, nll, pac, pac_score, quadratic_kappa, roc_auc, average_precision,<br>precision, recall, f1, precision_macro, precision_micro, precision_weighted, recall_macro, recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted |
| multiclass | accuracy, acc, balanced_accuracy, mcc, log_loss, nll, pac, pac_score, quadratic_kappa,<br>precision_macro, precision_micro, precision_weighted, recall_macro, recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted,<br>roc_auc_ovo, roc_auc_ovo_macro, roc_auc_ovo_weighted, roc_auc_ovr, roc_auc_ovr_macro, roc_auc_ovr_micro, roc_auc_ovr_weighted |
| regression | r2, mean_squared_error, mse, root_mean_squared_error, rmse, mean_absolute_error, mae, median_absolute_error,<br>mean_absolute_percentage_error, mape, symmetric_mean_absolute_percentage_error, smape, spearmanr, pearsonr |
| quantile | pinball_loss, pinball |
| softclass | soft_log_loss |

## MultiModalPredictor

| problem_type | metrics (eval_metric names) |
|---|---|
| binary | accuracy, acc, balanced_accuracy, mcc, log_loss, nll, pac, pac_score, quadratic_kappa, roc_auc, average_precision,<br>precision, recall, f1, precision_macro, precision_micro, precision_weighted, recall_macro, recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted, coverage |
| multiclass | accuracy, acc, balanced_accuracy, mcc, log_loss, nll, pac, pac_score, quadratic_kappa,<br>precision_macro, precision_micro, precision_weighted, recall_macro, recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted,<br>roc_auc_ovo, roc_auc_ovo_macro, roc_auc_ovo_weighted, roc_auc_ovr, roc_auc_ovr_macro, roc_auc_ovr_micro, roc_auc_ovr_weighted |
| regression | r2, mean_squared_error, mse, root_mean_squared_error, rmse, mean_absolute_error, mae, median_absolute_error,<br>mean_absolute_percentage_error, mape, symmetric_mean_absolute_percentage_error, smape, spearmanr, pearsonr |
| object_detection | map, mean_average_precision, map_50, map_75, map_small, map_medium, map_large, mar_1, mar_10, mar_100, mar_small, mar_medium, mar_large |
| semantic_segmentation | iou, ber, sm |
| ner / named_entity_recognition | overall_f1, ner_token_f1 |
| few_shot_classification | accuracy, acc, balanced_accuracy, mcc, log_loss, nll, pac, pac_score, quadratic_kappa,<br>precision_macro, precision_micro, precision_weighted, recall_macro, recall_micro, recall_weighted, f1_macro, f1_micro, f1_weighted,<br>roc_auc_ovo, roc_auc_ovo_macro, roc_auc_ovo_weighted, roc_auc_ovr, roc_auc_ovr_macro, roc_auc_ovr_micro, roc_auc_ovr_weighted |
| matching (text/image/image-text similarity) | acc, accuracy, direct_loss, rmse, root_mean_squared_error, r2, quadratic_kappa, roc_auc, log_loss, cross_entropy,<br>pearsonr, spearmanr, f1, f1_macro, f1_micro, f1_weighted, map, mean_average_precision, ner_token_f1, overall_f1,<br>recall, sm, iou, ber, coverage, ndcg, precision, mrr |

## TimeSeriesPredictor

| metrics (eval_metric names) |
|---|
| MASE, MAPE, SMAPE, RMSE, RMSLE, RMSSE, WAPE, SQL, WQL, MSE, MAE, WCD (experimental) |

## Task_type to Metric Mapping (train.py)

| mode | RunRequest.task_type | AutoGluon problem_type | metric list to use |
|---|---|---|---|
| tabular / multimodal | classification | binary or multiclass (label cardinality) | see binary or multiclass lists above |
| tabular / multimodal | binary | binary | see binary list above |
| tabular / multimodal | multiclass | multiclass | see multiclass list above |
| tabular / multimodal | regression | regression | see regression list above |
| tabular only | quantile | quantile | see quantile list above |
| tabular only | softclass | softclass | soft_log_loss |
| timeseries | any | ignored | see TimeSeries list above |

## Example Payloads (train.py)

### Tabular - binary classification

```json
{
  "trainer": "autogluon",
  "dataset": {
    "type": "tabular",
    "uri": "s3://bucket/path/data.csv",
    "label": "target"
  },
  "time_budget_s": 3600,
  "metric": "roc_auc",
  "task_type": "classification",
  "group_key": [],
  "split": {
    "method": "row_shuffle",
    "test_size": 0.2,
    "random_seed": 42
  },
  "run_name": "tabular-binary",
  "extras": {
    "autogluon": {
      "mode": "tabular",
      "fit_args": {
        "presets": "medium_quality_faster_train"
      },
      "leaderboard": true,
      "feature_importance": true,
      "fit_summary": true
    }
  }
}
```

### Multimodal - regression (tabular CSV with text/image columns)

```json
{
  "trainer": "autogluon",
  "dataset": {
    "type": "tabular",
    "uri": "s3://bucket/path/data.csv",
    "label": "target"
  },
  "time_budget_s": 3600,
  "metric": "rmse",
  "task_type": "regression",
  "group_key": [],
  "split": {
    "method": "row_shuffle",
    "test_size": 0.2,
    "random_seed": 42
  },
  "run_name": "multimodal-regression",
  "extras": {
    "autogluon": {
      "mode": "multimodal",
      "fit_args": {
        "presets": "medium_quality_faster_train"
      },
      "leaderboard": false,
      "feature_importance": false,
      "fit_summary": true
    }
  }
}
```

### TimeSeries

```json
{
  "trainer": "autogluon",
  "dataset": {
    "type": "tabular",
    "uri": "s3://bucket/path/ts.csv",
    "label": "target"
  },
  "time_budget_s": 3600,
  "metric": "MASE",
  "task_type": "regression",
  "group_key": [],
  "split": {
    "method": "row_shuffle",
    "test_size": 0.2,
    "random_seed": 42
  },
  "run_name": "timeseries",
  "extras": {
    "autogluon": {
      "mode": "timeseries",
      "fit_args": {},
      "timeseries": {
        "prediction_length": 24,
        "item_id": "item_id",
        "timestamp": "timestamp",
        "target": "target",
        "predictor_args": {
          "freq": "D"
        }
      }
    }
  }
}
```

## Notes

- Time series metric names are case-insensitive and normalized to upper case.
- mean_wQuantileLoss is deprecated and maps to WQL.
