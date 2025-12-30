# FLAML 2.3.6 Metrics by Task

## Task_type -> Available metrics (full list)

| task_type | available metrics |
|---|---|
| classification / binary / multiclass | accuracy, log_loss, f1, micro_f1, macro_f1, roc_auc, roc_auc_ovr, roc_auc_ovo,<br>roc_auc_weighted, roc_auc_ovr_weighted, roc_auc_ovo_weighted, ap |
| regression | r2, rmse, mae, mse, mape |
| rank | ndcg, ndcg@k |
| forecast / ts_forecast / ts_forecast_regression / ts_forecast_classification / ts_forecast_panel | r2, rmse, mae, mse, mape |
| seq-classification | HuggingFace metrics list (see below) |
| multichoice-classification | HuggingFace metrics list (see below) |
| token-classification | HuggingFace metrics list (see below) |
| seq-regression | HuggingFace metrics list (see below) |
| summarization | HuggingFace metrics list (see below) |

## Default metric when metric="auto"

| task_type | recommended metric (metric="auto") |
|---|---|
| classification | if 2 classes -> roc_auc; else -> log_loss |
| binary | roc_auc |
| multiclass | log_loss |
| regression | r2 |
| rank | ndcg |
| forecast / ts_forecast / ts_forecast_regression / ts_forecast_classification / ts_forecast_panel | mape |
| seq-classification | accuracy |
| multichoice-classification | accuracy |
| token-classification | seqeval |
| seq-regression | r2 |
| summarization | rouge1 |

## HuggingFace metrics list (complete, requires flaml[hf])

`accuracy`, `bertscore`, `bleu`, `bleurt`, `cer`, `chrf`, `code_eval`, `comet`,
`competition_math`, `coval`, `cuad`, `f1`, `gleu`, `google_bleu`,
`matthews_correlation`, `meteor`, `pearsonr`, `precision`, `recall`, `rouge`,
`sacrebleu`, `sari`, `seqeval`, `spearmanr`, `ter`, `wer`.

Notes:
- `rouge1` and `rouge2` are accepted and mapped to `rouge`.
- `seqeval` supports submetric syntax like `seqeval:overall_f1`.

## Spark metrics (Spark DataFrame only)

- Regression: `r2`, `rmse`, `mse`, `mae`, `var`
- Binary Classification: `pr_auc`, `roc_auc`
- Multi-class Classification: `accuracy`, `log_loss`, `f1`, `micro_f1`, `macro_f1`

## Notes

- If you want FLAML to select defaults, pass metric="auto". This repo forwards it unchanged.
- task_type="classification" is resolved by FLAML at runtime to binary or multiclass based on label count.
