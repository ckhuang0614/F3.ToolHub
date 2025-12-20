from __future__ import annotations
from typing import Dict

_METRIC_MAP: Dict[str, str] = {
    "accuracy": "accuracy",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "mae": "mae",
    "mse": "mse",
    "rmse": "rmse",
}


def normalize_metric(metric: str) -> str:
    if not metric:
        raise ValueError("metric is required")
    key = metric.lower()
    return _METRIC_MAP.get(key, metric)
