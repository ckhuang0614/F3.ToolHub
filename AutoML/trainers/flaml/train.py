from __future__ import annotations
import json
import os
import shutil
import tempfile
import threading
import time
import pandas as pd
import joblib

from clearml import Task
from clearml.storage import StorageManager

from flaml import AutoML

from shared_lib.run_request import RunRequest, TabularDatasetSpec
from shared_lib.grouping import make_group_id, group_shuffle_split, row_shuffle_split
from shared_lib.metrics import normalize_metric


def _load_run_request(task: Task) -> RunRequest:
    s = task.get_parameter("RunRequest/json")
    if not s:
        raise ValueError("Missing ClearML parameter: RunRequest/json")
    return RunRequest.from_dict(json.loads(s))

def _flaml_extras(rr: RunRequest) -> dict:
    extras = rr.extras or {}
    if not isinstance(extras, dict):
        return {}
    fl = extras.get("flaml") or {}
    if not isinstance(fl, dict):
        return {}
    return fl

def _flaml_fit_args(fl_extras: dict) -> dict:
    fit_args = fl_extras.get("fit_args") or {}
    if not isinstance(fit_args, dict):
        return {}
    reserved = {"X_train", "y_train", "time_budget", "task", "metric"}
    return {k: v for k, v in fit_args.items() if k not in reserved}

def _progress_interval_s() -> int:
    try:
        return max(10, int(os.getenv("PROGRESS_INTERVAL_S", "60")))
    except ValueError:
        return 60


def _start_progress_logger(logger, total_s: int, automl: AutoML) -> threading.Event:
    interval_s = _progress_interval_s()
    start = time.time()
    stop = threading.Event()

    logger.report_scalar(title="progress", series="elapsed_s", value=0.0, iteration=0)
    if total_s:
        logger.report_scalar(title="progress", series="pct", value=0.0, iteration=0)

    def _run() -> None:
        iteration = 1
        while not stop.wait(interval_s):
            elapsed = time.time() - start
            logger.report_scalar(title="progress", series="elapsed_s", value=float(elapsed), iteration=iteration)
            if total_s:
                pct = min(100.0, 100.0 * elapsed / float(total_s))
                logger.report_scalar(title="progress", series="pct", value=float(pct), iteration=iteration)
            try:
                best_loss = getattr(automl, "best_loss", None)
                if best_loss is not None:
                    logger.report_scalar(title="progress", series="best_loss", value=float(best_loss), iteration=iteration)
            except Exception:
                pass
            iteration += 1

    thread = threading.Thread(target=_run, name="progress-reporter", daemon=True)
    thread.start()
    return stop


def main():
    task = Task.current_task() or Task.init(project_name="AutoML-Tabular", task_name="flaml-train")
    logger = task.get_logger()

    rr = _load_run_request(task)
    task.set_name(rr.run_name or task.name)

    if not isinstance(rr.dataset, TabularDatasetSpec):
        raise ValueError(f"flaml trainer expects tabular dataset, got: {getattr(rr.dataset, 'type', type(rr.dataset))}")

    local_csv = StorageManager.get_local_copy(rr.dataset.uri)
    df = pd.read_csv(local_csv)

    if rr.split.method == "row_shuffle":
        train_df, val_df = row_shuffle_split(df, rr.split.test_size, rr.split.random_seed)
    else:
        group_id = make_group_id(df, rr.group_key)
        train_df, val_df = group_shuffle_split(df, group_id, rr.split.test_size, rr.split.random_seed)

    y_train = train_df[rr.dataset.label]
    X_train = train_df.drop(columns=[rr.dataset.label])
    y_val = val_df[rr.dataset.label]
    X_val = val_df.drop(columns=[rr.dataset.label])

    automl = AutoML()
    stop_progress = _start_progress_logger(logger, int(rr.time_budget_s), automl)
    fl_extras = _flaml_extras(rr)
    fit_args = _flaml_fit_args(fl_extras)
    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task=rr.task_type,
            time_budget=int(rr.time_budget_s),
            metric=normalize_metric(rr.metric),
            **fit_args,
        )
    finally:
        stop_progress.set()

    val_score = automl.score(X_val, y_val)
    logger.report_scalar(title="val", series="score", value=float(val_score), iteration=0)

    tmp_dir = tempfile.mkdtemp(prefix="flaml_artifacts_")
    try:
        model_path = os.path.join(tmp_dir, "flaml_automl.pkl")
        joblib.dump(automl, model_path)
        # wait_on_upload avoids deleting the file before ClearML finishes reading it
        task.upload_artifact(name="model", artifact_object=model_path, wait_on_upload=True)

        if bool(fl_extras.get("summary")):
            try:
                summary = {
                    "best_estimator": getattr(automl, "best_estimator", None),
                    "best_config": getattr(automl, "best_config", None),
                    "best_loss": getattr(automl, "best_loss", None),
                    "best_iteration": getattr(automl, "best_iteration", None),
                }
                summary_path = os.path.join(tmp_dir, "flaml_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=True, indent=2, default=str)
                task.upload_artifact(name="flaml_summary", artifact_object=summary_path, wait_on_upload=True)
            except Exception as exc:
                logger.report_text(f"flaml_summary failed: {exc}")

        if bool(fl_extras.get("feature_importance")):
            try:
                model = getattr(automl, "model", None)
                if model is None:
                    raise ValueError("best model is not available")
                if hasattr(model, "feature_importances_"):
                    values = model.feature_importances_
                elif hasattr(model, "coef_"):
                    values = model.coef_
                else:
                    raise ValueError("model does not expose feature importance")
                fi_df = pd.DataFrame(
                    {
                        "feature": X_train.columns,
                        "importance": list(values),
                    }
                )
                fi_path = os.path.join(tmp_dir, "flaml_feature_importance.csv")
                fi_df.to_csv(fi_path, index=False)
                task.upload_artifact(name="flaml_feature_importance", artifact_object=fi_path, wait_on_upload=True)
            except Exception as exc:
                logger.report_text(f"flaml_feature_importance failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
