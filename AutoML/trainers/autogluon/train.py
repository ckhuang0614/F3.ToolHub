from __future__ import annotations
import json
import os
import shutil
import tempfile
import threading
import time

# Disable ClearML's torch patch before ClearML is imported to avoid fastai pickling issues
os.environ.setdefault("CLEARML_PATCH_PYTORCH", "0")

import pandas as pd

from clearml import Task
from clearml.storage import StorageManager

from autogluon.tabular import TabularPredictor

from shared_lib.run_request import RunRequest, TabularDatasetSpec
from shared_lib.grouping import make_group_id, group_shuffle_split, row_shuffle_split
from shared_lib.metrics import normalize_metric


def _load_run_request(task: Task) -> RunRequest:
    s = task.get_parameter("RunRequest/json")
    if not s:
        raise ValueError("Missing ClearML parameter: RunRequest/json")
    return RunRequest.from_dict(json.loads(s))


def _autogluon_problem_type(rr: RunRequest, df: pd.DataFrame, label: str) -> str:
    """
    Map our generic task_type to AutoGluon expected values.
    AutoGluon accepts: binary, multiclass, regression, quantile.
    """
    if rr.task_type.lower() == "classification":
        unique_labels = df[label].dropna().unique()
        return "binary" if len(unique_labels) <= 2 else "multiclass"
    return rr.task_type


def _excluded_models() -> list[str]:
    # FastAI and NN_TORCH are noisy/failing with ClearML patching; skip them by default.
    # raw = os.getenv("EXCLUDED_MODELS", "FASTAI,NN_TORCH")
    raw = os.getenv("EXCLUDED_MODELS", "")
    return [m.strip() for m in raw.split(",") if m.strip()]

def _autogluon_extras(rr: RunRequest) -> dict:
    extras = rr.extras or {}
    if not isinstance(extras, dict):
        return {}
    ag = extras.get("autogluon") or {}
    if not isinstance(ag, dict):
        return {}
    return ag

def _autogluon_fit_args(ag_extras: dict) -> dict:
    fit_args = ag_extras.get("fit_args") or {}
    if not isinstance(fit_args, dict):
        return {}
    reserved = {"train_data", "time_limit", "excluded_model_types"}
    return {k: v for k, v in fit_args.items() if k not in reserved}

def _progress_interval_s() -> int:
    try:
        return max(10, int(os.getenv("PROGRESS_INTERVAL_S", "60")))
    except ValueError:
        return 60


def _start_progress_logger(logger, total_s: int) -> threading.Event:
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
            iteration += 1

    thread = threading.Thread(target=_run, name="progress-reporter", daemon=True)
    thread.start()
    return stop


def main():
    # Avoid ClearML's torch patch interfering with fastai save
    os.environ.setdefault("CLEARML_PATCH_PYTORCH", "0")

    task = Task.current_task() or Task.init(
        project_name="AutoML-Tabular",
        task_name="autogluon-train",
        auto_connect_frameworks=False,
    )
    logger = task.get_logger()

    rr = _load_run_request(task)
    task.set_name(rr.run_name or task.name)

    if not isinstance(rr.dataset, TabularDatasetSpec):
        raise ValueError(f"autogluon trainer expects tabular dataset, got: {getattr(rr.dataset, 'type', type(rr.dataset))}")

    local_csv = StorageManager.get_local_copy(rr.dataset.uri)
    df = pd.read_csv(local_csv)

    if rr.split.method == "row_shuffle":
        train_df, val_df = row_shuffle_split(df, rr.split.test_size, rr.split.random_seed)
    else:
        group_id = make_group_id(df, rr.group_key)
        train_df, val_df = group_shuffle_split(df, group_id, rr.split.test_size, rr.split.random_seed)

    label = rr.dataset.label
    time_limit = int(rr.time_budget_s)
    metric = normalize_metric(rr.metric)
    problem_type = _autogluon_problem_type(rr, train_df, label)
    ag_extras = _autogluon_extras(rr)
    fit_args = _autogluon_fit_args(ag_extras)

    predictor = TabularPredictor(
        label=label,
        eval_metric=metric,
        problem_type=problem_type,
    )
    stop_progress = _start_progress_logger(logger, time_limit)
    try:
        predictor.fit(
            train_data=train_df,
            time_limit=time_limit,
            excluded_model_types=_excluded_models(),
            **fit_args,
        )
    finally:
        stop_progress.set()

    perf = predictor.evaluate(val_df, silent=True)
    for k, v in perf.items():
        try:
            logger.report_scalar(title="val", series=k, value=float(v), iteration=0)
        except Exception:
            pass

    tmp_dir = tempfile.mkdtemp(prefix="ag_artifacts_")
    try:
        model_dir = os.path.join(tmp_dir, "autogluon_model")
        predictor.save(model_dir)

        rr_path = os.path.join(tmp_dir, "run_request.json")
        with open(rr_path, "w", encoding="utf-8") as f:
            json.dump(rr.__dict__, f, ensure_ascii=False, default=lambda o: o.__dict__)

        if bool(ag_extras.get("leaderboard")):
            try:
                leaderboard = predictor.leaderboard(val_df, silent=True)
                leaderboard_path = os.path.join(tmp_dir, "leaderboard.csv")
                leaderboard.to_csv(leaderboard_path, index=False)
                task.upload_artifact(name="leaderboard", artifact_object=leaderboard_path, wait_on_upload=True)
            except Exception as exc:
                logger.report_text(f"leaderboard failed: {exc}")

        if bool(ag_extras.get("feature_importance")):
            try:
                fi_args = ag_extras.get("feature_importance_args") or {}
                if not isinstance(fi_args, dict):
                    fi_args = {}
                feature_importance = predictor.feature_importance(val_df, **fi_args)
                fi_path = os.path.join(tmp_dir, "feature_importance.csv")
                feature_importance.to_csv(fi_path)
                task.upload_artifact(name="feature_importance", artifact_object=fi_path, wait_on_upload=True)
            except Exception as exc:
                logger.report_text(f"feature_importance failed: {exc}")

        if bool(ag_extras.get("fit_summary")):
            try:
                summary = predictor.fit_summary()
                summary_path = os.path.join(tmp_dir, "fit_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=True, indent=2, default=str)
                task.upload_artifact(name="fit_summary", artifact_object=summary_path, wait_on_upload=True)
            except Exception as exc:
                logger.report_text(f"fit_summary failed: {exc}")

        # wait_on_upload=True avoids cleanup before ClearML finishes reading the files
        task.upload_artifact(name="model_dir", artifact_object=model_dir, wait_on_upload=True)
        task.upload_artifact(name="run_request", artifact_object=rr_path, wait_on_upload=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
