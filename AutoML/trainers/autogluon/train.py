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

from shared_lib.run_request import RunRequest
from shared_lib.grouping import make_group_id, group_shuffle_split
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

    local_csv = StorageManager.get_local_copy(rr.dataset.csv_uri)
    df = pd.read_csv(local_csv)

    group_id = make_group_id(df, rr.group_key)
    train_df, val_df = group_shuffle_split(df, group_id, rr.split.test_size, rr.split.random_seed)

    label = rr.dataset.target
    time_limit = int(rr.time_budget_s)
    metric = normalize_metric(rr.metric)
    problem_type = _autogluon_problem_type(rr, train_df, label)

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

        # wait_on_upload=True avoids cleanup before ClearML finishes reading the files
        task.upload_artifact(name="model_dir", artifact_object=model_dir, wait_on_upload=True)
        task.upload_artifact(name="run_request", artifact_object=rr_path, wait_on_upload=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
