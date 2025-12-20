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

from shared_lib.run_request import RunRequest
from shared_lib.grouping import make_group_id, group_shuffle_split
from shared_lib.metrics import normalize_metric


def _load_run_request(task: Task) -> RunRequest:
    s = task.get_parameter("RunRequest/json")
    if not s:
        raise ValueError("Missing ClearML parameter: RunRequest/json")
    return RunRequest.from_dict(json.loads(s))

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

    local_csv = StorageManager.get_local_copy(rr.dataset.csv_uri)
    df = pd.read_csv(local_csv)

    group_id = make_group_id(df, rr.group_key)
    train_df, val_df = group_shuffle_split(df, group_id, rr.split.test_size, rr.split.random_seed)

    y_train = train_df[rr.dataset.target]
    X_train = train_df.drop(columns=[rr.dataset.target])
    y_val = val_df[rr.dataset.target]
    X_val = val_df.drop(columns=[rr.dataset.target])

    automl = AutoML()
    stop_progress = _start_progress_logger(logger, int(rr.time_budget_s), automl)
    try:
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task=rr.task_type,
            time_budget=int(rr.time_budget_s),
            metric=normalize_metric(rr.metric),
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
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
