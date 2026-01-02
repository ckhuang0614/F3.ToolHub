from __future__ import annotations
import json
import os
import shutil
import tempfile
import threading
import time
import inspect
from datetime import datetime
from typing import Optional
import pandas as pd
import joblib

from clearml import OutputModel, Task
from clearml.backend_interface.util import get_or_create_project

from flaml import AutoML

from shared_lib.clearml_s3 import apply_s3_overrides
from shared_lib.clearml_reporting import report_kv_table, report_table_from_csv
from shared_lib.run_request import RunRequest, TabularDatasetSpec
from shared_lib.dataset_resolver import resolve_tabular_dataset
from shared_lib.grouping import make_group_id, group_shuffle_split, row_shuffle_split
from shared_lib.metrics import normalize_metric
from shared_lib.task_metadata import apply_run_metadata, dataset_info


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


def _model_project() -> str:
    return os.getenv("CLEARML_MODEL_PROJECT", "AutoML-Models")


def _model_name(trainer: str, run_name: Optional[str]) -> str:
    raw = (run_name or "").strip()
    if raw:
        safe = "-".join(raw.split())
        return f"{trainer}-{safe}"
    return trainer


def _model_version(task: Task) -> str:
    date = datetime.utcnow().strftime("%Y%m%d")
    task_id = task.id or "unknown"
    return f"{date}-{task_id}"


def _resolve_output_uri(task: Task, logger) -> Optional[str]:
    env_uri = os.getenv("CLEARML_DEFAULT_OUTPUT_URI") or os.getenv("CLEARML_OUTPUT_URI")
    output_uri = env_uri.strip() if env_uri else None
    if output_uri:
        try:
            task.output_uri = output_uri
        except Exception as exc:
            logger.report_text(f"output_uri set failed: {exc}")
        return output_uri
    try:
        return task.output_uri
    except Exception:
        return None


def _resolve_project_id(task: Task, project_name: str) -> Optional[str]:
    if not project_name:
        return None
    try:
        session = task.session if task is not None else Task._get_default_session()
        return get_or_create_project(session=session, project_name=project_name)
    except Exception:
        return None


def _create_output_model(
    task: Task,
    name: str,
    project: str,
    version: str,
) -> tuple[OutputModel, Optional[str]]:
    params = inspect.signature(OutputModel.__init__).parameters
    kwargs = {"task": task, "name": name}
    version_tag = f"version:{version}"
    if "tags" in params:
        kwargs["tags"] = [version_tag]
    elif "comment" in params:
        kwargs["comment"] = f"version={version}"
    if "version" in params:
        kwargs["version"] = version
    if "project" in params:
        kwargs["project"] = project
    elif "project_name" in params:
        kwargs["project_name"] = project
    model = OutputModel(**kwargs)
    project_id = None
    if project and "project" not in params and "project_name" not in params:
        project_id = _resolve_project_id(task, project)
    return model, project_id


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
    apply_s3_overrides()
    task = Task.current_task() or Task.init(project_name="AutoML-Tabular", task_name="flaml-train")
    logger = task.get_logger()

    rr = _load_run_request(task)
    apply_run_metadata(task, rr)
    task.set_name(rr.run_name or task.name)

    if not isinstance(rr.dataset, TabularDatasetSpec):
        raise ValueError(f"flaml trainer expects tabular dataset, got: {getattr(rr.dataset, 'type', type(rr.dataset))}")

    local_csv = resolve_tabular_dataset(rr.dataset)
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
    summary = {
        "trainer": rr.trainer,
        "run_name": rr.run_name or task.name,
        "metric": rr.metric,
        "task_type": rr.task_type,
        "time_budget_s": rr.time_budget_s,
        "dataset": dataset_info(rr),
        "val_score": float(val_score),
    }

    tmp_dir = tempfile.mkdtemp(prefix="flaml_artifacts_")
    output_model_id = None
    model_name = None
    model_version = None
    model_project = None
    try:
        model_path = os.path.join(tmp_dir, "flaml_automl.pkl")
        joblib.dump(automl, model_path)
        # wait_on_upload avoids deleting the file before ClearML finishes reading it
        task.upload_artifact(name="model", artifact_object=model_path, wait_on_upload=True)
        try:
            output_uri = _resolve_output_uri(task, logger)
            model_project = _model_project()
            model_version = _model_version(task)
            model_name = _model_name("flaml", rr.run_name or task.name)
            model, project_id = _create_output_model(
                task=task,
                name=model_name,
                project=model_project,
                version=model_version,
            )
            model.update_weights(model_path, upload_uri=output_uri)
            output_model_id = getattr(model, "id", None)
            if output_model_id:
                try:
                    task.set_parameter("Model/id", output_model_id)
                    task.set_parameter("Model/name", model_name)
                    task.set_parameter("Model/project", model_project)
                    task.set_parameter("Model/version", model_version)
                except Exception:
                    pass
            if project_id:
                try:
                    model.project = project_id
                except Exception as exc:
                    logger.report_text(f"model registry set project failed: {exc}")
        except Exception as exc:
            logger.report_text(f"model registry update failed: {exc}")

        if bool(fl_extras.get("summary")):
            try:
                flaml_summary = {
                    "best_estimator": getattr(automl, "best_estimator", None),
                    "best_config": getattr(automl, "best_config", None),
                    "best_loss": getattr(automl, "best_loss", None),
                    "best_iteration": getattr(automl, "best_iteration", None),
                }
                summary_path = os.path.join(tmp_dir, "flaml_summary.json")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(flaml_summary, f, ensure_ascii=True, indent=2, default=str)
                task.upload_artifact(name="flaml_summary", artifact_object=summary_path, wait_on_upload=True)
                report_kv_table(logger, title="flaml_summary", series="flaml", data=flaml_summary)
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
                report_table_from_csv(logger, fi_path, title="feature_importance", series="flaml")
            except Exception as exc:
                logger.report_text(f"flaml_feature_importance failed: {exc}")

        if output_model_id:
            summary["model"] = {
                "id": output_model_id,
                "name": model_name,
                "project": model_project,
                "version": model_version,
            }
        summary_path = os.path.join(tmp_dir, "run_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=True, indent=2, default=str)
            task.upload_artifact(name="run_summary", artifact_object=summary_path, wait_on_upload=True)
            report_kv_table(logger, title="run_summary", series="flaml", data=summary)
        except Exception as exc:
            logger.report_text(f"run_summary failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
