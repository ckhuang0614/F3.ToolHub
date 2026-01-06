from __future__ import annotations
import json
import os
import shutil
import tempfile
import threading
import time
import functools
import importlib.util
import inspect
from datetime import datetime
from typing import Optional

# Disable ClearML's torch patch before ClearML is imported to avoid fastai pickling issues
os.environ.setdefault("CLEARML_PATCH_PYTORCH", "0")

import pandas as pd
try:
    import torch
except Exception:
    torch = None

from clearml import OutputModel, Task
from clearml.backend_interface.util import get_or_create_project

from shared_lib.clearml_s3 import apply_s3_overrides
from shared_lib.clearml_reporting import report_kv_table, report_table_from_csv
from shared_lib.run_request import RunRequest, TabularDatasetSpec
from shared_lib.dataset_resolver import resolve_tabular_dataset
from shared_lib.grouping import make_group_id, group_shuffle_split, row_shuffle_split
from shared_lib.metrics import normalize_metric
from shared_lib.task_metadata import apply_run_metadata, dataset_info
from shared_lib.model_registry import apply_model_status


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

def _analysis_flags(ag_extras: dict) -> dict:
    analysis = ag_extras.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}
    return {
        "summary": bool(analysis.get("summary")),
        "corr": bool(analysis.get("corr")),
        "mutual_info": bool(analysis.get("mutual_info")),
        "target_corr": bool(analysis.get("target_corr")),
        "shap": bool(analysis.get("shap")),
    }

def _autogluon_mode(ag_extras: dict) -> str:
    raw = ag_extras.get("mode")
    if raw in (None, ""):
        return "tabular"
    mode = str(raw).strip().lower()
    if mode not in {"tabular", "multimodal", "timeseries"}:
        raise ValueError("extras.autogluon.mode must be one of: tabular, multimodal, timeseries")
    return mode

def _split_tabular(rr: RunRequest, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rr.split.method == "row_shuffle":
        return row_shuffle_split(df, rr.split.test_size, rr.split.random_seed)
    group_id = make_group_id(df, rr.group_key)
    return group_shuffle_split(df, group_id, rr.split.test_size, rr.split.random_seed)

def _autogluon_timeseries_config(rr: RunRequest, ag_extras: dict) -> dict:
    ts = ag_extras.get("timeseries") or {}
    if not isinstance(ts, dict):
        raise ValueError("extras.autogluon.timeseries must be a dict")
    item_id = ts.get("item_id") or ts.get("id_column") or "item_id"
    timestamp = ts.get("timestamp") or ts.get("time_column") or "timestamp"
    target = ts.get("target") or rr.dataset.label
    prediction_length = ts.get("prediction_length")
    if prediction_length in (None, ""):
        raise ValueError("timeseries mode requires extras.autogluon.timeseries.prediction_length")
    predictor_args = ts.get("predictor_args") or {}
    if not isinstance(predictor_args, dict):
        predictor_args = {}
    allow_unsafe = bool(ts.get("allow_unsafe_torch_load") or ag_extras.get("allow_unsafe_torch_load"))
    return {
        "item_id": str(item_id),
        "timestamp": str(timestamp),
        "target": str(target),
        "prediction_length": int(prediction_length),
        "predictor_args": predictor_args,
        "allow_unsafe_torch_load": allow_unsafe,
    }

def _ensure_datetime(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors="coerce")

def _split_timeseries_data(ts_data, prediction_length: int):
    for name in ("split_train_test", "train_test_split"):
        split_fn = getattr(ts_data, name, None)
        if not callable(split_fn):
            continue
        try:
            return split_fn(prediction_length=prediction_length)
        except TypeError:
            try:
                return split_fn(prediction_length)
            except Exception:
                continue
        except Exception:
            continue
    return ts_data, None

def _evaluate_predictor(predictor, eval_data):
    if eval_data is None:
        return None
    try:
        return predictor.evaluate(eval_data, silent=True)
    except TypeError:
        return predictor.evaluate(eval_data)
    except Exception:
        return None

def _allow_torch_safe_globals() -> None:
    if torch is None:
        return
    try:
        serialization = getattr(torch, "serialization", None)
        add_safe = getattr(serialization, "add_safe_globals", None)
        if callable(add_safe):
            allowlist = [functools.partial, getattr, object]
            try:
                from gluonts.torch.distributions.quantile_output import QuantileOutput
            except Exception:
                QuantileOutput = None
            if QuantileOutput is not None:
                allowlist.append(QuantileOutput)
            add_safe(allowlist)
    except Exception:
        pass

def _disable_torch_weights_only() -> None:
    if torch is None:
        return
    if getattr(torch.load, "_clearml_weights_only_patched", False):
        return
    original = torch.load

    @functools.wraps(original)
    def _wrapped(*args, **kwargs):
        if "weights_only" not in kwargs or kwargs["weights_only"] is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    _wrapped._clearml_weights_only_patched = True
    torch.load = _wrapped

def _run_tabular_analysis(
    df: pd.DataFrame,
    label: str,
    problem_type: str,
    predictor,
    output_dir: str,
    flags: dict,
    logger,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}
    if df.empty:
        logger.report_text("analysis skipped: dataframe is empty")
        return outputs

    if flags.get("summary"):
        summary = df.describe(include="all").transpose()
        summary["missing_count"] = df.isna().sum()
        summary["missing_ratio"] = (df.isna().mean()).round(6)
        summary["dtype"] = df.dtypes.astype(str)
        summary["nunique"] = df.nunique(dropna=True)
        summary_path = os.path.join(output_dir, "summary.csv")
        summary.to_csv(summary_path)
        outputs["summary"] = summary_path

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    feature_numeric = [c for c in numeric_cols if c != label]

    if flags.get("corr"):
        if feature_numeric:
            corr_path = os.path.join(output_dir, "corr.csv")
            df[feature_numeric].corr().to_csv(corr_path)
            outputs["corr"] = corr_path
        else:
            logger.report_text("analysis corr skipped: no numeric features")

    mi_values = None
    if flags.get("mutual_info") or flags.get("target_corr"):
        if feature_numeric and label in df.columns:
            X = df[feature_numeric].copy()
            X = X.fillna(X.median(numeric_only=True))
            y = df[label]
            if problem_type in {"binary", "multiclass", "classification"}:
                y_enc = pd.Categorical(y).codes
                try:
                    from sklearn.feature_selection import mutual_info_classif
                    mi = mutual_info_classif(X, y_enc, discrete_features=False, random_state=42)
                    mi_values = pd.Series(mi, index=feature_numeric)
                except Exception as exc:
                    logger.report_text(f"analysis mutual_info failed: {exc}")
            else:
                if y.dtype.kind not in {"i", "u", "f"}:
                    y = pd.to_numeric(y, errors="coerce")
                try:
                    from sklearn.feature_selection import mutual_info_regression
                    mi = mutual_info_regression(X, y, random_state=42)
                    mi_values = pd.Series(mi, index=feature_numeric)
                except Exception as exc:
                    logger.report_text(f"analysis mutual_info failed: {exc}")
        else:
            logger.report_text("analysis mutual_info skipped: missing label or numeric features")

    if flags.get("mutual_info") and mi_values is not None:
        mi_path = os.path.join(output_dir, "mutual_info.csv")
        mi_values.sort_values(ascending=False).rename("mutual_info").to_csv(mi_path)
        outputs["mutual_info"] = mi_path

    if flags.get("target_corr"):
        if label in df.columns and feature_numeric:
            if df[label].dtype.kind in {"i", "u", "f"}:
                target_corr = df[feature_numeric].corrwith(df[label]).sort_values(ascending=False)
                target_path = os.path.join(output_dir, "target_corr.csv")
                target_corr.rename("pearson_corr").to_csv(target_path)
                outputs["target_corr"] = target_path
            elif mi_values is not None:
                target_path = os.path.join(output_dir, "target_corr.csv")
                mi_values.sort_values(ascending=False).rename("mutual_info").to_csv(target_path)
                outputs["target_corr"] = target_path
            else:
                logger.report_text("analysis target_corr skipped: no numeric target")
        else:
            logger.report_text("analysis target_corr skipped: missing label or numeric features")

    if flags.get("shap"):
        if importlib.util.find_spec("shap") is None:
            logger.report_text("analysis shap skipped: shap is not installed")
        else:
            try:
                import numpy as np
                import shap
                feature_cols = [c for c in df.columns if c != label]
                if not feature_cols:
                    raise ValueError("no feature columns for shap")
                non_numeric = df[feature_cols].select_dtypes(exclude="number").columns.tolist()
                if non_numeric:
                    preview = ", ".join(non_numeric[:5])
                    suffix = "..." if len(non_numeric) > 5 else ""
                    logger.report_text(
                        f"analysis shap skipped: non-numeric features present ({preview}{suffix})"
                    )
                    return outputs
                sample = df[feature_cols].head(200).copy()
                for col in sample.columns:
                    sample[col] = sample[col].fillna(sample[col].median())

                def _predict(data):
                    if not isinstance(data, pd.DataFrame):
                        data = pd.DataFrame(data, columns=feature_cols)
                    if problem_type in {"binary", "multiclass", "classification"}:
                        return predictor.predict_proba(data)
                    return predictor.predict(data)

                background = sample.head(min(50, len(sample)))
                explainer = shap.Explainer(_predict, background)
                shap_values = explainer(sample)
                values = getattr(shap_values, "values", None)
                if values is None:
                    raise ValueError("shap values missing")
                vals = np.array(values)
                if vals.ndim == 3:
                    mean_abs = np.mean(np.abs(vals), axis=(0, 1))
                elif vals.ndim == 2:
                    mean_abs = np.mean(np.abs(vals), axis=0)
                else:
                    raise ValueError(f"unsupported shap values shape: {vals.shape}")
                shap_path = os.path.join(output_dir, "shap_importance.csv")
                pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False).to_csv(shap_path, header=["shap_importance"])
                outputs["shap_importance"] = shap_path
            except Exception as exc:
                logger.report_text(f"analysis shap failed: {exc}")

    return outputs

def _save_predictor(predictor, mode: str, default_dir: str) -> str:
    if mode == "timeseries":
        predictor.save()
        ts_path = getattr(predictor, "path", None)
        if ts_path:
            return str(ts_path)
        return default_dir
    try:
        predictor.save()
    except TypeError:
        predictor.save(default_dir)
        if os.path.exists(default_dir):
            return default_dir
    save_path = getattr(predictor, "path", None)
    if save_path:
        return str(save_path)
    return default_dir

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


def main():
    # Avoid ClearML's torch patch interfering with fastai save
    os.environ.setdefault("CLEARML_PATCH_PYTORCH", "0")
    apply_s3_overrides()

    task = Task.current_task() or Task.init(
        project_name="AutoML-Tabular",
        task_name="autogluon-train",
        auto_connect_frameworks=False,
    )
    logger = task.get_logger()

    rr = _load_run_request(task)
    apply_run_metadata(task, rr)
    task.set_name(rr.run_name or task.name)

    if not isinstance(rr.dataset, TabularDatasetSpec):
        raise ValueError(f"autogluon trainer expects tabular dataset, got: {getattr(rr.dataset, 'type', type(rr.dataset))}")

    local_csv = resolve_tabular_dataset(rr.dataset)
    df = pd.read_csv(local_csv)

    ag_extras = _autogluon_extras(rr)
    fit_args = _autogluon_fit_args(ag_extras)
    analysis_flags = _analysis_flags(ag_extras)
    mode = _autogluon_mode(ag_extras)
    time_limit = int(rr.time_budget_s)
    metric = normalize_metric(rr.metric)

    predictor = None
    eval_data = None
    analysis_df = None
    analysis_label = None
    analysis_problem_type = None
    if mode in {"tabular", "multimodal"}:
        train_df, val_df = _split_tabular(rr, df)
        label = rr.dataset.label
        if label not in train_df.columns:
            raise ValueError(f"label column not found in dataset: {label}")
        problem_type = _autogluon_problem_type(rr, train_df, label)
        if mode == "multimodal":
            from autogluon.multimodal import MultiModalPredictor

            predictor = MultiModalPredictor(
                label=label,
                eval_metric=metric,
                problem_type=problem_type,
            )
        else:
            from autogluon.tabular import TabularPredictor

            predictor = TabularPredictor(
                label=label,
                eval_metric=metric,
                problem_type=problem_type,
            )
        eval_data = val_df
        analysis_df = train_df
        analysis_label = label
        analysis_problem_type = problem_type
    else:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        ts_cfg = _autogluon_timeseries_config(rr, ag_extras)
        for col in (ts_cfg["item_id"], ts_cfg["timestamp"], ts_cfg["target"]):
            if col not in df.columns:
                raise ValueError(f"timeseries column not found in dataset: {col}")
        _ensure_datetime(df, ts_cfg["timestamp"])
        ts_data = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column=ts_cfg["item_id"],
            timestamp_column=ts_cfg["timestamp"],
        )
        train_ts, val_ts = _split_timeseries_data(ts_data, ts_cfg["prediction_length"])
        predictor = TimeSeriesPredictor(
            target=ts_cfg["target"],
            prediction_length=ts_cfg["prediction_length"],
            eval_metric=metric,
            **ts_cfg["predictor_args"],
        )
        eval_data = val_ts
    stop_progress = _start_progress_logger(logger, time_limit)
    try:
        if mode == "tabular":
            predictor.fit(
                train_data=train_df,
                time_limit=time_limit,
                excluded_model_types=_excluded_models(),
                **fit_args,
            )
        elif mode == "multimodal":
            predictor.fit(
                train_data=train_df,
                time_limit=time_limit,
                tuning_data=eval_data,
                **fit_args,
            )
        else:
            _allow_torch_safe_globals()
            if ts_cfg.get("allow_unsafe_torch_load"):
                _disable_torch_weights_only()
            fit_kwargs = {
                "train_data": train_ts,
                "time_limit": time_limit,
                **fit_args,
            }
            if eval_data is not None:
                fit_kwargs["tuning_data"] = eval_data
            predictor.fit(**fit_kwargs)
    finally:
        stop_progress.set()

    perf = _evaluate_predictor(predictor, eval_data)
    summary = {
        "trainer": rr.trainer,
        "run_name": rr.run_name or task.name,
        "metric": rr.metric,
        "task_type": rr.task_type,
        "time_budget_s": rr.time_budget_s,
        "dataset": dataset_info(rr),
    }
    if perf:
        for k, v in perf.items():
            try:
                logger.report_scalar(title="val", series=k, value=float(v), iteration=0)
            except Exception:
                pass
        summary["eval_metrics"] = perf

    tmp_dir = tempfile.mkdtemp(prefix="ag_artifacts_")
    output_model_id = None
    model_name = None
    model_version = None
    model_project = None
    try:
        model_dir = os.path.join(tmp_dir, "autogluon_model")
        model_dir = _save_predictor(predictor, mode, model_dir)

        rr_path = os.path.join(tmp_dir, "run_request.json")
        with open(rr_path, "w", encoding="utf-8") as f:
            json.dump(rr.__dict__, f, ensure_ascii=False, default=lambda o: o.__dict__)

        if bool(ag_extras.get("leaderboard")):
            try:
                leaderboard = predictor.leaderboard(eval_data, silent=True)
                leaderboard_path = os.path.join(tmp_dir, "leaderboard.csv")
                leaderboard.to_csv(leaderboard_path, index=False)
                task.upload_artifact(name="leaderboard", artifact_object=leaderboard_path, wait_on_upload=True)
                report_table_from_csv(logger, leaderboard_path, title="leaderboard", series="autogluon")
                if not leaderboard.empty:
                    best = leaderboard.iloc[0].to_dict()
                    best_info = {}
                    for key in ("model", "score_val", "eval_metric", "metric"):
                        if key in best:
                            best_info[key] = best[key]
                    if best_info:
                        summary["best_model"] = best_info
            except Exception as exc:
                logger.report_text(f"leaderboard failed: {exc}")

        if bool(ag_extras.get("feature_importance")):
            try:
                fi_args = ag_extras.get("feature_importance_args") or {}
                if not isinstance(fi_args, dict):
                    fi_args = {}
                feature_importance = predictor.feature_importance(eval_data, **fi_args)
                fi_path = os.path.join(tmp_dir, "feature_importance.csv")
                feature_importance.to_csv(fi_path)
                task.upload_artifact(name="feature_importance", artifact_object=fi_path, wait_on_upload=True)
                report_table_from_csv(logger, fi_path, title="feature_importance", series="autogluon")
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

        if mode == "tabular" and analysis_df is not None and analysis_label and analysis_problem_type:
            if any(analysis_flags.values()):
                analysis_dir = os.path.join(tmp_dir, "analysis")
                outputs = _run_tabular_analysis(
                    df=analysis_df,
                    label=analysis_label,
                    problem_type=analysis_problem_type,
                    predictor=predictor,
                    output_dir=analysis_dir,
                    flags=analysis_flags,
                    logger=logger,
                )
                for name, path in outputs.items():
                    if os.path.exists(path):
                        task.upload_artifact(name=name, artifact_object=path, wait_on_upload=True)

        # wait_on_upload=True avoids cleanup before ClearML finishes reading the files
        if os.path.exists(model_dir):
            task.upload_artifact(name="model_dir", artifact_object=model_dir, wait_on_upload=True)
            try:
                zip_base = os.path.join(tmp_dir, "model_registry")
                zip_path = shutil.make_archive(zip_base, "zip", model_dir)
                if zip_path and os.path.exists(zip_path):
                    output_uri = _resolve_output_uri(task, logger)
                    model_project = _model_project()
                    model_version = _model_version(task)
                    model_name = _model_name("autogluon", rr.run_name or task.name)
                    model, project_id = _create_output_model(
                        task=task,
                        name=model_name,
                        project=model_project,
                        version=model_version,
                    )
                    model.update_weights(zip_path, upload_uri=output_uri)
                    apply_model_status(model, logger)
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
        else:
            logger.report_text(f"model_dir not found for upload: {model_dir}")
        task.upload_artifact(name="run_request", artifact_object=rr_path, wait_on_upload=True)

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
            report_kv_table(logger, title="run_summary", series="autogluon", data=summary)
        except Exception as exc:
            logger.report_text(f"run_summary failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    task.close()


if __name__ == "__main__":
    main()
