import argparse
import ast
import json
import os
import re
import unicodedata
import shutil
import tempfile
import zipfile
import inspect
from collections.abc import Mapping
from datetime import datetime
from typing import Optional, Tuple

from clearml import InputModel, OutputModel, Task
from clearml.backend_interface.util import get_or_create_project
from clearml.storage import StorageManager

from shared_lib.clearml_s3 import apply_s3_overrides
from shared_lib.clearml_reporting import report_kv_table
from shared_lib.dataset_resolver import resolve_yolo_dataset_uri
from shared_lib.run_request import RunRequest, YoloDatasetSpec
from shared_lib.task_metadata import apply_run_metadata, dataset_info

# Minimal ultralytics training script template using ClearML
# This script uses ultralytics (YOLOv8) as an example. Adjust imports and training code
# if you prefer another YOLO implementation.


def parse_args():
    parser = argparse.ArgumentParser(description="ultralytics training with ClearML")
    parser.add_argument("--data", type=str, help="Path or s3 uri to dataset yaml or zip")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


DEFAULTS = {
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "project": "ultralytics",
    "name": "yolov8-train-demo",
    "device": "0",
    "weights": "yolov8n.pt",
    "workers": 8,
}


def _strip_control_chars(value: str) -> str:
    return "".join(ch for ch in value if ch >= " " or ch in "\t\n\r")


def _strip_invisible_chars(value: str) -> str:
    cleaned = []
    for ch in value:
        cat = unicodedata.category(ch)
        if cat and cat[0] == "C":
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def _ascii_sanitize(value: str) -> str:
    return "".join(ch if 32 <= ord(ch) <= 126 or ch in "\t\n\r" else " " for ch in value)


def _coerce_raw_string(raw: object) -> Optional[str]:
    if raw is None:
        return None
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        return raw
    try:
        return json.dumps(raw, ensure_ascii=False)
    except Exception:
        return str(raw)


def _extract_uri_fallback(raw: str) -> Optional[str]:
    ascii_raw = _ascii_sanitize(raw)
    compact = re.sub(r"\s+", "", ascii_raw)
    for pattern in (
        r"(s3://[^\"',}\]]+)",
        r"((?:https?|file)://[^\"',}\]]+)",
        r"([^\"',}\]]+\.zip)",
        r"([^\"',}\]]+\.ya?ml)",
    ):
        match = re.search(pattern, compact)
        if match:
            return match.group(1)
    return None


def _extract_raw_string(raw: str, key: str) -> Optional[str]:
    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"]*)"'
    match = re.search(pattern, raw)
    if match:
        return match.group(1)
    pattern = rf"'{re.escape(key)}'\s*:\s*'([^']*)'"
    match = re.search(pattern, raw)
    if match:
        return match.group(1)
    return None


def _extract_raw_number(raw: str, key: str) -> Optional[float]:
    pattern = rf'"{re.escape(key)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
    match = re.search(pattern, raw)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    pattern = rf"'{re.escape(key)}'\s*:\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, raw)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            return None
    return None

def _maybe_parse_payload(raw: object) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        for item in raw:
            payload = _maybe_parse_payload(item)
            if payload:
                return payload
        return None
    if isinstance(raw, Mapping):
        raw_dict = dict(raw)
        if "trainer" in raw_dict and "dataset" in raw_dict:
            return raw_dict
        if "value" in raw_dict:
            payload = _maybe_parse_payload(raw_dict.get("value"))
            if payload:
                return payload
        for key in ("RunRequest", "run_request", "json"):
            if key in raw_dict:
                payload = _maybe_parse_payload(raw_dict.get(key))
                if payload:
                    return payload
        return None
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="ignore")
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("\ufeff"):
            raw = raw.lstrip("\ufeff")
        if not raw:
            return None
        candidates = [raw]
        cleaned = _strip_invisible_chars(_strip_control_chars(raw))
        ascii_cleaned = _ascii_sanitize(cleaned)
        if cleaned != raw:
            candidates.append(cleaned)
        if ascii_cleaned != cleaned:
            candidates.append(ascii_cleaned)
        if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
            candidates.append(raw[1:-1])
        if "{" in raw and "}" in raw:
            start = raw.find("{")
            end = raw.rfind("}")
            if end > start:
                candidates.append(raw[start : end + 1])
        for cand in candidates:
            cand = _strip_invisible_chars(_strip_control_chars(cand.strip()))
            if not cand:
                continue
            try:
                payload = json.loads(cand)
            except Exception:
                payload = None
            if payload is None:
                try:
                    payload = json.JSONDecoder(strict=False).decode(cand)
                except Exception:
                    payload = None
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, (list, tuple)):
                for item in payload:
                    nested = _maybe_parse_payload(item)
                    if nested:
                        return nested
            if isinstance(payload, str):
                nested = _maybe_parse_payload(payload)
                if nested:
                    return nested
            # Fallback for python-literal dicts (single quotes, etc.)
            try:
                payload = ast.literal_eval(cand)
            except Exception:
                payload = None
            if isinstance(payload, dict):
                return payload
            if isinstance(payload, (list, tuple)):
                for item in payload:
                    nested = _maybe_parse_payload(item)
                    if nested:
                        return nested
            if isinstance(payload, str):
                nested = _maybe_parse_payload(payload)
                if nested:
                    return nested
    return None


def _iter_param_values(params: object):
    if isinstance(params, dict):
        for value in params.values():
            if isinstance(value, dict):
                yield from _iter_param_values(value)
            else:
                yield value


def _find_payload_in_params(params: object) -> Optional[dict]:
    for value in _iter_param_values(params):
        payload = _maybe_parse_payload(value)
        if payload and "trainer" in payload and "dataset" in payload:
            return payload
    return None


def _load_run_request(task: Task) -> Tuple[Optional[RunRequest], Optional[dict]]:
    payload = None
    for key in ("RunRequest/json", "RunRequest.json", "RunRequest_json"):
        try:
            payload = _maybe_parse_payload(_get_task_parameter(task, key))
        except Exception:
            payload = None
        if payload:
            break

    if not payload:
        try:
            payload = _maybe_parse_payload(_get_task_parameter(task, "RunRequest"))
        except Exception:
            payload = None

    if not payload:
        params = None
        try:
            if hasattr(task, "get_parameters_as_dict"):
                params = task.get_parameters_as_dict()
            else:
                params = task.get_parameters()
        except Exception:
            params = None
        payload = _find_payload_in_params(params)

    if not payload:
        return None, None

    try:
        rr = RunRequest.from_dict(payload)
    except Exception as exc:
        print(f"Warning: failed to parse RunRequest payload ({exc}); continuing with raw payload")
        rr = None
    return rr, payload


def _yolo_extras(rr: Optional[RunRequest]) -> dict:
    if not rr:
        return {}
    extras = rr.extras or {}
    if not isinstance(extras, dict):
        return {}
    y = extras.get("yolo") or {}
    if not isinstance(y, dict):
        return {}
    return y


def _yolo_extras_from_payload(payload: dict) -> dict:
    extras = payload.get("extras") or {}
    if not isinstance(extras, dict):
        return {}
    y = extras.get("yolo") or {}
    if not isinstance(y, dict):
        return {}
    return y


def _yolo_extras_from_raw(raw: str) -> dict:
    raw = _strip_invisible_chars(_strip_control_chars(raw))
    extras = {}
    for key in ("epochs", "batch", "imgsz", "workers"):
        value = _extract_raw_number(raw, key)
        if value is not None:
            extras[key] = int(value)
    for key in ("device", "weights", "model"):
        value = _extract_raw_string(raw, key)
        if value:
            extras[key] = value
    return extras


def _dataset_from_payload(payload: dict) -> Tuple[Optional[str], Optional[str]]:
    dataset = payload.get("dataset")
    if isinstance(dataset, str):
        return dataset, None
    if isinstance(dataset, dict):
        uri = (
            dataset.get("uri")
            or dataset.get("csv_uri")
            or dataset.get("data")
            or dataset.get("dataset_uri")
            or dataset.get("path")
        )
        yaml_hint = dataset.get("yaml_path") or dataset.get("yaml") or dataset.get("target")
        return (str(uri) if uri else None, str(yaml_hint) if yaml_hint else None)

    uri = (
        payload.get("data")
        or payload.get("data_uri")
        or payload.get("dataset_uri")
        or payload.get("uri")
        or payload.get("csv_uri")
    )
    yaml_hint = payload.get("yaml_path") or payload.get("yaml") or payload.get("target")
    return (str(uri) if uri else None, str(yaml_hint) if yaml_hint else None)


def _dataset_from_raw(raw: str) -> Tuple[Optional[str], Optional[str]]:
    raw = _strip_invisible_chars(_strip_control_chars(raw))
    uri = (
        _extract_raw_string(raw, "uri")
        or _extract_raw_string(raw, "csv_uri")
        or _extract_raw_string(raw, "data")
        or _extract_raw_string(raw, "dataset_uri")
        or _extract_raw_string(raw, "path")
    )
    if not uri:
        uri = _extract_uri_fallback(raw)
    yaml_hint = _extract_raw_string(raw, "yaml_path") or _extract_raw_string(raw, "yaml") or _extract_raw_string(raw, "target")
    if not yaml_hint:
        ascii_raw = _ascii_sanitize(raw)
        match = re.search(r"([A-Za-z0-9_.-]+\.ya?ml)", ascii_raw)
        if match:
            yaml_hint = match.group(1)
    return (uri, yaml_hint)


def _collect_param_keys(params: object, prefix: str = "") -> list:
    keys = []
    if isinstance(params, dict):
        for key, value in params.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                keys.extend(_collect_param_keys(value, name))
            else:
                keys.append(name)
    return keys


def _task_arg(task: Optional[Task], name: str) -> Optional[str]:
    if task is None:
        return None
    try:
        value = task.get_parameter(f"Args/{name}")
    except Exception:
        return None
    if value in (None, ""):
        return None
    return str(value)


def _get_task_parameter(task: Optional[Task], key: str) -> Optional[object]:
    if task is None:
        return None
    try:
        value = task.get_parameter(key)
    except Exception:
        value = None
    if value not in (None, ""):
        return value
    params = None
    try:
        if hasattr(task, "get_parameters_as_dict"):
            params = task.get_parameters_as_dict()
        else:
            params = task.get_parameters()
    except Exception:
        params = None
    if isinstance(params, dict):
        if key in params:
            return params.get(key)
        parts = key.split("/")
        current = params
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current.get(part)
            else:
                current = None
                break
        if current not in (None, ""):
            return current
    return None


def _debug(message: str) -> None:
    print(f"[debug] {message}")


def _find_yaml(root_dir: str) -> Optional[str]:
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith((".yaml", ".yml")):
                return os.path.join(root, name)
    return None


def _find_named_file(root_dir: str, name: str) -> Optional[str]:
    target = name.lower()
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower() == target:
                return os.path.join(root, fname)
    return None


def _resolve_data_path(data_uri: str, target: Optional[str]) -> Tuple[str, Optional[str]]:
    local_path = data_uri
    if "://" in data_uri and not os.path.exists(data_uri):
        local_path = StorageManager.get_local_copy(data_uri)

    if os.path.isdir(local_path):
        if target:
            candidate = os.path.join(local_path, target)
            if os.path.exists(candidate):
                return candidate, None
            named = _find_named_file(local_path, target)
            if named:
                return named, None
        yaml_path = _find_yaml(local_path)
        if yaml_path:
            return yaml_path, None
        raise ValueError(f"Dataset directory does not contain a yaml file: {local_path}")

    if local_path.lower().endswith(".zip"):
        extract_dir = tempfile.mkdtemp(prefix="yolo_data_")
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(extract_dir)
        if target:
            candidate = os.path.join(extract_dir, target)
            if os.path.exists(candidate):
                return candidate, extract_dir
            raise ValueError(f"Dataset target not found in zip: {candidate}")
        yaml_path = _find_yaml(extract_dir)
        if yaml_path:
            return yaml_path, extract_dir
        raise ValueError("Dataset zip does not contain a yaml file")

    return local_path, None


def _patch_final_eval(data_uri: str, data_path: str) -> None:
    try:
        from ultralytics.engine import trainer as trainer_mod
    except Exception:
        return

    candidates = []
    for name in ("BaseTrainer", "Trainer"):
        cls = getattr(trainer_mod, name, None)
        if cls and hasattr(cls, "final_eval"):
            candidates.append(cls)
    if not candidates:
        return

    def _wrap_final_eval(orig_final_eval):
        def _final_eval(self):
            try:
                current_data = getattr(self.args, "data", None)
                if (
                    data_path
                    and os.path.exists(data_path)
                    and (current_data == data_uri or (isinstance(current_data, str) and "://" in current_data))
                ):
                    self.args.data = data_path
                    if getattr(self, "validator", None) and hasattr(self.validator, "args"):
                        self.validator.args.data = data_path
            except Exception:
                pass

            try:
                return orig_final_eval(self)
            except FileNotFoundError as exc:
                if data_uri and data_uri in str(exc):
                    try:
                        self.args.data = data_path
                        if getattr(self, "validator", None) and hasattr(self.validator, "args"):
                            self.validator.args.data = data_path
                    except Exception:
                        pass
                    return orig_final_eval(self)
                raise

        _final_eval._clearml_patched = True
        return _final_eval

    for cls in candidates:
        if getattr(cls.final_eval, "_clearml_patched", False):
            continue
        cls.final_eval = _wrap_final_eval(cls.final_eval)


def _build_train_args(
    data_path: str,
    epochs: int,
    batch: int,
    imgsz: int,
    device: str,
    project: str,
    name: str,
    workers: int,
    yolo_extras: Optional[dict],
) -> dict:
    if isinstance(yolo_extras, dict):
        extra_train_args = {
            k: v
            for k, v in yolo_extras.items()
            if v is not None and k not in {"weights", "model"}
        }
    else:
        extra_train_args = {}
    return {
        **extra_train_args,
        "data": data_path,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "project": project,
        "name": name,
        "workers": workers,
        # keep val True so callbacks set the local path before eval
        "val": True,
    }

def _is_remote_uri(value: str) -> bool:
    return "://" in value and not value.startswith("file://")


def _resolve_weights(weights: str, task: Task, logger) -> str:
    if not weights:
        return weights
    raw = str(weights)
    if not _is_remote_uri(raw):
        return raw
    try:
        local = StorageManager.get_local_copy(raw)
    except Exception as exc:
        raise ValueError(f"Failed to download weights: {raw} ({exc})") from exc
    if not local:
        raise ValueError(f"Failed to resolve weights: {raw}")
    try:
        input_model = InputModel.import_model(weights_url=raw, framework="PyTorch")
        if input_model and getattr(input_model, "id", None):
            task.set_input_model(model_id=input_model.id, name="weights")
    except Exception as exc:
        logger.report_text(f"input model import failed: {exc}")
    return local


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
    apply_s3_overrides()
    args = parse_args()

    raw_payload = None
    task = Task.current_task()
    if task is None:
        # local debug mode (not running under clearml-agent)
        task = Task.init(
            project_name=args.project or DEFAULTS["project"],
            task_name=args.name or DEFAULTS["name"],
            task_type=Task.TaskTypes.training,
        )
        rr = None
        payload = None
    else:
        rr, payload = _load_run_request(task)
        if rr:
            apply_run_metadata(task, rr)
        if rr and rr.run_name:
            task.set_name(rr.run_name)
        try:
            raw_payload = _get_task_parameter(task, "RunRequest/json")
        except Exception:
            raw_payload = None
        if payload is None and raw_payload is not None:
            payload = _maybe_parse_payload(raw_payload)
        if payload is None and raw_payload is not None:
            raw_text = _coerce_raw_string(raw_payload)
            if raw_text:
                try:
                    payload = json.loads(_ascii_sanitize(raw_text))
                except Exception:
                    payload = None
            if payload and rr is None:
                try:
                    rr = RunRequest.from_dict(payload)
                except Exception as exc:
                    print(f"Warning: failed to parse sanitized RunRequest payload ({exc}); continuing with raw payload")

    yolo_extras = _yolo_extras(rr)
    if not yolo_extras and payload:
        yolo_extras = _yolo_extras_from_payload(payload)
    raw_payload_text = _coerce_raw_string(raw_payload)
    if not yolo_extras and raw_payload_text:
        yolo_extras = _yolo_extras_from_raw(raw_payload_text)

    data_uri = None
    yaml_hint: Optional[str] = None
    data_source = None
    rr_data_uri = None
    rr_yaml_hint = None
    payload_data_uri = None
    payload_yaml_hint = None
    args_data_uri = args.data if args.data else None
    env_data_uri = os.getenv("YOLO_DATA_URI")
    env_data_alt = os.getenv("YOLO_DATA")
    env_data_uri_alt = os.getenv("DATA_URI")
    env_data_alt2 = os.getenv("DATA")
    task_arg_data = None
    raw_payload_uri = None
    raw_payload_yaml = None
    parsed_payload_uri = None
    parsed_payload_yaml = None
    if rr is not None:
        if not isinstance(rr.dataset, YoloDatasetSpec):
            raise ValueError(f"ultralytics trainer expects yolo dataset, got: {getattr(rr.dataset, 'type', type(rr.dataset))}")
        rr_data_uri = rr.dataset.uri
        rr_yaml_hint = rr.dataset.yaml_path
        try:
            data_uri = resolve_yolo_dataset_uri(rr.dataset)
        except Exception as exc:
            raise ValueError(f"Failed to resolve RunRequest dataset: {exc}") from exc
        yaml_hint = rr_yaml_hint
        if rr.dataset.clearml or (rr_data_uri and rr_data_uri.startswith("clearml://")):
            data_source = "rr.dataset.clearml"
        else:
            data_source = "rr.dataset.uri"
    else:
        data_uri = None
        yaml_hint = None

    if payload:
        raw_uri, raw_yaml = _dataset_from_payload(payload)
        payload_data_uri = raw_uri
        payload_yaml_hint = raw_yaml
    if not data_uri and payload_data_uri:
        data_uri = payload_data_uri
        yaml_hint = payload_yaml_hint or yaml_hint
        data_source = "payload.dataset.uri"

    if not data_uri and args_data_uri:
        data_uri = args_data_uri
        data_source = "args.data"

    if not data_uri:
        data_uri = env_data_uri or env_data_alt or env_data_uri_alt or env_data_alt2
        if data_uri:
            if env_data_uri:
                data_source = "env.YOLO_DATA_URI"
            elif env_data_alt:
                data_source = "env.YOLO_DATA"
            elif env_data_uri_alt:
                data_source = "env.DATA_URI"
            else:
                data_source = "env.DATA"

    if not data_uri and task is not None:
        try:
            arg_data = task.get_parameter("Args/data")
            if arg_data:
                task_arg_data = str(arg_data)
                data_uri = task_arg_data
                data_source = "Args/data"
        except Exception:
            pass

    if raw_payload_text:
        raw_uri, raw_yaml = _dataset_from_raw(raw_payload_text)
        raw_payload_uri = raw_uri
        raw_payload_yaml = raw_yaml
    if not data_uri and raw_payload_uri:
        data_uri = raw_payload_uri
        yaml_hint = raw_payload_yaml or yaml_hint
        data_source = "raw_payload.regex"
    if not data_uri and raw_payload_text:
        parsed_payload = _maybe_parse_payload(raw_payload_text)
        if isinstance(parsed_payload, dict):
            parsed_uri, parsed_yaml = _dataset_from_payload(parsed_payload)
            parsed_payload_uri = parsed_uri
            parsed_payload_yaml = parsed_yaml
            if parsed_payload_uri:
                data_uri = parsed_payload_uri
                yaml_hint = parsed_payload_yaml or yaml_hint
                data_source = "raw_payload.parse"
    if not data_uri and task is not None:
        late_raw = _get_task_parameter(task, "RunRequest/json")
        if late_raw is None:
            try:
                late_raw = task.get_parameter("RunRequest/json")
            except Exception:
                late_raw = None
        late_text = _coerce_raw_string(late_raw)
        if late_text:
            if raw_payload is None:
                raw_payload = late_raw
            if raw_payload_text is None:
                raw_payload_text = late_text
            late_payload = _maybe_parse_payload(late_raw) or _maybe_parse_payload(late_text)
            if late_payload and payload is None:
                payload = late_payload
                if rr is None:
                    try:
                        rr = RunRequest.from_dict(payload)
                    except Exception as exc:
                        print(f"Warning: failed to parse late RunRequest payload ({exc}); continuing with raw payload")
            if payload:
                late_uri, late_yaml = _dataset_from_payload(payload)
                if late_uri and not data_uri:
                    data_uri = late_uri
                    yaml_hint = late_yaml or yaml_hint
                    data_source = "RunRequest/json late payload"
            if not data_uri and late_text:
                late_uri, late_yaml = _dataset_from_raw(late_text)
                if late_uri:
                    data_uri = late_uri
                    yaml_hint = late_yaml or yaml_hint
                    data_source = "RunRequest/json late regex"
            if not yolo_extras and payload:
                yolo_extras = _yolo_extras_from_payload(payload)
            if not yolo_extras and raw_payload_text:
                yolo_extras = _yolo_extras_from_raw(raw_payload_text)
    if data_uri:
        _debug(f"Dataset resolved from {data_source}: {data_uri}")
        if yaml_hint:
            _debug(f"Dataset yaml hint: {yaml_hint}")

    if not data_uri:
        _debug(f"RunRequest/json raw type: {type(raw_payload).__name__}")
        if raw_payload_text:
            _debug(f"RunRequest/json raw length: {len(raw_payload_text)}")
            snippet = raw_payload_text if len(raw_payload_text) <= 300 else raw_payload_text[:300] + "..."
            _debug(f"RunRequest/json raw snippet: {snippet}")
        _debug(f"RunRequest parse status: rr={'ok' if rr else 'none'} payload={'ok' if payload else 'none'}")
        if isinstance(payload, dict):
            _debug(f"RunRequest payload keys: {sorted(payload.keys())}")
            ds = payload.get("dataset")
            if isinstance(ds, dict):
                _debug(f"RunRequest dataset keys: {sorted(ds.keys())}")
        _debug(f"Candidate rr.dataset.uri: {rr_data_uri}")
        _debug(f"Candidate payload.dataset.uri: {payload_data_uri}")
        _debug(f"Candidate args.data: {args_data_uri}")
        _debug(f"Candidate env.YOLO_DATA_URI: {env_data_uri}")
        _debug(f"Candidate env.YOLO_DATA: {env_data_alt}")
        _debug(f"Candidate env.DATA_URI: {env_data_uri_alt}")
        _debug(f"Candidate env.DATA: {env_data_alt2}")
        _debug(f"Candidate Args/data: {task_arg_data}")
        _debug(f"Candidate raw_payload.regex uri: {raw_payload_uri}")
        _debug(f"Candidate raw_payload.parse uri: {parsed_payload_uri}")
        if payload is None:
            print("RunRequest payload missing or invalid; dataset not found.")
            if task is None:
                print("Task.current_task() is None; check CLEARML_TASK_ID and agent execution context.")
            else:
                try:
                    raw_direct = task.get_parameter("RunRequest/json")
                    if raw_direct is not None:
                        print(f"RunRequest/json direct type: {type(raw_direct).__name__}")
                        if isinstance(raw_direct, (str, bytes, bytearray)):
                            snippet = raw_direct if len(str(raw_direct)) <= 400 else str(raw_direct)[:400] + "..."
                            print(f"RunRequest/json direct value: {snippet}")
                            print(f"RunRequest/json direct repr: {repr(str(raw_direct)[:200])}")
                            codepoints = [ord(ch) for ch in str(raw_direct)[:60]]
                            print(f"RunRequest/json first codepoints: {codepoints}")
                            try:
                                json.loads(_ascii_sanitize(str(raw_direct)))
                                print("RunRequest/json json.loads(ascii_sanitize) OK")
                            except Exception as exc:
                                print(f"RunRequest/json json.loads(ascii_sanitize) failed: {exc}")
                        else:
                            print(f"RunRequest/json direct value: {raw_direct}")
                    arg_data = task.get_parameter("Args/data")
                    if arg_data is not None:
                        print(f"Args/data value: {arg_data}")
                    params = task.get_parameters()
                    keys = sorted(set(_collect_param_keys(params)))
                    print(f"Available task parameter keys: {keys[:50]}")
                except Exception:
                    print("Failed to read task parameters for debug.")
        else:
            ds = payload.get("dataset")
            if isinstance(ds, dict):
                print(f"RunRequest dataset keys: {sorted(ds.keys())}")
            elif ds is not None:
                print(f"RunRequest dataset type: {type(ds).__name__}")
        raise ValueError("Missing dataset; provide RunRequest/json.dataset.uri or dataset.clearml (ClearML) or --data (local)")

    task_epochs = _task_arg(task, "epochs")
    task_batch = _task_arg(task, "batch")
    task_imgsz = _task_arg(task, "imgsz")
    task_device = _task_arg(task, "device")
    task_weights = _task_arg(task, "weights")
    task_model = _task_arg(task, "model")
    task_workers = _task_arg(task, "workers")
    task_name = _task_arg(task, "name")
    task_project = _task_arg(task, "project")

    extra_name = yolo_extras.get("name") if isinstance(yolo_extras, dict) else None
    extra_project = yolo_extras.get("project") if isinstance(yolo_extras, dict) else None
    extra_model = None
    if isinstance(yolo_extras, dict):
        extra_model = yolo_extras.get("model") or yolo_extras.get("weights")

    epochs = args.epochs if args.epochs is not None else int(task_epochs if task_epochs is not None else yolo_extras.get("epochs", DEFAULTS["epochs"]))
    batch = args.batch if args.batch is not None else int(task_batch if task_batch is not None else yolo_extras.get("batch", DEFAULTS["batch"]))
    imgsz = args.imgsz if args.imgsz is not None else int(task_imgsz if task_imgsz is not None else yolo_extras.get("imgsz", DEFAULTS["imgsz"]))
    device = args.device if args.device is not None else str(task_device if task_device is not None else yolo_extras.get("device", DEFAULTS["device"]))
    weights = args.weights if args.weights is not None else str(task_weights if task_weights is not None else (task_model if task_model is not None else (extra_model if extra_model is not None else DEFAULTS["weights"])))
    workers = args.workers if args.workers is not None else int(task_workers if task_workers is not None else yolo_extras.get("workers", DEFAULTS["workers"]))
    project = args.project if args.project is not None else (task_project if task_project else (str(extra_project) if extra_project else os.getenv("YOLO_OUTPUT_DIR", "runs/ultralytics")))
    name = args.name if args.name is not None else (task_name if task_name else (rr.run_name if rr and rr.run_name else None))
    if not name:
        if payload and payload.get("run_name"):
            name = str(payload.get("run_name"))
        elif extra_name:
            name = str(extra_name)
        elif isinstance(raw_payload, str):
            raw_name = _extract_raw_string(raw_payload, "run_name")
            if raw_name:
                name = raw_name
    if not name:
        name = DEFAULTS["name"]

    # get logger to report scalars
    logger = task.get_logger()

    # connect resolved config so it appears in ClearML console
    weights_source = weights
    weights = _resolve_weights(weights, task, logger)
    task.connect(
        {
            "data": data_uri,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "weights": weights_source,
            "workers": workers,
            "project": project,
            "name": name,
        }
    )

    # Local training using ultralytics (YOLOv8). Import lazily so script fails early if missing.
    try:
        from ultralytics import YOLO
    except Exception:
        task.close()
        raise

    cleanup_dir = None
    try:
        data_path, cleanup_dir = _resolve_data_path(data_uri, yaml_hint)
        print(f"Resolved data_path: {data_path}")
        _patch_final_eval(data_uri, data_path)
        # Use resolved local dataset path in training args.

        # create simple training config and run
        model = YOLO(weights)
        # force ultralytics overrides to use resolved local data path (prevents final eval from reusing s3 uri)
        try:
            model.overrides["data"] = data_path
            model.overrides["val"] = True
        except Exception:
            pass
        # Ensure validator uses the resolved data_path during any validation hooks
        def _force_validator_data(trainer):
            try:
                trainer.validator.args.data = data_path
            except Exception:
                pass
        model.add_callback("on_val_start", _force_validator_data)
        model.add_callback("on_predict_start", _force_validator_data)

        # ultralytics accepts dict or cli args
        train_args = _build_train_args(
            data_path=data_path,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            workers=workers,
            yolo_extras=yolo_extras,
        )
        print(f"Ultralytics train_args: {train_args}")

        # Run training
        results = model.train(**train_args)
        if not train_args.get("val", True):
            results = model.val(data=data_path)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    metrics = {}
    try:
        raw_metrics = results.metrics if hasattr(results, "metrics") else None
        if raw_metrics:
            metrics = dict(raw_metrics)
        if metrics and "mAP50" in metrics:
            logger.report_scalar("val/mAP50", "mAP50", iteration=epochs, value=float(metrics["mAP50"]))
    except Exception:
        metrics = {}

    summary = {
        "trainer": "ultralytics",
        "run_name": rr.run_name if rr and rr.run_name else name,
        "metric": rr.metric if rr else None,
        "task_type": rr.task_type if rr else None,
        "time_budget_s": rr.time_budget_s if rr else None,
        "dataset": dataset_info(rr) if rr else {"type": "yolo", "uri": data_uri, "yaml_path": yaml_hint},
        "weights": weights_source,
        "data_source": data_source,
        "metrics": metrics,
    }

    # upload final weights if exists
    out_dir = train_args['project'] + '/' + train_args['name']
    weights_path = os.path.join(out_dir, 'weights', 'best.pt')
    output_model_id = None
    model_name = None
    model_version = None
    model_project = None
    if os.path.exists(weights_path):
        task.upload_artifact('model', weights_path, wait_on_upload=True)
        try:
            output_uri = _resolve_output_uri(task, logger)
            model_project = _model_project()
            model_version = _model_version(task)
            model_name = _model_name("ultralytics", rr.run_name if rr and rr.run_name else name)
            model, project_id = _create_output_model(
                task=task,
                name=model_name,
                project=model_project,
                version=model_version,
            )
            model.update_weights(weights_path, upload_uri=output_uri)
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

    if output_model_id:
        summary["model"] = {
            "id": output_model_id,
            "name": model_name,
            "project": model_project,
            "version": model_version,
        }
    summary_dir = tempfile.mkdtemp(prefix="yolo_summary_")
    try:
        summary_path = os.path.join(summary_dir, "run_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=True, indent=2, default=str)
        task.upload_artifact(name="run_summary", artifact_object=summary_path, wait_on_upload=True)
        report_kv_table(logger, title="run_summary", series="ultralytics", data=summary)
    except Exception as exc:
        logger.report_text(f"run_summary failed: {exc}")
    finally:
        shutil.rmtree(summary_dir, ignore_errors=True)

    task.close()


if __name__ == '__main__':
    main()
