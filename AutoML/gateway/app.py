from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from clearml import Dataset, Task
from clearml.storage import StorageManager

from shared_lib.pipeline_builder import start_pipeline
from shared_lib.run_request import RunRequest

app = FastAPI(title="AutoML Gateway", version="0.1.0")


def _trainer_images() -> Dict[str, str]:
    return {
        "autogluon": os.getenv("AUTOGLOUON_IMAGE", "f3.autogluon-trainer:latest"),
        "flaml": os.getenv("FLAML_IMAGE", "f3.flaml-trainer:latest"),
        "ultralytics": os.getenv("ULTRALYTICS_IMAGE", "f3.ultralytics-trainer:latest"),
    }


def _default_queue() -> str:
    return os.getenv("CLEARML_DEFAULT_QUEUE", "default")


def _queue_for_trainer(trainer: str) -> str | None:
    queue_map = {
        "autogluon": os.getenv("CLEARML_QUEUE_AUTOGLOUON"),
        "flaml": os.getenv("CLEARML_QUEUE_FLAML"),
        "ultralytics": os.getenv("CLEARML_QUEUE_ULTRALYTICS"),
    }
    value = queue_map.get(trainer)
    return value.strip() if value else None


def _default_output_uri() -> str | None:
    value = os.getenv("CLEARML_DEFAULT_OUTPUT_URI") or os.getenv("CLEARML_OUTPUT_URI")
    return value.strip() if value else None


def _normalize_queues(raw: Any) -> list[Dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict) and "queues" in raw:
        raw = raw.get("queues")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, (list, tuple, set)):
        items = []
        for item in raw:
            if isinstance(item, dict):
                items.append(item)
            else:
                name = str(item).strip()
                if name:
                    items.append({"name": name})
        return items
    name = str(raw).strip()
    return [{"name": name}] if name else []


def _list_queues() -> list[Dict[str, Any]]:
    for attr in ("get_all_queues", "get_queue_names"):
        getter = getattr(Task, attr, None)
        if getter is None:
            continue
        try:
            result = getter()
        except Exception:
            continue
        queues = _normalize_queues(result)
        if queues:
            return queues

    try:
        from clearml.backend_api.session import Session
    except Exception as exc:  # pragma: no cover - optional import
        raise RuntimeError(f"clearml backend_api unavailable: {exc}") from exc

    session = Session()
    response = session.send_request("queues", "get_all", {})
    if not getattr(response, "ok", False):
        status = getattr(response, "status_code", "unknown")
        raise RuntimeError(f"clearml queue query failed: {status}")
    payload = response.json() if hasattr(response, "json") else {}
    data = payload.get("data") if isinstance(payload, dict) else None
    queues = None
    if isinstance(data, dict):
        queues = data.get("queues") or data.get("queue")
    if queues is None and isinstance(payload, dict):
        queues = payload.get("queues") or payload.get("queue")
    return _normalize_queues(queues)


def _parse_list(value: Any, split_commas: bool = False) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        if split_commas:
            parts = [part.strip() for part in value.split(",") if part.strip()]
            return parts if parts else [value]
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _is_uri(value: str) -> bool:
    return bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", value))


def _normalize_file_uri(value: str) -> str:
    if value.startswith("file://"):
        return value[7:]
    return value


def _split_dataset_inputs(payload: Dict[str, Any]) -> tuple[list[str], list[str]]:
    local_paths: list[str] = []
    uri_paths: list[str] = []

    paths = payload.get("files") or payload.get("paths") or payload.get("path") or payload.get("file")
    for item in _parse_list(paths):
        item = _normalize_file_uri(str(item))
        if _is_uri(item):
            uri_paths.append(item)
        else:
            local_paths.append(item)

    uris = payload.get("uris") or payload.get("external_uris") or payload.get("uri")
    for item in _parse_list(uris, split_commas=True):
        item = _normalize_file_uri(str(item))
        if _is_uri(item):
            uri_paths.append(item)
        else:
            local_paths.append(item)

    return local_paths, uri_paths


def _add_external_files(dataset: Dataset, uris: list[str], recursive: bool) -> None:
    if not hasattr(dataset, "add_external_files"):
        raise ValueError("clearml SDK does not support add_external_files; set external=false to download")

    add_fn = dataset.add_external_files
    try:
        add_fn(uris, recursive=recursive)
        return
    except TypeError:
        pass
    try:
        add_fn(uris)
        return
    except TypeError:
        pass
    try:
        add_fn(path=uris, recursive=recursive)
        return
    except TypeError:
        pass
    add_fn(path=uris)


def _create_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
    project = payload.get("project") or payload.get("dataset_project")
    name = payload.get("name") or payload.get("dataset_name")
    version = payload.get("version") or payload.get("dataset_version")
    storage = payload.get("storage") or payload.get("dataset_storage")

    if not project:
        raise ValueError("dataset project is required (project or dataset_project)")
    if not name:
        raise ValueError("dataset name is required (name or dataset_name)")

    tags = _parse_list(payload.get("tags") or payload.get("dataset_tags"), split_commas=True)
    parents = _parse_list(
        payload.get("parents") or payload.get("parent_dataset_ids") or payload.get("parent_dataset_id")
    )
    recursive = bool(payload.get("recursive", True))
    upload = bool(payload.get("upload", True))
    finalize = bool(payload.get("finalize", True))
    allow_empty = bool(payload.get("allow_empty", False))
    use_external = bool(payload.get("external", True))

    local_paths, uri_paths = _split_dataset_inputs(payload)
    if not local_paths and not uri_paths and not allow_empty:
        raise ValueError("dataset files are required (files/path or uris)")

    for path in local_paths:
        if not os.path.exists(path):
            raise ValueError(f"dataset path not found: {path}")

    parent_datasets = []
    for dataset_id in parents:
        parent_datasets.append(Dataset.get(dataset_id=dataset_id))

    create_kwargs = {
        "dataset_project": str(project),
        "dataset_name": str(name),
        "dataset_version": str(version) if version else None,
        "parent_datasets": parent_datasets or None,
    }
    if storage:
        create_kwargs["dataset_storage"] = str(storage)

    try:
        dataset = Dataset.create(**create_kwargs)
    except TypeError:
        if "dataset_storage" in create_kwargs:
            create_kwargs.pop("dataset_storage", None)
            dataset = Dataset.create(**create_kwargs)
        else:
            raise

    if tags:
        dataset.add_tags(tags)

    for path in local_paths:
        dataset.add_files(path=path, recursive=recursive)

    if uri_paths:
        if use_external:
            _add_external_files(dataset, uri_paths, recursive=recursive)
        else:
            for uri in uri_paths:
                local_copy = StorageManager.get_local_copy(uri)
                dataset.add_files(path=local_copy, recursive=recursive)

    if upload:
        dataset.upload()
    if finalize:
        dataset.finalize()

    file_count = None
    try:
        file_count = dataset.get_num_files()
    except Exception:
        file_count = None

    return {
        "id": dataset.id,
        "name": dataset.name,
        "project": dataset.project,
        "version": dataset.version,
        "file_count": file_count,
    }

# Constant runner script (no per-payload argv bootstrapping).
# Trainers must read Task parameter: RunRequest/json.
RUNNER_SCRIPT = (
    "import runpy, sys\n"
    "try:\n"
    "    import site\n"
    "    base_prefix = getattr(sys, 'base_prefix', sys.prefix)\n"
    "    for path in site.getsitepackages([base_prefix]):\n"
    "        if path not in sys.path:\n"
    "            sys.path.append(path)\n"
    "    if '/app' not in sys.path:\n"
    "        sys.path.insert(0, '/app')\n"
    "except Exception:\n"
    "    pass\n"
    "runpy.run_path('/app/train.py', run_name='__main__')\n"
)


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/debug")
async def debug_env() -> Dict[str, str | None]:
    return {
        "clearml_default_output_uri": os.getenv("CLEARML_DEFAULT_OUTPUT_URI"),
        "clearml_output_uri": os.getenv("CLEARML_OUTPUT_URI"),
        "resolved_output_uri": _default_output_uri(),
    }


@app.get("/queues")
async def list_queues() -> Dict[str, Any]:
    try:
        queues = _list_queues()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"default_queue": _default_queue(), "queues": queues}


@app.post("/runs")
async def submit_run(payload: Dict[str, Any]):
    try:
        rr = RunRequest.from_dict(payload)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    images = _trainer_images()
    docker_image = images.get(rr.trainer)
    if not docker_image:
        raise HTTPException(status_code=400, detail=f"Unsupported trainer: {rr.trainer}")

    project = os.getenv("CLEARML_PROJECT", "AutoML-Tabular")
    # Use different project for yolo if desired
    if rr.trainer == "ultralytics":
        project = os.getenv("CLEARML_PROJECT_YOLO", "AutoML-ULTRALYTICS")

    task = Task.create(project_name=project, task_name=rr.run_name or f"{rr.trainer}-run")
    output_uri = _default_output_uri()
    if output_uri:
        task.output_uri = output_uri
    task.set_parameter("RunRequest/json", json.dumps(payload))
    task.set_parameter("RunRequest/schema_version", str(getattr(rr, 'schema_version', 2)))
    if rr.trainer == "ultralytics":
        try:
            data_uri = getattr(rr.dataset, "uri", None)
        except Exception:
            data_uri = None
        if data_uri:
            task.set_parameter("Args/data", data_uri)
        if rr.run_name:
            task.set_parameter("Args/name", rr.run_name)
        yolo_extras = rr.extras.get("yolo") if isinstance(rr.extras, dict) else {}
        if isinstance(yolo_extras, dict):
            for key in ("epochs", "batch", "imgsz", "device", "weights", "workers"):
                if key in yolo_extras and yolo_extras[key] is not None:
                    task.set_parameter(f"Args/{key}", yolo_extras[key])
    task.set_base_docker(docker_image=docker_image)

    # No per-payload bootstrap: trainers read RunRequest/json by themselves.
    task.set_script(diff=RUNNER_SCRIPT, entry_point="runner.py")

    queue = rr.queue or _queue_for_trainer(rr.trainer) or _default_queue()
    Task.enqueue(task=task, queue_name=queue)
    task_id = task.id

    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "queue": queue, "docker_image": docker_image},
    )


@app.post("/datasets")
async def create_dataset(payload: Dict[str, Any]):
    try:
        info = _create_dataset(payload)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(status_code=201, content=info)


@app.post("/pipelines")
async def create_pipeline(payload: Dict[str, Any]):
    try:
        info = start_pipeline(payload, allow_payload_paths=False, controller_remote=False)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(status_code=201, content=info)
