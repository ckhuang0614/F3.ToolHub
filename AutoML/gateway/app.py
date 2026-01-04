from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from clearml import Task

from shared_lib.dataset_manager import (
    create_dataset as create_dataset_record,
    get_dataset,
    list_datasets,
    lineage,
    list_versions,
)
from shared_lib.pipeline_builder import start_pipeline
from shared_lib.run_request import RunRequest
from shared_lib.task_metadata import apply_run_metadata

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

    project = rr.project or os.getenv("CLEARML_PROJECT", "AutoML-Tabular")
    # Use different project for yolo if desired
    if rr.trainer == "ultralytics":
        project = rr.project or os.getenv("CLEARML_PROJECT_YOLO", "AutoML-ULTRALYTICS")

    task = Task.create(project_name=project, task_name=rr.run_name or f"{rr.trainer}-run")
    output_uri = _default_output_uri()
    if output_uri:
        task.output_uri = output_uri
    apply_run_metadata(task, rr)
    if not rr.project:
        try:
            task.set_parameter("Run/project", project)
        except Exception:
            pass
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
        info = create_dataset_record(payload)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(status_code=201, content=info)


@app.get("/datasets")
async def list_datasets_api(
    project: str | None = None,
    name: str | None = None,
    version: str | None = None,
    tags: str | None = None,
):
    tag_list = _parse_list(tags, split_commas=True)
    try:
        info = list_datasets(project=project, name=name, version=version, tags=tag_list)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"datasets": info}


@app.get("/datasets/lookup")
async def get_dataset_api(
    project: str | None = None,
    name: str | None = None,
    version: str | None = None,
    tags: str | None = None,
    dataset_id: str | None = None,
):
    tag_list = _parse_list(tags, split_commas=True)
    try:
        info = get_dataset(
            dataset_id=dataset_id,
            project=project,
            name=name,
            version=version,
            tags=tag_list or None,
        )
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return info


@app.get("/datasets/{dataset_id}")
async def get_dataset_by_id(dataset_id: str):
    try:
        info = get_dataset(dataset_id=dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return info


@app.get("/datasets/{dataset_id}/versions")
async def list_dataset_versions(dataset_id: str):
    try:
        info = list_versions(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"versions": info}


@app.get("/datasets/{dataset_id}/lineage")
async def dataset_lineage(dataset_id: str):
    try:
        info = lineage(dataset_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"lineage": info}


@app.post("/pipelines")
async def create_pipeline(payload: Dict[str, Any]):
    try:
        info = start_pipeline(payload, allow_payload_paths=False, controller_remote=False)
    except ValueError as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(status_code=201, content=info)
