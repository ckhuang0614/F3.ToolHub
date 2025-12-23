from __future__ import annotations

import json
import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from clearml import Task

from shared_lib.run_request import RunRequest

app = FastAPI(title="AutoML Gateway", version="0.1.0")


def _trainer_images() -> Dict[str, str]:
    return {
        "autogluon": os.getenv("AUTOGLOUON_IMAGE", "f3.autogluon-trainer:latest"),
        "flaml": os.getenv("FLAML_IMAGE", "f3.flaml-trainer:latest"),
        "yolo": os.getenv("YOLO_IMAGE", "f3.yolo-trainer:latest"),
    }


def _default_queue() -> str:
    return os.getenv("CLEARML_DEFAULT_QUEUE", "default")


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
    if rr.trainer == "yolo":
        project = os.getenv("CLEARML_PROJECT_YOLO", "AutoML-YOLO")

    task = Task.create(project_name=project, task_name=rr.run_name or f"{rr.trainer}-run")
    task.set_parameter("RunRequest/json", json.dumps(payload))
    task.set_parameter("RunRequest/schema_version", str(getattr(rr, 'schema_version', 2)))
    task.set_base_docker(docker_image=docker_image)

    # No per-payload bootstrap: trainers read RunRequest/json by themselves.
    task.set_script(diff=RUNNER_SCRIPT, entry_point="runner.py")

    queue = _default_queue()
    Task.enqueue(task=task, queue_name=queue)
    task_id = task.id

    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "queue": queue, "docker_image": docker_image},
    )
