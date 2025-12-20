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
    }


def _default_queue() -> str:
    return os.getenv("CLEARML_DEFAULT_QUEUE", "default")

def _bootstrap_script() -> str:
    return (
        "import runpy\n"
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
    task = Task.create(project_name=project, task_name=rr.run_name or f"{rr.trainer}-run")
    task.set_parameter("RunRequest/json", json.dumps(payload))
    task.set_base_docker(docker_image=docker_image)
    task.set_script(diff=_bootstrap_script(), entry_point="bootstrap.py")

    queue = _default_queue()
    Task.enqueue(task=task, queue_name=queue)
    task_id = task.id

    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "queue": queue, "docker_image": docker_image},
    )
