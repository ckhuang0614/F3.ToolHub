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


def _bootstrap_script_from_payload(payload: Dict[str, Any]) -> str:
    """Generate a bootstrap script that sets sys.argv for train.py based on payload."""
    # embed payload as JSON literal
    payload_json = json.dumps(payload)
    # basic script: load payload, build argv, run train.py
    script = (
        "import json, sys, runpy\n"
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
        f"cfg = json.loads('''{payload_json}''')\n"
        "argv = ['train.py']\n"
        "# default data arg if present\n"
        "if 'dataset' in cfg and cfg['dataset'].get('csv_uri'):\n"
        "    argv += ['--data', cfg['dataset']['csv_uri']]\n"
        "# pass extras for specific trainers\n"
        "extras = cfg.get('extras', {})\n"
        "# YOLO extras mapping\n"
        "if cfg.get('trainer') == 'yolo':\n"
        "    y = extras.get('yolo', {})\n"
        "    if 'epochs' in y:\n"
        "        argv += ['--epochs', str(y['epochs'])]\n"
        "    if 'batch' in y:\n"
        "        argv += ['--batch', str(y['batch'])]\n"
        "    if 'imgsz' in y:\n"
        "        argv += ['--imgsz', str(y['imgsz'])]\n"
        "    if 'device' in y:\n"
        "        argv += ['--device', str(y['device'])]\n"
        "    if 'weights' in y:\n"
        "        argv += ['--weights', str(y['weights'])]\n"
        "# other trainers can be extended similarly\n"
        "sys.argv = argv\n"
        "# avoid logging sensitive URIs\n"
        "def _sanitize_arg(a):\n"
        "    if isinstance(a, str) and ('s3://' in a or 'http://' in a or 'https://' in a):\n"
        "        return '<uri-redacted>'\n"
        "    return a\n"
        "safe_argv = [_sanitize_arg(a) for a in sys.argv]\n"
        "print('Bootstrap argv:', safe_argv)\n"
        "runpy.run_path('/app/train.py', run_name='__main__')\n"
    )
    return script


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
    task.set_base_docker(docker_image=docker_image)

    # Generate a bootstrap script that passes command-line args to train.py
    bootstrap = _bootstrap_script_from_payload(payload)
    task.set_script(diff=bootstrap, entry_point="bootstrap.py")

    queue = _default_queue()
    Task.enqueue(task=task, queue_name=queue)
    task_id = task.id

    return JSONResponse(
        status_code=202,
        content={"task_id": task_id, "queue": queue, "docker_image": docker_image},
    )
