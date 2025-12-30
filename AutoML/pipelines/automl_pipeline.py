from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from clearml import Task
from clearml.automation.controller import PipelineController

from shared_lib.run_request import RunRequest


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


def _trainer_images() -> Dict[str, str]:
    return {
        "autogluon": os.getenv("AUTOGLOUON_IMAGE", "f3.autogluon-trainer:latest"),
        "flaml": os.getenv("FLAML_IMAGE", "f3.flaml-trainer:latest"),
        "ultralytics": os.getenv("ULTRALYTICS_IMAGE", "f3.ultralytics-trainer:latest"),
    }


def _default_queue() -> str:
    return os.getenv("CLEARML_DEFAULT_QUEUE", "default")


def _default_project() -> str:
    return os.getenv("CLEARML_PROJECT", "AutoML-Tabular")


def _default_yolo_project() -> Optional[str]:
    return os.getenv("CLEARML_PROJECT_YOLO")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(base_dir: str, value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(base_dir, value))


def _load_payload(payload_spec: Any, base_dir: str) -> Dict[str, Any]:
    if isinstance(payload_spec, str):
        payload_path = _resolve_path(base_dir, payload_spec)
        return _load_json(payload_path)
    if isinstance(payload_spec, dict):
        return payload_spec
    raise ValueError("payload must be a path or object")


def _project_for_trainer(trainer: str, default_project: str, default_yolo_project: Optional[str]) -> str:
    if trainer == "ultralytics" and default_yolo_project:
        return default_yolo_project
    return default_project


def _normalize_parents(parents: Any) -> Optional[List[str]]:
    if parents in (None, ""):
        return None
    if isinstance(parents, str):
        return [parents]
    return [str(p) for p in list(parents)]


def _build_overrides(rr: RunRequest, payload: Dict[str, Any]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "RunRequest/json": json.dumps(payload, ensure_ascii=True),
        "RunRequest/schema_version": str(getattr(rr, "schema_version", 2)),
    }
    if rr.trainer == "ultralytics":
        dataset_uri = getattr(rr.dataset, "uri", None)
        if dataset_uri:
            overrides["Args/data"] = dataset_uri
        if rr.run_name:
            overrides["Args/name"] = rr.run_name
        yolo_extras = rr.extras.get("yolo") if isinstance(rr.extras, dict) else None
        if isinstance(yolo_extras, dict):
            for key in ("epochs", "batch", "imgsz", "device", "weights", "workers"):
                if key in yolo_extras and yolo_extras[key] is not None:
                    overrides[f"Args/{key}"] = yolo_extras[key]
    return overrides


def _create_template_task(project: str, trainer: str, docker_image: str) -> Task:
    task = Task.create(
        project_name=project,
        task_name=f"pipeline-template-{trainer}",
        task_type=Task.TaskTypes.training,
    )
    task.set_base_docker(docker_image=docker_image)
    task.set_script(diff=RUNNER_SCRIPT, entry_point="runner.py")
    task.add_tags(["pipeline-template"])
    return task


def _load_steps(config_path: str) -> Dict[str, Any]:
    config = _load_json(config_path)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    steps = config.get("steps") or []
    if not steps:
        raise ValueError("config.steps is required")
    base_dir = os.path.dirname(os.path.abspath(config_path))
    normalized_steps = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("each step must be an object")
        if step.get("enabled", True) is False:
            continue
        name = step.get("name")
        if not name:
            raise ValueError("step.name is required")
        payload = _load_payload(step.get("payload"), base_dir)
        rr = RunRequest.from_dict(payload)
        normalized_steps.append(
            {
                "name": str(name),
                "payload": payload,
                "rr": rr,
                "queue": step.get("queue"),
                "project": step.get("project"),
                "parents": _normalize_parents(step.get("parents")),
            }
        )
    if not normalized_steps:
        raise ValueError("no enabled steps found")
    config["steps"] = normalized_steps
    return config


def run_pipeline(config_path: str) -> None:
    config = _load_steps(config_path)
    default_project = str(config.get("project") or _default_project())
    default_yolo_project = config.get("yolo_project") or _default_yolo_project()
    default_queue = str(config.get("queue") or _default_queue())

    pipeline_name = str(config.get("name") or "AutoML Pipeline")
    pipeline_version = str(config.get("version") or "0.1")

    pipe = PipelineController(
        name=pipeline_name,
        project=default_project,
        version=pipeline_version,
        add_pipeline_tags=True,
    )

    trainer_images = _trainer_images()
    templates: Dict[tuple[str, str], Task] = {}
    previous_step_name: Optional[str] = None

    for step in config["steps"]:
        rr: RunRequest = step["rr"]
        trainer = rr.trainer
        docker_image = trainer_images.get(trainer)
        if not docker_image:
            raise ValueError(f"Unsupported trainer: {trainer}")

        step_project = str(step.get("project") or _project_for_trainer(trainer, default_project, default_yolo_project))
        step_queue = str(step.get("queue") or default_queue)

        template_key = (step_project, trainer)
        template_task = templates.get(template_key)
        if template_task is None:
            template_task = _create_template_task(step_project, trainer, docker_image)
            templates[template_key] = template_task

        parents = step.get("parents")
        if parents is None and previous_step_name:
            parents = [previous_step_name]

        pipe.add_step(
            name=step["name"],
            base_task_id=template_task.id,
            parameter_override=_build_overrides(rr, step["payload"]),
            execution_queue=step_queue,
            parents=parents,
        )
        previous_step_name = step["name"]

    pipe.start()
    pipe.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoML ClearML pipeline skeleton")
    parser.add_argument(
        "--config",
        required=True,
        help="Pipeline config JSON (see pipelines/pipeline_example.json)",
    )
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
