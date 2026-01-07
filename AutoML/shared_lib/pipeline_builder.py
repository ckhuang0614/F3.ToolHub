from __future__ import annotations

import json
import os
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

from clearml import Task

try:
    from clearml.automation.controller import PipelineController
except Exception:  # pragma: no cover - handled at runtime
    PipelineController = None  # type: ignore[assignment]

from shared_lib.dataset_manager import create_dataset
from shared_lib.run_request import RunRequest
from shared_lib.task_metadata import dataset_params


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


def _queue_for_trainer(trainer: str) -> Optional[str]:
    queue_map = {
        "autogluon": os.getenv("CLEARML_QUEUE_AUTOGLOUON"),
        "flaml": os.getenv("CLEARML_QUEUE_FLAML"),
        "ultralytics": os.getenv("CLEARML_QUEUE_ULTRALYTICS"),
    }
    value = queue_map.get(trainer)
    return value.strip() if value else None


def _default_output_uri() -> Optional[str]:
    value = os.getenv("CLEARML_DEFAULT_OUTPUT_URI") or os.getenv("CLEARML_OUTPUT_URI")
    return value.strip() if value else None


def _default_project() -> str:
    return os.getenv("CLEARML_PROJECT", "AutoML-Tabular")


def _default_yolo_project() -> Optional[str]:
    return os.getenv("CLEARML_PROJECT_YOLO")


def _require_pipeline_controller() -> None:
    if PipelineController is None:
        raise RuntimeError("clearml PipelineController not available; upgrade clearml SDK")


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(base_dir: Optional[str], value: str) -> str:
    if os.path.isabs(value):
        return value
    if base_dir:
        return os.path.normpath(os.path.join(base_dir, value))
    return value


def _load_payload(payload_spec: Any, base_dir: Optional[str], allow_paths: bool) -> Dict[str, Any]:
    if isinstance(payload_spec, str):
        if not allow_paths:
            raise ValueError("payload path is not supported; provide inline payload object")
        payload_path = _resolve_path(base_dir, payload_spec)
        return _load_json(payload_path)
    if isinstance(payload_spec, dict):
        return payload_spec
    raise ValueError("payload must be a path or object")


def _extract_dataset_ref(payload: Dict[str, Any]) -> Optional[str]:
    ref = payload.get("dataset_ref") or payload.get("datasetRef")
    if ref:
        return str(ref)
    dataset = payload.get("dataset")
    if isinstance(dataset, dict):
        ref = dataset.get("ref") or dataset.get("dataset_ref") or dataset.get("datasetRef")
        if ref:
            return str(ref)
    return None


def _apply_dataset_ref(payload: Dict[str, Any], dataset_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ref = _extract_dataset_ref(payload)
    if not ref:
        return payload
    dataset_info = dataset_outputs.get(ref)
    if not dataset_info:
        raise ValueError(f"dataset_ref '{ref}' not found")
    ds = payload.get("dataset") or {}
    if not isinstance(ds, dict):
        ds = {"uri": ds} if ds else {}
    clearml_ref = ds.get("clearml") if isinstance(ds.get("clearml"), dict) else {}
    clearml_ref["id"] = dataset_info.get("id")
    ds["clearml"] = clearml_ref
    payload = dict(payload)
    payload["dataset"] = ds
    payload.pop("dataset_ref", None)
    return payload


def _project_for_trainer(trainer: str, default_project: str, default_yolo_project: Optional[str]) -> str:
    if trainer == "autogluon":
        return os.getenv("CLEARML_PROJECT_AUTOGLUON", "AutoML-AUTOGLUON")
    if trainer == "flaml":
        return os.getenv("CLEARML_PROJECT_FLAML", "AutoML-FLAML")
    if trainer == "ultralytics":
        return os.getenv("CLEARML_PROJECT_ULTRALYTICS", default_yolo_project or "AutoML-ULTRALYTICS")
    return default_project


def _normalize_parents(parents: Any) -> Optional[List[str]]:
    if parents in (None, ""):
        return None
    if isinstance(parents, str):
        return [parents]
    return [str(p) for p in list(parents)]


def _build_overrides(
    rr: RunRequest,
    payload: Dict[str, Any],
    pipeline_name: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    pipeline_project: Optional[str] = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "RunRequest/json": json.dumps(payload, ensure_ascii=True),
        "RunRequest/schema_version": str(getattr(rr, "schema_version", 2)),
    }
    if pipeline_name:
        overrides["Pipeline/name"] = pipeline_name
    if pipeline_version:
        overrides["Pipeline/version"] = pipeline_version
    if pipeline_project:
        overrides["Pipeline/project"] = pipeline_project
    overrides.update(dataset_params(rr))
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
    output_uri = _default_output_uri()
    if output_uri:
        task.output_uri = output_uri
    task.set_base_docker(docker_image=docker_image)
    task.set_script(diff=RUNNER_SCRIPT, entry_point="runner.py")
    task.add_tags(["pipeline-template"])
    return task


def _normalize_config(
    config: Dict[str, Any],
    base_dir: Optional[str],
    allow_payload_paths: bool,
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    steps = config.get("steps") or []
    if not steps:
        raise ValueError("config.steps is required")

    normalized_steps = []
    for step in steps:
        if not isinstance(step, dict):
            raise ValueError("each step must be an object")
        if step.get("enabled", True) is False:
            continue
        name = step.get("name")
        if not name:
            raise ValueError("step.name is required")
        if "payload" not in step:
            raise ValueError("step.payload is required")

        payload = _load_payload(step.get("payload"), base_dir, allow_payload_paths)
        step_type = str(step.get("type") or step.get("kind") or "run").lower()
        rr = None
        if step_type not in {"dataset", "data"}:
            dataset_ref = _extract_dataset_ref(payload)
            if not dataset_ref:
                rr = RunRequest.from_dict(payload)
        normalized_steps.append(
            {
                "name": str(name),
                "payload": payload,
                "rr": rr,
                "type": step_type,
                "queue": step.get("queue"),
                "project": step.get("project"),
                "parents": _normalize_parents(step.get("parents")),
            }
        )

    if not normalized_steps:
        raise ValueError("no enabled steps found")

    normalized = dict(config)
    normalized["steps"] = normalized_steps
    return normalized


def _pipeline_task_id(pipe: Any) -> Optional[str]:
    for attr in ("pipeline_task_id", "pipeline_task", "_pipeline_task"):
        value = getattr(pipe, attr, None)
        if isinstance(value, str):
            return value
        if value is not None:
            task_id = getattr(value, "id", None)
            if task_id:
                return task_id
    return None


def start_pipeline(
    config: Dict[str, Any],
    base_dir: Optional[str] = None,
    allow_payload_paths: bool = True,
    wait: Optional[bool] = None,
    controller_remote: Optional[bool] = None,
) -> Dict[str, Any]:
    _require_pipeline_controller()
    normalized = _normalize_config(config, base_dir, allow_payload_paths)

    default_project = str(normalized.get("project") or _default_project())
    default_yolo_project = normalized.get("yolo_project") or _default_yolo_project()
    default_queue = str(normalized.get("queue") or _default_queue())
    controller_queue = str(
        normalized.get("controller_queue")
        or normalized.get("pipeline_queue")
        or os.getenv("CLEARML_PIPELINE_QUEUE")
        or default_queue
    )
    config_controller_remote = bool(
        normalized.get("controller_remote")
        or normalized.get("remote_controller")
        or normalized.get("run_controller_remotely")
    )
    controller_remote_flag = config_controller_remote if controller_remote is None else bool(controller_remote)

    pipeline_name = str(normalized.get("name") or "AutoML Pipeline")
    pipeline_version = str(normalized.get("version") or "0.1")
    wait_flag = bool(normalized.get("wait", False)) if wait is None else bool(wait)

    dataset_outputs: Dict[str, Dict[str, Any]] = {}
    for step in normalized["steps"]:
        if step.get("type") not in {"dataset", "data"}:
            continue
        payload = dict(step["payload"])
        if step.get("project") and not payload.get("project"):
            payload["project"] = step.get("project")
        parents = step.get("parents") or []
        if parents:
            parent_ids = []
            for parent_name in parents:
                parent_info = dataset_outputs.get(parent_name)
                if not parent_info:
                    raise ValueError(f"dataset step '{step['name']}' depends on unknown parent '{parent_name}'")
                parent_id = parent_info.get("id")
                if parent_id:
                    parent_ids.append(parent_id)
            if parent_ids:
                payload["parents"] = parent_ids
        dataset_outputs[step["name"]] = create_dataset(payload)

    pipe = PipelineController(
        name=pipeline_name,
        project=default_project,
        version=pipeline_version,
        add_pipeline_tags=True,
    )

    trainer_images = _trainer_images()
    templates: Dict[Tuple[str, str], Task] = {}
    previous_step_name: Optional[str] = None

    pipeline_step_names = {step["name"] for step in normalized["steps"] if step.get("type") not in {"dataset", "data"}}

    for step in normalized["steps"]:
        if step.get("type") in {"dataset", "data"}:
            continue
        payload = _apply_dataset_ref(step["payload"], dataset_outputs)
        rr = step["rr"] or RunRequest.from_dict(payload)
        step["payload"] = payload
        step["rr"] = rr

        trainer = rr.trainer
        docker_image = trainer_images.get(trainer)
        if not docker_image:
            raise ValueError(f"Unsupported trainer: {trainer}")

        step_project = str(
            step.get("project")
            or rr.project
            or _project_for_trainer(trainer, default_project, default_yolo_project)
        )
        step_queue = step.get("queue") or rr.queue or _queue_for_trainer(trainer) or default_queue
        step_queue = str(step_queue)

        template_key = (step_project, trainer)
        template_task = templates.get(template_key)
        if template_task is None:
            template_task = _create_template_task(step_project, trainer, docker_image)
            templates[template_key] = template_task

        parents = step.get("parents")
        if parents:
            parents = [parent for parent in parents if parent in pipeline_step_names]
            if not parents:
                parents = None
        if parents is None and previous_step_name:
            parents = [previous_step_name]

        pipe.add_step(
            name=step["name"],
            base_task_id=template_task.id,
            parameter_override=_build_overrides(
                rr,
                payload,
                pipeline_name=pipeline_name,
                pipeline_version=pipeline_version,
                pipeline_project=default_project,
            ),
            execution_queue=step_queue,
            parents=parents,
        )
        previous_step_name = step["name"]

    if controller_remote_flag:
        try:
            pipe.start(queue=controller_queue)
        except TypeError:
            pipe.start()
        if wait_flag:
            pipe.wait()
    else:
        if wait_flag:
            pipe.start_locally()
        else:
            thread = Thread(target=pipe.start_locally, name="pipeline-controller", daemon=True)
            thread.start()

    pipeline_id = _pipeline_task_id(pipe)
    if pipeline_id:
        try:
            pipeline_task = Task.get_task(task_id=pipeline_id)
            pipeline_task.set_parameter("Pipeline/id", pipeline_id)
            pipeline_task.set_parameter("Pipeline/name", pipeline_name)
            pipeline_task.set_parameter("Pipeline/version", pipeline_version)
            pipeline_task.set_parameter("Pipeline/project", default_project)
        except Exception:
            pass

    return {
        "pipeline_id": pipeline_id,
        "name": pipeline_name,
        "project": default_project,
        "version": pipeline_version,
        "steps": [step["name"] for step in normalized["steps"]],
        "queue": default_queue,
        "controller_queue": controller_queue,
        "controller_remote": controller_remote_flag,
        "wait": wait_flag,
    }


def start_pipeline_from_path(config_path: str, wait: bool = True) -> Dict[str, Any]:
    config = _load_json(config_path)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    return start_pipeline(config, base_dir=base_dir, allow_payload_paths=True, wait=wait)
