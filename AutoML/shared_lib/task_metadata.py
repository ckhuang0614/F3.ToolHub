from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared_lib.run_request import RunRequest


def _normalize_tags(raw: Any) -> List[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, str):
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, (list, tuple, set)):
        tags = []
        for item in raw:
            tag = str(item).strip()
            if tag:
                tags.append(tag)
        return tags
    tag = str(raw).strip()
    return [tag] if tag else []


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def build_task_tags(rr: RunRequest, extra: Optional[List[str]] = None) -> List[str]:
    tags = [
        "automl",
        f"trainer:{rr.trainer}",
        f"schema:v{rr.schema_version}",
    ]
    tags.extend(_normalize_tags(rr.tags))
    if extra:
        tags.extend(_normalize_tags(extra))
    return _dedupe(tags)


def dataset_info(rr: RunRequest) -> Dict[str, Any]:
    ds = rr.dataset
    info: Dict[str, Any] = {"type": getattr(ds, "type", None)}

    uri = getattr(ds, "uri", None)
    if uri:
        info["uri"] = uri
    label = getattr(ds, "label", None)
    if label:
        info["label"] = label
    yaml_path = getattr(ds, "yaml_path", None)
    if yaml_path:
        info["yaml_path"] = yaml_path

    clearml_ref = getattr(ds, "clearml", None)
    if clearml_ref:
        info["clearml"] = {
            "id": getattr(clearml_ref, "id", None),
            "name": getattr(clearml_ref, "name", None),
            "project": getattr(clearml_ref, "project", None),
            "version": getattr(clearml_ref, "version", None),
            "tags": getattr(clearml_ref, "tags", None),
        }
    return info


def dataset_params(rr: RunRequest) -> Dict[str, str]:
    info = dataset_info(rr)
    params: Dict[str, str] = {}

    def _set(key: str, value: Any) -> None:
        if value in (None, ""):
            return
        params[key] = str(value)

    _set("Dataset/type", info.get("type"))
    _set("Dataset/uri", info.get("uri"))
    _set("Dataset/label", info.get("label"))
    _set("Dataset/yaml_path", info.get("yaml_path"))

    clearml = info.get("clearml")
    if isinstance(clearml, dict):
        _set("Dataset/clearml_id", clearml.get("id"))
        _set("Dataset/clearml_name", clearml.get("name"))
        _set("Dataset/clearml_project", clearml.get("project"))
        _set("Dataset/clearml_version", clearml.get("version"))
        tags = clearml.get("tags")
        if tags:
            if isinstance(tags, (list, tuple, set)):
                tag_str = ",".join([str(t) for t in tags if str(t).strip()])
                _set("Dataset/clearml_tags", tag_str)
            else:
                _set("Dataset/clearml_tags", tags)
    return params


def apply_run_metadata(task, rr: RunRequest, extra_tags: Optional[List[str]] = None) -> None:
    tags = build_task_tags(rr, extra=extra_tags)
    if tags:
        try:
            task.add_tags(tags)
        except Exception:
            pass

    params: Dict[str, Any] = {
        "Run/trainer": rr.trainer,
        "Run/schema_version": rr.schema_version,
    }
    if rr.project:
        params["Run/project"] = rr.project
    if rr.queue:
        params["Run/queue"] = rr.queue
    if tags:
        params["Run/tags"] = ",".join(tags)

    params.update(dataset_params(rr))

    for key, value in params.items():
        if value in (None, ""):
            continue
        try:
            task.set_parameter(key, value)
        except Exception:
            continue
