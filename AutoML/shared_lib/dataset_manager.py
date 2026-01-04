from __future__ import annotations

import inspect
import os
import re
from typing import Any, Dict, List, Optional

from clearml import Dataset
from clearml.storage import StorageManager


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


def _dataset_meta(dataset: Dataset) -> Dict[str, Any]:
    info = {
        "id": getattr(dataset, "id", None),
        "name": getattr(dataset, "name", None),
        "project": getattr(dataset, "project", None),
        "version": getattr(dataset, "version", None),
        "tags": getattr(dataset, "tags", None),
    }
    for attr in ("created", "created_at", "time"):
        value = getattr(dataset, attr, None)
        if value:
            info["created"] = value
            break
    parents = _parent_ids(dataset)
    if parents:
        info["parents"] = parents
    return info


def _parent_ids(dataset: Dataset) -> List[str]:
    for attr in ("parent_datasets", "parents", "parent_ids", "parent_dataset_ids"):
        parents = getattr(dataset, attr, None)
        if parents:
            return _normalize_parent_list(parents)
    get_parents = getattr(dataset, "get_parents", None)
    if callable(get_parents):
        try:
            parents = get_parents()
            return _normalize_parent_list(parents)
        except Exception:
            return []
    return []


def _normalize_parent_list(parents: Any) -> List[str]:
    if parents in (None, ""):
        return []
    output: List[str] = []
    if isinstance(parents, (list, tuple, set)):
        items = parents
    else:
        items = [parents]
    for item in items:
        if isinstance(item, Dataset):
            if getattr(item, "id", None):
                output.append(item.id)
        elif isinstance(item, dict):
            value = item.get("id") or item.get("dataset_id")
            if value:
                output.append(str(value))
        else:
            value = str(item).strip()
            if value:
                output.append(value)
    return output


def create_dataset(payload: Dict[str, Any]) -> Dict[str, Any]:
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

    info = _dataset_meta(dataset)
    info["file_count"] = file_count
    return info


def _call_with_supported_kwargs(func, **kwargs):
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters and v not in (None, "")}
    return func(**supported)


def list_datasets(
    project: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
    allow_partial: bool = True,
) -> List[Dict[str, Any]]:
    for method_name in ("list_datasets", "get_datasets", "get_all"):
        func = getattr(Dataset, method_name, None)
        if not callable(func):
            continue
        try:
            result = _call_with_supported_kwargs(
                func,
                dataset_project=project,
                dataset_name=name,
                dataset_version=version,
                dataset_tags=tags,
                project=project,
                name=name,
                version=version,
                tags=tags,
                allow_partial_name=allow_partial,
            )
        except Exception:
            continue
        return _normalize_dataset_list(result)
    raise RuntimeError("clearml SDK does not support dataset listing")


def _normalize_dataset_list(result: Any) -> List[Dict[str, Any]]:
    if result in (None, ""):
        return []
    if isinstance(result, dict):
        for key in ("datasets", "data", "result", "queue"):
            value = result.get(key)
            if value is not None:
                result = value
                break
    if isinstance(result, Dataset):
        return [_dataset_meta(result)]
    if isinstance(result, (list, tuple, set)):
        output = []
        for item in result:
            if isinstance(item, Dataset):
                output.append(_dataset_meta(item))
            elif isinstance(item, dict):
                output.append(item)
            else:
                value = str(item).strip()
                if value:
                    try:
                        ds = Dataset.get(dataset_id=value)
                        output.append(_dataset_meta(ds))
                    except Exception:
                        output.append({"id": value})
        return output
    return []


def get_dataset(
    dataset_id: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    version: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    if dataset_id:
        dataset = Dataset.get(dataset_id=dataset_id)
        return _dataset_meta(dataset)
    if not name:
        raise ValueError("dataset name is required when dataset_id is not provided")
    kwargs = {
        "dataset_name": name,
        "dataset_project": project,
        "dataset_version": version,
        "dataset_tags": tags,
    }
    dataset = Dataset.get(**{k: v for k, v in kwargs.items() if v not in (None, "")})
    return _dataset_meta(dataset)


def list_versions(dataset_id: str) -> List[Dict[str, Any]]:
    dataset = Dataset.get(dataset_id=dataset_id)
    project = getattr(dataset, "project", None)
    name = getattr(dataset, "name", None)
    tags = getattr(dataset, "tags", None)
    tags_list = tags if isinstance(tags, list) else None
    return list_datasets(project=project, name=name, tags=tags_list, allow_partial=False)


def lineage(dataset_id: str) -> List[Dict[str, Any]]:
    visited = set()
    results: List[Dict[str, Any]] = []

    def _walk(current_id: str) -> None:
        if not current_id or current_id in visited:
            return
        visited.add(current_id)
        ds = Dataset.get(dataset_id=current_id)
        info = _dataset_meta(ds)
        results.append(info)
        for parent_id in info.get("parents", []) or []:
            _walk(parent_id)

    _walk(dataset_id)
    return results
