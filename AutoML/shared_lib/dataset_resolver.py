from __future__ import annotations

import os
from typing import Optional, Sequence

from clearml import Dataset
from clearml.storage import StorageManager

from shared_lib.run_request import ClearMLDatasetRef, TabularDatasetSpec, YoloDatasetSpec

_CLEARML_URI_PREFIX = "clearml://"


def resolve_tabular_dataset(spec: TabularDatasetSpec) -> str:
    ref = _clearml_ref_from_spec(spec)
    if ref:
        root = _get_clearml_dataset_local_copy(ref)
        return _resolve_file_from_root(root, spec.path, (".csv",))
    if not spec.uri:
        raise ValueError("tabular dataset requires dataset.uri or dataset.clearml")
    return StorageManager.get_local_copy(spec.uri)


def resolve_yolo_dataset_uri(spec: YoloDatasetSpec) -> str:
    ref = _clearml_ref_from_spec(spec)
    if ref:
        return _get_clearml_dataset_local_copy(ref)
    if not spec.uri:
        raise ValueError("yolo dataset requires dataset.uri or dataset.clearml")
    return spec.uri


def _clearml_ref_from_spec(spec: object) -> Optional[ClearMLDatasetRef]:
    ref = getattr(spec, "clearml", None)
    if ref:
        return ref
    uri = getattr(spec, "uri", None)
    return _clearml_ref_from_uri(uri) if isinstance(uri, str) else None


def _clearml_ref_from_uri(uri: Optional[str]) -> Optional[ClearMLDatasetRef]:
    if not uri or not uri.startswith(_CLEARML_URI_PREFIX):
        return None
    dataset_id = uri[len(_CLEARML_URI_PREFIX) :].strip("/")
    if not dataset_id:
        return None
    return ClearMLDatasetRef(id=dataset_id)


def _get_clearml_dataset_local_copy(ref: ClearMLDatasetRef) -> str:
    if not (ref.id or ref.name):
        raise ValueError("dataset.clearml requires id or name")
    if ref.id:
        dataset = Dataset.get(dataset_id=ref.id)
    else:
        kwargs = {"dataset_name": ref.name}
        if ref.project:
            kwargs["dataset_project"] = ref.project
        if ref.version:
            kwargs["dataset_version"] = ref.version
        if ref.tags:
            kwargs["dataset_tags"] = ref.tags
        dataset = Dataset.get(**kwargs)
    return dataset.get_local_copy()


def _resolve_file_from_root(root: str, path_hint: Optional[str], extensions: Sequence[str]) -> str:
    if os.path.isfile(root):
        return root
    if not os.path.isdir(root):
        raise ValueError(f"ClearML dataset local copy not found: {root}")
    if path_hint:
        candidate = path_hint if os.path.isabs(path_hint) else os.path.join(root, path_hint)
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            matches = _find_files(candidate, extensions, limit=2)
            return _select_single_file(matches, candidate)
        raise ValueError(f"dataset.path not found: {candidate}")

    matches = _find_files(root, extensions, limit=2)
    return _select_single_file(matches, root)


def _find_files(root: str, extensions: Sequence[str], limit: int) -> list[str]:
    matches: list[str] = []
    exts = tuple(ext.lower() for ext in extensions)
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                matches.append(os.path.join(dirpath, name))
                if len(matches) >= limit:
                    return matches
    return matches


def _select_single_file(matches: list[str], root: str) -> str:
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"No matching files found under {root}; provide dataset.path")
    raise ValueError(f"Multiple matching files found under {root}; provide dataset.path")
