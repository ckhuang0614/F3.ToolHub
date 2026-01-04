from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# -------------------------
# Dataset specs (v2 schema)
# -------------------------


@dataclass(frozen=True)
class ClearMLDatasetRef:
    id: Optional[str] = None
    name: Optional[str] = None
    project: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None


@dataclass(frozen=True)
class TabularDatasetSpec:
    """Tabular dataset.

    uri: typically s3://datasets/xxx.csv (or any ClearML StorageManager-supported URI)
    label: target column name
    path: optional relative path inside a ClearML Dataset
    """

    uri: Optional[str]
    label: str
    path: Optional[str] = None
    clearml: Optional[ClearMLDatasetRef] = None
    type: str = "tabular"


@dataclass(frozen=True)
class YoloDatasetSpec:
    """YOLO dataset.

    uri: can be a dataset.yaml (or a directory) or a zip containing dataset.yaml
    yaml_path: optional hint for yaml path inside zip (or inside directory)
    """

    uri: Optional[str]
    yaml_path: Optional[str] = None
    clearml: Optional[ClearMLDatasetRef] = None
    type: str = "yolo"


DatasetSpec = Union[TabularDatasetSpec, YoloDatasetSpec]


@dataclass(frozen=True)
class SplitSpec:
    """Split spec.

    - group_shuffle: split by group_id derived from group_key columns
    - row_shuffle: split by rows (random shuffle)

    NOTE: If method=group_shuffle but group_key is empty (or collapses to a single group),
    the split implementation is expected to fall back to row_shuffle.
    """

    method: str = "group_shuffle"
    test_size: float = 0.2
    random_seed: int = 42


@dataclass
class RunRequest:
    """Unified request contract for AutoML trainers.

    Backward compatibility:
    - Old tabular payload: dataset.{csv_uri,target}
    - Old yolo payload: dataset.{csv_uri,target} where target acted as yaml path inside zip
    """

    trainer: str
    dataset: DatasetSpec
    time_budget_s: int
    metric: str
    task_type: str = "classification"

    # Optional for tabular; used when split.method=group_shuffle.
    group_key: List[str] = field(default_factory=list)

    split: SplitSpec = field(default_factory=SplitSpec)
    run_name: Optional[str] = None
    queue: Optional[str] = None
    project: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extras: Optional[Dict[str, Any]] = None

    schema_version: int = 2

    @property
    def is_yolo(self) -> bool:
        return getattr(self.dataset, "type", "") == "yolo" or self.trainer == "ultralytics" or self.task_type == "detection"

    @property
    def is_tabular(self) -> bool:
        return getattr(self.dataset, "type", "") == "tabular" or self.trainer in {"autogluon", "flaml"}

    @staticmethod
    def _parse_clearml_ref(d: Dict[str, Any]) -> Optional[ClearMLDatasetRef]:
        ref = d.get("clearml") or d.get("clearml_dataset") or d.get("dataset_ref")
        ref_dict: Dict[str, Any] = {}
        if isinstance(ref, str):
            ref_dict["id"] = ref
        elif isinstance(ref, dict):
            ref_dict = dict(ref)

        def _pick(*keys: str) -> Optional[Any]:
            for key in keys:
                value = ref_dict.get(key)
                if value not in (None, ""):
                    return value
            for key in keys:
                value = d.get(key)
                if value not in (None, ""):
                    return value
            return None

        dataset_id = _pick("id", "dataset_id", "clearml_dataset_id")
        name = _pick("name", "dataset_name", "clearml_dataset_name")
        project = _pick("project", "dataset_project", "clearml_dataset_project")
        version = _pick("version", "dataset_version", "clearml_dataset_version")
        tags = _pick("tags", "dataset_tags", "clearml_dataset_tags")

        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        elif tags is not None:
            if isinstance(tags, (list, tuple, set)):
                tags = [str(t).strip() for t in tags if str(t).strip()]
            else:
                tag = str(tags).strip()
                tags = [tag] if tag else None

        if not any([dataset_id, name, project, version, tags]):
            return None

        return ClearMLDatasetRef(
            id=str(dataset_id) if dataset_id else None,
            name=str(name) if name else None,
            project=str(project) if project else None,
            version=str(version) if version else None,
            tags=tags if tags else None,
        )

    @staticmethod
    def _parse_dataset(d: Dict[str, Any]) -> DatasetSpec:
        if not d:
            raise ValueError("dataset is required")

        clearml_ref = RunRequest._parse_clearml_ref(d)
        tabular_path = d.get("path") or d.get("file") or d.get("file_path") or d.get("csv_path")

        # v2 explicit typing
        ds_type = (d.get("type") or "").strip().lower()
        if ds_type in {"tabular", "yolo"}:
            if ds_type == "tabular":
                label = d.get("label") or d.get("target")
                uri = d.get("uri") or d.get("csv_uri")
                if not label:
                    raise ValueError("tabular dataset requires dataset.label (or target)")
                if not uri and not clearml_ref:
                    raise ValueError("tabular dataset requires dataset.uri (or csv_uri) or dataset.clearml")
                return TabularDatasetSpec(
                    uri=str(uri) if uri else None,
                    label=str(label),
                    path=str(tabular_path) if tabular_path else None,
                    clearml=clearml_ref,
                )

            uri = d.get("uri") or d.get("csv_uri")
            yaml_path = (
                d.get("yaml_path")
                or d.get("yaml")
                or d.get("target")
                or d.get("path")
                or d.get("file")
                or d.get("file_path")
            )
            if not uri and not clearml_ref:
                raise ValueError("yolo dataset requires dataset.uri (or csv_uri) or dataset.clearml")
            return YoloDatasetSpec(
                uri=str(uri) if uri else None,
                yaml_path=str(yaml_path) if yaml_path else None,
                clearml=clearml_ref,
            )

        # v1 (implicit) - decide later in from_dict based on trainer/task_type
        # We still parse keys so we can up-convert.
        uri = d.get("uri") or d.get("csv_uri")
        target = d.get("label") or d.get("target")
        yaml_path = d.get("yaml_path") or d.get("yaml")
        return {  # type: ignore[return-value]
            "_uri": uri,
            "_target": target,
            "_path": tabular_path,
            "_yaml_path": yaml_path,
            "_clearml": clearml_ref,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunRequest":
        if not d:
            raise ValueError("RunRequest payload is empty")

        trainer = str(d.get("trainer") or "").strip().lower()
        if trainer not in {"autogluon", "flaml", "ultralytics"}:
            raise ValueError("trainer must be autogluon, flaml or ultralytics")

        task_type = str(d.get("task_type", "classification")).strip().lower()
        if not task_type:
            task_type = "classification"

        ds_raw = d.get("dataset") or {}
        parsed = RunRequest._parse_dataset(ds_raw)

        # Handle v1 implicit dataset
        if isinstance(parsed, dict):
            uri = parsed.get("_uri")
            target = parsed.get("_target")
            path = parsed.get("_path")
            yaml_path = parsed.get("_yaml_path")
            clearml_ref = parsed.get("_clearml")

            is_yolo = (trainer == "ultralytics") or (task_type == "detection")
            if is_yolo:
                if not uri and not clearml_ref:
                    raise ValueError("yolo dataset requires dataset.csv_uri (or dataset.uri) or dataset.clearml")
                dataset = YoloDatasetSpec(
                    uri=str(uri) if uri else None,
                    yaml_path=str(yaml_path) if yaml_path else (str(target) if target else None),
                    clearml=clearml_ref,
                )
            else:
                if not target:
                    raise ValueError("tabular dataset requires dataset.target (or dataset.label)")
                if not uri and not clearml_ref:
                    raise ValueError("tabular dataset requires dataset.csv_uri (or dataset.uri) or dataset.clearml")
                dataset = TabularDatasetSpec(
                    uri=str(uri) if uri else None,
                    label=str(target),
                    path=str(path) if path else None,
                    clearml=clearml_ref,
                )
        else:
            dataset = parsed

        sp = d.get("split") or {}
        split = SplitSpec(
            method=str(sp.get("method", "group_shuffle")).strip().lower(),
            test_size=float(sp.get("test_size", 0.2)),
            random_seed=int(sp.get("random_seed", 42)),
        )

        group_key_raw = d.get("group_key")
        if group_key_raw in (None, ""):
            group_key = []
        elif isinstance(group_key_raw, str):
            group_key = [group_key_raw]
        else:
            # expect list/tuple
            group_key = list(group_key_raw)

        queue_raw = d.get("queue") or d.get("execution_queue") or d.get("clearml_queue")
        queue = str(queue_raw).strip() if queue_raw not in (None, "") else None
        project_raw = d.get("project") or d.get("project_name")
        project = str(project_raw).strip() if project_raw not in (None, "") else None

        tags_raw = d.get("tags") or d.get("task_tags") or d.get("clearml_tags")
        if tags_raw in (None, ""):
            tags = []
        elif isinstance(tags_raw, str):
            if "," in tags_raw:
                tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
            else:
                tags = [tags_raw.strip()] if tags_raw.strip() else []
        elif isinstance(tags_raw, (list, tuple, set)):
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
        else:
            tag = str(tags_raw).strip()
            tags = [tag] if tag else []

        req = RunRequest(
            trainer=trainer,
            dataset=dataset,
            time_budget_s=int(d.get("time_budget_s")),
            metric=str(d.get("metric")),
            task_type=task_type,
            group_key=group_key,
            split=split,
            run_name=d.get("run_name"),
            queue=queue,
            project=project,
            tags=tags,
            extras=d.get("extras"),
            schema_version=int(d.get("schema_version", 2)),
        )

        # -------- validation --------
        if req.time_budget_s < 30:
            raise ValueError("time_budget_s must be >= 30")

        if req.split.method not in {"group_shuffle", "row_shuffle"}:
            raise ValueError("split.method must be group_shuffle or row_shuffle")

        if not (0.05 <= req.split.test_size <= 0.5):
            raise ValueError("split.test_size out of range (0.05~0.5)")

        # Dataset-type specific checks
        if isinstance(req.dataset, TabularDatasetSpec):
            if not req.dataset.label:
                raise ValueError("tabular dataset requires dataset.label")
            if not req.dataset.uri and not req.dataset.clearml:
                raise ValueError("tabular dataset requires dataset.uri or dataset.clearml")
            if req.dataset.clearml and not (req.dataset.clearml.id or req.dataset.clearml.name):
                raise ValueError("dataset.clearml requires id or name")
        elif isinstance(req.dataset, YoloDatasetSpec):
            if not req.dataset.uri and not req.dataset.clearml:
                raise ValueError("yolo dataset requires dataset.uri or dataset.clearml")
            if req.dataset.clearml and not (req.dataset.clearml.id or req.dataset.clearml.name):
                raise ValueError("dataset.clearml requires id or name")
        else:
            raise ValueError("dataset parsing failed")

        return req
