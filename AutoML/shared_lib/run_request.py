from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# -------------------------
# Dataset specs (v2 schema)
# -------------------------


@dataclass(frozen=True)
class TabularDatasetSpec:
    """Tabular dataset.

    uri: typically s3://datasets/xxx.csv (or any ClearML StorageManager-supported URI)
    label: target column name
    """

    uri: str
    label: str
    type: str = "tabular"


@dataclass(frozen=True)
class YoloDatasetSpec:
    """YOLO dataset.

    uri: can be a dataset.yaml (or a directory) or a zip containing dataset.yaml
    yaml_path: optional hint for yaml path inside zip (or inside directory)
    """

    uri: str
    yaml_path: Optional[str] = None
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
    extras: Optional[Dict[str, Any]] = None

    schema_version: int = 2

    @property
    def is_yolo(self) -> bool:
        return getattr(self.dataset, "type", "") == "yolo" or self.trainer == "yolo" or self.task_type == "detection"

    @property
    def is_tabular(self) -> bool:
        return getattr(self.dataset, "type", "") == "tabular" or self.trainer in {"autogluon", "flaml"}

    @staticmethod
    def _parse_dataset(d: Dict[str, Any]) -> DatasetSpec:
        if not d:
            raise ValueError("dataset is required")

        # v2 explicit typing
        ds_type = (d.get("type") or "").strip().lower()
        if ds_type in {"tabular", "yolo"}:
            if ds_type == "tabular":
                uri = d.get("uri") or d.get("csv_uri")
                label = d.get("label") or d.get("target")
                if not uri or not label:
                    raise ValueError("tabular dataset requires dataset.uri (or csv_uri) and dataset.label (or target)")
                return TabularDatasetSpec(uri=str(uri), label=str(label))

            uri = d.get("uri") or d.get("csv_uri")
            if not uri:
                raise ValueError("yolo dataset requires dataset.uri (or csv_uri)")
            yaml_path = d.get("yaml_path") or d.get("yaml") or d.get("target")
            return YoloDatasetSpec(uri=str(uri), yaml_path=str(yaml_path) if yaml_path else None)

        # v1 (implicit) - decide later in from_dict based on trainer/task_type
        # We still parse keys so we can up-convert.
        uri = d.get("uri") or d.get("csv_uri")
        target = d.get("label") or d.get("target")
        return {"_uri": uri, "_target": target}  # type: ignore[return-value]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunRequest":
        if not d:
            raise ValueError("RunRequest payload is empty")

        trainer = str(d.get("trainer") or "").strip().lower()
        if trainer not in {"autogluon", "flaml", "yolo"}:
            raise ValueError("trainer must be autogluon, flaml or yolo")

        task_type = str(d.get("task_type", "classification")).strip().lower()
        if not task_type:
            task_type = "classification"

        ds_raw = d.get("dataset") or {}
        parsed = RunRequest._parse_dataset(ds_raw)

        # Handle v1 implicit dataset
        if isinstance(parsed, dict):
            uri = parsed.get("_uri")
            target = parsed.get("_target")

            is_yolo = (trainer == "yolo") or (task_type == "detection")
            if is_yolo:
                if not uri:
                    raise ValueError("yolo dataset requires dataset.csv_uri (or dataset.uri)")
                dataset: DatasetSpec = YoloDatasetSpec(uri=str(uri), yaml_path=str(target) if target else None)
            else:
                if not uri or not target:
                    raise ValueError("tabular dataset requires dataset.csv_uri (or dataset.uri) and dataset.target (or dataset.label)")
                dataset = TabularDatasetSpec(uri=str(uri), label=str(target))
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

        req = RunRequest(
            trainer=trainer,
            dataset=dataset,
            time_budget_s=int(d.get("time_budget_s")),
            metric=str(d.get("metric")),
            task_type=task_type,
            group_key=group_key,
            split=split,
            run_name=d.get("run_name"),
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
            if not req.dataset.uri or not req.dataset.label:
                raise ValueError("tabular dataset requires dataset.uri and dataset.label")
        elif isinstance(req.dataset, YoloDatasetSpec):
            if not req.dataset.uri:
                raise ValueError("yolo dataset requires dataset.uri")
        else:
            raise ValueError("dataset parsing failed")

        return req
