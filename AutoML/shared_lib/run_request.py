from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class DatasetSpec:
    csv_uri: str
    target: str


@dataclass
class SplitSpec:
    method: str = "group_shuffle"
    test_size: float = 0.2
    random_seed: int = 42


@dataclass
class RunRequest:
    trainer: str
    dataset: DatasetSpec
    group_key: List[str]
    time_budget_s: int
    metric: str
    task_type: str = "classification"
    split: SplitSpec = SplitSpec()
    run_name: Optional[str] = None
    extras: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunRequest":
        if not d:
            raise ValueError("RunRequest payload is empty")
        ds = d.get("dataset") or {}
        sp = d.get("split") or {}
        req = RunRequest(
            trainer=d["trainer"],
            dataset=DatasetSpec(csv_uri=ds["csv_uri"], target=ds["target"]),
            group_key=list(d["group_key"]),
            time_budget_s=int(d["time_budget_s"]),
            metric=str(d["metric"]),
            task_type=str(d.get("task_type", "classification")),
            split=SplitSpec(
                method=str(sp.get("method", "group_shuffle")),
                test_size=float(sp.get("test_size", 0.2)),
                random_seed=int(sp.get("random_seed", 42)),
            ),
            run_name=d.get("run_name"),
            extras=d.get("extras"),
        )
        if req.time_budget_s < 30:
            raise ValueError("time_budget_s must be >= 30")
        if not req.group_key:
            raise ValueError("group_key must not be empty")
        if req.split.method not in {"group_shuffle"}:
            raise ValueError("split.method must be group_shuffle")
        if not (0.05 <= req.split.test_size <= 0.5):
            raise ValueError("split.test_size out of range (0.05~0.5)")
        if req.trainer not in {"autogluon", "flaml"}:
            raise ValueError("trainer must be autogluon or flaml")
        return req
