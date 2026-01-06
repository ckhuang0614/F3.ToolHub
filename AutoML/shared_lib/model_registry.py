from __future__ import annotations

import os
from typing import Iterable


def _parse_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        return parts or [raw]
    if isinstance(value, Iterable):
        items = [str(item).strip() for item in value]
        return [item for item in items if item]
    return [str(value).strip()]


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_model_status(model, logger=None) -> dict:
    """
    Optionally tag or publish a ClearML model based on env flags.

    - CLEARML_MODEL_TAGS: comma-separated tags to apply (e.g. "candidate")
    - CLEARML_MODEL_PUBLISH: if truthy, call model.publish()
    """
    if model is None:
        return {"tags": [], "published": False}

    tags = _parse_tags(os.getenv("CLEARML_MODEL_TAGS"))
    published = False

    if tags:
        adder = getattr(model, "add_tags", None)
        setter = getattr(model, "set_tags", None)
        try:
            if callable(adder):
                adder(tags)
            elif callable(setter):
                setter(tags)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                try:
                    logger.report_text(f"model tag update failed: {exc}")
                except Exception:
                    pass

    if _truthy(os.getenv("CLEARML_MODEL_PUBLISH")):
        try:
            model.publish()
            published = True
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                try:
                    logger.report_text(f"model publish failed: {exc}")
                except Exception:
                    pass

    return {"tags": tags, "published": published}
