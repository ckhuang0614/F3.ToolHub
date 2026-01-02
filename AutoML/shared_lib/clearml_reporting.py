from __future__ import annotations

import json
from typing import Any, Dict


def report_table_from_csv(logger, path: str, title: str, series: str, max_rows: int = 200) -> None:
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        logger.report_table(title=title, series=series, iteration=0, table_plot=df)
        return
    except Exception as exc:
        try:
            logger.report_text(f"{title} table report failed: {exc}")
        except Exception:
            pass


def _stringify_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        import pandas as pd

        if isinstance(value, (pd.DataFrame, pd.Series)):
            preview = value.head(10)
            return preview.to_string(index=False)
    except Exception:
        pass
    try:
        return json.dumps(value, ensure_ascii=True, default=str)
    except Exception:
        return str(value)


def report_kv_table(logger, title: str, series: str, data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    rows = []
    for key, value in data.items():
        text = _stringify_value(value)
        if text in (None, ""):
            continue
        rows.append({"key": str(key), "value": text})
    if not rows:
        return
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        logger.report_table(title=title, series=series, iteration=0, table_plot=df)
    except Exception as exc:
        try:
            lines = [f"{row['key']}: {row['value']}" for row in rows]
            logger.report_text("\n".join(lines))
        except Exception:
            pass
