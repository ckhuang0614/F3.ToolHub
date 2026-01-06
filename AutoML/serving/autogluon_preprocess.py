import json
import os
import tempfile
import zipfile
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _find_predictor_dir(root_dir: str) -> str:
    for root, _, files in os.walk(root_dir):
        if "predictor.pkl" in files:
            return root
    return root_dir


def _to_records(data: Any) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError("Request must be a dict or list of dicts")


def _dataframe_from_request(request: Any) -> pd.DataFrame:
    if isinstance(request, list):
        return pd.DataFrame(_to_records(request))

    if not isinstance(request, dict):
        raise ValueError("Request must be a dict or list of dicts")

    if "records" in request:
        return pd.DataFrame(_to_records(request["records"]))
    if "rows" in request:
        return pd.DataFrame(_to_records(request["rows"]))
    if "data" in request:
        data = request["data"]
        if isinstance(data, list):
            if data and isinstance(data[0], (list, tuple)):
                columns = request.get("columns")
                if not columns:
                    raise ValueError("columns is required when data is list of lists")
                return pd.DataFrame(data, columns=columns)
            return pd.DataFrame(_to_records(data))
    if "features" in request and isinstance(request["features"], dict):
        return pd.DataFrame([request["features"]])

    meta_keys = {"return_proba", "proba", "meta", "metadata"}
    features = {k: v for k, v in request.items() if k not in meta_keys}
    return pd.DataFrame([features])


def _to_serializable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


class Preprocess:
    def __init__(self):
        self.model_endpoint = None
        self._predictor = None
        self._model_dir = None

    def load(self, model_path: Optional[str]):
        if not model_path:
            raise ValueError("model_path is empty")

        local_path = model_path
        if os.path.isfile(local_path) and local_path.lower().endswith(".zip"):
            extract_dir = tempfile.mkdtemp(prefix="ag_model_")
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(extract_dir)
            local_path = extract_dir
        elif os.path.isfile(local_path):
            raise ValueError("Autogluon model must be a directory or zip archive")

        model_dir = _find_predictor_dir(local_path)
        self._model_dir = model_dir

        from autogluon.tabular import TabularPredictor

        self._predictor = TabularPredictor.load(model_dir)
        return self._predictor

    def preprocess(
        self,
        request: Dict[str, Any],
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> pd.DataFrame:
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        return_proba = bool(request.get("return_proba") or request.get("proba"))
        state["return_proba"] = return_proba
        return _dataframe_from_request(request)

    def process(
        self,
        data: pd.DataFrame,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Dict[str, Any]:
        if self._predictor is None:
            raise ValueError("Predictor is not loaded")
        predictions = self._predictor.predict(data)
        result = {"predictions": _to_serializable(predictions)}

        if state.get("return_proba"):
            try:
                proba = self._predictor.predict_proba(data)
                result["probabilities"] = _to_serializable(proba)
            except Exception as exc:
                result["probabilities_error"] = str(exc)

        return result

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        return data
