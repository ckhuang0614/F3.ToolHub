import os
import tempfile
import zipfile
from typing import Any, Dict, Optional

import joblib
import pandas as pd


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


def _find_model_file(root_dir: str) -> str:
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".pkl"):
                return os.path.join(root, name)
    raise ValueError("No .pkl model file found")


class Preprocess:
    def __init__(self):
        self.model_endpoint = None
        self._model = None

    def load(self, model_path: Optional[str]):
        if not model_path:
            raise ValueError("model_path is empty")

        local_path = model_path
        if os.path.isfile(local_path) and local_path.lower().endswith(".zip"):
            extract_dir = tempfile.mkdtemp(prefix="flaml_model_")
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(extract_dir)
            local_path = extract_dir

        if os.path.isdir(local_path):
            model_file = _find_model_file(local_path)
        elif os.path.isfile(local_path):
            model_file = local_path
        else:
            raise ValueError("FLAML model path is invalid")

        self._model = joblib.load(model_file)
        return self._model

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
        if self._model is None:
            raise ValueError("Model is not loaded")
        predictions = self._model.predict(data)
        result = {"predictions": _to_serializable(predictions)}

        if state.get("return_proba"):
            try:
                proba = self._model.predict_proba(data)
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
