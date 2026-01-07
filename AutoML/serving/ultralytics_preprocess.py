import base64
import io
import os
import tempfile
import zipfile
from typing import Any, Dict, Iterable, Optional

from PIL import Image


def _decode_base64_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        _, data = data.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(data)))


def _load_image(item: Any) -> Image.Image:
    if isinstance(item, Image.Image):
        return item
    if isinstance(item, bytes):
        return Image.open(io.BytesIO(item))
    if not isinstance(item, str):
        raise ValueError("image must be a path, url, or base64 string")
    if item.startswith("http://") or item.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(item) as resp:
            return Image.open(io.BytesIO(resp.read()))
    if os.path.isfile(item):
        return Image.open(item)
    return _decode_base64_image(item)


def _iter_images(request: Dict[str, Any]) -> Iterable[Image.Image]:
    if "image" in request:
        yield _load_image(request["image"])
        return
    if "image_base64" in request:
        yield _decode_base64_image(request["image_base64"])
        return
    if "image_url" in request:
        yield _load_image(request["image_url"])
        return
    if "image_path" in request:
        yield _load_image(request["image_path"])
        return
    if "images" in request and isinstance(request["images"], list):
        for item in request["images"]:
            yield _load_image(item)
        return
    raise ValueError("Missing image input (image/image_base64/image_url/image_path/images)")


def _find_model_file(root_dir: str) -> str:
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".pt"):
                return os.path.join(root, name)
    raise ValueError("No .pt model file found")


def _to_serializable(values: Any) -> Any:
    to_list = getattr(values, "tolist", None)
    if callable(to_list):
        return to_list()
    if isinstance(values, dict):
        return {k: _to_serializable(v) for k, v in values.items()}
    if isinstance(values, (list, tuple)):
        return [_to_serializable(v) for v in values]
    return values


class Preprocess:
    def __init__(self):
        self.model_endpoint = None
        self._model = None

    def load(self, model_path: Optional[str]):
        if not model_path:
            raise ValueError("model_path is empty")

        local_path = model_path
        if os.path.isfile(local_path) and local_path.lower().endswith(".zip"):
            extract_dir = tempfile.mkdtemp(prefix="yolo_model_")
            with zipfile.ZipFile(local_path, "r") as zf:
                zf.extractall(extract_dir)
            local_path = extract_dir

        if os.path.isdir(local_path):
            model_file = _find_model_file(local_path)
        elif os.path.isfile(local_path):
            model_file = local_path
        else:
            raise ValueError("Ultralytics model path is invalid")

        from ultralytics import YOLO

        self._model = YOLO(model_file)
        return self._model

    def preprocess(
        self,
        request: Dict[str, Any],
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        if not isinstance(state, dict):
            raise ValueError("state must be a dict")
        state["infer_args"] = {
            "conf": request.get("conf"),
            "iou": request.get("iou"),
            "imgsz": request.get("imgsz"),
            "max_det": request.get("max_det"),
        }
        return list(_iter_images(request))

    def process(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Dict[str, Any]:
        if self._model is None:
            raise ValueError("Model is not loaded")
        infer_args = {k: v for k, v in state.get("infer_args", {}).items() if v is not None}
        results = self._model.predict(data, **infer_args)

        outputs = []
        for res in results:
            boxes = getattr(res, "boxes", None)
            outputs.append(
                {
                    "boxes": _to_serializable(getattr(boxes, "xyxy", [])),
                    "scores": _to_serializable(getattr(boxes, "conf", [])),
                    "classes": _to_serializable(getattr(boxes, "cls", [])),
                    "names": getattr(res, "names", None),
                }
            )
        return {"results": outputs}

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        return data
