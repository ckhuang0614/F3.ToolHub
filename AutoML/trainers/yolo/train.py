import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from typing import Optional, Tuple

from clearml import Task
from clearml.storage import StorageManager

from shared_lib.run_request import RunRequest

# Minimal YOLO training script template using ClearML
# This script uses ultralytics (YOLOv8) as an example. Adjust imports and training code
# if you prefer another YOLO implementation.


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO training with ClearML")
    parser.add_argument("--data", type=str, help="Path or s3 uri to dataset yaml or zip")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    return parser.parse_args()


DEFAULTS = {
    "epochs": 50,
    "batch": 16,
    "imgsz": 640,
    "project": "YOLO",
    "name": "yolov8-train-demo",
    "device": "0",
    "weights": "yolov8n.pt",
    "workers": 8,
}


def _load_run_request(task: Task) -> Optional[RunRequest]:
    s = task.get_parameter("RunRequest/json")
    if not s:
        return None
    return RunRequest.from_dict(json.loads(s))


def _yolo_extras(rr: Optional[RunRequest]) -> dict:
    if not rr:
        return {}
    extras = rr.extras or {}
    if not isinstance(extras, dict):
        return {}
    y = extras.get("yolo") or {}
    if not isinstance(y, dict):
        return {}
    return y


def _find_yaml(root_dir: str) -> Optional[str]:
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith((".yaml", ".yml")):
                return os.path.join(root, name)
    return None


def _find_named_file(root_dir: str, name: str) -> Optional[str]:
    target = name.lower()
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.lower() == target:
                return os.path.join(root, fname)
    return None


def _resolve_data_path(data_uri: str, target: Optional[str]) -> Tuple[str, Optional[str]]:
    local_path = data_uri
    if "://" in data_uri and not os.path.exists(data_uri):
        local_path = StorageManager.get_local_copy(data_uri)

    if os.path.isdir(local_path):
        if target:
            candidate = os.path.join(local_path, target)
            if os.path.exists(candidate):
                return candidate, None
            named = _find_named_file(local_path, target)
            if named:
                return named, None
        yaml_path = _find_yaml(local_path)
        if yaml_path:
            return yaml_path, None
        raise ValueError(f"Dataset directory does not contain a yaml file: {local_path}")

    if local_path.lower().endswith(".zip"):
        extract_dir = tempfile.mkdtemp(prefix="yolo_data_")
        with zipfile.ZipFile(local_path, "r") as zf:
            zf.extractall(extract_dir)
        if target:
            candidate = os.path.join(extract_dir, target)
            if os.path.exists(candidate):
                return candidate, extract_dir
            raise ValueError(f"Dataset target not found in zip: {candidate}")
        yaml_path = _find_yaml(extract_dir)
        if yaml_path:
            return yaml_path, extract_dir
        raise ValueError("Dataset zip does not contain a yaml file")

    return local_path, None


def _patch_final_eval(data_uri: str, data_path: str) -> None:
    try:
        from ultralytics.engine import trainer as trainer_mod
    except Exception:
        return

    candidates = []
    for name in ("BaseTrainer", "Trainer"):
        cls = getattr(trainer_mod, name, None)
        if cls and hasattr(cls, "final_eval"):
            candidates.append(cls)
    if not candidates:
        return

    def _wrap_final_eval(orig_final_eval):
        def _final_eval(self):
            try:
                current_data = getattr(self.args, "data", None)
                if (
                    data_path
                    and os.path.exists(data_path)
                    and (current_data == data_uri or (isinstance(current_data, str) and "://" in current_data))
                ):
                    self.args.data = data_path
                    if getattr(self, "validator", None) and hasattr(self.validator, "args"):
                        self.validator.args.data = data_path
            except Exception:
                pass

            try:
                return orig_final_eval(self)
            except FileNotFoundError as exc:
                if data_uri and data_uri in str(exc):
                    try:
                        self.args.data = data_path
                        if getattr(self, "validator", None) and hasattr(self.validator, "args"):
                            self.validator.args.data = data_path
                    except Exception:
                        pass
                    return orig_final_eval(self)
                raise

        _final_eval._clearml_patched = True
        return _final_eval

    for cls in candidates:
        if getattr(cls.final_eval, "_clearml_patched", False):
            continue
        cls.final_eval = _wrap_final_eval(cls.final_eval)


def main():
    args = parse_args()

    task = Task.current_task() or Task.init(
        project_name=args.project or DEFAULTS["project"],
        task_name=args.name or DEFAULTS["name"],
        task_type=Task.TaskTypes.training,
    )
    rr = _load_run_request(task)
    if rr and rr.run_name:
        task.set_name(rr.run_name)

    yolo_extras = _yolo_extras(rr)
    data_uri = args.data or (rr.dataset.csv_uri if rr else None)
    if not data_uri:
        raise ValueError("Missing dataset; provide --data or RunRequest/json.dataset.csv_uri")

    epochs = args.epochs if args.epochs is not None else int(yolo_extras.get("epochs", DEFAULTS["epochs"]))
    batch = args.batch if args.batch is not None else int(yolo_extras.get("batch", DEFAULTS["batch"]))
    imgsz = args.imgsz if args.imgsz is not None else int(yolo_extras.get("imgsz", DEFAULTS["imgsz"]))
    device = args.device if args.device is not None else str(yolo_extras.get("device", DEFAULTS["device"]))
    weights = args.weights if args.weights is not None else str(yolo_extras.get("weights", DEFAULTS["weights"]))
    workers = args.workers if args.workers is not None else int(yolo_extras.get("workers", DEFAULTS["workers"]))
    name = args.name if args.name is not None else (rr.run_name if rr and rr.run_name else DEFAULTS["name"])

    # connect resolved config so it appears in ClearML console
    task.connect(
        {
            "data": data_uri,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "weights": weights,
            "workers": workers,
            "name": name,
        }
    )

    # get logger to report scalars
    logger = task.get_logger()

    # Local training using ultralytics (YOLOv8). Import lazily so script fails early if missing.
    try:
        from ultralytics import YOLO
    except Exception:
        task.close()
        raise

    cleanup_dir = None
    try:
        data_path, cleanup_dir = _resolve_data_path(data_uri, rr.dataset.target if rr else None)
        print(f"Resolved data_path: {data_path}")
        _patch_final_eval(data_uri, data_path)
        # Ensure Ultralytics sees the resolved local dataset path.
        args.data = data_path
        sys.argv = [
            "train.py",
            "--data",
            data_path,
            "--epochs",
            str(epochs),
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
            "--device",
            str(device),
            "--weights",
            str(weights),
            "--workers",
            str(workers),
            "--name",
            str(name),
        ]

        # create simple training config and run
        model = YOLO(weights)
        # force ultralytics overrides to use resolved local data path (prevents final eval from reusing s3 uri)
        try:
            model.overrides["data"] = data_path
            model.overrides["val"] = True
        except Exception:
            pass
        # Ensure validator uses the resolved data_path during any validation hooks
        def _force_validator_data(trainer):
            try:
                trainer.validator.args.data = data_path
            except Exception:
                pass
        model.add_callback("on_val_start", _force_validator_data)
        model.add_callback("on_predict_start", _force_validator_data)

        # ultralytics accepts dict or cli args
        train_args = {
            "data": data_path,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "project": os.getenv("YOLO_OUTPUT_DIR", "runs/ultralytics"),
            "name": name,
            "workers": workers,
            # keep val True so callbacks set the local path before eval
            "val": True,
        }

        # Run training
        results = model.train(**train_args)
        if not train_args.get("val", True):
            results = model.val(data=data_path)
    finally:
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    # example: report final mAP if available in results
    try:
        metrics = results.metrics if hasattr(results, 'metrics') else None
        if metrics and "mAP50" in metrics:
            logger.report_scalar("val/mAP50", "mAP50", iteration=epochs, value=float(metrics["mAP50"]))
    except Exception:
        pass

    # upload final weights if exists
    out_dir = train_args['project'] + '/' + train_args['name']
    weights_path = os.path.join(out_dir, 'weights', 'best.pt')
    if os.path.exists(weights_path):
        task.upload_artifact('model', weights_path)

    task.close()


if __name__ == '__main__':
    main()
