import importlib.util
import os
import sys
import types
import unittest


def _install_clearml_stub():
    if "clearml" in sys.modules:
        return

    clearml = types.ModuleType("clearml")

    class _TaskStub:
        class TaskTypes:
            training = "training"

        @staticmethod
        def current_task():
            return None

        @staticmethod
        def init(*_args, **_kwargs):
            return None

    clearml.Task = _TaskStub

    storage = types.ModuleType("clearml.storage")

    class _StorageManagerStub:
        @staticmethod
        def get_local_copy(uri):
            return uri

    storage.StorageManager = _StorageManagerStub
    clearml.storage = storage

    sys.modules["clearml"] = clearml
    sys.modules["clearml.storage"] = storage


def _load_train_module():
    _install_clearml_stub()
    root_dir = os.path.dirname(os.path.dirname(__file__))
    train_path = os.path.join(root_dir, "trainers", "yolo", "train.py")
    spec = importlib.util.spec_from_file_location("yolo_train", train_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["yolo_train"] = module
    spec.loader.exec_module(module)
    return module


class TestYoloExtrasPassThrough(unittest.TestCase):
    def test_extras_passthrough(self):
        train = _load_train_module()
        extras = {
            "optimizer": "AdamW",
            "patience": 7,
            "data": "s3://should/not/use.yaml",
            "val": False,
            "project": "runs/override",
            "name": "override-name",
            "epochs": 99,
            "drop": None,
        }

        train_args = train._build_train_args(
            data_path="local.yaml",
            epochs=3,
            batch=4,
            imgsz=640,
            device="0",
            project="runs/ultralytics",
            name="exp1",
            workers=0,
            yolo_extras=extras,
        )

        self.assertEqual(train_args["optimizer"], "AdamW")
        self.assertEqual(train_args["patience"], 7)
        self.assertEqual(train_args["data"], "local.yaml")
        self.assertEqual(train_args["val"], True)
        self.assertEqual(train_args["project"], "runs/ultralytics")
        self.assertEqual(train_args["name"], "exp1")
        self.assertEqual(train_args["epochs"], 3)
        self.assertNotIn("drop", train_args)


if __name__ == "__main__":
    unittest.main()
