# Ultralytics YOLO (trainers/ultralytics/train.py) - Implementation & Params

## Implementation Summary

- Reads ClearML `RunRequest/json`; falls back to raw payload/Task parameters/env/CLI.
- `trainer` must be `ultralytics` to pass `RunRequest.from_dict` validation.
- YOLO dataset uses `YoloDatasetSpec` (yaml/zip/dir/remote URI). Zip is auto-extracted and a yaml is searched.
- Patches Ultralytics `final_eval` to force local dataset path (avoids s3:// issues).
- Forces `val=True`.
- Uploads `project/name/weights/best.pt` to ClearML if present.
- Reports `mAP50` to ClearML when available.

## Required RunRequest JSON

- `trainer`: `ultralytics`
- `dataset`:
  - `type`: `yolo`
  - `uri`: yaml/zip/dir/remote URI
  - `yaml_path` / `yaml` / `target` (optional, for zip/dir)
- `run_name` (optional)
- `extras.yolo` (optional; see below)

## Supported Parameters

### extras.yolo
All keys are passed through to `YOLO.train(**train_args)`.
The wrapper also reads these keys as defaults:
- `epochs`, `batch`, `imgsz`, `device`, `weights`, `workers`, `project`, `name`

### CLI args
- `--data`, `--epochs`, `--batch`, `--imgsz`, `--project`, `--name`, `--device`, `--weights`, `--workers`

### ClearML Task Args
- `Args/data`, `Args/epochs`, `Args/batch`, `Args/imgsz`, `Args/device`, `Args/weights`, `Args/workers`, `Args/project`, `Args/name`

### Environment Variables
- Dataset: `YOLO_DATA_URI` / `YOLO_DATA` / `DATA_URI` / `DATA`
- Output dir: `YOLO_OUTPUT_DIR`

## Train Args Actually Passed to Ultralytics

- Always includes: `data`, `epochs`, `batch`, `imgsz`, `device`, `project`, `name`, `workers`, `val=True`
- Plus all non-None keys from `extras.yolo`

Note: `data/epochs/batch/imgsz/device/project/name/workers/val` are overwritten by the wrapper,
so `extras.yolo` cannot override them.

## Priority Order (Key Fields)

- `epochs/batch/imgsz/device/weights/workers`: CLI > ClearML Args > extras.yolo > DEFAULTS
- `project`: CLI > ClearML Args > extras.yolo.project > YOLO_OUTPUT_DIR > runs/ultralytics
- `name`: CLI > ClearML Args > RunRequest.run_name > payload.run_name > extras.yolo.name > raw payload > DEFAULTS
- `data`: RunRequest.dataset > payload > CLI > env > ClearML Args > raw payload fallback

## Defaults

- `epochs=50`, `batch=16`, `imgsz=640`, `project="YOLO"`, `name="yolov8-train-demo"`,
  `device="0"`, `weights="yolov8n.pt"`, `workers=8`

## Notes

- `weights` is used to initialize `YOLO(weights)`, not a `train()` arg.
- `val=True` is forced and cannot be disabled.
- This wrapper is YOLO-only; `dataset.type` should be `yolo`.
