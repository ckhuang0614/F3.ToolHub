#!/usr/bin/env python3
"""
Prepare YOLO dataset from a simple annotation list and package into yolo_dataset.zip.

Expected input:
 - annotations/train_annotations.txt with lines like:
   images/train/demo_image.jpg 0 50 50 200 200
   (path class x_min y_min x_max y_max)

This script will:
 - create directories `images/train`, `images/val`, `labels/train`, `labels/val` if missing
 - for each annotation, read image to get its width/height, convert bbox to YOLO format
   (class x_center y_center width height) with normalized coordinates
 - write label files next to images under `labels/train` or `labels/val`
 - create `yolo_dataset.zip` containing `images`, `labels`, and `labels.yaml`

Usage:
  cd AutoML/dataset
  python prepare_yolo_dataset.py --annotations annotations/train_annotations.txt --out yolo_dataset.zip

"""
import argparse
import os
from pathlib import Path
import shutil
import zipfile

try:
    from PIL import Image
except Exception:
    raise SystemExit("Pillow is required. Install with: pip install Pillow")


def ensure_dirs(base: Path):
    for d in (base / 'images' / 'train', base / 'images' / 'val', base / 'labels' / 'train', base / 'labels' / 'val'):
        d.mkdir(parents=True, exist_ok=True)


def parse_annotation_line(line: str):
    parts = line.strip().split()
    if len(parts) != 6:
        raise ValueError(f"Bad annotation line: {line}")
    img_path, cls, x_min, y_min, x_max, y_max = parts
    return img_path, int(cls), int(x_min), int(y_min), int(x_max), int(y_max)


def convert_and_write_label(base: Path, img_path: Path, cls: int, x_min: int, y_min: int, x_max: int, y_max: int, subset: str = 'train'):
    # open image to get size
    with Image.open(img_path) as im:
        w, h = im.size

    x_center = (x_min + x_max) / 2.0 / w
    y_center = (y_min + y_max) / 2.0 / h
    width = (x_max - x_min) / w
    height = (y_max - y_min) / h

    # ensure label dir
    label_dir = base / 'labels' / subset
    label_dir.mkdir(parents=True, exist_ok=True)

    label_file = label_dir / (img_path.stem + '.txt')
    with open(label_file, 'w', encoding='utf-8') as f:
        f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', type=str, default='annotations/train_annotations.txt')
    p.add_argument('--dataset-dir', type=str, default='.')
    p.add_argument('--out', type=str, default='yolo_dataset.zip')
    p.add_argument('--val-split', type=float, default=0.0, help='Fraction of images to move to val set (0..1)')
    args = p.parse_args()

    base = Path(args.dataset_dir).resolve()
    ann_file = base / args.annotations
    if not ann_file.exists():
        raise SystemExit(f"Annotations file not found: {ann_file}")

    ensure_dirs(base)

    lines = [l.strip() for l in ann_file.read_text(encoding='utf-8').splitlines() if l.strip()]

    # simple logic: keep all annotated images in train unless val_split > 0
    for line in lines:
        img_rel, cls, x_min, y_min, x_max, y_max = parse_annotation_line(line)
        img_path = (base / img_rel).resolve()
        if not img_path.exists():
            print(f"Warning: image not found: {img_path}, skipping")
            continue
        subset = 'train'
        # copy image into images/train or images/val (if not already there)
        dest_img_dir = base / 'images' / subset
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_img = dest_img_dir / img_path.name
        if img_path.resolve() != dest_img.resolve():
            shutil.copy2(img_path, dest_img)

        convert_and_write_label(base, dest_img, cls, x_min, y_min, x_max, y_max, subset=subset)
        print(f"Processed {dest_img} -> labels/{subset}/{dest_img.stem}.txt")

    # create zip
    out_zip = base / args.out
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        # add images and labels and labels.yaml
        for folder in ('images', 'labels'):
            folder_path = base / folder
            for root, _, files in os.walk(folder_path):
                for f in files:
                    fp = Path(root) / f
                    arcname = fp.relative_to(base)
                    zf.write(fp, arcname.as_posix())
        # labels.yaml if exists
        ly = base / 'labels.yaml'
        if ly.exists():
            zf.write(ly, 'labels.yaml')

    print(f"Created {out_zip}")


if __name__ == '__main__':
    main()
