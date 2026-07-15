"""Validate a YOLO pose dataset with strict checks and Ultralytics parsing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".yolo-config"))

from ultralytics.data.dataset import YOLODataset


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def label_dir_for(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    try:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"Image path does not contain an 'images' directory: {image_dir}") from exc
    return Path(*parts)


def validate_split(root: Path, split: str, split_path: str, data: dict) -> tuple[int, int]:
    image_dir = root / split_path
    label_dir = label_dir_for(image_dir)
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(label_dir.glob("*.txt"))
    nkpt, ndim = data["kpt_shape"]
    expected_columns = 5 + nkpt * ndim
    errors = []
    object_count = 0

    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    for stem in sorted(image_stems - label_stems):
        errors.append(f"{split}: image has no label: {stem}")
    for stem in sorted(label_stems - image_stems):
        errors.append(f"{split}: label has no image: {stem}")

    for label_path in labels:
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            values = np.asarray(line.split(), dtype=np.float32)
            object_count += 1
            if values.size != expected_columns:
                errors.append(
                    f"{label_path}:{line_number}: expected {expected_columns} columns, got {values.size}"
                )
                continue
            if not np.isfinite(values).all():
                errors.append(f"{label_path}:{line_number}: contains NaN or infinity")
                continue
            if int(values[0]) != values[0] or not 0 <= int(values[0]) < len(data["names"]):
                errors.append(f"{label_path}:{line_number}: invalid class id {values[0]}")

            bbox = values[1:5]
            keypoints = values[5:].reshape(nkpt, ndim)
            if np.any((bbox < 0) | (bbox > 1)) or np.any(bbox[2:] <= 0):
                errors.append(f"{label_path}:{line_number}: invalid normalized bounding box")
            if np.any((keypoints[:, :2] < 0) | (keypoints[:, :2] > 1)):
                errors.append(f"{label_path}:{line_number}: keypoint x/y outside [0, 1]")
            if ndim == 3 and np.any(~np.isin(keypoints[:, 2], (0, 1, 2))):
                errors.append(f"{label_path}:{line_number}: visibility must be 0, 1, or 2")

    if errors:
        raise ValueError("\n".join(errors[:50]))

    parsed = YOLODataset(
        img_path=str(image_dir),
        data=data,
        task="pose",
        imgsz=640,
        augment=False,
        cache=False,
        prefix=f"{split}: ",
    )
    parsed_objects = sum(len(item["cls"]) for item in parsed.labels)
    if len(parsed.labels) != len(images):
        raise ValueError(
            f"{split}: Ultralytics retained {len(parsed.labels)}/{len(images)} images; "
            "inspect its corrupt-label warnings"
        )
    if parsed_objects != object_count:
        raise ValueError(
            f"{split}: text files contain {object_count} objects but Ultralytics loaded {parsed_objects}"
        )

    print(f"{split}: {len(images)} images, {len(labels)} labels, {object_count} objects")
    return len(images), object_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Path to a YOLO pose data.yaml")
    args = parser.parse_args()

    yaml_path = args.data.resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if "kpt_shape" not in data:
        raise ValueError(f"{yaml_path} does not define kpt_shape")
    root = Path(data.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    total_images = 0
    total_objects = 0
    splits = ["train", "val"] + (["test"] if data.get("test") else [])
    for split in splits:
        images, objects = validate_split(root, split, data[split], data)
        total_images += images
        total_objects += objects
    print(f"OK: {total_images} images and {total_objects} pose objects passed")


if __name__ == "__main__":
    main()
