"""Create deterministic, sequence-aware train/val/test splits for a YOLO dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def sequence_and_index(stem: str) -> tuple[str, int]:
    """Return a recording/view group and sortable frame index."""
    if stem.startswith("daniel_IMG_"):
        return "daniel", int(stem.rsplit("_", 1)[1])
    if stem.startswith("joaquin_button_"):
        return "joaquin_button", int(stem.rsplit("_", 1)[1])
    if stem.startswith("joaquin_top_"):
        return "joaquin_top", int(stem.rsplit("_", 1)[1])
    raise ValueError(f"Unknown sequence naming pattern: {stem}")


def allocate_group(items: list[tuple[int, Path, Path]]) -> dict[str, list[tuple[int, Path, Path]]]:
    """Allocate contiguous frame blocks so neighboring frames do not randomly leak across splits."""
    items = sorted(items)
    count = len(items)
    val_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    train_count = count - val_count - test_count
    if train_count < 1:
        raise ValueError(f"Sequence has too few images for three splits: {count}")
    return {
        "train": items[:train_count],
        "val": items[train_count : train_count + val_count],
        "test": items[train_count + val_count :],
    }


def split_dataset(source: Path, output: Path) -> dict:
    cfg = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    groups: dict[str, list[tuple[int, Path, Path]]] = defaultdict(list)

    for original_split in ("train", "val", "test"):
        image_dir = source / "images" / original_split
        label_dir = source / "labels" / original_split
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}")
            sequence, frame_index = sequence_and_index(image_path.stem)
            groups[sequence].append((frame_index, image_path, label_path))

    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "test"):
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    manifest = {"strategy": "contiguous 70/15/15 blocks per recording/view", "groups": {}, "splits": {}}
    seen_stems = set()
    for sequence, items in sorted(groups.items()):
        allocation = allocate_group(items)
        manifest["groups"][sequence] = {}
        for split, records in allocation.items():
            manifest["groups"][sequence][split] = [record[1].stem for record in records]
            for _, image_path, label_path in records:
                if image_path.stem in seen_stems:
                    raise ValueError(f"Duplicate image stem across source splits: {image_path.stem}")
                seen_stems.add(image_path.stem)
                shutil.copy2(image_path, output / "images" / split / image_path.name)
                shutil.copy2(label_path, output / "labels" / split / label_path.name)

    for split in ("train", "val", "test"):
        images = list((output / "images" / split).iterdir())
        objects = sum(
            1
            for label_path in (output / "labels" / split).glob("*.txt")
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        manifest["splits"][split] = {"images": len(images), "objects": objects}

    cfg.update(
        {
            "path": str(output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
        }
    )
    (output / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = split_dataset(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
