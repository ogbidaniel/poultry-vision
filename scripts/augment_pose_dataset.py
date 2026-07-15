"""
Augment the pose training split (keypoint-aware) and write a YOLO-pose dataset.

Reproduces the augmentation recipe used previously: geometric (affine,
perspective), photometric (brightness/contrast, hue/sat), and light
blur/noise, applied only to the training split. Horizontal flips are left to
the trainer (data.yaml flip_idx) so left/right keypoints stay correct.
Validation and test splits are copied unchanged.

Usage:
    python scripts/augment_pose_dataset.py \
        --src dataset/pose/pose-dataset-merged-split \
        --dst dataset/pose/pose-dataset-merged-aug-split \
        --target-train 360
"""
from __future__ import annotations

import argparse
import shutil
from math import ceil
from pathlib import Path

import cv2
import numpy as np
import yaml
import albumentations as A

N_KPTS = 10


def build_augmenter() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.0),  # trainer applies flips via flip_idx
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.06, 0.06),
                     rotate=(-12, 12), shear=(-6, 6), p=0.75),
            A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.25),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussNoise(std_range=(0.08, 0.2), p=1.0),
                A.ISONoise(p=1.0),
            ], p=0.25),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20,
                                     val_shift_limit=15, p=1.0),
            ], p=0.6),
        ],
        bbox_params=A.BboxParams(format='yolo',
                                 label_fields=['class_labels', 'instance_ids'],
                                 min_visibility=0.0, clip=True),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )


def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def sanitize_bbox(bbox, eps=1e-6):
    cx, cy, bw, bh = [float(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        return None
    x1, y1 = clip01(cx - bw / 2), clip01(cy - bh / 2)
    x2, y2 = clip01(cx + bw / 2), clip01(cy + bh / 2)
    if (x2 - x1) <= eps or (y2 - y1) <= eps:
        return None
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def parse_label(path: Path) -> list[dict]:
    instances = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5 + N_KPTS * 3:
            continue
        cls = int(float(p[0]))
        bbox = [float(v) for v in p[1:5]]
        kpts = np.array([float(v) for v in p[5:5 + N_KPTS * 3]],
                        dtype=np.float32).reshape(N_KPTS, 3)
        instances.append({'cls': cls, 'bbox': bbox, 'kpts': kpts})
    return instances


def serialize(instances: list[dict]) -> list[str]:
    lines = []
    for inst in instances:
        vals = [str(int(inst['cls']))]
        vals += [f'{float(v):.6f}' for v in inst['bbox']]
        vals += [f'{float(v):.6f}' for v in inst['kpts'].reshape(-1)]
        lines.append(' '.join(vals))
    return lines


def augment(image_bgr, instances, aug: A.Compose):
    h, w = image_bgr.shape[:2]
    valid = []
    for inst in instances:
        sb = sanitize_bbox(inst['bbox'])
        if sb is not None:
            valid.append({'cls': int(inst['cls']), 'bbox': sb, 'kpts': inst['kpts']})
    if not valid:
        return image_bgr, []

    kpts_px = [(float(x) * w, float(y) * h)
               for inst in valid for x, y, _ in inst['kpts']]
    out = aug(image=image_bgr,
              bboxes=[i['bbox'] for i in valid],
              class_labels=[i['cls'] for i in valid],
              instance_ids=list(range(len(valid))),
              keypoints=kpts_px)

    img = out['image']
    oh, ow = img.shape[:2]
    if len(out['keypoints']) != len(valid) * N_KPTS:
        return img, []
    kps = np.array(out['keypoints'], dtype=np.float32).reshape(len(valid), N_KPTS, 2)

    res = []
    for i in range(len(out['bboxes'])):
        src = int(out['instance_ids'][i])
        sb = sanitize_bbox(out['bboxes'][i])
        if sb is None:
            continue
        arr = np.zeros((N_KPTS, 3), dtype=np.float32)
        for j in range(N_KPTS):
            xn, yn = kps[src, j, 0] / max(ow, 1), kps[src, j, 1] / max(oh, 1)
            old_v = float(valid[src]['kpts'][j, 2])
            v = 0.0 if (xn < 0 or xn > 1 or yn < 0 or yn > 1) else old_v
            arr[j] = [clip01(xn), clip01(yn), v]
        res.append({'cls': int(out['class_labels'][i]), 'bbox': [clip01(v) for v in sb], 'kpts': arr})
    return img, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/pose/pose-dataset-merged-split")
    ap.add_argument("--dst", default="dataset/pose/pose-dataset-merged-aug-split")
    ap.add_argument("--target-train", type=int, default=360)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    aug = build_augmenter()

    if dst.exists():
        shutil.rmtree(dst)
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_imgs = sorted((src / "images" / "train").glob("*.*"))
    repeats = max(1, ceil((args.target_train - len(train_imgs)) / max(len(train_imgs), 1)))
    print(f"train images: {len(train_imgs)}  repeats/image: {repeats}  "
          f"-> ~{len(train_imgs) * (repeats + 1)} train samples")

    created = 0
    for img_path in train_imgs:
        lbl = src / "labels" / "train" / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        instances = parse_label(lbl)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # original
        shutil.copy2(img_path, dst / "images" / "train" / img_path.name)
        shutil.copy2(lbl, dst / "labels" / "train" / lbl.name)
        created += 1
        # augmented copies
        for k in range(repeats):
            aimg, ainst = augment(img, instances, aug)
            if not ainst:
                continue
            name = f"aug{k:02d}_{img_path.stem}"
            cv2.imwrite(str(dst / "images" / "train" / f"{name}.jpg"), aimg)
            (dst / "labels" / "train" / f"{name}.txt").write_text("\n".join(serialize(ainst)))
            created += 1

    for split in ("val", "test"):
        n = 0
        for img_path in sorted((src / "images" / split).glob("*.*")):
            lbl = src / "labels" / split / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copy2(img_path, dst / "images" / split / img_path.name)
            shutil.copy2(lbl, dst / "labels" / split / lbl.name)
            n += 1
        print(f"copied {split}: {n}")

    # data.yaml
    cfg = yaml.safe_load((src / "data.yaml").read_text())
    cfg["path"] = str(dst.resolve())
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if (src / "split_manifest.json").exists():
        shutil.copy2(src / "split_manifest.json", dst / "split_manifest.json")

    print(f"train samples written: {created}")
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
