"""
Build the v6 pose dataset by extending the v3 merged split with Fyaz's newly
labeled top+bottom control-pen pose frames.

Starts from the existing merged base (dataset/pose/pose-dataset-merged-v3-split,
already base-order and containing base frames + earlier crops), then folds in
dataset/pose/pose-dataset-fyaz (Roboflow keypoint order -> remapped to base order
with the shared PC_FROM_BASE map). For each fyaz frame we add both:
  - the full frame with remapped keypoints, and
  - per-hen crops (every visible hen inside the crop labeled, keypoints carried
    into crop coordinates) matching the crop distribution the model sees at
    deployment.
Fyaz has only train/val, so those fold into train/val; the held-out test stays
the base-only v3 test for comparability with v5.

Usage:
    python scripts/build_pose_v6.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pose_v5 import remap_line, BASE_YAML, N_KPTS  # noqa: E402
from extract_pose_crops import parse_instances, label_for_crop  # noqa: E402

FYAZ_CAMS = ["Top", "Bottom"]


def copy_split(src, dst, split, counts, key):
    """Plain copy of an already-base-order split (v3 base)."""
    si, sl = src / "images" / split, src / "labels" / split
    di, dl = dst / "images" / split, dst / "labels" / split
    di.mkdir(parents=True, exist_ok=True)
    dl.mkdir(parents=True, exist_ok=True)
    if not si.exists():
        return
    for img in sorted(si.glob("*.*")):
        lbl = sl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, di / img.name)
        shutil.copy2(lbl, dl / lbl.name)
        counts[f"{key}_{split}"] += 1


def fold_fyaz(fyaz_root, out, counts, pad, min_frac, min_kpts):
    """Add fyaz frames (remapped) + per-hen crops to the train/val splits."""
    for cam in FYAZ_CAMS:
        for split in ("train", "val"):
            si = fyaz_root / "Pose_Data" / cam / "images" / split
            sl = fyaz_root / "Pose_Data" / cam / "labels" / split
            if not si.exists():
                continue
            di, dl = out / "images" / split, out / "labels" / split
            di.mkdir(parents=True, exist_ok=True)
            dl.mkdir(parents=True, exist_ok=True)
            for img in sorted(si.glob("*.*")):
                lbl = sl / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                remapped = [remap_line(ln) for ln in lbl.read_text().splitlines() if ln.strip()]
                remapped = [r for r in remapped if r]
                if not remapped:
                    continue
                stem = f"fyaz_{cam}_{img.stem}"
                # 1) full frame with remapped keypoints
                shutil.copy2(img, di / f"{stem}{img.suffix}")
                (dl / f"{stem}.txt").write_text("\n".join(remapped))
                counts[f"fyaz_{split}"] += 1
                # 2) per-hen crops from the remapped label
                frame = cv2.imread(str(img))
                if frame is None:
                    continue
                H, W = frame.shape[:2]
                insts = parse_instances("\n".join(remapped), W, H)
                for ti, target in enumerate(insts):
                    x1, y1, x2, y2 = target["box"]
                    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
                    cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
                    cx2, cy2 = int(min(W, x2 + pw)), int(min(H, y2 + ph))
                    if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                        continue
                    crop = (cx1, cy1, cx2, cy2)
                    cw, ch = cx2 - cx1, cy2 - cy1
                    lines = []
                    for j, inst in enumerate(insts):
                        ln = label_for_crop(inst, crop, cw, ch, is_target=(j == ti),
                                            min_frac=min_frac, min_kpts=min_kpts)
                        if ln:
                            lines.append(ln)
                    if not lines:
                        continue
                    cname = f"{stem}_h{ti}"
                    cv2.imwrite(str(di / f"{cname}.jpg"), frame[cy1:cy2, cx1:cx2])
                    (dl / f"{cname}.txt").write_text("\n".join(lines))
                    counts[f"fyazcrop_{split}"] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset/pose/pose-dataset-merged-v3-split")
    ap.add_argument("--fyaz", default="dataset/pose/pose-dataset-fyaz")
    ap.add_argument("--out", default="dataset/pose/pose-dataset-merged-v4-split")
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--min-visible-frac", type=float, default=0.60)
    ap.add_argument("--min-visible-kpts", type=int, default=3)
    args = ap.parse_args()

    root = Path.cwd()
    for _ in range(6):
        if (root / "pen_config.json").exists():
            break
        root = root.parent
    os.chdir(root)

    base, fyaz, out = Path(args.base), Path(args.fyaz), Path(args.out)
    if out.exists():
        shutil.rmtree(out)

    counts = Counter()
    for split in ("train", "val", "test"):
        copy_split(base, out, split, counts, "base")
    fold_fyaz(fyaz, out, counts, args.pad, args.min_visible_frac, args.min_visible_kpts)

    cfg = dict(BASE_YAML)
    cfg["path"] = str(out.resolve())
    cfg["train"], cfg["val"], cfg["test"] = "images/train", "images/val", "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("=== v6 dataset composition ===")
    for split in ("train", "val", "test"):
        total = len(list((out / "images" / split).glob("*.*")))
        print(f"{split}: {total} images   (base:{counts[f'base_{split}']} "
              f"fyaz:{counts[f'fyaz_{split}']} fyazcrop:{counts[f'fyazcrop_{split}']})")
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
