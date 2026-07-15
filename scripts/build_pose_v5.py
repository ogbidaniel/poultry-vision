"""
Build the v5 pose dataset by merging, with split preservation:
  - base full frames   : dataset/pose/pose-dataset-merged-split        (base order)
  - all-instance crops : dataset/pose/pose-crops-fromframes            (base order)
  - labeled video crops: dataset/pose/pose-crops                       (Roboflow order -> remap)

Video-sourced crops are de-duplicated spatially/temporally: crops from the same
camera+video+detection-slot within a frame window are near-identical, so only one
is kept. Reports the resulting composition.

Usage:
    python scripts/build_pose_v5.py
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

N_KPTS = 10
PC_FROM_BASE = [4, 5, 6, 8, 7, 9, 2, 3, 1, 0]   # Roboflow crop order -> base order
BASE_YAML = {
    "kpt_shape": [10, 3],
    "flip_idx": [0, 1, 3, 2, 5, 4, 6, 7, 8, 9],
    "names": {0: "hen"},
    "kpt_names": {0: ["tail_base", "tail_tip", "left_hock", "right_hock",
                      "left_foot", "right_foot", "neck_back", "middle_back",
                      "comb", "beak"]},
}
_VID_RE = re.compile(r"cam(\d)_(\d{8}_\d{6})_f(\d+)_h(\d+)")


def remap_line(line):
    p = line.split()
    if len(p) < 5 + N_KPTS * 3:
        return None
    head = p[:5]
    kp = [p[5 + 3*j: 5 + 3*j + 3] for j in range(N_KPTS)]
    return " ".join(head + [v for i in range(N_KPTS) for v in kp[PC_FROM_BASE[i]]])


def copy_plain(src, dst, split, counts, key):
    si, sl = src / "images" / split, src / "labels" / split
    di, dl = dst / "images" / split, dst / "labels" / split
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    if not si.exists():
        return
    for img in sorted(si.glob("*.*")):
        lbl = sl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        shutil.copy2(img, di / img.name)
        shutil.copy2(lbl, dl / lbl.name)
        counts[f"{key}_{split}"] += 1


def copy_video_crops(src, dst, split, counts, key, frame_window):
    """Remap to base order + spatial/temporal dedup by (cam,video,slot) frame window."""
    si, sl = src / "images" / split, src / "labels" / split
    di, dl = dst / "images" / split, dst / "labels" / split
    di.mkdir(parents=True, exist_ok=True); dl.mkdir(parents=True, exist_ok=True)
    if not si.exists():
        return
    imgs = sorted(si.glob("*.*"))
    kept_by_group = {}     # (cam, video, slot) -> [frame, ...] already kept
    dropped = 0
    for img in imgs:
        lbl = sl / (img.stem + ".txt")
        if not lbl.exists():
            continue
        m = _VID_RE.search(img.stem)
        if m:
            cam, vid, frame, slot = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            g = (cam, vid, slot)
            if any(abs(frame - f) < frame_window for f in kept_by_group.get(g, [])):
                dropped += 1
                continue
            kept_by_group.setdefault(g, []).append(frame)
        shutil.copy2(img, di / img.name)
        out = [remap_line(ln) for ln in lbl.read_text().splitlines() if ln.strip()]
        (dl / lbl.name).write_text("\n".join(o for o in out if o))
        counts[f"{key}_{split}"] += 1
    counts[f"{key}_{split}_dropped"] += dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dataset/pose/pose-dataset-merged-split")
    ap.add_argument("--fromframes", default="dataset/pose/pose-crops-fromframes")
    ap.add_argument("--video-crops", default="dataset/pose/pose-crops")
    ap.add_argument("--out", default="dataset/pose/pose-dataset-merged-v3-split")
    ap.add_argument("--frame-window", type=int, default=30, help="dedup window (frames) for video crops")
    args = ap.parse_args()

    root = Path.cwd()
    for _ in range(6):
        if (root / "pen_config.json").exists():
            break
        root = root.parent
    os.chdir(root)

    base, ff, vc, out = (Path(args.base), Path(args.fromframes),
                         Path(args.video_crops), Path(args.out))
    if out.exists():
        shutil.rmtree(out)

    counts = Counter()
    for split in ("train", "val", "test"):
        copy_plain(base, out, split, counts, "base")
        copy_plain(ff, out, split, counts, "fromframes")
    # video crops: train -> train, val -> val (keep val held out for the crop eval)
    copy_video_crops(vc, out, "train", counts, "videocrops", args.frame_window)
    copy_video_crops(vc, out, "val", counts, "videocrops", args.frame_window)

    cfg = dict(BASE_YAML)
    cfg["path"] = str(out.resolve())
    cfg["train"], cfg["val"], cfg["test"] = "images/train", "images/val", "images/test"
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("=== v5 dataset composition ===")
    for split in ("train", "val", "test"):
        total = len(list((out / "images" / split).glob("*.*")))
        parts = " + ".join(f"{k.split('_')[0]}:{counts[f'{k.split(chr(95))[0]}_{split}']}"
                            for k in ["base_x", "fromframes_x", "videocrops_x"])
        print(f"{split}: {total} images   ({parts})")
    print(f"video-crop dedup dropped: train={counts['videocrops_train_dropped']} "
          f"val={counts['videocrops_val_dropped']}")
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
