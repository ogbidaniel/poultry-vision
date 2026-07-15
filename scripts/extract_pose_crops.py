"""
Track 1 - Extract crops of every labeled hen instance from the pose dataset,
labeling ALL hens that fall inside each crop (not just the centered one).

The pose dataset images are full-frame lateral shots with multiple labeled hens.
A crop centered on one hen frequently also contains neighbours. Labeling only
the centered hen would leave those visible neighbours unlabeled, training the
model to treat real hens as background. So for each crop we include every hen
whose keypoints land inside the crop, subject to a visibility filter: a hen is
labeled only if at least `--min-visible-frac` of its originally-visible keypoints
remain inside the crop and at least `--min-visible-kpts` are present. Slivers at
the crop edge are dropped to avoid degenerate partial labels. Keypoints carried
into crop coordinates so the crops need no re-labeling. Split membership
preserved.

Usage:
    python scripts/extract_pose_crops.py
    python scripts/extract_pose_crops.py --pad 0.10 --min-visible-frac 0.60
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

N_KPTS = 10
BASE_YAML = {
    "kpt_shape": [10, 3],
    "flip_idx": [0, 1, 3, 2, 5, 4, 6, 7, 8, 9],
    "names": {0: "hen"},
    "kpt_names": {0: ["tail_base", "tail_tip", "left_hock", "right_hock",
                      "left_foot", "right_foot", "neck_back", "middle_back",
                      "comb", "beak"]},
}
SKELETON = [(9, 8), (8, 6), (6, 7), (7, 0), (0, 1), (7, 2), (7, 3), (2, 4), (3, 5)]


def clip01(v):
    return max(0.0, min(1.0, float(v)))


def parse_instances(label_text, W, H):
    """Return list of {cls, box(px xyxy), kpts(10,3 px+vis)}."""
    out = []
    for line in label_text.splitlines():
        p = line.split()
        if len(p) < 5 + N_KPTS * 3:
            continue
        cls = int(float(p[0]))
        cx, cy, bw, bh = [float(v) for v in p[1:5]]
        box = [(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H]
        kp = np.array([float(v) for v in p[5:5+N_KPTS*3]], np.float32).reshape(N_KPTS, 3)
        kp[:, 0] *= W
        kp[:, 1] *= H
        out.append({"cls": cls, "box": box, "kpts": kp})
    return out


def label_for_crop(inst, crop, cw, ch, is_target, min_frac, min_kpts):
    """Return a YOLO-pose label line for `inst` within `crop`, or None to skip."""
    cx1, cy1 = crop[0], crop[1]
    kp = inst["kpts"]
    orig_vis = int((kp[:, 2] > 0).sum())
    # how many originally-visible keypoints land inside the crop
    inside = 0
    for j in range(N_KPTS):
        if kp[j, 2] <= 0:
            continue
        lx, ly = kp[j, 0] - cx1, kp[j, 1] - cy1
        if 0 <= lx <= cw and 0 <= ly <= ch:
            inside += 1
    if not is_target:
        if orig_vis == 0 or inside < min_kpts or inside / orig_vis < min_frac:
            return None
    # clamp the box to the crop
    bx1 = max(inst["box"][0], cx1); by1 = max(inst["box"][1], cy1)
    bx2 = min(inst["box"][2], crop[2]); by2 = min(inst["box"][3], crop[3])
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return None
    ncx = clip01(((bx1 + bx2) / 2 - cx1) / cw)
    ncy = clip01(((by1 + by2) / 2 - cy1) / ch)
    nbw = clip01((bx2 - bx1) / cw)
    nbh = clip01((by2 - by1) / ch)
    vals = [str(inst["cls"]), f"{ncx:.6f}", f"{ncy:.6f}", f"{nbw:.6f}", f"{nbh:.6f}"]
    for j in range(N_KPTS):
        lx, ly, v = (kp[j, 0] - cx1) / cw, (kp[j, 1] - cy1) / ch, kp[j, 2]
        if v <= 0 or lx < 0 or lx > 1 or ly < 0 or ly > 1:
            vals += [f"{clip01(lx):.6f}", f"{clip01(ly):.6f}", "0.000000"]
        else:
            vals += [f"{lx:.6f}", f"{ly:.6f}", f"{v:.6f}"]
    return " ".join(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset/pose/pose-dataset-merged-split")
    ap.add_argument("--dst", default="dataset/pose/pose-crops-fromframes")
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--min-visible-frac", type=float, default=0.60)
    ap.add_argument("--min-visible-kpts", type=int, default=3)
    ap.add_argument("--viz", type=int, default=8)
    args = ap.parse_args()

    root = Path.cwd()
    for _ in range(6):
        if (root / "pen_config.json").exists():
            break
        root = root.parent
    os.chdir(root)

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)

    counts = {}
    multi = 0
    viz_items = []
    for split in ("train", "val", "test"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = 0
        for img_path in sorted((src / "images" / split).glob("*.*")):
            lbl = src / "labels" / split / (img_path.stem + ".txt")
            if not lbl.exists():
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            H, W = frame.shape[:2]
            insts = parse_instances(lbl.read_text(), W, H)
            for ti, target in enumerate(insts):
                x1, y1, x2, y2 = target["box"]
                pw, ph = (x2-x1)*args.pad, (y2-y1)*args.pad
                cx1, cy1 = int(max(0, x1-pw)), int(max(0, y1-ph))
                cx2, cy2 = int(min(W, x2+pw)), int(min(H, y2+ph))
                if cx2-cx1 < 8 or cy2-cy1 < 8:
                    continue
                crop = (cx1, cy1, cx2, cy2)
                cw, ch = cx2-cx1, cy2-cy1
                lines = []
                for j, inst in enumerate(insts):
                    ln = label_for_crop(inst, crop, cw, ch, is_target=(j == ti),
                                        min_frac=args.min_visible_frac, min_kpts=args.min_visible_kpts)
                    if ln:
                        lines.append(ln)
                if not lines:
                    continue
                name = f"{img_path.stem}_h{ti}"
                cv2.imwrite(str(dst / "images" / split / f"{name}.jpg"), frame[cy1:cy2, cx1:cx2])
                (dst / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                n += 1
                if len(lines) > 1:
                    multi += 1
                if split == "train" and len(viz_items) < args.viz and len(lines) >= 1:
                    viz_items.append((frame[cy1:cy2, cx1:cx2].copy(), lines))
        counts[split] = n
        print(f"{split}: {n} crops")

    cfg = dict(BASE_YAML)
    cfg["path"] = str(dst.resolve())
    cfg["train"], cfg["val"], cfg["test"] = "images/train", "images/val", "images/test"
    (dst / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    if viz_items:
        tiles = []
        for crop_img, lines in viz_items:
            t = cv2.resize(crop_img, (256, 256))
            for ln in lines:
                p = ln.split()
                kp = [(float(p[5+3*j])*256, float(p[5+3*j+1])*256, float(p[5+3*j+2])) for j in range(N_KPTS)]
                for a, b in SKELETON:
                    if kp[a][2] > 0 and kp[b][2] > 0:
                        cv2.line(t, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), (255, 140, 0), 2)
                for x, y, v in kp:
                    if v > 0:
                        cv2.circle(t, (int(x), int(y)), 3, (0, 255, 255), -1)
            tiles.append(t)
        cols = min(4, len(tiles)); rows = (len(tiles)+cols-1)//cols
        while len(tiles) < rows*cols:
            tiles.append(np.zeros((256, 256, 3), np.uint8))
        grid = np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(rows)])
        cv2.imwrite(str(dst / "_keypoint_check.jpg"), grid)
        print(f"verification montage -> {dst / '_keypoint_check.jpg'}")

    print(f"total crops: {sum(counts.values())}   multi-hen crops: {multi}   -> {dst}")


if __name__ == "__main__":
    main()
