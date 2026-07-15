"""
Track 2a/2b - Extract UNLABELED hen crops (cam1) for Roboflow labeling, from the
user-recommended video regions plus an optional random spread of cam1 videos.

Crops are 0.10-padded hen detections; one frame is sampled per `stride` frames
within each region. Output crops go to a candidate pool to be labeled and then
merged into the pose training set.

Usage:
    python scripts/extract_crops_from_regions.py
    python scripts/extract_crops_from_regions.py --random 15
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO

HEN_CLASS = 1
CAM1 = "dataset/treatment1/cam1"

# Recommended regions (cam1, 15 fps). Each range = (start_frame, end_frame, stride).
SPEC = {
    "20260605_141448": [(525, 630, 15), (1575, 1800, 15)],          # 0:35-0:42, 1:45-2:00
    "20260605_142503": [(0, 1800, 60)],                              # hens standing, spaced
    "20260605_145103": [(0, 1800, 30)],                              # occlusion / feather-pecking
    "20260605_150703": [(225, 450, 10), (1140, 1185, 5)],            # 0:15-0:30 walking, ~1:17
    "20260605_152503": [(0, 1800, 60), (1620, 1800, 15)],            # 3 hens feeding + near end
}


def crop_box(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    cx2, cy2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def frames_for(stem):
    idxs = set()
    for (a, b, s) in SPEC[stem]:
        idxs.update(range(a, b, s))
    return sorted(idxs)


def random_spread(n, exclude):
    vids = sorted(Path(CAM1).glob("*.mp4"), key=lambda p: p.stem)
    vids = [v for v in vids if v.stem not in exclude]
    if not vids:
        return []
    step = max(1, len(vids) // n)
    return [(v.stem, [v_mid(v)]) for v in vids[::step][:n]]


def v_mid(path):
    c = cv2.VideoCapture(str(path)); n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)); c.release()
    return max(0, n // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/pose/pose-crops-candidates")
    ap.add_argument("--det", default="models/det0-yolo12s.pt")
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--min-dim", type=int, default=40)
    ap.add_argument("--random", type=int, default=0, help="also sample N random cam1 videos (1 mid frame each)")
    args = ap.parse_args()

    root = Path.cwd()
    for _ in range(6):
        if (root / "pen_config.json").exists():
            break
        root = root.parent
    os.chdir(root)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    det = YOLO(args.det)

    jobs = [(stem, frames_for(stem)) for stem in SPEC]
    if args.random > 0:
        jobs += random_spread(args.random, exclude=set(SPEC))

    manifest, saved = [], 0
    for stem, idxs in jobs:
        path = f"{CAM1}/{stem}.mp4"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"  WARN cannot open {path}"); continue
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            r = det.predict(frame, conf=args.conf, verbose=False)[0]
            if r.boxes is None:
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            hi = 0
            for i in range(len(xyxy)):
                if cls[i] != HEN_CLASS:
                    continue
                crop, cbox = crop_box(frame, xyxy[i], args.pad)
                if crop.size == 0 or min(crop.shape[:2]) < args.min_dim:
                    continue
                name = f"cam1_{stem}_f{idx}_h{hi}.jpg"
                cv2.imwrite(str(out / name), crop)
                manifest.append({"file": name, "video": path, "frame": int(idx),
                                 "box_xyxy": [float(b) for b in xyxy[i]],
                                 "crop_xyxy": [int(b) for b in cbox], "conf": float(conf[i])})
                hi += 1; saved += 1
        cap.release()
        print(f"  {stem}: {len(idxs)} frames sampled")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved {saved} candidate crops -> {out}")
    print(f"Manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
