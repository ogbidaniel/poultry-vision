"""
Build a small hen-crop dataset for labeling and pose retraining.

Samples begin/middle/end frames from the corresponding cam0/cam1 pair plus a
handful of other videos at varying times, detects the hens, and writes padded
hen crops to a folder. The crops are meant to be hand-labeled with the
ten-keypoint schema to give the pose model more diverse postures.

Usage:
    python scripts/build_hen_crops.py
    python scripts/build_hen_crops.py --target 70 --conf 0.4 --pad 0.10
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO

HEN_CLASS = 1
FRAME_FRACS = (0.10, 0.50, 0.90)   # begin / middle / end

# The corresponding pair the user identified.
PAIRED = [
    "dataset/treatment1/cam0/20260606_000102.mp4",
    "dataset/treatment1/cam1/20260606_000103.mp4",
]


def cam_of(path: Path) -> str:
    return path.parent.name


def pick_other_videos(n: int, exclude: set[str]) -> list[Path]:
    """Evenly spaced across the sorted cam0+cam1 timeline (varying times)."""
    vids = sorted(
        [p for p in Path("dataset/treatment1/cam0").glob("*.mp4")] +
        [p for p in Path("dataset/treatment1/cam1").glob("*.mp4")],
        key=lambda p: (p.stem, p.parent.name),
    )
    vids = [p for p in vids if str(p).replace("\\", "/") not in exclude]
    if not vids:
        return []
    step = max(1, len(vids) // n)
    chosen = vids[::step][:n]
    return chosen


def grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


def crop_box(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    cx2, cy2 = int(min(w, x2 + pad * bw)), int(min(h, y2 + pad * bh))
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1, cx2, cy2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="hen_crops")
    ap.add_argument("--det", default="models/det0-yolo12s.pt")
    ap.add_argument("--target", type=int, default=70)
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--pad", type=float, default=0.10)
    ap.add_argument("--min-dim", type=int, default=40, help="drop crops smaller than this (px)")
    ap.add_argument("--n-other", type=int, default=6)
    args = ap.parse_args()

    root = Path.cwd()
    for _ in range(6):
        if (root / "pen_config.json").exists():
            break
        root = root.parent
    import os
    os.chdir(root)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paired = [Path(p) for p in PAIRED]
    others = pick_other_videos(args.n_other,
                               exclude={p.replace("\\", "/") for p in PAIRED})
    videos = paired + others
    print("Videos used:")
    for v in videos:
        print(f"  {cam_of(v)}/{v.name}")

    det = YOLO(args.det)

    # Collect crops grouped by (video, frame) so we can sample with spread.
    groups: dict[str, list] = {}
    manifest = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"  WARN cannot open {v}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cam = cam_of(v)
        for frac in FRAME_FRACS:
            idx = min(total - 1, int(frac * total))
            frame = grab(cap, idx)
            if frame is None:
                continue
            r = det.predict(frame, conf=args.conf, verbose=False)[0]
            if r.boxes is None:
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            key = f"{cam}_{v.stem}_f{idx}"
            for hi, i in enumerate([j for j in range(len(xyxy)) if cls[j] == HEN_CLASS]):
                crop, cbox = crop_box(frame, xyxy[i], args.pad)
                if crop.size == 0 or min(crop.shape[:2]) < args.min_dim:
                    continue
                name = f"{cam}_{v.stem}_f{idx}_h{hi}.jpg"
                groups.setdefault(key, []).append((name, crop, {
                    "video": str(v).replace("\\", "/"), "cam": cam,
                    "frame": idx, "box_xyxy": [float(b) for b in xyxy[i]],
                    "crop_xyxy": [int(b) for b in cbox], "conf": float(conf[i]),
                }))
        cap.release()

    pool_total = sum(len(v) for v in groups.values())
    print(f"\nDetected {pool_total} hen crops across {len(groups)} frames.")

    # Round-robin across frames to spread the sample to the target count.
    saved = 0
    order = list(groups.values())
    while saved < args.target and any(order):
        for g in order:
            if not g:
                continue
            name, crop, meta = g.pop(0)
            cv2.imwrite(str(out_dir / name), crop)
            manifest.append({"file": name, **meta})
            saved += 1
            if saved >= args.target:
                break

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {saved} crops -> {out_dir}")
    print(f"Manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
