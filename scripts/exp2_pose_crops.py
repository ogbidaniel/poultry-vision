"""
Experiment 2 — Pose model on individual side-view crops vs full frame.

For each detected bird in the side view, this script:
  1. Runs the pose model on the entire side frame (full-frame inference)
  2. Crops each detection with 20 px padding, runs the pose model on the crop,
     then maps keypoint coords back to full-frame space
  3. Displays all three views side by side so you can compare keypoint quality

Use this to decide whether to run the pose model per-crop in the pipeline.

Controls
--------
Space / any key : advance one frame
q               : quit
s               : save current combined image to output/exp2/

Usage
-----
    python scripts/exp2_pose_crops.py --config pen_config.json --pair 20260605_141702
    python scripts/exp2_pose_crops.py --config pen_config.json --pair 20260605_141702 --start-frame 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.capture import CameraSourceConfig, FrameSource
from src.detect import Detector

_KP_NAMES = [
    "tail_base", "tail_tip", "left_hock", "right_hock",
    "left_foot", "right_foot", "neck_back", "middle_back",
    "comb", "beak",
]
_SKELETON = [
    (9, 8), (8, 6), (6, 7), (7, 0), (0, 1),
    (0, 2), (0, 3), (7, 2), (7, 3), (2, 4), (3, 5),
]
_KP_CONF_THRESHOLD = 0.3
_CROP_PAD = 20
_PANEL_H = 540
_CROP_DISPLAY_H = 180  # height of each crop thumbnail in Panel C


def _draw_skeleton(img: np.ndarray, kps: np.ndarray,
                   box_bgr: tuple = (100, 100, 255),
                   offset_xy: tuple = (0, 0)) -> None:
    ox, oy = offset_xy
    if kps is None:
        return
    for i1, i2 in _SKELETON:
        if i1 >= len(kps) or i2 >= len(kps):
            continue
        x1, y1, c1 = kps[i1]
        x2, y2, c2 = kps[i2]
        if c1 >= _KP_CONF_THRESHOLD and c2 >= _KP_CONF_THRESHOLD:
            cv2.line(img, (int(x1 + ox), int(y1 + oy)), (int(x2 + ox), int(y2 + oy)),
                     (255, 140, 0), 2, cv2.LINE_AA)
    for k, name in enumerate(_KP_NAMES):
        if k >= len(kps):
            continue
        xk, yk, ck = kps[k]
        pt = (int(xk + ox), int(yk + oy))
        if ck >= _KP_CONF_THRESHOLD:
            cv2.circle(img, pt, 4, (0, 255, 255), -1)
            cv2.putText(img, f"{k}", (pt[0] + 4, pt[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
        else:
            cv2.circle(img, pt, 3, (60, 60, 60), -1)


def _annotate_full(frame: np.ndarray, dets, label: str) -> np.ndarray:
    out = frame.copy()
    for det in dets:
        box = det.box.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (100, 100, 255), 2)
        if det.kps is not None:
            _draw_skeleton(out, det.kps)
    cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def _make_crop_panel(crops_with_kps: list[tuple[np.ndarray, np.ndarray | None]],
                     target_h: int) -> np.ndarray:
    """Build a horizontal strip of annotated crops."""
    if not crops_with_kps:
        blank = np.zeros((target_h, target_h, 3), dtype=np.uint8)
        cv2.putText(blank, "no detections", (8, target_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return blank

    tiles = []
    for crop_img, kps in crops_with_kps:
        tile = crop_img.copy()
        if kps is not None:
            _draw_skeleton(tile, kps)
        tile = _resize_h(tile, target_h)
        # Add thin separator
        sep = np.zeros((target_h, 4, 3), dtype=np.uint8)
        tiles.append(tile)
        tiles.append(sep)

    return np.hstack(tiles[:-1])  # drop trailing separator


def run(cfg: dict, pair: str, start_frame: int, output_dir: Path) -> None:
    dataset_root = Path(cfg["dataset_root"])
    side_video = dataset_root / cfg["cameras"]["side"] / f"{pair}.mp4"

    pose_model = Detector(cfg["models"]["pose"])

    win = ("Exp 2 — Pose: full-frame (A) vs crop-mapped (B) vs crop grid (C)"
           "  |  space=next  s=save  q=quit")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved_count = 0

    side_cfg = CameraSourceConfig(name="side", source=str(side_video), camera_type="file")

    with FrameSource(side_cfg) as side_src:
        side_gen = iter(side_src)

        for _ in range(start_frame):
            try:
                next(side_gen)
                frame_idx += 1
            except StopIteration:
                print(f"Video ended before frame {start_frame}.")
                return

        while True:
            try:
                side_f = next(side_gen)
            except StopIteration:
                print("End of video.")
                break

            H, W = side_f.shape[:2]

            # ── Full-frame inference ──────────────────────────────────────────
            full_dets = pose_model.predict(side_f)

            # ── Per-crop inference ────────────────────────────────────────────
            crops_with_kps: list[tuple[np.ndarray, np.ndarray | None]] = []
            crop_dets_remapped = []  # detections with kps remapped to full-frame space

            for det in full_dets:
                x1 = max(0, int(det.box[0]) - _CROP_PAD)
                y1 = max(0, int(det.box[1]) - _CROP_PAD)
                x2 = min(W, int(det.box[2]) + _CROP_PAD)
                y2 = min(H, int(det.box[3]) + _CROP_PAD)

                crop = side_f[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_dets = pose_model.predict(crop)

                # Pick highest-confidence detection from crop result
                if crop_dets:
                    best = max(crop_dets, key=lambda d: d.conf)
                    crop_kps = best.kps

                    # Remap keypoints back to full-frame coordinates
                    if crop_kps is not None:
                        remapped = crop_kps.copy()
                        remapped[:, 0] += x1
                        remapped[:, 1] += y1
                    else:
                        remapped = None

                    crops_with_kps.append((crop, crop_kps))
                    crop_dets_remapped.append({
                        "box": det.box,
                        "kps": remapped,
                        "crop_kps": crop_kps,
                        "conf_full": det.conf,
                        "conf_crop": best.conf,
                        "crop_origin": (x1, y1),
                    })
                else:
                    crops_with_kps.append((crop, None))
                    crop_dets_remapped.append({
                        "box": det.box,
                        "kps": None,
                        "crop_kps": None,
                        "conf_full": det.conf,
                        "conf_crop": 0.0,
                        "crop_origin": (x1, y1),
                    })

            # ── Terminal output ───────────────────────────────────────────────
            print(f"\nframe {frame_idx:5d}  |  {len(full_dets)} detection(s)")
            for i, cd in enumerate(crop_dets_remapped):
                full_kps = full_dets[i].kps
                crop_kps = cd["crop_kps"]
                print(f"  det {i}: full_conf={cd['conf_full']:.2f}  "
                      f"crop_conf={cd['conf_crop']:.2f}")
                for k, name in enumerate(_KP_NAMES):
                    fc = float(full_kps[k, 2]) if (full_kps is not None
                                                    and k < len(full_kps)) else 0.0
                    cc = float(crop_kps[k, 2]) if (crop_kps is not None
                                                    and k < len(crop_kps)) else 0.0
                    marker = " ★" if k == 6 else ""  # highlight neck_back
                    print(f"    kp{k:2d} {name:<12}  full={fc:.2f}  crop={cc:.2f}{marker}")

            # ── Panel A: full-frame inference ─────────────────────────────────
            panel_a = _annotate_full(
                side_f, full_dets,
                f"A) Full-frame  dets={len(full_dets)}  f={frame_idx}"
            )

            # ── Panel B: crop-mapped inference ────────────────────────────────
            panel_b = side_f.copy()
            for cd in crop_dets_remapped:
                box = cd["box"].astype(int)
                cv2.rectangle(panel_b, (box[0], box[1]), (box[2], box[3]), (0, 220, 0), 2)
                if cd["kps"] is not None:
                    _draw_skeleton(panel_b, cd["kps"])
            cv2.putText(panel_b,
                        f"B) Crop-mapped  dets={len(crop_dets_remapped)}  f={frame_idx}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(panel_b,
                        f"B) Crop-mapped  dets={len(crop_dets_remapped)}  f={frame_idx}",
                        (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)

            # ── Panel C: crop grid ────────────────────────────────────────────
            panel_c = _make_crop_panel(crops_with_kps, _CROP_DISPLAY_H)

            # ── Stack all panels ──────────────────────────────────────────────
            ab = np.hstack([_resize_h(panel_a, _PANEL_H), _resize_h(panel_b, _PANEL_H)])
            # Pad panel_c to match ab width
            c_resized = _resize_h(panel_c, _CROP_DISPLAY_H)
            pad_w = ab.shape[1] - c_resized.shape[1]
            if pad_w > 0:
                c_resized = np.hstack([c_resized,
                                       np.zeros((c_resized.shape[0], pad_w, 3), dtype=np.uint8)])
            else:
                c_resized = c_resized[:, :ab.shape[1]]

            combined = np.vstack([ab, c_resized])

            cv2.imshow(win, combined)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            if key == ord('s'):
                save_path = output_dir / f"exp2_f{frame_idx:05d}.png"
                cv2.imwrite(str(save_path), combined)
                saved_count += 1
                print(f"  saved → {save_path}")

            frame_idx += 1

    cv2.destroyAllWindows()
    print(f"Done. {saved_count} frame(s) saved to {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 2 — pose on crops vs full frame")
    ap.add_argument("--config", default="pen_config.json")
    ap.add_argument("--pair", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--output-dir", default="output/exp2")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(cfg, args.pair, args.start_frame, Path(args.output_dir))


if __name__ == "__main__":
    main()
