"""
Experiment 1 — Raw detection viewer.

Runs the segmentation model on the top/cam0 frame and the pose model on the
side/cam1 frame, so you can see exactly what each model produces before any
homography, tracking, or fusion is applied.

Controls
--------
Space           : pause / resume
Right arrow / . : step one frame (while paused)
Left arrow  / , : step back one frame (while paused — re-seeks)
+  / =          : speed up
-               : slow down
s               : save current combined image to output/exp1/
q               : quit

Usage
-----
    # Via config + matching timestamp stem (loads cam0 + cam1 automatically):
    python scripts/exp1_detect.py --config pen_config.json --pair 20260605_141702

    # Explicit paths (two cameras side by side):
    python scripts/exp1_detect.py --cam0 dataset/treatment1/cam0/20260605_141702.mp4 \
                                  --cam1 dataset/treatment1/cam1/20260605_141702.mp4

    # Single video only:
    python scripts/exp1_detect.py --cam0 dataset/treatment1/cam0/20260605_141702.mp4
    python scripts/exp1_detect.py --cam1 dataset/treatment1/cam1/20260605_141702.mp4

    # Start at a specific frame:
    python scripts/exp1_detect.py --config pen_config.json --pair 20260605_141702 --start-frame 50
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

# ── Keypoint schema (hen_pose_yolo12s_scratch.pt) ────────────────────────────
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

# ── Segmentation class colours (BGR) ─────────────────────────────────────────
_CLASS_NAMES = {0: "feeder", 1: "hen", 2: "waterer"}
_CLASS_BGR   = {0: (0, 165, 255), 1: (220, 80, 0), 2: (255, 200, 0)}

_DISPLAY_H = 720
_DEFAULT_FPS = 10   # playback speed in frames/sec; adjust with +/-


def _draw_seg(frame: np.ndarray, dets) -> np.ndarray:
    out = frame.copy()
    for det in dets:
        cid  = det.class_id
        bgr  = _CLASS_BGR.get(cid, (200, 200, 200))
        name = _CLASS_NAMES.get(cid, str(cid))
        box  = det.box.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), bgr, 2)
        label = f"{name} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(box[1] - 3, th + 3)
        cv2.rectangle(out, (box[0], ty - th - 3), (box[0] + tw + 2, ty + 1), bgr, -1)
        cv2.putText(out, label, (box[0] + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_pose(frame: np.ndarray, dets) -> np.ndarray:
    out = frame.copy()
    for det in dets:
        box = det.box.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), (100, 100, 255), 2)
        cv2.putText(out, f"{det.conf:.2f}", (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1, cv2.LINE_AA)

        kps = det.kps
        if kps is None:
            continue

        for i1, i2 in _SKELETON:
            if i1 >= len(kps) or i2 >= len(kps):
                continue
            x1, y1, c1 = kps[i1]
            x2, y2, c2 = kps[i2]
            if c1 >= _KP_CONF_THRESHOLD and c2 >= _KP_CONF_THRESHOLD:
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 140, 0), 2, cv2.LINE_AA)

        for k, name in enumerate(_KP_NAMES):
            if k >= len(kps):
                continue
            xk, yk, ck = kps[k]
            if ck >= _KP_CONF_THRESHOLD:
                cv2.circle(out, (int(xk), int(yk)), 4, (0, 255, 255), -1)
                abbrev = name[:3]
                cv2.putText(out, f"{abbrev}{ck:.1f}", (int(xk) + 5, int(yk) - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1)
            else:
                cv2.circle(out, (int(xk), int(yk)), 3, (80, 80, 80), -1)
    return out


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _overlay_hud(img: np.ndarray, frame_idx: int, fps: float, paused: bool) -> np.ndarray:
    out = img.copy()
    state = "PAUSED" if paused else f"{fps:.0f} fps"
    text = f"f={frame_idx}  {state}  [space]=pause  [+/-]=speed  [s]=save  [q]=quit"
    cv2.putText(out, text, (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (8, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def _read_next(gen):
    try:
        return next(gen)
    except StopIteration:
        return None


def run(
    cam0_path: Path | None,
    cam1_path: Path | None,
    seg_model_path: str,
    pose_model_path: str,
    start_frame: int,
    output_dir: Path,
    playback_fps: float = _DEFAULT_FPS,
) -> None:
    dual = cam0_path is not None and cam1_path is not None
    if cam0_path is None and cam1_path is None:
        print("Error: at least one of --cam0 or --cam1 must be provided.")
        sys.exit(1)

    seg_model  = Detector(seg_model_path)
    pose_model = Detector(pose_model_path)

    if dual:
        win_title = ("Exp 1 — cam0 (seg, left)  |  cam1 (pose, right)"
                     "  |  space=pause  +/-=speed  s=save  q=quit")
    elif cam0_path:
        win_title = f"Exp 1 — cam0 (seg)  |  {cam0_path.name}"
    else:
        win_title = f"Exp 1 — cam1 (pose)  |  {cam1_path.name}"

    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx   = 0
    saved_count = 0
    paused      = False
    fps         = playback_fps

    def _open(path: Path):
        cfg = CameraSourceConfig(name=path.stem, source=str(path), camera_type="file")
        return FrameSource(cfg)

    # We use cv2 directly for single-video seek (FrameSource generators are forward-only)
    # For the initial start_frame seek we just skip via the generator.
    sources = []
    if cam0_path:
        sources.append(_open(cam0_path))
    if cam1_path:
        sources.append(_open(cam1_path))

    with sources[0] as s0, *(([sources[1]] if dual else [])):
        gen0 = iter(s0)
        gen1 = iter(sources[1]) if dual else None

        # Seek to start frame
        for _ in range(start_frame):
            f0 = _read_next(gen0)
            f1 = _read_next(gen1) if gen1 else None
            if f0 is None or (dual and f1 is None):
                print(f"Video ended before reaching frame {start_frame}.")
                return
            frame_idx += 1

        last_combined = None

        while True:
            if not paused:
                f0 = _read_next(gen0) if gen0 else None
                f1 = _read_next(gen1) if gen1 else None

                if (cam0_path and f0 is None) or (cam1_path and f1 is None):
                    print("End of video.")
                    break

                # Detect
                if cam0_path and f0 is not None:
                    top_dets = seg_model.predict(f0)
                    ann0 = _draw_seg(f0, top_dets)
                    n_hen    = sum(1 for d in top_dets if d.class_id == 1)
                    n_feeder = sum(1 for d in top_dets if d.class_id == 0)
                    ann0 = _add_label(ann0, f"CAM0 (top)  f={frame_idx}  hens={n_hen}  feeder={n_feeder}")
                else:
                    ann0 = None

                if cam1_path and f1 is not None:
                    side_dets = pose_model.predict(f1)
                    ann1 = _draw_pose(f1, side_dets)
                    n_side = len(side_dets)
                    ann1 = _add_label(ann1, f"CAM1 (side)  f={frame_idx}  dets={n_side}")
                else:
                    ann1 = None

                if dual:
                    combined = np.hstack([
                        _resize_h(ann0, _DISPLAY_H),
                        _resize_h(ann1, _DISPLAY_H),
                    ])
                else:
                    combined = _resize_h(ann0 if ann0 is not None else ann1, _DISPLAY_H)

                combined = _overlay_hud(combined, frame_idx, fps, paused)
                last_combined = combined

                if cam0_path:
                    print(f"frame {frame_idx:5d}  |  cam0: {n_hen} hen, {n_feeder} feeder"
                          + (f"  |  cam1: {n_side} pose det(s)" if dual else ""))
                else:
                    print(f"frame {frame_idx:5d}  |  cam1: {n_side} pose det(s)")

                frame_idx += 1

            if last_combined is not None:
                display = last_combined
                if paused:
                    display = _overlay_hud(last_combined.copy(), frame_idx - 1, fps, paused)
                cv2.imshow(win_title, display)

            wait_ms = max(1, int(1000 / fps)) if not paused else 30
            key = cv2.waitKey(wait_ms) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")
            elif key in (ord('+'), ord('=')):
                fps = min(fps + 5, 120)
                print(f"Speed: {fps:.0f} fps")
            elif key == ord('-'):
                fps = max(fps - 5, 1)
                print(f"Speed: {fps:.0f} fps")
            elif key == ord('s') and last_combined is not None:
                save_path = output_dir / f"exp1_f{frame_idx - 1:05d}.png"
                cv2.imwrite(str(save_path), last_combined)
                saved_count += 1
                print(f"  saved → {save_path}")

    cv2.destroyAllWindows()
    print(f"Done. {saved_count} frame(s) saved to {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp 1 — raw detection viewer")

    # Config+pair mode (original)
    ap.add_argument("--config", default="pen_config.json",
                    help="pen_config.json path (used with --pair)")
    ap.add_argument("--pair",
                    help="Video stem shared by both cameras, e.g. 20260605_141702")

    # Direct path mode
    ap.add_argument("--cam0", help="Direct path to cam0 (top) video file")
    ap.add_argument("--cam1", help="Direct path to cam1 (side) video file")

    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--fps", type=float, default=_DEFAULT_FPS,
                    help=f"Playback speed in frames/sec (default {_DEFAULT_FPS})")
    ap.add_argument("--output-dir", default="output/exp1")
    args = ap.parse_args()

    # Resolve video paths
    cam0_path: Path | None = None
    cam1_path: Path | None = None

    if args.pair:
        with open(args.config) as f:
            cfg = json.load(f)
        root = Path(cfg["dataset_root"])
        cam0_path = root / cfg["cameras"]["top"]  / f"{args.pair}.mp4"
        cam1_path = root / cfg["cameras"]["side"] / f"{args.pair}.mp4"
        seg_model  = cfg["models"]["segmentation"]
        pose_model = cfg["models"]["pose"]
    else:
        if not (args.cam0 or args.cam1):
            ap.error("Provide either --pair or at least one of --cam0 / --cam1")
        cam0_path = Path(args.cam0) if args.cam0 else None
        cam1_path = Path(args.cam1) if args.cam1 else None
        # Load model paths from config when available, else fall back to defaults
        try:
            with open(args.config) as f:
                cfg = json.load(f)
            seg_model  = cfg["models"]["segmentation"]
            pose_model = cfg["models"]["pose"]
        except Exception:
            seg_model  = "models/poultry-seg-large.pt"
            pose_model = "models/hen_pose_yolo12s_scratch.pt"

    for p in filter(None, [cam0_path, cam1_path]):
        if not p.exists():
            print(f"Error: video not found: {p}")
            sys.exit(1)

    run(
        cam0_path=cam0_path,
        cam1_path=cam1_path,
        seg_model_path=seg_model,
        pose_model_path=pose_model,
        start_frame=args.start_frame,
        output_dir=Path(args.output_dir),
        playback_fps=args.fps,
    )


if __name__ == "__main__":
    main()
