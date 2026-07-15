"""
Dual-camera detection + tracking with colour identity and floor gating.

This is the first milestone of the multiview system: clean DETECTION + TRACKING
only (no pose, no behaviour).  For every aligned frame pair it:

    1. INGEST  — read one frame from each camera (cam0=top, cam1=side)
    2. ALIGN   — pair by nearest timestamp (RingBufferSynchronizer)
    3. DETECT  — YOLO detect + ByteTrack, independently per camera
    4. GATE    — project each box to the metric floor; drop anything that lands
                 outside the pen rectangle (netting / wall / shadow = false +ve)
    5. COLOR   — classify each hen's neck marker (red/blue/black/green/yellow)
    6. FUSE    — ColorIdentityTracker merges per-colour into 5 fixed slots,
                 averaging a colour seen by both cameras and keeping last-known
                 position for a colour seen by neither (graceful out-of-frame)
    7. FEEDER/WATERER — accumulate a stable in-bounds floor position for each
    8. SHOW    — annotated cam views + a live top-down metric floor map
    9. LOG     — optional per-frame CSV of the pen state

Works identically on recorded videos and on live USB cameras (the Raspberry Pi
deployment) — only the source paths change.

Usage
-----
    # config + matching timestamp stem (loads cam0 + cam1):
    python scripts/track_pen.py --config pen_config.json --pair 20260605_171703

    # explicit paths:
    python scripts/track_pen.py --cam0 dataset/treatment1/cam0/X.mp4 \
                                --cam1 dataset/treatment1/cam1/X.mp4

    # write CSV, no windows (batch):
    python scripts/track_pen.py --config pen_config.json --pair X \
                                --csv output/X_track.csv --no-show

Controls:  [space] pause   [q] quit
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO

from src.color_marker import classify_neck_color
from src.floor import FloorModel
from src.identity import ColorIdentityTracker, HenSlot, Observation
from src.multiview import RingBufferSynchronizer, SyncSettings, TimestampedFrame
from src.pen_overlay import PenProjector, draw_floor_boundary, draw_walls, load_intrinsics

# Detection class ids (det0-yolo12s.pt: {0: feeder, 1: hen, 2: waterer})
_CLS_FEEDER, _CLS_HEN, _CLS_WATERER = 0, 1, 2

_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    "red":       (0,   0, 220),
    "blue":      (220, 80,  0),
    "black":     (50,  50, 50),
    "green":     (0,  200,  0),
    "yellow":    (0, 220, 220),
    "uncertain": (160, 160, 160),
}
_DISPLAY_H = 540


# ── Fixed-object (feeder / waterer) accumulator ───────────────────────────────

class FixedObjectAccumulator:
    """Average in-bounds floor positions, then lock a stable estimate."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._buf: list[tuple[float, float]] = []
        self.position: Optional[tuple[float, float]] = None

    def add(self, xy: tuple[float, float]) -> None:
        if self.position is not None:
            return
        self._buf.append(xy)
        if len(self._buf) >= self._n:
            self.position = (float(np.mean([p[0] for p in self._buf])),
                             float(np.mean([p[1] for p in self._buf])))


# ── Per-camera detect + track + gate + colour ─────────────────────────────────

def _process_camera(
    model: YOLO,
    frame: np.ndarray,
    cam: str,
    floor: FloorModel,
    feeder_acc: FixedObjectAccumulator,
    waterer_acc: FixedObjectAccumulator,
    conf: float,
    iou: float,
) -> tuple[list[Observation], int]:
    """Return (hen observations, n_rejected) for one camera frame."""
    res = model.track(
        frame, persist=True, verbose=False,
        conf=conf, iou=iou, classes=[_CLS_FEEDER, _CLS_HEN, _CLS_WATERER],
    )[0]

    observations: list[Observation] = []
    rejected = 0
    if res.boxes is None:
        return observations, rejected

    xyxy = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    ids = (res.boxes.id.cpu().numpy().astype(int)
           if res.boxes.id is not None else None)

    # Track best in-bounds feeder/waterer detection this frame.
    best_feeder: Optional[tuple[float, tuple[float, float]]] = None
    best_waterer: Optional[tuple[float, tuple[float, float]]] = None

    for i in range(len(xyxy)):
        box = xyxy[i]
        fxy = floor.to_floor(cam, box)
        if not floor.in_bounds(fxy):
            rejected += 1                      # off-floor → false positive
            continue
        c = clss[i]
        if c == _CLS_HEN:
            if ids is None:                    # need a stable id to vote on colour
                continue
            color = classify_neck_color(frame, box=box)
            observations.append(Observation(
                camera=cam, track_id=int(ids[i]), color=color,
                floor_xy=fxy, box=box, conf=float(confs[i]),
            ))
        elif c == _CLS_FEEDER:
            if best_feeder is None or confs[i] > best_feeder[0]:
                best_feeder = (float(confs[i]), fxy)
        elif c == _CLS_WATERER:
            if best_waterer is None or confs[i] > best_waterer[0]:
                best_waterer = (float(confs[i]), fxy)

    if best_feeder is not None:
        feeder_acc.add(best_feeder[1])
    if best_waterer is not None:
        waterer_acc.add(best_waterer[1])

    return observations, rejected


# ── Drawing ───────────────────────────────────────────────────────────────────

def _annotate_camera(
    frame: np.ndarray,
    cam: str,
    slots: list[HenSlot],
    rejected: int,
    frame_idx: int,
    floor: Optional[FloorModel] = None,
    proj: Optional[PenProjector] = None,
    wall_h: float = 0.0,
) -> np.ndarray:
    out = frame.copy()
    # Calibration overlays first, so detections draw on top.
    if floor is not None:
        draw_floor_boundary(out, floor, cam)
        if proj is not None and wall_h > 0:
            draw_walls(out, floor, proj, wall_h)
    for slot in slots:
        box = slot.boxes.get(cam)
        if box is None:
            continue
        bgr = _COLOR_BGR.get(slot.color, _COLOR_BGR["uncertain"])
        b = box.astype(int)
        cv2.rectangle(out, (b[0], b[1]), (b[2], b[3]), bgr, 2)
        text = slot.color
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(b[1] - 4, th + 4)
        cv2.rectangle(out, (b[0], ty - th - 3), (b[0] + tw + 2, ty + 1), bgr, -1)
        txt = (0, 0, 0) if slot.color == "yellow" else (255, 255, 255)
        cv2.putText(out, text, (b[0] + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt, 1, cv2.LINE_AA)

    label = f"{'CAM0 top' if cam == 'top' else 'CAM1 side'}  f={frame_idx}  rejected={rejected}"
    cv2.putText(out, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _render_floor_map(
    floor: FloorModel,
    slots: list[HenSlot],
    feeder: Optional[tuple[float, float]],
    waterer: Optional[tuple[float, float]],
    height: int,
) -> np.ndarray:
    """Top-down metric schematic of the pen and current pen state."""
    pad = 40
    pen_h = height - 2 * pad
    scale = pen_h / floor.length_m
    pen_w = int(floor.width_m * scale)
    img = np.full((height, pen_w + 2 * pad, 3), 30, dtype=np.uint8)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return (int(pad + x * scale), int(pad + y * scale))

    # Pen rectangle (Y=0 near/top-cam wall at the visual top).
    cv2.rectangle(img, to_px(0, 0), to_px(floor.width_m, floor.length_m),
                  (200, 200, 200), 2)
    cv2.putText(img, "near (cam0)", (pad, pad - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(img, "far (cam1)", (pad, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    for label, pos, marker in [("feeder", feeder, "F"), ("waterer", waterer, "W")]:
        if pos is not None:
            px = to_px(*floor.clamp(pos))
            cv2.rectangle(img, (px[0] - 9, px[1] - 9), (px[0] + 9, px[1] + 9),
                          (0, 165, 255), 2)
            cv2.putText(img, marker, (px[0] - 5, px[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    for slot in slots:
        if slot.floor_xy is None:
            continue
        bgr = _COLOR_BGR.get(slot.color, _COLOR_BGR["uncertain"])
        px = to_px(*floor.clamp(slot.floor_xy))
        if slot.visible:
            cv2.circle(img, px, 10, bgr, -1)
            cv2.circle(img, px, 10, (255, 255, 255), 1)
            tag = slot.color[0].upper()
            cv2.putText(img, tag, (px[0] - 5, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0) if slot.color == "yellow" else (255, 255, 255), 1)
        else:
            cv2.circle(img, px, 9, (90, 90, 90), 1)      # last-known, missing
    return img


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(
    cam0_path: Path,
    cam1_path: Path,
    model_path: str,
    cfg: dict,
    conf: float,
    iou: float,
    margin_m: float,
    show: bool,
    csv_path: Optional[Path],
    walls: bool,
    save_video: Optional[Path],
    max_frames: Optional[int] = None,
) -> None:
    floor = FloorModel.from_config(cfg, margin_m=margin_m)
    wall_h = float(cfg.get("pen", {}).get("wall_height_m", 0.0))
    colors = cfg.get("marker_colors", None)
    tracker = ColorIdentityTracker(
        colors=colors,
        timeout_s=float(cfg.get("track_timeout_s", 2.0)),
    )
    stabilize_n = int(cfg.get("feeder_stabilize_frames", 30))
    feeder_acc = FixedObjectAccumulator(stabilize_n)
    waterer_acc = FixedObjectAccumulator(stabilize_n)

    # One model instance per camera so ByteTrack keeps separate state per stream.
    model_top = YOLO(model_path)
    model_side = YOLO(model_path)

    cap0 = cv2.VideoCapture(str(cam0_path))
    cap1 = cv2.VideoCapture(str(cam1_path))
    if not cap0.isOpened() or not cap1.isOpened():
        sys.exit(f"Could not open videos:\n  {cam0_path}\n  {cam1_path}")

    # Estimated camera projectors for the raised-wall wireframe (optional).
    proj_top = proj_side = None
    if walls:
        w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH)); h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)); h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        K_top, K_side = load_intrinsics(cfg, "top"), load_intrinsics(cfg, "side")
        print("walls: using",
              "calibrated K" if K_top is not None else "ESTIMATED K (run calibrate_intrinsics.py)")
        proj_top = PenProjector.build(floor, "top", w0, h0, K=K_top)
        proj_side = PenProjector.build(floor, "side", w1, h1, K=K_side)

    syncer = RingBufferSynchronizer(
        SyncSettings(tolerance_ms=float(cfg.get("sync_tolerance_ms", 75.0)),
                     buffer_size=30))

    csv_file = csv_writer = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        csv_writer = _csv.writer(csv_file)
        csv_writer.writerow(
            ["frame", "timestamp", "color", "visible", "floor_x", "floor_y", "views"])

    win = "Pen tracking — cam0 | cam1 | floor"
    if show:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    writer: Optional[cv2.VideoWriter] = None

    frame_idx = 0
    paused = False
    print(f"Model: {model_path}")
    print(f"Cam0 : {cam0_path}\nCam1 : {cam1_path}")
    print("Controls: [space] pause  [q] quit")

    while True:
        if not paused:
            ok0, f0 = cap0.read()
            ok1, f1 = cap1.read()
            if not ok0 or not ok1:
                print("End of video.")
                break
            if max_frames is not None and frame_idx >= max_frames:
                print(f"Reached max-frames={max_frames}.")
                break

            ts = time.monotonic()
            syncer.add_frame(TimestampedFrame(f0, ts * 1000.0, "top", frame_idx))
            syncer.add_frame(TimestampedFrame(f1, ts * 1000.0, "side", frame_idx))
            pair = syncer.get_pair("top", "side")
            if pair is None:
                frame_idx += 1
                continue
            top_f, side_f = pair.primary.frame, pair.secondary.frame
            ts_s = pair.primary.timestamp_ms / 1000.0

            obs_top, rej_top = _process_camera(
                model_top, top_f, "top", floor, feeder_acc, waterer_acc, conf, iou)
            obs_side, rej_side = _process_camera(
                model_side, side_f, "side", floor, feeder_acc, waterer_acc, conf, iou)

            tracker.update(obs_top + obs_side, ts_s)
            slots = tracker.states()

            if csv_writer is not None:
                for slot in slots:
                    fx = "" if slot.floor_x is None else f"{slot.floor_x:.4f}"
                    fy = "" if slot.floor_y is None else f"{slot.floor_y:.4f}"
                    csv_writer.writerow(
                        [frame_idx, f"{ts_s:.4f}", slot.color, int(slot.visible),
                         fx, fy, "+".join(sorted(slot.views))])

            if show or save_video is not None:
                ann0 = _annotate_camera(top_f, "top", slots, rej_top, frame_idx,
                                        floor, proj_top, wall_h)
                ann1 = _annotate_camera(side_f, "side", slots, rej_side, frame_idx,
                                        floor, proj_side, wall_h)
                fmap = _render_floor_map(
                    floor, slots, feeder_acc.position, waterer_acc.position, _DISPLAY_H)
                combined = np.hstack([
                    _resize_h(ann0, _DISPLAY_H),
                    _resize_h(ann1, _DISPLAY_H),
                    fmap,
                ])
                if show:
                    cv2.imshow(win, combined)
                if save_video is not None:
                    if writer is None:
                        save_video.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(
                            str(save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                            15.0, (combined.shape[1], combined.shape[0]))
                    writer.write(combined)

            if frame_idx % 50 == 0:
                vis = sum(1 for s in slots if s.visible)
                print(f"f={frame_idx:5d}  visible={vis}/{len(slots)}  "
                      f"missing={tracker.missing()}  "
                      f"feeder={'ok' if feeder_acc.position else '...'}  "
                      f"waterer={'ok' if waterer_acc.position else '...'}")
            frame_idx += 1

        if show:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
                print("PAUSED" if paused else "RESUMED")

    cap0.release()
    cap1.release()
    if writer is not None:
        writer.release()
        print(f"Video written -> {save_video}")
    if csv_file is not None:
        csv_file.close()
        print(f"CSV written -> {csv_path}")
    if show:
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dual-camera detection + tracking")
    ap.add_argument("--config", default="pen_config.json")
    ap.add_argument("--pair", help="Shared video stem, e.g. 20260605_171703")
    ap.add_argument("--cam0", help="cam0 (top) video path")
    ap.add_argument("--cam1", help="cam1 (side) video path")
    ap.add_argument("--model", default=None, help="Detection .pt (default: det0-yolo12s.pt)")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.15,
                    help="Floor-gate tolerance in metres outside the pen")
    ap.add_argument("--csv", default=None, help="Write per-frame pen state CSV")
    ap.add_argument("--walls", action="store_true",
                    help="Draw the ESTIMATED raised-wall wireframe (z axis)")
    ap.add_argument("--save-video", default=None,
                    help="Write the annotated combined view to this .mp4")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Stop after N frames (for quick previews)")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    if args.pair:
        root = Path(cfg["dataset_root"])
        cam0 = root / cfg["cameras"]["top"] / f"{args.pair}.mp4"
        cam1 = root / cfg["cameras"]["side"] / f"{args.pair}.mp4"
    elif args.cam0 and args.cam1:
        cam0, cam1 = Path(args.cam0), Path(args.cam1)
    else:
        ap.error("Provide --pair, or both --cam0 and --cam1")

    for p in (cam0, cam1):
        if not p.exists():
            sys.exit(f"Video not found: {p}")

    model_path = args.model or cfg.get("models", {}).get("detection", "models/det0-yolo12s.pt")

    run(
        cam0_path=cam0, cam1_path=cam1, model_path=model_path, cfg=cfg,
        conf=args.conf, iou=args.iou, margin_m=args.margin,
        show=not args.no_show, csv_path=Path(args.csv) if args.csv else None,
        walls=args.walls, save_video=Path(args.save_video) if args.save_video else None,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
