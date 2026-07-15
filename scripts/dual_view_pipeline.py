"""
Dual-camera multi-view pen-state pipeline — Treatment 1.

Processes matched video pairs from cam0 (top-down) and cam1 (side), aligns
frames temporally, runs segmentation + pose inference, fuses both views into a
metric floor coordinate system, and maintains a live pen state for up to 5
color-marked hens.

Usage
-----
    python scripts/dual_view_pipeline.py --config pen_config.json
    python scripts/dual_view_pipeline.py --config pen_config.json --pair 20260605_141702
    python scripts/dual_view_pipeline.py --config pen_config.json --no-viz --output-dir out/

Steps per aligned frame pair (strict order):
    1. INGEST  — read one frame from each source
    2. ALIGN   — pair by nearest timestamp (RingBufferSynchronizer)
    3. INFER   — top→seg, side→pose
    4. TRACK   — Ultralytics persist=True tracker on top view (cap at num_birds)
    5. COLOR   — classify neck-marker HSV patch (neck_back index 6)
    6. FEEDER/WATERER — detect and accumulate floor positions until stable
    7. FUSE    — project floor XY; associate side poses to top tracks
    8. PEN STATE — assemble PenState dataclass
    9. LOG     — write top and side rows to CSV
   10. VIZ    — update 2.5D matplotlib Axes3D
"""

from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root first so src/ modules are importable and so the local
# ultralytics/ copy is used (its AAttn.forward now dispatches on qkv vs qk,
# supporting both checkpoint styles in the same process).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.capture import CameraSourceConfig, FrameSource
from src.color_marker import classify_neck_color
from src.csv_logger import CSVLogger
from src.detect import Detector
from src.geometry import euclidean_distance, get_box_center, get_box_height, transform_point
from src.multiview import RingBufferSynchronizer, SyncSettings, TimestampedFrame
from src.pen_state import HenEntry, PenState
from src.pose import PoseDetector
from src.viz_3d import PenViz3D

# Keypoint index for neck-back (color marker) in the new schema
_KP_NECK_BACK = 6
# Ground-contact keypoint indices (hocks and feet)
_GROUND_KPS = [2, 3, 4, 5]
_KP_CONF_THRESHOLD = 0.3
# Cross-view association distance gate (metres)
_ASSOC_GATE_M = 0.30

# Segmentation class IDs
_CLASS_HEN     = 1
_CLASS_FEEDER  = 0
_CLASS_WATERER = 2


# ── Config helpers ────────────────────────────────────────────────────────────

def _load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = json.load(f)
    h_top = cfg["homographies"].get("H_top_floor")
    if h_top is None:
        sys.exit(
            "pen_config.json has no H_top_floor. "
            "Run calibrate_corners.py first."
        )
    cfg["H_top_floor"]  = np.array(h_top,  dtype=np.float64)
    cfg["H_side_top"]   = np.array(cfg["homographies"]["H_side_top"],   dtype=np.float64)
    cfg["H_side_floor"] = np.array(cfg["homographies"]["H_side_floor"], dtype=np.float64)
    return cfg


def _matched_pairs(dataset_root: Path, top_dir: str, side_dir: str) -> list[str]:
    cam0 = dataset_root / top_dir
    cam1 = dataset_root / side_dir
    stems0 = {p.stem for p in cam0.glob("*.mp4")}
    stems1 = {p.stem for p in cam1.glob("*.mp4")}
    return sorted(stems0 & stems1)


# ── Track state ───────────────────────────────────────────────────────────────

class TrackState:
    """
    Maps Ultralytics track_id → color label using majority-vote across frames.

    Per-frame classification is noisy; accumulating votes per track ID ensures
    the most consistently observed color wins.
    """

    def __init__(self, timeout_s: float) -> None:
        self._timeout = timeout_s
        self._votes: dict[int, dict[str, int]] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, color: str, ts: float) -> None:
        self._last_seen[track_id] = ts
        if color != "uncertain":
            if track_id not in self._votes:
                self._votes[track_id] = {}
            self._votes[track_id][color] = self._votes[track_id].get(color, 0) + 1

    def color(self, track_id: int) -> str:
        votes = self._votes.get(track_id, {})
        if not votes:
            return "uncertain"
        return max(votes, key=lambda c: votes[c])

    def expire(self, ts: float) -> None:
        stale = [tid for tid, t in self._last_seen.items() if ts - t > self._timeout]
        for tid in stale:
            self._votes.pop(tid, None)
            self._last_seen.pop(tid, None)


# ── Feeder / waterer accumulator ──────────────────────────────────────────────

class FixedObjectAccumulator:
    """Accumulate projected floor positions and return stable estimate after N frames."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._buffer: list[tuple[float, float]] = []
        self._stable: Optional[tuple[float, float]] = None

    @property
    def position(self) -> Optional[tuple[float, float]]:
        return self._stable

    def add(self, xy: tuple[float, float]) -> None:
        if self._stable is not None:
            return
        self._buffer.append(xy)
        if len(self._buffer) >= self._n:
            xs = [p[0] for p in self._buffer]
            ys = [p[1] for p in self._buffer]
            self._stable = (float(np.mean(xs)), float(np.mean(ys)))


# ── Annotation / display helpers ─────────────────────────────────────────────

_LABEL_BGR: dict[str, tuple[int, int, int]] = {
    "green":     (0,   200,   0),
    "black":     (50,   50,  50),
    "red":       (0,     0, 220),
    "blue":      (220,  80,   0),
    "yellow":    (0,   220, 220),
    "uncertain": (160, 160, 160),
}

# Skeleton connections — keypoint indices matching the pose model schema
_SIDE_SKELETON = [
    (9, 8), (8, 6), (6, 7), (7, 0), (0, 1),
    (0, 2), (0, 3), (7, 2), (7, 3), (2, 4), (3, 5),
]


def _draw_top_annotations(
    frame: np.ndarray,
    top_hens: list[dict],
    track_state: TrackState,
) -> np.ndarray:
    out = frame.copy()
    for hen in top_hens:
        tid   = hen["track_id"]
        label = track_state.color(tid)
        bgr   = _LABEL_BGR.get(label, _LABEL_BGR["uncertain"])
        box   = hen["box"].astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), bgr, 2)
        text = f"#{tid} {label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(box[1] - 4, th + 4)
        cv2.rectangle(out, (box[0], ty - th - 3), (box[0] + tw + 2, ty + 1), bgr, -1)
        txt_bgr = (0, 0, 0) if label == "yellow" else (255, 255, 255)
        cv2.putText(out, text, (box[0] + 1, ty - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_bgr, 1, cv2.LINE_AA)
        # Dot at bbox centre (approximate colour-sampling location)
        cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
        cv2.circle(out, (cx, cy), 4, bgr, -1)
    return out


def _draw_side_annotations(
    frame: np.ndarray,
    side_dets,
    matched_indices: set[int],
) -> np.ndarray:
    out = frame.copy()
    for idx, det in enumerate(side_dets):
        matched = idx in matched_indices
        box_bgr = (0, 220, 0) if matched else (0, 60, 220)
        box = det.box.astype(int)
        cv2.rectangle(out, (box[0], box[1]), (box[2], box[3]), box_bgr, 2)

        kps = det.kps
        if kps is None:
            continue
        # Skeleton
        for i1, i2 in _SIDE_SKELETON:
            if i1 >= len(kps) or i2 >= len(kps):
                continue
            x1, y1, c1 = kps[i1]
            x2, y2, c2 = kps[i2]
            if c1 >= _KP_CONF_THRESHOLD and c2 >= _KP_CONF_THRESHOLD:
                cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 140, 0), 2, cv2.LINE_AA)
        # Keypoint dots
        for k in range(len(kps)):
            xk, yk, ck = kps[k]
            if ck >= _KP_CONF_THRESHOLD:
                cv2.circle(out, (int(xk), int(yk)), 3, (0, 255, 255), -1)
    return out


def _show_combined(top_ann: np.ndarray, side_ann: np.ndarray, win_name: str) -> None:
    target_h = 540

    def _resize_h(img: np.ndarray) -> np.ndarray:
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h))

    combined = np.hstack([_resize_h(top_ann), _resize_h(side_ann)])
    cv2.imshow(win_name, combined)
    cv2.waitKey(1)


# ── Step 7: spatial fusion ────────────────────────────────────────────────────

def _ground_contact(det_box: np.ndarray, kps: Optional[np.ndarray]) -> tuple[float, float]:
    """Return the best floor-contact pixel from a side-view detection."""
    if kps is not None:
        pts = []
        for idx in _GROUND_KPS:
            if idx < len(kps) and float(kps[idx][2]) >= _KP_CONF_THRESHOLD:
                pts.append((float(kps[idx][0]), float(kps[idx][1])))
        if pts:
            return (float(np.mean([p[0] for p in pts])),
                    float(np.mean([p[1] for p in pts])))
    # Fallback: bottom-centre of bbox
    cx = float((det_box[0] + det_box[2]) / 2)
    cy = float(det_box[3])
    return (cx, cy)


def _approx_z(det_box: np.ndarray, wall_height_m: float, frame_height: int) -> float:
    """Approximate Z from the bounding-box height in the side view."""
    box_h_px = float(det_box[3] - det_box[1])
    scale = wall_height_m / frame_height
    return box_h_px * scale / 2.0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    cfg: dict,
    pairs: list[str],
    output_dir: Path,
    show_viz: bool,
    show_frames: bool,
) -> None:
    dataset_root = Path(cfg["dataset_root"])
    top_dir  = cfg["cameras"]["top"]
    side_dir = cfg["cameras"]["side"]
    num_birds      = int(cfg["num_birds"])
    timeout_s      = float(cfg["track_timeout_s"])
    stabilize_n    = int(cfg["feeder_stabilize_frames"])
    tolerance_ms   = float(cfg["sync_tolerance_ms"])
    wall_h         = float(cfg["pen"]["wall_height_m"])
    H_top_floor    = cfg["H_top_floor"]
    H_side_floor   = cfg["H_side_floor"]

    seg_model_path  = cfg["models"]["segmentation"]
    pose_model_path = cfg["models"]["pose"]

    seg_detector  = Detector(seg_model_path)
    pose_detector = PoseDetector(pose_model_path)

    # Load pose model separately for tracking (persist=True requires YOLO instance)
    seg_tracker = YOLO(seg_model_path)

    sync_settings = SyncSettings(tolerance_ms=tolerance_ms, buffer_size=30)

    track_state = TrackState(timeout_s)
    feeder_acc  = FixedObjectAccumulator(stabilize_n)
    waterer_acc = FixedObjectAccumulator(stabilize_n)

    viz: Optional[PenViz3D] = None
    if show_viz:
        viz = PenViz3D(
            pen_width_m=float(cfg["pen"]["width_m"]),
            pen_length_m=float(cfg["pen"]["length_m"]),
            wall_height_m=wall_h,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for stem in pairs:
        top_video  = dataset_root / top_dir  / f"{stem}.mp4"
        side_video = dataset_root / side_dir / f"{stem}.mp4"

        print(f"\nProcessing pair: {stem}")

        csv_path = output_dir / f"{stem}.csv"
        with CSVLogger(csv_path) as logger:
            _process_pair(
                top_video=top_video,
                side_video=side_video,
                seg_tracker=seg_tracker,
                pose_detector=pose_detector,
                sync_settings=sync_settings,
                track_state=track_state,
                feeder_acc=feeder_acc,
                waterer_acc=waterer_acc,
                H_top_floor=H_top_floor,
                H_side_floor=H_side_floor,
                num_birds=num_birds,
                wall_h=wall_h,
                logger=logger,
                viz=viz,
                show_frames=show_frames,
            )

    if viz is not None:
        viz.close()


def _process_pair(
    *,
    top_video: Path,
    side_video: Path,
    seg_tracker: YOLO,
    pose_detector: PoseDetector,
    sync_settings: SyncSettings,
    track_state: TrackState,
    feeder_acc: FixedObjectAccumulator,
    waterer_acc: FixedObjectAccumulator,
    H_top_floor: np.ndarray,
    H_side_floor: np.ndarray,
    num_birds: int,
    wall_h: float,
    logger: CSVLogger,
    viz: Optional[PenViz3D],
    show_frames: bool,
) -> None:
    top_cfg  = CameraSourceConfig(name="top",  source=str(top_video),  camera_type="file")
    side_cfg = CameraSourceConfig(name="side", source=str(side_video), camera_type="file")

    syncer   = RingBufferSynchronizer(sync_settings)
    frame_id = 0

    with FrameSource(top_cfg) as top_src, FrameSource(side_cfg) as side_src:
        top_gen  = iter(top_src)
        side_gen = iter(side_src)
        top_fid  = 0
        side_fid = 0

        while True:
            # ── 1. INGEST ─────────────────────────────────────────────────────
            try:
                top_frame = next(top_gen)
            except StopIteration:
                break
            try:
                side_frame = next(side_gen)
            except StopIteration:
                break

            ts = time.monotonic()
            top_tf  = TimestampedFrame(top_frame,  ts * 1000.0, "top",  top_fid)
            side_tf = TimestampedFrame(side_frame, ts * 1000.0, "side", side_fid)
            top_fid  += 1
            side_fid += 1

            # ── 2. ALIGN ──────────────────────────────────────────────────────
            syncer.add_frame(top_tf)
            syncer.add_frame(side_tf)
            pair = syncer.get_pair("top", "side")
            if pair is None:
                continue

            top_f  = pair.primary.frame
            side_f = pair.secondary.frame
            ts_s   = pair.primary.timestamp_ms / 1000.0
            frame_h, frame_w = side_f.shape[:2]

            # ── 3. INFER ──────────────────────────────────────────────────────
            seg_results = seg_tracker.track(
                top_f, persist=True, verbose=False,
                conf=0.4, iou=0.5,
            )
            side_dets = pose_detector.predict(side_f)

            # ── 4. TRACK (cap at num_birds on top view) ───────────────────────
            result = seg_results[0] if seg_results else None
            top_hens: list[dict] = []     # {track_id, box, conf, kps}
            top_feeder_dets: list[np.ndarray] = []
            top_waterer_dets: list[np.ndarray] = []

            if result is not None and result.boxes is not None:
                boxes      = result.boxes
                class_ids  = boxes.cls.cpu().numpy().astype(int)
                confs      = boxes.conf.cpu().numpy()
                xyxy       = boxes.xyxy.cpu().numpy()
                track_ids  = (boxes.id.cpu().numpy().astype(int)
                              if boxes.id is not None
                              else np.zeros(len(boxes), dtype=int))
                kps_arr    = (result.keypoints.xy.cpu().numpy()
                              if result.keypoints is not None else None)
                kps_conf   = (result.keypoints.conf.cpu().numpy()
                              if result.keypoints is not None else None)

                for i, (cid, conf, box, tid) in enumerate(
                        zip(class_ids, confs, xyxy, track_ids)):
                    if cid == _CLASS_FEEDER:
                        top_feeder_dets.append(box)
                    elif cid == _CLASS_WATERER:
                        top_waterer_dets.append(box)
                    elif cid == _CLASS_HEN:
                        kps: Optional[np.ndarray] = None
                        if kps_arr is not None and i < len(kps_arr):
                            xy  = kps_arr[i]         # (N, 2)
                            vc  = kps_conf[i]        # (N,) or None
                            kps = np.concatenate(
                                [xy, (vc[:, None] if vc is not None
                                      else np.ones((len(xy), 1)))],
                                axis=1,
                            )
                        top_hens.append({"track_id": int(tid), "box": box,
                                         "conf": float(conf), "kps": kps})

                # Keep at most num_birds by confidence
                top_hens.sort(key=lambda h: h["conf"], reverse=True)
                top_hens = top_hens[:num_birds]

            track_state.expire(ts_s)

            # ── 5. COLOR ──────────────────────────────────────────────────────
            for hen in top_hens:
                color = classify_neck_color(top_f, kps=hen["kps"], box=hen["box"])
                track_state.update(hen["track_id"], color, ts_s)

            # ── 6. FEEDER / WATERER ───────────────────────────────────────────
            for box in top_feeder_dets:
                cx = float((box[0] + box[2]) / 2)
                cy = float(box[3])
                floor_xy = transform_point((cx, cy), H_top_floor)
                feeder_acc.add(floor_xy)

            for box in top_waterer_dets:
                cx = float((box[0] + box[2]) / 2)
                cy = float(box[3])
                floor_xy = transform_point((cx, cy), H_top_floor)
                waterer_acc.add(floor_xy)

            # ── 7. FUSE ───────────────────────────────────────────────────────
            # Project top-hen floor positions
            hen_entries: list[HenEntry] = []
            for hen in top_hens:
                box  = hen["box"]
                cx   = float((box[0] + box[2]) / 2)
                cy   = float(box[3])
                fxy  = transform_point((cx, cy), H_top_floor)
                entry = HenEntry(
                    track_id=hen["track_id"],
                    color_label=track_state.color(hen["track_id"]),
                    floor_x=fxy[0],
                    floor_y=fxy[1],
                    top_box=box,
                    views={"top"},
                )
                hen_entries.append(entry)

            # Associate side pose detections to top tracks
            unmatched_side: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
            matched_side_idx: set[int] = set()
            for s_idx, side_det in enumerate(side_dets):
                box  = side_det.box
                kps  = side_det.kps
                gc   = _ground_contact(box, kps)
                fxy_s = transform_point(gc, H_side_floor)

                best_entry: Optional[HenEntry] = None
                best_dist  = float("inf")
                for entry in hen_entries:
                    d = euclidean_distance((entry.floor_x, entry.floor_y), fxy_s)
                    if d < best_dist and d <= _ASSOC_GATE_M:
                        best_dist  = d
                        best_entry = entry

                if best_entry is not None:
                    best_entry.side_kps  = kps
                    best_entry.approx_z  = _approx_z(box, wall_h, frame_h)
                    best_entry.views.add("side")
                    matched_side_idx.add(s_idx)
                else:
                    unmatched_side.append((box, kps))

            # ── 8. PEN STATE ──────────────────────────────────────────────────
            pen_state = PenState(
                timestamp=ts_s,
                hens=hen_entries,
                feeder_floor=feeder_acc.position,
                waterer_floor=waterer_acc.position,
            )

            # ── 9. LOG ────────────────────────────────────────────────────────
            logger.write_top(pen_state, frame_id)
            logger.write_side(pen_state, unmatched_side, frame_id)

            # ── 10. VIZ ───────────────────────────────────────────────────────
            if show_frames:
                ann_top  = _draw_top_annotations(top_f, top_hens, track_state)
                ann_side = _draw_side_annotations(side_f, side_dets, matched_side_idx)
                _show_combined(ann_top, ann_side, "Raw output | top (left)  side (right)")

            if viz is not None:
                viz.update(pen_state)

            frame_id += 1

            if frame_id % 100 == 0:
                print(f"  frame {frame_id}  hens={len(hen_entries)}"
                      f"  feeder={'ok' if feeder_acc.position else 'searching'}"
                      f"  waterer={'ok' if waterer_acc.position else 'searching'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-camera pen-state pipeline for Treatment 1."
    )
    parser.add_argument("--config", default="pen_config.json",
                        help="Path to pen_config.json")
    parser.add_argument("--pair", default=None,
                        help="Process a single video stem (e.g. 20260605_141702). "
                             "Omit to process all matched pairs.")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for CSV output files")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable the 2.5D visualization window")
    parser.add_argument("--no-frames", action="store_true",
                        help="Disable the annotated side-by-side frame window")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Config not found: {config_path}")

    cfg = _load_config(config_path)

    dataset_root = Path(cfg["dataset_root"])
    top_dir  = cfg["cameras"]["top"]
    side_dir = cfg["cameras"]["side"]

    if args.pair:
        pairs = [args.pair]
    else:
        pairs = _matched_pairs(dataset_root, top_dir, side_dir)
        if not pairs:
            sys.exit("No matching video pairs found.")
        print(f"Found {len(pairs)} matched video pairs.")

    run_pipeline(
        cfg=cfg,
        pairs=pairs,
        output_dir=Path(args.output_dir),
        show_viz=not args.no_viz,
        show_frames=not args.no_frames,
    )


if __name__ == "__main__":
    main()
