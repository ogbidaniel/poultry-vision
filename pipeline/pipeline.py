"""
Core inference pipeline.

YOLO pose model → per-frame detections + 10 keypoints → ReID → DB write → annotated frame.

Keypoint schema (must match trained model):
    0  beak
    1  crown
    2  back_neck   ← painted spot, ReID anchor ★
    3  middle_back
    4  tail_base
    5  tail_tip
    6  left_hock
    7  left_foot
    8  right_hock
    9  right_foot

Usage:
    from pipeline.pipeline import Pipeline
    pipe = Pipeline(config)
    for annotated_frame, detections in pipe.run("rtsp://..."):
        cv2.imshow("live", annotated_frame)
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

from .reid import BirdRegistry
from .database import Database


# Skeleton edges as (kp_index_a, kp_index_b)
_SKELETON = [
    (0, 1),   # beak       → crown
    (0, 2),   # beak       → back_neck
    (2, 3),   # back_neck  → middle_back
    (3, 4),   # middle_back → tail_base
    (4, 5),   # tail_base  → tail_tip
    (4, 6),   # tail_base  → left_hock
    (4, 8),   # tail_base  → right_hock
    (3, 6),   # middle_back → left_hock
    (3, 8),   # middle_back → right_hock
    (6, 7),   # left_hock  → left_foot
    (8, 9),   # right_hock → right_foot
]

# BGR colors per keypoint index
_KP_COLORS = [
    (60,  57,  230),   # 0 beak        crimson
    (65, 162, 244),    # 1 crown       warm orange
    (74, 196, 233),    # 2 back_neck   golden ★
    (136, 183, 82),    # 3 middle_back sage green
    (239, 149, 72),    # 4 tail_base   blue
    (212, 232, 173),   # 5 tail_tip    light cyan
    (28,  159, 255),   # 6 left_hock   amber
    (105, 191, 255),   # 7 left_foot   light amber
    (229,  93, 155),   # 8 right_hock  violet
    (255, 125, 199),   # 9 right_foot  light violet
]

# Minimum keypoint confidence to draw / use
_KP_CONF_THRESHOLD = 0.3


class Pipeline:
    """Single-camera YOLO pose pipeline with ReID and live DB writes."""

    def __init__(self, config: dict):
        self.conf   = config.get("confidence", 0.45)
        self.iou    = config.get("iou", 0.45)
        self.imgsz  = config.get("imgsz", 640)
        self.device = config.get("device", "cpu")

        self.model    = YOLO(config["model_path"])
        self.registry = BirdRegistry(config.get("reid", {}))
        self.db       = Database(config.get("db_path", "poultry.db"))

    def run(self, source):
        """
        Generator — yields (annotated_frame, detections) per frame.
        `source`: int (camera index), file path, or RTSP URL string.

        detections: list of dicts with keys bird_id, box, conf, kps
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                results = self.model.predict(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )

                detections = self._process(frame, results[0])
                annotated  = self._draw(frame.copy(), detections)

                yield annotated, detections
        finally:
            cap.release()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray, result) -> list[dict]:
        """Extract detections, run ReID, write to DB. Returns detection list."""
        detections = []

        if result.boxes is None or result.keypoints is None:
            return detections

        boxes   = result.boxes.xyxy.cpu().numpy()   # (N, 4)
        confs   = result.boxes.conf.cpu().numpy()   # (N,)
        kps_all = result.keypoints.data.cpu().numpy()  # (N, 10, 3) — x, y, conf

        ts = time.time()

        for box, conf, kps in zip(boxes, confs, kps_all):
            # Primary false-positive filter:
            # back_neck keypoint (index 2) must be visible.
            # Background wood produces no valid neck keypoint.
            if kps[2, 2] < _KP_CONF_THRESHOLD:
                continue

            bird_id = self.registry.identify(frame, kps)
            if bird_id == 0:
                continue  # filtered by ReID (e.g. unsaturated patch)

            self.db.write_detection(ts, bird_id, box, float(conf), kps)

            detections.append({
                "bird_id": bird_id,
                "box":     box,
                "conf":    float(conf),
                "kps":     kps,
            })

        return detections

    def _draw(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Draw bounding boxes, skeleton, and keypoints onto frame in-place."""
        for d in detections:
            box     = d["box"].astype(int)
            bird_id = d["bird_id"]
            kps     = d["kps"]
            color   = _bird_color(bird_id)

            # Bounding box + label
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                frame,
                f"Bird {bird_id}  {d['conf']:.2f}",
                (box[0], box[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )

            # Skeleton edges
            for idx_a, idx_b in _SKELETON:
                xa, ya, ca = kps[idx_a]
                xb, yb, cb = kps[idx_b]
                if ca < _KP_CONF_THRESHOLD or cb < _KP_CONF_THRESHOLD:
                    continue
                cv2.line(
                    frame,
                    (int(xa), int(ya)), (int(xb), int(yb)),
                    (200, 200, 200), 1, cv2.LINE_AA,
                )

            # Keypoints
            for idx, (x, y, c) in enumerate(kps):
                if c < _KP_CONF_THRESHOLD:
                    continue
                # ReID anchor drawn larger with a white ring
                radius = 7 if idx == 2 else 4
                kp_color = _KP_COLORS[idx]
                if idx == 2:
                    cv2.circle(frame, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
                cv2.circle(frame, (int(x), int(y)), radius, kp_color, -1)

        return frame


def _bird_color(bird_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per bird ID."""
    rng = np.random.default_rng(bird_id * 37 + 13)
    return tuple(int(c) for c in rng.integers(80, 220, 3))
