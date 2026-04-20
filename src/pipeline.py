"""
Core inference pipeline.

YOLO pose model → per-frame detections → ReID (BirdRegistry) → behavior
classification → DB write → annotated frame.

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

Usage::

    from src.pipeline import Pipeline
    pipe = Pipeline(config)
    for result in pipe.run("rtsp://..."):
        cv2.imshow("live", result.annotated)
"""

from __future__ import annotations

import time
from typing import Generator

import numpy as np

from .pose import PoseDetector
from .track import BirdRegistry
from .behavior import HenBehaviorMonitor
from .render import draw_birds
from .io_utils import Database
from .capture import FrameSource
from .types import TrackedBird, BehaviorState, FrameResult


class Pipeline:
    """
    Single-camera YOLO pose pipeline with ReID, behavior monitoring, and
    live SQLite writes.

    Parameters
    ----------
    config : dict
        Loaded from ``config/system.yaml`` (or ``config/pipeline.yaml``).
        Supported keys:

        model_path    (str)   — required
        confidence    (float) — default 0.45
        iou           (float) — default 0.45
        imgsz         (int)   — default 640
        device        (str)   — default "cpu"
        db_path       (str)   — default "poultry.db"
        fps           (float) — default 30.0
        reid          (dict)  — BirdRegistry kwargs
    """

    def __init__(self, config: dict) -> None:
        self.detector = PoseDetector(
            model_path=config["model_path"],
            conf=config.get("confidence", 0.45),
            iou=config.get("iou", 0.45),
            imgsz=config.get("imgsz", 640),
            device=config.get("device", "cpu"),
        )
        self.registry = BirdRegistry(config.get("reid"))
        self.db       = Database(config.get("db_path", "poultry.db"))
        self.behavior = HenBehaviorMonitor(fps=config.get("fps", 30.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, source: int | str | FrameSource) -> Generator[FrameResult, None, None]:
        """
        Process *source* frame by frame, yielding a :class:`FrameResult` for
        each frame.

        *source* may be a camera index (int), a file path, or an RTSP URL.
        """
        cam = source if isinstance(source, FrameSource) else FrameSource(source)
        with cam:
            for frame in cam:
                result = self._process_frame(frame)
                yield result

    # ── Internal ─────────────────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        ts = time.time()

        # 1. Detect + filter by back_neck confidence
        detections = self.detector.predict(frame)

        # 2. Assign persistent bird IDs via colour ReID
        birds: list[TrackedBird] = []
        hens_for_behavior: list[dict] = []

        for det in detections:
            bird_id = self.registry.assign(frame, det)
            if bird_id == 0:
                continue  # background false-positive

            bird = TrackedBird(
                bird_id=bird_id,
                box=det.box,
                conf=det.conf,
                kps=det.kps,
            )
            birds.append(bird)
            hens_for_behavior.append({
                "id": bird_id,
                "box": tuple(det.box),
            })

            # Persist to DB
            self.db.write_detection(ts, bird_id, det.box, det.conf, det.kps)

        # 3. Update behavior (no feeder/waterer detections in pose-only model,
        #    so all birds remain IDLE unless a multi-class model is wired in)
        self.behavior.update(hens_for_behavior, [], [])
        for bird in birds:
            bstate = self.behavior.get_stats(bird.bird_id).get(
                "current_behavior", BehaviorState.IDLE.value
            )
            bird.behavior = BehaviorState(bstate)

        # 4. Render overlays
        annotated = draw_birds(frame.copy(), birds)

        return FrameResult(annotated=annotated, birds=birds)
