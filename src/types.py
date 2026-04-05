"""
Shared dataclasses and enums used across all pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class BehaviorState(str, Enum):
    """Observable behavior state of a tracked bird."""
    IDLE = "idle"
    FEEDING = "feeding"
    DRINKING = "drinking"


@dataclass
class Detection:
    """
    Raw detection from a single frame (before ReID assignment).

    box     : (x1, y1, x2, y2) in pixel coordinates.
    kps     : (N_kp, 3) float array — x, y, confidence per keypoint.
    conf    : detection confidence.
    class_id: YOLO class index.
    """
    box: np.ndarray
    conf: float
    class_id: int = 0
    kps: Optional[np.ndarray] = None

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.box[0] + self.box[2]) / 2),
            float((self.box[1] + self.box[3]) / 2),
        )


@dataclass
class TrackedBird:
    """
    Detection enriched with a persistent ReID-assigned bird ID.

    bird_id : 1-indexed stable identifier across frames.
    behavior: current inferred behavior state.
    """
    bird_id: int
    box: np.ndarray
    conf: float
    kps: Optional[np.ndarray] = None
    behavior: BehaviorState = BehaviorState.IDLE

    @property
    def center(self) -> tuple[float, float]:
        return (
            float((self.box[0] + self.box[2]) / 2),
            float((self.box[1] + self.box[3]) / 2),
        )


@dataclass
class FrameResult:
    """
    Output produced by the pipeline for a single processed frame.

    annotated : BGR frame with overlays drawn.
    birds     : tracked birds visible in this frame.
    """
    annotated: np.ndarray
    birds: list[TrackedBird] = field(default_factory=list)
