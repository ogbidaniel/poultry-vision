"""
Pose estimation wrapper.

``PoseDetector`` is a thin subclass of :class:`~src.detect.Detector` that
adds keypoint-specific filtering.  It uses the same YOLO model interface but
expects the model to be a pose variant (e.g. ``yolo11-pose.pt``).

The poultry back_neck keypoint at index 2 is the primary false-positive filter:
any detection where that keypoint confidence is below *kp_conf_threshold* is
discarded because background wood produces no valid neck keypoint.
"""

from __future__ import annotations

import numpy as np

from .detect import Detector
from .types import Detection

# Index of the back_neck keypoint in the 10-kp poultry schema
BACK_NECK_IDX = 2


class PoseDetector(Detector):
    """
    YOLO pose detector with back_neck keypoint confidence filtering.

    Parameters
    ----------
    kp_conf_threshold : float
        Minimum confidence required for the back_neck keypoint (index 2).
        Detections where this keypoint is below the threshold are dropped.
    """

    def __init__(self, *args, kp_conf_threshold: float = 0.3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.kp_conf_threshold = kp_conf_threshold

    def predict(self, frame: np.ndarray) -> list[Detection]:
        detections = super().predict(frame)
        return [
            d for d in detections
            if self._has_valid_neck(d)
        ]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _has_valid_neck(self, det: Detection) -> bool:
        if det.kps is None:
            return False
        if det.kps.shape[0] <= BACK_NECK_IDX:
            return False
        return float(det.kps[BACK_NECK_IDX, 2]) >= self.kp_conf_threshold
