"""
Object detection wrapper around the Ultralytics YOLO model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ultralytics import YOLO

from .types import Detection


class Detector:
    """
    Thin wrapper around a YOLO model that returns typed ``Detection`` objects.

    Parameters
    ----------
    model_path : str
        Path to a ``.pt`` model file.
    conf : float
        Minimum detection confidence threshold.
    iou : float
        NMS IoU threshold.
    imgsz : int
        Inference image size.
    device : str
        Inference device, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.45,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a single BGR frame.

        Returns a list of :class:`Detection` objects (no keypoints unless the
        model is a pose model — see :class:`PoseDetector`).
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        return self._parse(results[0])

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(result) -> list[Detection]:
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        kps_all: Optional[np.ndarray] = None
        if result.keypoints is not None:
            kps_all = result.keypoints.data.cpu().numpy()

        detections = []
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
            kps = kps_all[i] if kps_all is not None else None
            detections.append(Detection(
                box=box,
                conf=float(conf),
                class_id=int(cls_id),
                kps=kps,
            ))
        return detections
