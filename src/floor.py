"""
Floor model — turns camera pixels into metric pen-floor coordinates and gates
detections to the physical pen.

Built from the homographies produced by ``calibrate_corners.py``:

    H_top_floor   cam0 (top)  pixel -> metric floor (x, y)
    H_side_floor  cam1 (side) pixel -> metric floor (x, y)

Metric frame: origin at the near-left corner, X across the width (0 -> width_m),
Y along the length (0 -> length_m).

Because the calibration covers the *whole* floor (near corners from the top cam,
far corners from the side cam, stitched through the overlap band), every
detection can be placed in one shared metric frame and tested against the pen
rectangle.  A box that projects outside the rectangle is a false positive
(netting, wall, shadow, reflection) and is dropped — this is the main accuracy
win the calibration buys us.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import transform_point


@dataclass
class FloorModel:
    H_top_floor: np.ndarray
    H_side_floor: np.ndarray
    width_m: float
    length_m: float
    margin_m: float = 0.15   # tolerance outside the rectangle before rejecting

    @classmethod
    def from_config(cls, cfg: dict, margin_m: float = 0.15) -> "FloorModel":
        H = cfg["homographies"]
        return cls(
            H_top_floor=np.asarray(H["H_top_floor"], dtype=np.float64),
            H_side_floor=np.asarray(H["H_side_floor"], dtype=np.float64),
            width_m=float(cfg["pen"]["width_m"]),
            length_m=float(cfg["pen"]["length_m"]),
            margin_m=margin_m,
        )

    def _H(self, cam: str) -> np.ndarray:
        if cam == "top":
            return self.H_top_floor
        if cam == "side":
            return self.H_side_floor
        raise ValueError(f"unknown camera {cam!r} (expected 'top' or 'side')")

    def ground_point(self, cam: str, box: np.ndarray) -> tuple[float, float]:
        """
        Best floor-contact pixel for a box in a given view.

        Top camera looks down, so the body sits over its bbox centre.  The side
        camera sees the bird upright, so the feet are at the bottom-centre.
        """
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx = (x1 + x2) / 2.0
        if cam == "top":
            return (cx, (y1 + y2) / 2.0)
        return (cx, y2)   # side: bottom-centre (feet)

    def to_floor(self, cam: str, box: np.ndarray) -> tuple[float, float]:
        """Project a detection box to its metric floor (x, y)."""
        return transform_point(self.ground_point(cam, box), self._H(cam))

    def floor_to_pixel(self, cam: str, xy: tuple[float, float]) -> tuple[float, float]:
        """Inverse of :meth:`to_floor` — metric floor (x, y) -> camera pixel."""
        return transform_point(xy, np.linalg.inv(self._H(cam)))

    def H(self, cam: str) -> np.ndarray:
        """Public accessor for a camera's pixel->metric homography."""
        return self._H(cam)

    def in_bounds(self, xy: tuple[float, float]) -> bool:
        """True if a metric point lies inside the pen rectangle (+ margin)."""
        x, y = xy
        m = self.margin_m
        return (-m <= x <= self.width_m + m) and (-m <= y <= self.length_m + m)

    def clamp(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Clamp a metric point into the pen rectangle (for display)."""
        x, y = xy
        return (
            float(min(max(x, 0.0), self.width_m)),
            float(min(max(y, 0.0), self.length_m)),
        )
