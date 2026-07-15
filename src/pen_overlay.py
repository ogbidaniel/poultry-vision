"""
Draw the calibrated pen geometry onto camera frames.

Two layers:

* **Floor boundary** — accurate.  The four metric floor corners are projected
  back into the image with the inverse of the camera's floor homography.  Lines
  to corners a camera cannot see simply run off-frame (OpenCV clips them).

* **Raised walls (Z)** — an *estimate*.  A floor homography only describes the
  z=0 plane, so to lift the corners to the wall height we need a full camera
  projection.  We approximate the C920 intrinsics from its field of view and
  decompose ``H_floor->image = K[r1 r2 t]`` to recover the out-of-plane axis
  ``r3 = r1 x r2``.  This is good enough for a visual wireframe; it is not a
  metric reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

from .floor import FloorModel

# Logitech C920 horizontal FOV at 4:3 (degrees) — used to estimate focal length.
_C920_HFOV_DEG = 70.4


def estimate_intrinsics(w: int, h: int, hfov_deg: float = _C920_HFOV_DEG) -> np.ndarray:
    f = w / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    return np.array([[f, 0, w / 2.0],
                     [0, f, h / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def load_intrinsics(cfg: dict, cam: str) -> Optional[np.ndarray]:
    """Return the calibrated 3x3 K for *cam* from config, or None if absent."""
    intr = (cfg.get("camera_intrinsics") or {}).get(cam)
    if not intr or "K" not in intr:
        return None
    return np.asarray(intr["K"], dtype=np.float64)


@dataclass
class PenProjector:
    """Projects metric pen points (x, y, z) into one camera's pixels."""

    project: Callable[[float, float, float], Optional[tuple[float, float]]]

    @classmethod
    def build(
        cls,
        floor: FloorModel,
        cam: str,
        img_w: int,
        img_h: int,
        hfov_deg: float = _C920_HFOV_DEG,
        K: Optional[np.ndarray] = None,
    ) -> "PenProjector":
        # Prefer a real calibrated K (from a checkerboard); fall back to the
        # FOV estimate. With a real K the wall/Z wireframe becomes accurate.
        if K is None:
            K = estimate_intrinsics(img_w, img_h, hfov_deg)
        K = np.asarray(K, dtype=np.float64)
        H_f2i = np.linalg.inv(floor.H(cam))          # floor (x,y,1) -> image
        Kinv = np.linalg.inv(K)
        B = Kinv @ H_f2i
        b1, b2, b3 = B[:, 0], B[:, 1], B[:, 2]
        s = 1.0 / np.linalg.norm(b1)
        r1, r2, t = b1 * s, b2 * s, b3 * s
        r3 = np.cross(r1, r2)

        def _project(x: float, y: float, z: float) -> Optional[tuple[float, float]]:
            cam_pt = K @ (r1 * x + r2 * y + r3 * z + t)
            if cam_pt[2] <= 1e-6:                     # behind the camera
                return None
            return (float(cam_pt[0] / cam_pt[2]), float(cam_pt[1] / cam_pt[2]))

        # Fix the Z sign so that "up" (z>0) renders ABOVE the floor in the image.
        base = _project(floor.width_m / 2, floor.length_m / 2, 0.0)
        top = _project(floor.width_m / 2, floor.length_m / 2, 1.0)
        if base is not None and top is not None and top[1] > base[1]:
            r3 = -r3

            def _project_flipped(x, y, z):
                cam_pt = K @ (r1 * x + r2 * y + r3 * z + t)
                if cam_pt[2] <= 1e-6:
                    return None
                return (float(cam_pt[0] / cam_pt[2]), float(cam_pt[1] / cam_pt[2]))

            return cls(project=_project_flipped)
        return cls(project=_project)


def _corners(floor: FloorModel) -> list[tuple[float, float]]:
    """Floor corners in draw order (origin first)."""
    return [(0.0, 0.0), (floor.width_m, 0.0),
            (floor.width_m, floor.length_m), (0.0, floor.length_m)]


def _ipt(p: tuple[float, float]) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def draw_floor_boundary(
    frame: np.ndarray,
    floor: FloorModel,
    cam: str,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the accurate z=0 floor rectangle (from the homography)."""
    pts = [floor.floor_to_pixel(cam, c) for c in _corners(floor)]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        cv2.line(frame, _ipt(a), _ipt(b), color, thickness, cv2.LINE_AA)
    # Mark the metric origin so the frame is unambiguous.
    o = floor.floor_to_pixel(cam, (0.0, 0.0))
    cv2.circle(frame, _ipt(o), 5, (0, 0, 255), -1)
    cv2.putText(frame, "(0,0)", (_ipt(o)[0] + 6, _ipt(o)[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    return frame


def draw_walls(
    frame: np.ndarray,
    floor: FloorModel,
    proj: PenProjector,
    wall_h: float,
    color: tuple[int, int, int] = (255, 180, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw an ESTIMATED wireframe of the pen walls raised to ``wall_h``."""
    h, w = frame.shape[:2]

    def _visible(p: Optional[tuple[float, float]]) -> bool:
        return p is not None and -w <= p[0] <= 2 * w and -h <= p[1] <= 2 * h

    base = [proj.project(x, y, 0.0) for x, y in _corners(floor)]
    top = [proj.project(x, y, wall_h) for x, y in _corners(floor)]

    # Vertical posts at each corner.
    for b, t in zip(base, top):
        if _visible(b) and _visible(t):
            cv2.line(frame, _ipt(b), _ipt(t), color, thickness, cv2.LINE_AA)
    # Top rectangle.
    for i in range(4):
        a, c = top[i], top[(i + 1) % 4]
        if _visible(a) and _visible(c):
            cv2.line(frame, _ipt(a), _ipt(c), color, 1, cv2.LINE_AA)
    return frame
