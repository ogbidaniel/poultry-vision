"""
Geometry helpers: bounding-box operations, homography, distance.

All functions are stateless and depend only on numpy / opencv.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ── Bounding box helpers ──────────────────────────────────────────────────────

def get_box_center(box: tuple) -> tuple[float, float]:
    """Return (cx, cy) float center of a (x1, y1, x2, y2) box."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_box_center_int(box: tuple) -> tuple[int, int]:
    """Return (cx, cy) integer center of a (x1, y1, x2, y2) box."""
    cx, cy = get_box_center(box)
    return (int(cx), int(cy))


def get_box_width(box: tuple) -> float:
    return box[2] - box[0]


def get_box_height(box: tuple) -> float:
    return box[3] - box[1]


def get_box_area(box: tuple) -> float:
    return get_box_width(box) * get_box_height(box)


# ── Overlap / IoU ─────────────────────────────────────────────────────────────

def calculate_overlap_area(box1: tuple, box2: tuple) -> float:
    """Return the pixel area of the intersection of two boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def check_overlap(box1: tuple, box2: tuple) -> tuple[bool, float]:
    """Return (overlaps: bool, overlap_area: float)."""
    area = calculate_overlap_area(box1, box2)
    return (area > 0, area)


def calculate_iou(box1: tuple, box2: tuple) -> float:
    """Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    inter = calculate_overlap_area(box1, box2)
    if inter == 0:
        return 0.0
    union = get_box_area(box1) + get_box_area(box2) - inter
    return inter / union if union > 0 else 0.0


# ── Circle / distance ─────────────────────────────────────────────────────────

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    return float(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)


def is_point_in_circle(
    point: tuple,
    center: Optional[tuple],
    radius: float,
) -> bool:
    """Return True when *point* lies within *radius* of *center*."""
    if center is None:
        return False
    return euclidean_distance(point, center) <= radius


# ── Homography / coordinate transform ────────────────────────────────────────

def get_homography_matrix(
    pen_corners: np.ndarray,
    pen_width_cm: float,
    pen_height_cm: float,
) -> np.ndarray:
    """
    Compute a 3×3 homography from 4 pixel pen corners to a
    (pen_width_cm × pen_height_cm) world rectangle.

    pen_corners : (4, 2) float32 array ordered
                  [top-left, top-right, bottom-right, bottom-left].
    """
    dst = np.array([
        [0,             0],
        [pen_width_cm,  0],
        [pen_width_cm,  pen_height_cm],
        [0,             pen_height_cm],
    ], dtype=np.float32)

    src = pen_corners.astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_point(
    point: tuple[float, float],
    matrix: Optional[np.ndarray],
) -> tuple[float, float]:
    """
    Apply a 3×3 homography to a single point.
    Returns the original point unchanged when *matrix* is None.
    """
    if matrix is None:
        return point

    pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, matrix)
    return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))


# ── Unit conversions ──────────────────────────────────────────────────────────

def pixels_to_cm(pixels: float, scale: float) -> float:
    """Convert pixels to centimetres using a linear scale factor."""
    return pixels * scale


def cm_to_pixels(cm: float, scale: float) -> float:
    """Convert centimetres to pixels using a linear scale factor.

    Returns 0 when *scale* is zero to avoid division by zero.
    """
    if scale == 0:
        return 0
    return cm / scale
