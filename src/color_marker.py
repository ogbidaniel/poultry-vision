"""
HSV-based neck-marker color classifier.

The five hens wear a painted marker at the back of the neck.  This module
first tries to sample an HSV patch around the neck_back keypoint (index 6)
if one is available with sufficient confidence.  If not (e.g. when called
with a segmentation-only result that has no keypoints), it falls back to
scanning the entire bounding box for high-saturation paint pixels.

Color label vocabulary: green, black, red, blue, yellow, uncertain.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

NECK_BACK_IDX = 6

_COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red":    [(np.array([0,   80,  60]), np.array([10,  255, 255])),
               (np.array([170, 80,  60]), np.array([180, 255, 255]))],
    "green":  [(np.array([35,  60,  60]), np.array([85,  255, 255]))],
    "blue":   [(np.array([100, 60,  60]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20,  80,  80]), np.array([35,  255, 255]))],
    "black":  [(np.array([0,   0,   0]),  np.array([180, 255,  50]))],
}

_MIN_SATURATION      = 30.0   # mean-saturation threshold for kp-patch method
_MIN_MATCH_RATIO     = 0.15   # kp-patch: fraction of pixels that must match
_UNCERTAIN_THRESHOLD = 0.10   # kp-patch: minimum fraction to accept a color
_HIGH_SAT_THRESHOLD  = 80     # bbox method: saturation cutoff to isolate paint pixels
_BOX_MIN_SAT_PIXELS  = 20     # bbox method: min high-sat pixels to consider reliable
_BOX_BLACK_RATIO     = 0.04   # bbox method: fraction of very-dark pixels → black paint
_BOX_MIN_COLOR_PIXELS = 15    # bbox method: min pixels matching a color range


def classify_neck_color(
    frame: np.ndarray,
    kps: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
    patch_radius: int = 12,
    min_kp_conf: float = 0.3,
) -> str:
    """
    Return the marker color label for a detection, or 'uncertain'.

    Parameters
    ----------
    frame        : BGR image (top camera frame)
    kps          : (10, 3) keypoint array [x, y, conf], or None
    box          : (4,) xyxy bounding box used as fallback when kps unavailable
    patch_radius : pixel crop radius around neck_back keypoint
    min_kp_conf  : minimum keypoint confidence to attempt kp-based classification
    """
    label = _classify(frame, kps, box, patch_radius, min_kp_conf)
    return label if label is not None else "uncertain"


def _classify(
    frame: np.ndarray,
    kps: Optional[np.ndarray],
    box: Optional[np.ndarray],
    patch_radius: int,
    min_kp_conf: float,
) -> Optional[str]:
    # ── Try keypoint-based patch first ────────────────────────────────────────
    if kps is not None and len(kps) > NECK_BACK_IDX:
        x, y, conf = kps[NECK_BACK_IDX]
        if conf >= min_kp_conf:
            result = _classify_kp_patch(frame, int(x), int(y), patch_radius)
            if result is not None:
                return result

    # ── Fallback: scan whole bbox for high-saturation paint pixels ────────────
    if box is not None:
        return _classify_from_box(frame, box)

    return None


def _classify_kp_patch(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
) -> Optional[str]:
    h, w = frame.shape[:2]
    x1, y1 = max(0, cx - radius), max(0, cy - radius)
    x2, y2 = min(w, cx + radius), min(h, cy + radius)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    if float(hsv[:, :, 1].mean()) < _MIN_SATURATION:
        return _check_black(hsv, _MIN_MATCH_RATIO)

    scores: dict[str, float] = {}
    total_pixels = hsv.shape[0] * hsv.shape[1]
    for color, ranges in _COLOR_RANGES.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        scores[color] = float(mask.sum() / 255) / total_pixels

    best_color = max(scores, key=lambda c: scores[c])
    best_score = scores[best_color]

    if best_score < _UNCERTAIN_THRESHOLD:
        return None

    if best_color in ("black", "blue"):
        if abs(scores.get("black", 0) - scores.get("blue", 0)) < _UNCERTAIN_THRESHOLD:
            return None

    return best_color


def _classify_from_box(frame: np.ndarray, box: np.ndarray) -> Optional[str]:
    """
    Identify paint-marker color from the entire bounding box.

    Strategy: paint markers are highly saturated compared to feathers.
    Isolate high-saturation pixels and find the dominant hue among them.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    total_pixels = patch.shape[0] * patch.shape[1]

    # High-saturation pixels = paint (feathers are low-saturation browns/whites)
    high_sat = (hsv[:, :, 1] > _HIGH_SAT_THRESHOLD).astype(np.uint8)
    n_high_sat = int(high_sat.sum())

    # Check for black paint (dark pixels, regardless of saturation)
    very_dark = (hsv[:, :, 2] < 50).astype(np.uint8)
    black_ratio = float(very_dark.sum()) / total_pixels

    if n_high_sat < _BOX_MIN_SAT_PIXELS:
        # No strongly-coloured pixels — only option is black
        return "black" if black_ratio >= _BOX_BLACK_RATIO else None

    # Score each non-black color against the high-saturation mask
    scores: dict[str, int] = {}
    for color, ranges in _COLOR_RANGES.items():
        if color == "black":
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        # Only count pixels that are BOTH in the color range AND high-saturation
        scores[color] = int((mask & (high_sat * 255)).sum() / 255)

    best_color = max(scores, key=lambda c: scores[c])
    best_count = scores[best_color]

    if best_count < _BOX_MIN_COLOR_PIXELS:
        return "black" if black_ratio >= _BOX_BLACK_RATIO else None

    # Ambiguity guard for visually similar pairs
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    if best_color in ("blue", "black") and (best_count - second_best) < _BOX_MIN_COLOR_PIXELS:
        return None

    return best_color


def _check_black(hsv: np.ndarray, min_ratio: float) -> Optional[str]:
    lo, hi = _COLOR_RANGES["black"][0]
    mask = cv2.inRange(hsv, lo, hi)
    ratio = float(mask.sum() / 255) / (hsv.shape[0] * hsv.shape[1])
    return "black" if ratio > min_ratio else None
