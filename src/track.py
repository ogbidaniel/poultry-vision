"""
Bird re-identification via painted back_neck spot (keypoint index 2).

Strategy
--------
Each hen has paint applied to the back of its neck.  On each detection:

1. Crop a small patch around the back_neck keypoint in HSV colour space.
2. Compute a normalised hue-saturation histogram as the colour signature.
3. Cosine-match against all registered birds.
4. If best match ≥ ``match_threshold``, update that bird's signature with
   an exponential moving average (handles lighting drift).
5. Otherwise register as a new bird.

False-positive filter
---------------------
Background wood has low HSV saturation.  Detections whose back_neck patch has
mean S < ``min_saturation`` are returned as ``bird_id=0`` and dropped by the
pipeline before the DB write.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import Detection, TrackedBird

_BACK_NECK_IDX = 2


class BirdRegistry:
    """
    Manages per-bird colour signatures for persistent re-identification.

    Parameters (all tunable via *config* dict)
    ------------------------------------------
    match_threshold : float  Cosine similarity for positive match.  Default 0.82.
    min_saturation  : float  HSV-S mean below which patch = background.  Default 40.
    patch_radius    : int    Pixel crop radius around keypoint.  Default 12.
    smooth_alpha    : float  EMA weight for signature updates.  Default 0.15.
    """

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        self.match_threshold = cfg.get("match_threshold", 0.82)
        self.min_saturation  = cfg.get("min_saturation", 40)
        self.patch_radius    = cfg.get("patch_radius", 12)
        self.smooth_alpha    = cfg.get("smooth_alpha", 0.15)

        self._signatures: dict[int, np.ndarray] = {}
        self._next_id = 1

    # ── Public ────────────────────────────────────────────────────────────────

    def assign(self, frame: np.ndarray, detection: Detection) -> int:
        """
        Assign a persistent bird ID to *detection*.

        Returns a 1-indexed bird ID, or **0** if the detection should be
        discarded (unsaturated patch = background false-positive).
        """
        if detection.kps is None:
            return 0

        sig = self._extract_signature(frame, detection.kps)
        if sig is None:
            return 0

        best_id, best_score = self._best_match(sig)

        if best_id is not None and best_score >= self.match_threshold:
            self._update_signature(best_id, sig)
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self._signatures[new_id] = sig
        return new_id

    def identify(self, frame: np.ndarray, kps: np.ndarray) -> int:
        """
        Legacy interface used by callers that pass raw keypoints directly.
        Returns bird_id or 0.
        """
        sig = self._extract_signature(frame, kps)
        if sig is None:
            return 0

        best_id, best_score = self._best_match(sig)
        if best_id is not None and best_score >= self.match_threshold:
            self._update_signature(best_id, sig)
            return best_id

        new_id = self._next_id
        self._next_id += 1
        self._signatures[new_id] = sig
        return new_id

    @property
    def bird_count(self) -> int:
        return len(self._signatures)

    def registered_ids(self) -> list[int]:
        return list(self._signatures.keys())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_signature(
        self,
        frame: np.ndarray,
        kps: np.ndarray,
    ) -> Optional[np.ndarray]:
        x, y, conf = kps[_BACK_NECK_IDX]
        if conf < 0.3:
            return None

        r = self.patch_radius
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x) - r), max(0, int(y) - r)
        x2, y2 = min(w, int(x) + r), min(h, int(y) + r)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        if float(hsv[:, :, 1].mean()) < self.min_saturation:
            return None

        hist = cv2.calcHist(
            [hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]
        ).flatten().astype(np.float32)

        norm = np.linalg.norm(hist)
        if norm == 0:
            return None
        return hist / norm

    def _best_match(self, sig: np.ndarray) -> tuple[Optional[int], float]:
        best_id, best_score = None, -1.0
        for bird_id, stored in self._signatures.items():
            score = float(np.dot(sig, stored))
            if score > best_score:
                best_score, best_id = score, bird_id
        return best_id, best_score

    def _update_signature(self, bird_id: int, sig: np.ndarray) -> None:
        a = self.smooth_alpha
        updated = (1.0 - a) * self._signatures[bird_id] + a * sig
        norm = np.linalg.norm(updated)
        self._signatures[bird_id] = updated / norm if norm > 0 else updated
