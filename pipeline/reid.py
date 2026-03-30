"""
Bird re-identification via painted back_neck spot (keypoint index 2).

Strategy
--------
Each hen has paint applied to the back of its neck. That location corresponds
to keypoint index 2 (back_neck) in the pose schema. On each detection we:

  1. Crop a small patch around the back_neck keypoint in HSV colour space.
  2. Compute a normalised hue-saturation histogram as the bird's colour signature.
  3. Cosine-match the signature against all registered birds.
  4. If the best match exceeds `match_threshold`, update that bird's signature
     with an exponential moving average (handles lighting drift).
  5. Otherwise register as a new bird.

False-positive filter
---------------------
Background wood has low HSV saturation. Any detection whose back_neck patch
has mean S < `min_saturation` is returned as bird_id=0 and dropped by the
pipeline before writing to the DB.
"""

import cv2
import numpy as np
from typing import Optional


_BACK_NECK_IDX = 2


class BirdRegistry:
    """Manages per-bird colour signatures for re-identification."""

    def __init__(self, config: dict = {}):
        # Cosine similarity threshold — 0.82 is tight; lower if same pen birds
        # are being over-split.
        self.match_threshold = config.get("match_threshold", 0.82)
        # Mean HSV-S value below which a patch is treated as background.
        self.min_saturation  = config.get("min_saturation", 40)
        # Radius (pixels) of the crop around the keypoint.
        self.patch_radius    = config.get("patch_radius", 12)
        # EMA weight for signature updates (higher = adapts faster to lighting).
        self.smooth_alpha    = config.get("smooth_alpha", 0.15)

        self._signatures: dict[int, np.ndarray] = {}
        self._next_id = 1

    # ── Public ───────────────────────────────────────────────────────────────

    def identify(self, frame: np.ndarray, kps: np.ndarray) -> int:
        """
        Match keypoints against registered birds.

        Returns bird_id (1-indexed) or 0 if the detection should be discarded
        (low-confidence keypoint or unsaturated patch — background filter).
        """
        sig = self._extract_signature(frame, kps)
        if sig is None:
            return 0

        best_id, best_score = self._best_match(sig)

        if best_id is not None and best_score >= self.match_threshold:
            self._update_signature(best_id, sig)
            return best_id

        # New bird
        new_id = self._next_id
        self._next_id += 1
        self._signatures[new_id] = sig
        return new_id

    @property
    def bird_count(self) -> int:
        return len(self._signatures)

    def registered_ids(self) -> list[int]:
        return list(self._signatures.keys())

    # ── Internal ─────────────────────────────────────────────────────────────

    def _extract_signature(
        self, frame: np.ndarray, kps: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Crop the back_neck patch and return a normalised HSV histogram,
        or None if the keypoint is not confident / patch is unsaturated.
        """
        x, y, conf = kps[_BACK_NECK_IDX]
        if conf < 0.3:
            return None

        r  = self.patch_radius
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x) - r), max(0, int(y) - r)
        x2, y2 = min(w, int(x) + r), min(h, int(y) + r)

        patch = frame[y1:y2, x1:x2]
        if patch.size == 0:
            return None

        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

        # Background filter — wood/litter has low saturation
        if float(hsv[:, :, 1].mean()) < self.min_saturation:
            return None

        # 2-D histogram: Hue (18 bins, 0–180) × Saturation (8 bins, 0–256)
        hist = cv2.calcHist(
            [hsv], [0, 1], None,
            [18, 8],
            [0, 180, 0, 256],
        ).flatten().astype(np.float32)

        norm = np.linalg.norm(hist)
        if norm == 0:
            return None
        return hist / norm

    def _best_match(self, sig: np.ndarray) -> tuple[Optional[int], float]:
        """Return (bird_id, cosine_score) of the best stored match."""
        best_id    = None
        best_score = -1.0
        for bird_id, stored in self._signatures.items():
            score = float(np.dot(sig, stored))
            if score > best_score:
                best_score = score
                best_id    = bird_id
        return best_id, best_score

    def _update_signature(self, bird_id: int, sig: np.ndarray) -> None:
        """EMA update then re-normalise."""
        a = self.smooth_alpha
        updated = (1.0 - a) * self._signatures[bird_id] + a * sig
        norm = np.linalg.norm(updated)
        self._signatures[bird_id] = updated / norm if norm > 0 else updated
