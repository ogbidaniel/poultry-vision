"""
Frame rendering / overlay drawing.

All functions accept a BGR numpy array and draw in-place, returning it.
"""

from __future__ import annotations

import cv2
import numpy as np

from .types import TrackedBird, BehaviorState

# Skeleton edges as (kp_index_a, kp_index_b)
_SKELETON = [
    (0, 1),   # beak        → crown
    (0, 2),   # beak        → back_neck
    (2, 3),   # back_neck   → middle_back
    (3, 4),   # middle_back → tail_base
    (4, 5),   # tail_base   → tail_tip
    (4, 6),   # tail_base   → left_hock
    (4, 8),   # tail_base   → right_hock
    (3, 6),   # middle_back → left_hock
    (3, 8),   # middle_back → right_hock
    (6, 7),   # left_hock   → left_foot
    (8, 9),   # right_hock  → right_foot
]

# BGR colours per keypoint index
_KP_COLORS = [
    (60,  57,  230),    # 0 beak         crimson
    (65, 162, 244),     # 1 crown        warm orange
    (74, 196, 233),     # 2 back_neck    golden ★
    (136, 183,  82),    # 3 middle_back  sage green
    (239, 149,  72),    # 4 tail_base    blue
    (212, 232, 173),    # 5 tail_tip     light cyan
    (28,  159, 255),    # 6 left_hock    amber
    (105, 191, 255),    # 7 left_foot    light amber
    (229,  93, 155),    # 8 right_hock   violet
    (255, 125, 199),    # 9 right_foot   light violet
]

_KP_CONF_THRESHOLD = 0.3

# Label colours per behavior
_BEHAVIOR_COLORS: dict[BehaviorState, tuple[int, int, int]] = {
    BehaviorState.IDLE:     (200, 200, 200),
    BehaviorState.FEEDING:  (60,  200,  60),
    BehaviorState.DRINKING: (60,  150, 240),
}


def bird_color(bird_id: int) -> tuple[int, int, int]:
    """Deterministic BGR colour for a given bird ID."""
    rng = np.random.default_rng(bird_id * 37 + 13)
    return tuple(int(c) for c in rng.integers(80, 220, 3))


def draw_birds(frame: np.ndarray, birds: list[TrackedBird]) -> np.ndarray:
    """
    Draw bounding boxes, skeleton, and keypoints for all tracked birds.

    Modifies *frame* in-place and returns it.
    """
    for bird in birds:
        _draw_bird(frame, bird)
    return frame


def _draw_bird(frame: np.ndarray, bird: TrackedBird) -> None:
    box    = bird.box.astype(int)
    color  = bird_color(bird.bird_id)
    label  = f"Bird {bird.bird_id}  {bird.conf:.2f}  [{bird.behavior.value}]"
    b_color = _BEHAVIOR_COLORS.get(bird.behavior, color)

    # Bounding box
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

    # Label
    cv2.putText(
        frame, label,
        (box[0], box[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, b_color, 2, cv2.LINE_AA,
    )

    if bird.kps is None:
        return

    kps = bird.kps

    # Skeleton edges
    for idx_a, idx_b in _SKELETON:
        if idx_a >= len(kps) or idx_b >= len(kps):
            continue
        xa, ya, ca = kps[idx_a]
        xb, yb, cb = kps[idx_b]
        if ca < _KP_CONF_THRESHOLD or cb < _KP_CONF_THRESHOLD:
            continue
        cv2.line(
            frame,
            (int(xa), int(ya)), (int(xb), int(yb)),
            (200, 200, 200), 1, cv2.LINE_AA,
        )

    # Keypoints
    for idx, (x, y, c) in enumerate(kps):
        if c < _KP_CONF_THRESHOLD:
            continue
        radius = 7 if idx == 2 else 4
        kp_color = _KP_COLORS[idx] if idx < len(_KP_COLORS) else (200, 200, 200)
        if idx == 2:
            cv2.circle(frame, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
        cv2.circle(frame, (int(x), int(y)), radius, kp_color, -1)
