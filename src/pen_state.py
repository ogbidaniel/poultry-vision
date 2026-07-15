"""
Pen-state dataclasses for the dual-camera pipeline.

PenState is a plain data container built once per aligned frame pair.
The pipeline script assembles it; nothing here performs inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HenEntry:
    """Per-hen state for a single frame."""

    track_id: int
    color_label: str
    floor_x: float
    floor_y: float
    top_box: Optional[np.ndarray] = None
    side_kps: Optional[np.ndarray] = None
    approx_z: float = 0.0
    views: set = field(default_factory=set)


@dataclass
class PenState:
    """Complete pen state for a single aligned frame pair."""

    timestamp: float
    hens: list[HenEntry] = field(default_factory=list)
    feeder_floor: Optional[tuple[float, float]] = None
    waterer_floor: Optional[tuple[float, float]] = None
