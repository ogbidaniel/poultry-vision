"""
Lightweight behavior classification based on bounding-box overlap.

A hen is FEEDING when its bounding box overlaps a feeder box.
A hen is DRINKING when its bounding box overlaps a waterer box.
FEEDING takes priority over DRINKING when both overlap simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from .geometry import check_overlap


class BehaviorType(str, Enum):
    IDLE = "idle"
    FEEDING = "feeding"
    DRINKING = "drinking"


@dataclass
class HenStats:
    """Accumulated behavioral statistics for a single hen."""

    hen_id: int
    feeding_time_s: float = 0.0
    drinking_time_s: float = 0.0
    feeding_events: int = 0
    drinking_events: int = 0
    current_behavior: BehaviorType = BehaviorType.IDLE

    def to_dict(self) -> dict:
        return {
            "hen_id": self.hen_id,
            "feeding_time_s": round(self.feeding_time_s, 2),
            "drinking_time_s": round(self.drinking_time_s, 2),
            "feeding_events": self.feeding_events,
            "drinking_events": self.drinking_events,
            "current_behavior": self.current_behavior.value,
        }


# Callback signature: (hen_id, old_behavior, new_behavior, duration_s)
BehaviorCallback = Callable[[int, BehaviorType, BehaviorType, float], None]


class HenBehaviorMonitor:
    """
    Frame-by-frame behavior monitor for multiple hens.

    Usage::

        monitor = HenBehaviorMonitor(fps=30.0)
        # each frame:
        monitor.update(hens, feeders, waterers)
        stats = monitor.get_stats(hen_id)
    """

    def __init__(self, fps: float = 30.0) -> None:
        self.fps = fps
        self._frame_count: int = 0
        self._stats: dict[int, HenStats] = {}
        self._state_start_frame: dict[int, int] = {}
        self._callbacks: list[BehaviorCallback] = []

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def tracked_hen_ids(self) -> set[int]:
        return set(self._stats.keys())

    def register_callback(self, cb: BehaviorCallback) -> None:
        """Register a function called whenever a hen's behavior changes."""
        self._callbacks.append(cb)

    def update(
        self,
        hens: list[dict],
        feeders: list[dict],
        waterers: list[dict],
    ) -> None:
        """
        Process one frame.

        hens / feeders / waterers are lists of dicts with at minimum::
            {"id": int, "box": (x1, y1, x2, y2)}  # hens must have "id"
            {"box": (x1, y1, x2, y2)}              # resources
        """
        self._frame_count += 1
        dt = 1.0 / self.fps

        for hen in hens:
            hen_id = hen["id"]
            hen_box = hen["box"]

            if hen_id not in self._stats:
                self._stats[hen_id] = HenStats(hen_id=hen_id)
                self._state_start_frame[hen_id] = self._frame_count

            stats = self._stats[hen_id]
            new_behavior = self._classify(hen_box, feeders, waterers)

            # Accumulate time in current behavior
            if stats.current_behavior == BehaviorType.FEEDING:
                stats.feeding_time_s += dt
            elif stats.current_behavior == BehaviorType.DRINKING:
                stats.drinking_time_s += dt

            # Detect transitions
            if new_behavior != stats.current_behavior:
                duration_s = (
                    self._frame_count - self._state_start_frame[hen_id]
                ) / self.fps

                # Count entry events
                if new_behavior == BehaviorType.FEEDING:
                    stats.feeding_events += 1
                elif new_behavior == BehaviorType.DRINKING:
                    stats.drinking_events += 1

                for cb in self._callbacks:
                    cb(hen_id, stats.current_behavior, new_behavior, duration_s)

                stats.current_behavior = new_behavior
                self._state_start_frame[hen_id] = self._frame_count

    def get_stats(self, hen_id: int) -> dict:
        """Return stats dict for *hen_id*, or empty dict if unknown."""
        if hen_id not in self._stats:
            return {}
        return self._stats[hen_id].to_dict()

    def get_current_behaviors(self) -> dict[int, str]:
        """Return {hen_id: behavior_value} for all tracked hens."""
        return {
            hid: s.current_behavior.value
            for hid, s in self._stats.items()
        }

    def get_summary(self) -> dict:
        """Return aggregate summary across all hens."""
        total_feeding = sum(s.feeding_time_s for s in self._stats.values())
        total_drinking = sum(s.drinking_time_s for s in self._stats.values())
        return {
            "total_hens": len(self._stats),
            "frames_processed": self._frame_count,
            "total_feeding_time_s": round(total_feeding, 2),
            "total_drinking_time_s": round(total_drinking, 2),
        }

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._stats.clear()
        self._state_start_frame.clear()
        self._frame_count = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _classify(
        hen_box: tuple,
        feeders: list[dict],
        waterers: list[dict],
    ) -> BehaviorType:
        """Determine behavior based on overlap — feeding takes priority."""
        for feeder in feeders:
            overlaps, _ = check_overlap(hen_box, feeder["box"])
            if overlaps:
                return BehaviorType.FEEDING

        for waterer in waterers:
            overlaps, _ = check_overlap(hen_box, waterer["box"])
            if overlaps:
                return BehaviorType.DRINKING

        return BehaviorType.IDLE
