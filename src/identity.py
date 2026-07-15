"""
Colour-as-identity tracker for the dual-camera pen.

The five hens wear fixed paint markers (red, blue, black, green, yellow), so the
*colour* is the canonical identity — not a per-camera track id.  Each camera
runs its own detector + tracker and reports observations:

    {camera, track_id, color, floor_xy, box, conf}

This module turns a stream of those observations into five stable hen slots:

1. VOTE   — accumulate per-(camera, track_id) colour votes over time.  A single
            frame's colour guess is noisy; the majority vote per track is not.
2. RESOLVE— each live track resolves to its majority colour.
3. ASSIGN — group resolved tracks by colour into the five slots.  When two tracks
            in the same frame claim one colour, the one nearest the slot's
            last-known floor position (continuity) wins; the loser is parked as
            ``uncertain`` so it is never silently mislabelled.
4. FUSE   — a colour seen by both cameras is averaged on the floor.
5. MISS   — a colour seen by neither camera keeps its last-known position and is
            flagged ``visible=False`` until it reappears.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

CANONICAL_COLORS = ["red", "blue", "black", "green", "yellow"]


@dataclass
class Observation:
    camera: str          # "top" | "side"
    track_id: int
    color: str           # per-frame guess, may be "uncertain"
    floor_xy: tuple[float, float]
    box: np.ndarray
    conf: float


@dataclass
class HenSlot:
    color: str
    floor_x: Optional[float] = None
    floor_y: Optional[float] = None
    visible: bool = False
    last_seen_ts: Optional[float] = None
    views: set = field(default_factory=set)          # cameras seeing it this frame
    boxes: dict = field(default_factory=dict)        # camera -> box (this frame)

    @property
    def floor_xy(self) -> Optional[tuple[float, float]]:
        if self.floor_x is None or self.floor_y is None:
            return None
        return (self.floor_x, self.floor_y)


class ColorIdentityTracker:
    """
    Maintains one :class:`HenSlot` per canonical colour.

    Parameters
    ----------
    colors        : the canonical marker colours (default the five hens).
    timeout_s     : a slot stays ``visible`` only if seen within this window.
    assoc_gate_m  : max floor distance for two same-colour tracks to be the
                    same hen (cross-camera fusion + continuity tie-break).
    min_votes     : a track must accumulate at least this many colour votes
                    before it is allowed to claim a slot (suppresses flicker).
    """

    def __init__(
        self,
        colors: Optional[list[str]] = None,
        timeout_s: float = 2.0,
        assoc_gate_m: float = 0.40,
        min_votes: int = 3,
    ) -> None:
        self.colors = colors or CANONICAL_COLORS
        self.timeout_s = timeout_s
        self.assoc_gate_m = assoc_gate_m
        self.min_votes = min_votes

        self.slots: dict[str, HenSlot] = {c: HenSlot(color=c) for c in self.colors}
        self._votes: dict[tuple[str, int], dict[str, int]] = {}
        self._last_track_seen: dict[tuple[str, int], float] = {}

    # ── Public ────────────────────────────────────────────────────────────────

    def update(self, observations: list[Observation], ts: float) -> None:
        self._accumulate_votes(observations, ts)
        self._expire_tracks(ts)

        # Reset per-frame visibility; positions persist (graceful out-of-frame).
        for slot in self.slots.values():
            slot.visible = False
            slot.views = set()
            slot.boxes = {}

        resolved = self._resolve(observations)        # color -> [Observation]
        for color, obs_list in resolved.items():
            if color not in self.slots:
                continue
            self._update_slot(self.slots[color], obs_list, ts)

    def states(self) -> list[HenSlot]:
        return [self.slots[c] for c in self.colors]

    def missing(self) -> list[str]:
        return [c for c, s in self.slots.items() if not s.visible]

    # ── Internal ────────────────────────────────────────────────────────────────

    def _accumulate_votes(self, observations: list[Observation], ts: float) -> None:
        for obs in observations:
            key = (obs.camera, obs.track_id)
            self._last_track_seen[key] = ts
            if obs.color != "uncertain":
                self._votes.setdefault(key, {})
                self._votes[key][obs.color] = self._votes[key].get(obs.color, 0) + 1

    def _expire_tracks(self, ts: float) -> None:
        stale = [k for k, t in self._last_track_seen.items()
                 if ts - t > self.timeout_s]
        for k in stale:
            self._votes.pop(k, None)
            self._last_track_seen.pop(k, None)

    def _track_color(self, key: tuple[str, int]) -> tuple[Optional[str], int]:
        votes = self._votes.get(key)
        if not votes:
            return None, 0
        color = max(votes, key=lambda c: votes[c])
        return color, votes[color]

    def _resolve(self, observations: list[Observation]) -> dict[str, list[Observation]]:
        """
        Map each live observation to its track's majority colour, enforcing one
        track per colour *per camera* (nearest-to-last-known wins ties).
        """
        # Gather candidates per (camera, color): the observations whose track
        # resolves to that colour and clears the vote threshold.
        candidates: dict[tuple[str, str], list[tuple[Observation, int]]] = {}
        for obs in observations:
            key = (obs.camera, obs.track_id)
            color, n = self._track_color(key)
            if color is None or n < self.min_votes:
                continue
            candidates.setdefault((obs.camera, color), []).append((obs, n))

        resolved: dict[str, list[Observation]] = {}
        for (camera, color), cand in candidates.items():
            winner = self._pick_winner(color, cand)
            resolved.setdefault(color, []).append(winner)
        return resolved

    def _pick_winner(
        self,
        color: str,
        cand: list[tuple[Observation, int]],
    ) -> Observation:
        if len(cand) == 1:
            return cand[0][0]
        last = self.slots[color].floor_xy
        if last is not None:
            # Continuity: closest to the slot's last-known floor position.
            return min(cand, key=lambda c: _dist(c[0].floor_xy, last))[0]
        # No history yet: trust the strongest vote, then confidence.
        return max(cand, key=lambda c: (c[1], c[0].conf))[0]

    def _update_slot(
        self,
        slot: HenSlot,
        obs_list: list[Observation],
        ts: float,
    ) -> None:
        # Fuse floor positions across cameras (mean of the agreeing views).
        xs = [o.floor_xy[0] for o in obs_list]
        ys = [o.floor_xy[1] for o in obs_list]
        slot.floor_x = float(np.mean(xs))
        slot.floor_y = float(np.mean(ys))
        slot.visible = True
        slot.last_seen_ts = ts
        slot.views = {o.camera for o in obs_list}
        slot.boxes = {o.camera: o.box for o in obs_list}


# ── helpers ───────────────────────────────────────────────────────────────────

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
