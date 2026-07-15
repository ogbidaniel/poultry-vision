"""
Long-format CSV logger for the dual-camera pen-state pipeline.

One row per detected instance per frame:
  - top rows: track_id, color_label, floor_x_m, floor_y_m, bbox; kp columns blank
  - side rows: all 30 keypoint columns; floor_x_m/y_m populated only when matched

Unmatched side detections are still written (matched=0) so no data is silently
dropped.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

from .pen_state import PenState

_KP_NAMES = [
    "tail_base", "tail_tip", "left_hock", "right_hock",
    "left_foot", "right_foot", "neck_back", "middle_back",
    "comb", "beak",
]

_KP_COLS: list[str] = []
for _name in _KP_NAMES:
    _KP_COLS += [f"kp_{_name}_x", f"kp_{_name}_y", f"kp_{_name}_v"]

_FIELDNAMES = [
    "timestamp", "frame_id", "view",
    "track_id", "color_label",
    "floor_x_m", "floor_y_m", "approx_z_m",
    "box_x1", "box_y1", "box_x2", "box_y2",
    *_KP_COLS,
    "matched",
]


class CSVLogger:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=_FIELDNAMES)
        self._writer.writeheader()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "CSVLogger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def write_top(
        self,
        pen_state: PenState,
        frame_id: int,
    ) -> None:
        """Write one row per hen visible in the top view."""
        for hen in pen_state.hens:
            if "top" not in hen.views:
                continue
            row: dict = _empty_row()
            row["timestamp"]  = pen_state.timestamp
            row["frame_id"]   = frame_id
            row["view"]       = "top"
            row["track_id"]   = hen.track_id
            row["color_label"] = hen.color_label
            row["floor_x_m"]  = _fmt(hen.floor_x)
            row["floor_y_m"]  = _fmt(hen.floor_y)
            row["approx_z_m"] = _fmt(hen.approx_z)
            if hen.top_box is not None:
                row["box_x1"] = _fmt(hen.top_box[0])
                row["box_y1"] = _fmt(hen.top_box[1])
                row["box_x2"] = _fmt(hen.top_box[2])
                row["box_y2"] = _fmt(hen.top_box[3])
            row["matched"] = 1 if "side" in hen.views else 0
            self._writer.writerow(row)

    def write_side(
        self,
        pen_state: PenState,
        unmatched_side: list[tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        frame_id: int,
    ) -> None:
        """
        Write side-view rows.

        Matched side detections are written as part of the hen entries
        (track_id populated, matched=1). Unmatched side detections are
        written separately with matched=0 and no track_id.

        Parameters
        ----------
        pen_state       : current PenState (matched hens already carry side_kps)
        unmatched_side  : list of (box, kps) for side dets not matched to any top track
        frame_id        : current frame index
        """
        for hen in pen_state.hens:
            if "side" not in hen.views:
                continue
            row = _empty_row()
            row["timestamp"]  = pen_state.timestamp
            row["frame_id"]   = frame_id
            row["view"]       = "side"
            row["track_id"]   = hen.track_id
            row["color_label"] = hen.color_label
            row["floor_x_m"]  = _fmt(hen.floor_x)
            row["floor_y_m"]  = _fmt(hen.floor_y)
            row["approx_z_m"] = _fmt(hen.approx_z)
            if hen.side_kps is not None:
                _fill_kps(row, hen.side_kps)
            row["matched"] = 1
            self._writer.writerow(row)

        for box, kps in unmatched_side:
            row = _empty_row()
            row["timestamp"] = pen_state.timestamp
            row["frame_id"]  = frame_id
            row["view"]      = "side"
            row["matched"]   = 0
            if box is not None:
                row["box_x1"] = _fmt(box[0])
                row["box_y1"] = _fmt(box[1])
                row["box_x2"] = _fmt(box[2])
                row["box_y2"] = _fmt(box[3])
            if kps is not None:
                _fill_kps(row, kps)
            self._writer.writerow(row)


def _empty_row() -> dict:
    return {f: "" for f in _FIELDNAMES}


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def _fill_kps(row: dict, kps: np.ndarray) -> None:
    for i, name in enumerate(_KP_NAMES):
        if i < len(kps):
            row[f"kp_{name}_x"] = _fmt(float(kps[i][0]))
            row[f"kp_{name}_y"] = _fmt(float(kps[i][1]))
            row[f"kp_{name}_v"] = _fmt(float(kps[i][2]))
