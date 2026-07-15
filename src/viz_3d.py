"""
2.5D matplotlib visualization for the dual-camera pen-state pipeline.

Floor position (X, Y) is metric (computed from H_top_floor).
Z height is a single-view approximation from the side-camera bounding-box
height — NOT triangulated depth.  This is clearly labelled as 2.5D.

The visualization updates once per aligned frame pair by clearing the
Axes3D and redrawing everything from the current PenState.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("TkAgg")   # use TkAgg so the window appears outside notebooks

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers the projection)

from .pen_state import PenState

# Keypoint indices — must match the pose model schema
_KP_TAIL_BASE   = 0
_KP_TAIL_TIP    = 1
_KP_LEFT_HOCK   = 2
_KP_RIGHT_HOCK  = 3
_KP_LEFT_FOOT   = 4
_KP_RIGHT_FOOT  = 5
_KP_NECK_BACK   = 6
_KP_MIDDLE_BACK = 7
_KP_COMB        = 8
_KP_BEAK        = 9

_SKELETON = [
    (_KP_BEAK,        _KP_COMB),
    (_KP_COMB,        _KP_NECK_BACK),
    (_KP_NECK_BACK,   _KP_MIDDLE_BACK),
    (_KP_MIDDLE_BACK, _KP_TAIL_BASE),
    (_KP_TAIL_BASE,   _KP_TAIL_TIP),
    (_KP_TAIL_BASE,   _KP_LEFT_HOCK),
    (_KP_TAIL_BASE,   _KP_RIGHT_HOCK),
    (_KP_MIDDLE_BACK, _KP_LEFT_HOCK),
    (_KP_MIDDLE_BACK, _KP_RIGHT_HOCK),
    (_KP_LEFT_HOCK,   _KP_LEFT_FOOT),
    (_KP_RIGHT_HOCK,  _KP_RIGHT_FOOT),
]

_MARKER_COLOR_MAP: dict[str, str] = {
    "green":     "#2ecc71",
    "black":     "#2c3e50",
    "red":       "#e74c3c",
    "blue":      "#3498db",
    "yellow":    "#f1c40f",
    "uncertain": "#95a5a6",
}

_KP_CONF_THRESHOLD = 0.3


class PenViz3D:
    """
    Live 2.5D pen visualization.

    Parameters
    ----------
    pen_width_m   : pen width in metres (X axis)
    pen_length_m  : pen length in metres (Y axis)
    wall_height_m : pen wall height in metres (Z axis)
    title         : window title
    """

    def __init__(
        self,
        pen_width_m: float,
        pen_length_m: float,
        wall_height_m: float,
        title: str = "Pen State — 2.5D Reconstruction",
    ) -> None:
        self._W = pen_width_m
        self._L = pen_length_m
        self._H = wall_height_m

        plt.ion()
        self._fig = plt.figure(figsize=(9, 7))
        self._ax: Axes3D = self._fig.add_subplot(111, projection="3d")
        self._fig.canvas.manager.set_window_title(title)
        self._fig.text(
            0.5, 0.01,
            "2.5D reconstruction — floor position is metric; Z is single-view approximation",
            ha="center", fontsize=8, color="gray",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 1])

    def update(self, pen_state: PenState) -> None:
        """Redraw the visualization for the current PenState."""
        ax = self._ax
        ax.cla()
        self._draw_pen_box(ax)

        if pen_state.feeder_floor is not None:
            fx, fy = pen_state.feeder_floor
            ax.scatter([fx], [fy], [0.0], marker="*", s=200, color="#e67e22",
                       zorder=5, label="Feeder")

        if pen_state.waterer_floor is not None:
            wx, wy = pen_state.waterer_floor
            ax.scatter([wx], [wy], [0.0], marker="o", s=150, color="#3498db",
                       zorder=5, label="Waterer")

        for hen in pen_state.hens:
            self._draw_hen(ax, hen)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_xlim(0, self._W)
        ax.set_ylim(0, self._L)
        ax.set_zlim(0, self._H)
        ax.set_title(
            f"t={pen_state.timestamp:.2f}s  |  {len(pen_state.hens)} hen(s)"
        )
        if pen_state.feeder_floor or pen_state.waterer_floor:
            ax.legend(loc="upper right", fontsize=8)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    def close(self) -> None:
        plt.close(self._fig)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _draw_pen_box(self, ax: Axes3D) -> None:
        W, L, H = self._W, self._L, self._H
        gray = "#cccccc"
        alpha = 0.25

        # Floor
        xs = [0, W, W, 0, 0]
        ys = [0, 0, L, L, 0]
        ax.plot(xs, ys, [0] * 5, color=gray)

        # Four vertical edges
        for (x, y) in [(0, 0), (W, 0), (W, L), (0, L)]:
            ax.plot([x, x], [y, y], [0, H], color=gray, alpha=alpha)

        # Ceiling outline
        ax.plot(xs, ys, [H] * 5, color=gray, alpha=alpha)

        # Floor grid
        for xi in np.linspace(0, W, 4):
            ax.plot([xi, xi], [0, L], [0, 0], color=gray, linewidth=0.5, alpha=0.5)
        for yi in np.linspace(0, L, 6):
            ax.plot([0, W], [yi, yi], [0, 0], color=gray, linewidth=0.5, alpha=0.5)

    def _draw_hen(self, ax: Axes3D, hen) -> None:
        color = _MARKER_COLOR_MAP.get(hen.color_label, "#95a5a6")
        fx, fy = hen.floor_x, hen.floor_y

        ax.scatter([fx], [fy], [0.0], color=color, s=60, zorder=4)
        ax.text(fx + 0.02, fy, 0.05, f"#{hen.track_id}", fontsize=7, color=color)

        if hen.side_kps is None:
            ax.plot([fx, fx], [fy, fy], [0.0, hen.approx_z],
                    color=color, linewidth=1, linestyle="--", alpha=0.4)
            return

        kps = hen.side_kps
        z_scale = hen.approx_z if hen.approx_z > 0 else 0.3

        kp_z = _build_kp_z(kps, z_scale)

        # Draw skeleton edges
        for i, j in _SKELETON:
            if (i >= len(kps) or j >= len(kps)):
                continue
            ci, cj = float(kps[i][2]), float(kps[j][2])
            if ci < _KP_CONF_THRESHOLD or cj < _KP_CONF_THRESHOLD:
                continue
            ax.plot(
                [fx, fx], [fy, fy], [kp_z[i], kp_z[j]],
                color=color, linewidth=1.2, alpha=0.8,
            )

        # Keypoint dots
        for i in range(len(kps)):
            if float(kps[i][2]) >= _KP_CONF_THRESHOLD:
                ax.scatter([fx], [fy], [kp_z[i]], color=color, s=15, zorder=5)


def _build_kp_z(kps: np.ndarray, z_scale: float) -> list[float]:
    """
    Map each keypoint's image-Y coordinate to an approximate world-Z height.

    The side camera sees the hen from the side.  Higher image-Y → closer to
    the floor (lower Z).  We normalise over the bounding-box image extent.
    """
    valid_y = [float(kps[i][1]) for i in range(len(kps)) if float(kps[i][2]) >= _KP_CONF_THRESHOLD]
    if not valid_y:
        return [z_scale * 0.5] * len(kps)

    y_min, y_max = min(valid_y), max(valid_y)
    y_range = y_max - y_min if y_max > y_min else 1.0

    result = []
    for i in range(len(kps)):
        if float(kps[i][2]) >= _KP_CONF_THRESHOLD:
            # Flip: small image-Y (top of image) → high Z; large image-Y → Z≈0
            norm = 1.0 - (float(kps[i][1]) - y_min) / y_range
            result.append(norm * z_scale)
        else:
            result.append(0.0)
    return result
