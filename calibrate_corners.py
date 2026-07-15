"""
calibrate_corners.py — dual-frame interactive homography calibration.

Both cam0 (top) and cam1 (side) are shown SIMULTANEOUSLY in one split window
so you can correlate features across views.

Geometry
--------
Neither camera sees the whole floor; together they cover it:

    Y=0     ┌───────────────┐   near wall  — cam0 (TOP) sees this base
            │   cam0 only   │
            │~~~ overlap ~~~│   middle band — BOTH cameras see this
            │   cam1 only   │
    Y=L     └───────────────┘   far wall   — cam1 (SIDE) sees this base

Both cameras are mounted at the SAME (near) end of the pen: the side camera is
low at the front looking down the length toward the far wall; the top camera is
overhead near the centre looking straight down.

Metric floor frame (matches the physical Spatial_Setup): origin (0,0) at the
front-left PHYSICAL corner = the near-RIGHT corner in the video, nearest the
side camera (the side cam cannot see it; the top cam sees its base).  X
increases toward video-left (0 → pen width); Y toward the far wall
(0 → pen length).

Because neither view contains all four floor corners, every click below is a
point the camera *actually sees*.  Each camera then borrows the two corners it
cannot see from the other camera through the overlap tie-point homography, so
no corner is ever guessed.

To rebuild homographies after editing the convention or pen dimensions without
re-clicking::

    python calibrate_corners.py --recompute

Usage
-----
    python calibrate_corners.py [--config pen_config.json] [--frame-idx 30]

Click workflow
--------------
Step 1  cam0 (LEFT, top)   — 2 NEAR floor corners: near-LEFT (W,0), near-RIGHT (0,0)=origin
Step 2  cam1 (RIGHT, side) — 2 FAR  floor corners: far-LEFT (W,L),  far-RIGHT (0,L)
Step 3  BOTH panels        — 4+ paired tie-points in the overlap band.
                             Click LEFT (cam0) then RIGHT (cam1) for each pair.
                             Spread them out and push some toward the far end.

Outputs (written to pen_config.json):
  H_top_floor   cam0 pixel → metric floor (x, y)
  H_side_top    cam1 pixel → cam0 pixel   (from overlap tie-points)
  H_side_floor  cam1 pixel → metric floor (x, y)   — direct 4-corner fit

Controls
--------
  Left-click   place a point in the active panel(s)
  u            undo last point
  r            reset all points for this step
  s            confirm step (once minimum points are placed)
  q            quit without saving
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_PANEL_HEIGHT = 480


# ── Frame helpers ─────────────────────────────────────────────────────────────

def _load_frame(path: Path, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Frame {idx} unreadable from {path}")
    return frame


def _first_pair(root: Path, top_dir: str, side_dir: str) -> tuple[Path, Path]:
    s0 = {p.stem for p in (root / top_dir).glob("*.mp4")}
    s1 = {p.stem for p in (root / side_dir).glob("*.mp4")}
    common = sorted(s0 & s1)
    if not common:
        raise RuntimeError("No matching video pairs found.")
    stem = common[0]
    return root / top_dir / f"{stem}.mp4", root / side_dir / f"{stem}.mp4"


# ── Dual-panel session ────────────────────────────────────────────────────────

class DualSession:
    """
    Side-by-side interactive click session.

    mode
    ----
    "left"    only left panel (cam0) accepts clicks
    "right"   only right panel (cam1) accepts clicks
    "paired"  alternates LEFT → RIGHT → LEFT → RIGHT …
              A coloured border shows which panel is active.
    """

    _DIV = 4           # divider width (px)
    _BANNER = 70       # banner height below the images (px)

    def __init__(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        mode: str,
        n_min: int,
        n_exact: int = 0,
        title: str = "",
        hint: str = "",
        color0: tuple = (0, 220, 255),
        color1: tuple = (80, 255, 120),
        prior0: Optional[list] = None,
        prior1: Optional[list] = None,
        prior_labels0: Optional[list[str]] = None,
        prior_labels1: Optional[list[str]] = None,
    ) -> None:
        self.mode    = mode
        self.n_min   = n_min
        self.n_exact = n_exact
        self.title   = title
        self.hint    = hint
        self.c0      = color0
        self.c1      = color1

        h = _PANEL_HEIGHT
        self.f0 = cv2.resize(frame0, (int(frame0.shape[1] * h / frame0.shape[0]), h))
        self.f1 = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
        self.w0 = self.f0.shape[1]
        self.w1 = self.f1.shape[1]

        self.pts0: list[tuple[float, float]] = []
        self.pts1: list[tuple[float, float]] = []
        self.prior0 = prior0 or []
        self.prior1 = prior1 or []
        self.plabels0 = prior_labels0 or []
        self.plabels1 = prior_labels1 or []

        self._next = 0       # 0=left, 1=right (paired mode only)
        self._done = False
        self._quit = False

        self._wn = "Calibration (cam0 LEFT | cam1 RIGHT)"
        cv2.namedWindow(self._wn, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._wn,
                         self.w0 + self._DIV + self.w1,
                         h + self._BANNER)
        cv2.setMouseCallback(self._wn, self._on_mouse)

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> tuple[Optional[list], Optional[list]]:
        self._draw()
        while not self._done and not self._quit:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("u"):
                self._undo()
            elif key == ord("r"):
                self.pts0.clear()
                self.pts1.clear()
                self._next = 0
                self._draw()
            elif key == ord("s"):
                if self._ready():
                    self._done = True
                else:
                    req = self.n_exact or self.n_min
                    print(f"  Need {'exactly' if self.n_exact else 'at least'} "
                          f"{req} {'pairs' if self.mode=='paired' else 'points'}.")
            elif key == ord("q"):
                self._quit = True
        cv2.setMouseCallback(self._wn, lambda *_: None)
        if self._quit:
            return None, None
        return self.pts0, self.pts1

    def show_overlay(
        self,
        extra0: Optional[np.ndarray] = None,
        extra1: Optional[np.ndarray] = None,
        message: str = "",
    ) -> None:
        """Display a non-interactive overlay (e.g. computed projection result)."""
        self._draw(extra0, extra1, message)
        cv2.waitKey(1)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_mouse(self, event: int, px: int, py: int, *_) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        cam, cx, cy = self._panel_to_cam(px, py)
        if cam is None:
            return

        if self.mode == "left" and cam == 0:
            if self.n_exact and len(self.pts0) >= self.n_exact:
                return
            self.pts0.append((float(cx), float(cy)))

        elif self.mode == "right" and cam == 1:
            if self.n_exact and len(self.pts1) >= self.n_exact:
                return
            self.pts1.append((float(cx), float(cy)))

        elif self.mode == "paired":
            if self._next == 0 and cam == 0:
                self.pts0.append((float(cx), float(cy)))
                self._next = 1
            elif self._next == 1 and cam == 1:
                self.pts1.append((float(cx), float(cy)))
                self._next = 0

        self._draw()

    def _undo(self) -> None:
        if self.mode == "paired":
            if len(self.pts1) == len(self.pts0) and self.pts1:
                self.pts1.pop()
                self._next = 1
            elif self.pts0:
                self.pts0.pop()
                self._next = 0
        elif self.mode == "left" and self.pts0:
            self.pts0.pop()
        elif self.mode == "right" and self.pts1:
            self.pts1.pop()
        self._draw()

    def _ready(self) -> bool:
        req = self.n_exact or self.n_min
        if self.mode == "paired":
            return (len(self.pts0) >= req
                    and len(self.pts0) == len(self.pts1))
        if self.mode == "left":
            return len(self.pts0) >= req
        return len(self.pts1) >= req

    def _panel_to_cam(self, px: int, py: int) -> tuple:
        if px < self.w0:
            return 0, px, py
        if px >= self.w0 + self._DIV:
            return 1, px - self.w0 - self._DIV, py
        return None, px, py

    def _cam0_panel_xy(self, x: float, y: float) -> tuple[int, int]:
        return int(x), int(y)

    def _cam1_panel_xy(self, x: float, y: float) -> tuple[int, int]:
        return int(x + self.w0 + self._DIV), int(y)

    def _draw(
        self,
        extra0: Optional[np.ndarray] = None,
        extra1: Optional[np.ndarray] = None,
        extra_msg: str = "",
    ) -> None:
        o0 = extra0.copy() if extra0 is not None else self.f0.copy()
        o1 = extra1.copy() if extra1 is not None else self.f1.copy()

        # Camera labels
        _label(o0, "cam0  (top-down)")
        _label(o1, "cam1  (side)")

        # Prior points
        for pts, ovr, labels in [(self.prior0, o0, self.plabels0),
                                   (self.prior1, o1, self.plabels1)]:
            for i, (px, py) in enumerate(pts):
                cv2.circle(ovr, (int(px), int(py)), 6, (120, 120, 120), -1)
                lbl = labels[i] if i < len(labels) else ""
                if lbl:
                    cv2.putText(ovr, lbl, (int(px)+8, int(py)-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

        # Current points
        for i, (px, py) in enumerate(self.pts0):
            cv2.circle(o0, (int(px), int(py)), 8, self.c0, -1)
            cv2.putText(o0, str(i+1), (int(px)+9, int(py)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.c0, 1)
        for i, (px, py) in enumerate(self.pts1):
            cv2.circle(o1, (int(px), int(py)), 8, self.c1, -1)
            cv2.putText(o1, str(i+1), (int(px)+9, int(py)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.c1, 1)

        # Polygon outline for cam0 corners (left, step 1)
        if self.mode == "left" and len(self.pts0) >= 2:
            for a, b in zip(self.pts0, self.pts0[1:]):
                cv2.line(o0, _ip(a), _ip(b), self.c0, 1)
            if len(self.pts0) == 4:
                cv2.line(o0, _ip(self.pts0[3]), _ip(self.pts0[0]), self.c0, 1)

        # Paired mode border highlight
        if self.mode == "paired":
            active_border = self.c0 if self._next == 0 else self.c1
            if self._next == 0:
                cv2.rectangle(o0, (2, 2), (o0.shape[1]-3, o0.shape[0]-3), active_border, 3)
            else:
                cv2.rectangle(o1, (2, 2), (o1.shape[1]-3, o1.shape[0]-3), active_border, 3)

        # Compose combined image
        div = np.full((_PANEL_HEIGHT, self._DIV, 3), 50, dtype=np.uint8)
        combined = np.hstack([o0, div, o1])

        # Draw paired connecting lines on the combined image
        if self.mode == "paired":
            for i in range(min(len(self.pts0), len(self.pts1))):
                px0 = self._cam0_panel_xy(*self.pts0[i])
                px1 = self._cam1_panel_xy(*self.pts1[i])
                cv2.line(combined, px0, px1, (180, 180, 60), 1)
                # label midpoint
                mx = (px0[0] + px1[0]) // 2
                my = (px0[1] + px1[1]) // 2
                cv2.putText(combined, str(i+1), (mx-6, my-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 60), 1)

        # Banner
        total_w = combined.shape[1]
        banner = np.zeros((self._BANNER, total_w, 3), dtype=np.uint8)

        if self.mode == "paired":
            n_pairs = min(len(self.pts0), len(self.pts1))
            req = self.n_exact or self.n_min
            status = (f"Pairs: {n_pairs}/{req}   "
                      f"-> click {'LEFT (cam0)' if self._next==0 else 'RIGHT (cam1)'} next")
        elif self.mode == "left":
            req = self.n_exact or self.n_min
            status = f"cam0 points: {len(self.pts0)}/{req}"
        else:
            req = self.n_exact or self.n_min
            status = f"cam1 points: {len(self.pts1)}/{req}"

        if self._ready():
            status += "   OK - press [s] to confirm"

        for i, line in enumerate([
            self.title,
            self.hint,
            status + "   |   [u] undo  [r] reset  [s] confirm  [q] quit",
            extra_msg,
        ]):
            if line:
                cv2.putText(banner, line, (10, 18 + i * 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        cv2.imshow(self._wn, np.vstack([combined, banner]))


# ── Drawing utilities ─────────────────────────────────────────────────────────

def _ip(pt: tuple) -> tuple[int, int]:
    return (int(pt[0]), int(pt[1]))


def _label(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)


# ── Geometry ──────────────────────────────────────────────────────────────────

def _homography(src: list, dst: list) -> np.ndarray:
    H, _ = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
    )
    if H is None:
        raise RuntimeError("findHomography returned None")
    return H


def _ransac_homography(src: list, dst: list) -> tuple[np.ndarray, np.ndarray]:
    H, mask = cv2.findHomography(
        np.array(src, dtype=np.float32),
        np.array(dst, dtype=np.float32),
        cv2.RANSAC, 5.0,
    )
    if H is None:
        raise RuntimeError("findHomography (RANSAC) returned None")
    return H, mask


def _project(pt: tuple, H: np.ndarray) -> tuple[float, float]:
    r = cv2.perspectiveTransform(
        np.array([[[pt[0], pt[1]]]], dtype=np.float32), H
    )[0, 0]
    return (float(r[0]), float(r[1]))


def _extend_line_to_y(p1: tuple, p2: tuple, target_y: float) -> tuple[float, float]:
    """Return the point on the line through p1→p2 at y=target_y."""
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1e-6:
        return (x1, target_y)
    t = (target_y - y1) / (y2 - y1)
    return (x1 + t * (x2 - x1), target_y)


# ── Main calibration ──────────────────────────────────────────────────────────

def calibrate(config_path: Path, frame_idx: int) -> None:
    with config_path.open() as f:
        cfg = json.load(f)

    root     = Path(cfg["dataset_root"])
    top_dir  = cfg["cameras"]["top"]
    side_dir = cfg["cameras"]["side"]
    pen_w    = float(cfg["pen"]["width_m"])
    pen_l    = float(cfg["pen"]["length_m"])

    top_vid, side_vid = _first_pair(root, top_dir, side_dir)
    print(f"Sample pair: {top_vid.name}")

    f0 = _load_frame(top_vid,  frame_idx)
    f1 = _load_frame(side_vid, frame_idx)

    # ── Step 1: cam0 (top) — 2 NEAR corners it actually sees ──────────────────
    # Origin (0,0) is the front-left PHYSICAL corner = the near-RIGHT corner in
    # the video (nearest the side camera; side cam can't see it, top sees its
    # base).  X increases toward video-left, Y toward the far wall. See
    # _solve_floor for the metric mapping.
    sess1 = DualSession(
        f0, f1, mode="left", n_min=2, n_exact=2,
        title="STEP 1/3  |  cam0 (TOP): click the 2 NEAR floor corners",
        hint="(1) near-LEFT = (W, 0)    (2) near-RIGHT = (0, 0)=ORIGIN   — wall base cam0 sees",
        color0=(0, 220, 255),
    )
    near, _ = sess1.run()
    if near is None:
        sys.exit("Calibration cancelled.")
    BL_px, BR_px = near
    print(f"\nNear corners (cam0): BL={BL_px}  BR={BR_px}")

    # ── Step 2: cam1 (side) — 2 FAR corners it actually sees ──────────────────
    sess2 = DualSession(
        f0, f1, mode="right", n_min=2, n_exact=2,
        title="STEP 2/3  |  cam1 (SIDE): click the 2 FAR floor corners",
        hint="(1) far-LEFT = (W, L)    (2) far-RIGHT = (0, L)   — the far wall base cam1 sees",
        color1=(0, 180, 255),
        prior0=near, prior_labels0=["BL", "BR"],
    )
    _, far = sess2.run()
    if far is None:
        sys.exit("Calibration cancelled.")
    FL_px, FR_px = far     # far corners in cam1 (side) pixels
    print(f"Far corners (cam1): FL={FL_px}  FR={FR_px}")

    # ── Step 3: overlap-band tie-points ───────────────────────────────────────
    print("\nOverlap tie-points: click LEFT (cam0) then the SAME floor point in RIGHT (cam1).")
    print("Pick identifiable specks both cameras see; spread them across the band.")
    sess3 = DualSession(
        f0, f1, mode="paired", n_min=4,
        title="STEP 3/3  |  Overlap band: 4+ paired tie-points (same floor point in both)",
        hint="Click LEFT (cam0) first, then the SAME physical point in RIGHT (cam1)",
        color0=(0, 200, 255), color1=(80, 255, 120),
        prior0=near, prior_labels0=["BL", "BR"],
        prior1=far, prior_labels1=["FL", "FR"],
    )
    tie0, tie1 = sess3.run()
    if tie0 is None:
        sys.exit("Calibration cancelled.")
    cv2.destroyAllWindows()
    print(f"  {len(tie0)} tie-pairs collected.")

    H_top_floor, H_side_top, H_side_floor = _solve_floor(
        nL_px=BL_px, nR_px=BR_px, fL_px=FL_px, fR_px=FR_px,
        tie0=tie0, tie1=tie1, pen_w=pen_w, pen_l=pen_l,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    cfg["homographies"]["H_top_floor"]  = H_top_floor.tolist()
    cfg["homographies"]["H_side_top"]   = H_side_top.tolist()
    cfg["homographies"]["H_side_floor"] = H_side_floor.tolist()
    cfg["calibration_points"] = {
        "frame_idx": frame_idx,
        "cam0_near_corners": {"near_left": list(BL_px), "near_right": list(BR_px)},
        "cam1_far_corners":  {"far_left": list(FL_px),  "far_right": list(FR_px)},
        "tie_points": {"cam0": [list(p) for p in tie0],
                       "cam1": [list(p) for p in tie1]},
    }

    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\nHomographies + calibration points saved to {config_path}")
    print("Calibration complete.")


def _solve_floor(
    *,
    nL_px: tuple, nR_px: tuple, fL_px: tuple, fR_px: tuple,
    tie0: list, tie1: list, pen_w: float, pen_l: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Borrow-solve the two floor homographies from the corners each camera sees
    plus the overlap tie-points.

    Inputs are named by their position IN THE VIDEO:
        nL/nR  near-left / near-right floor corner (clicked in cam0 top view)
        fL/fR  far-left  / far-right  floor corner (clicked in cam1 side view)

    Metric frame (matches the physical Spatial_Setup): origin (0,0) at the
    front-left physical corner = the near-RIGHT video corner, nearest the side
    camera.  X increases toward video-left (0 → pen_w); Y toward the far wall
    (0 → pen_l).  Hence the RIGHT video corners map to X=0 and the LEFT video
    corners map to X=pen_w.
    """
    nL_m = (pen_w, 0.0)     # near-left  video
    nR_m = (0.0,   0.0)     # near-right video  == ORIGIN
    fL_m = (pen_w, pen_l)   # far-left   video
    fR_m = (0.0,   pen_l)   # far-right  video

    # H_side_top from tie-points (cam1 px → cam0 px).
    print("\nComputing H_side_top from overlap tie-points ...")
    H_side_top, mask = _ransac_homography(tie1, tie0)
    H_top_side = np.linalg.inv(H_side_top)
    n_in = int(mask.sum()) if mask is not None else len(tie1)
    print(f"  Inliers: {n_in}/{len(tie1)}")

    # Borrow the corners each camera cannot see from the other view.
    fL_top = _project(fL_px, H_side_top)
    fR_top = _project(fR_px, H_side_top)
    nL_side = _project(nL_px, H_top_side)
    nR_side = _project(nR_px, H_top_side)

    print(f"  Far corners borrowed into cam0 px:  fL={_fmt(fL_top)}  fR={_fmt(fR_top)}"
          "  (above frame is expected)")
    print(f"  Near corners borrowed into cam1 px: nL={_fmt(nL_side)}  nR={_fmt(nR_side)}"
          "  (below frame is expected)")

    print("\nComputing H_top_floor (cam0 px → metric) ...")
    H_top_floor = _homography(
        [nL_px, nR_px, fR_top, fL_top],
        [nL_m,  nR_m,  fR_m,   fL_m],
    )
    print("Computing H_side_floor (cam1 px → metric) — direct 4-corner fit ...")
    H_side_floor = _homography(
        [nL_side, nR_side, fR_px, fL_px],
        [nL_m,    nR_m,    fR_m,  fL_m],
    )

    # Diagnostics: corner reprojection.
    print("\nCorner reprojection (clicked → metric):")
    for H, pts, mets, lbls, cam in [
        (H_top_floor,  [nL_px, nR_px], [nL_m, nR_m], ["nL", "nR"], "cam0"),
        (H_side_floor, [fL_px, fR_px], [fL_m, fR_m], ["fL", "fR"], "cam1"),
    ]:
        for pt, met, lbl in zip(pts, mets, lbls):
            proj = _project(pt, H)
            err  = float(np.linalg.norm(np.array(proj) - np.array(met)))
            print(f"  {cam} {lbl}: → ({proj[0]:.3f}, {proj[1]:.3f}) m"
                  f"  expect ({met[0]:.2f}, {met[1]:.2f})  err={err:.4f} m")

    # Diagnostics: cross-view agreement on tie-points (the real quality metric).
    print("\nCross-view tie-point agreement (top metric vs side metric):")
    agree_err = []
    for i, (t0, t1) in enumerate(zip(tie0, tie1)):
        m_top  = _project(t0, H_top_floor)
        m_side = _project(t1, H_side_floor)
        d = float(np.linalg.norm(np.array(m_top) - np.array(m_side)))
        agree_err.append(d)
        flag = "inlier" if (mask is None or mask[i]) else "OUTLIER"
        print(f"  pair {i+1}: top {_fmt_m(m_top)}  side {_fmt_m(m_side)}"
              f"  Δ={d:.4f} m  [{flag}]")
    if agree_err:
        mean_err = float(np.mean(agree_err))
        print(f"  mean agreement error: {mean_err:.4f} m  (max {max(agree_err):.4f} m)")
        if mean_err > 0.10:
            print("  WARNING: agreement > 0.10 m. Add tie-points, spread them wider, "
                  "and push some toward the far end before trusting the floor map.")

    return H_top_floor, H_side_top, H_side_floor


def recompute(config_path: Path) -> None:
    """
    Rebuild the homographies from the saved ``calibration_points`` — no
    re-clicking.  Use after changing the metric convention or pen dimensions.
    """
    with config_path.open() as f:
        cfg = json.load(f)

    cp = cfg.get("calibration_points")
    if not cp:
        sys.exit("No 'calibration_points' in config — run an interactive calibration first.")

    pen_w = float(cfg["pen"]["width_m"])
    pen_l = float(cfg["pen"]["length_m"])
    near = cp["cam0_near_corners"]
    far  = cp["cam1_far_corners"]
    # Accept both the old (BL/BR/FL/FR) and new (near_left/...) key names.
    nL = tuple(near.get("near_left",  near.get("BL")))
    nR = tuple(near.get("near_right", near.get("BR")))
    fL = tuple(far.get("far_left",    far.get("FL")))
    fR = tuple(far.get("far_right",   far.get("FR")))
    tie0 = [tuple(p) for p in cp["tie_points"]["cam0"]]
    tie1 = [tuple(p) for p in cp["tie_points"]["cam1"]]

    print(f"Recomputing from saved points (pen {pen_w} x {pen_l} m) ...")
    H_top_floor, H_side_top, H_side_floor = _solve_floor(
        nL_px=nL, nR_px=nR, fL_px=fL, fR_px=fR,
        tie0=tie0, tie1=tie1, pen_w=pen_w, pen_l=pen_l,
    )

    cfg["homographies"]["H_top_floor"]  = H_top_floor.tolist()
    cfg["homographies"]["H_side_top"]   = H_side_top.tolist()
    cfg["homographies"]["H_side_floor"] = H_side_floor.tolist()
    # Normalise to the new key names on save.
    cfg["calibration_points"]["cam0_near_corners"] = {"near_left": list(nL), "near_right": list(nR)}
    cfg["calibration_points"]["cam1_far_corners"]  = {"far_left": list(fL),  "far_right": list(fR)}

    with config_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nHomographies recomputed and saved to {config_path}")


def _fmt(pt: tuple) -> str:
    return f"({pt[0]:.0f}, {pt[1]:.0f})"


def _fmt_m(pt: tuple) -> str:
    return f"({pt[0]:.3f}, {pt[1]:.3f})"


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Windows consoles default to cp1252; our diagnostics use a few non-ASCII
    # glyphs.  Make stdout tolerant so it never crashes on the Pi or Windows.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="pen_config.json")
    parser.add_argument("--frame-idx", default=30, type=int)
    parser.add_argument("--recompute", action="store_true",
                        help="Rebuild homographies from saved calibration_points "
                             "(no re-clicking).")
    args = parser.parse_args()
    path = Path(args.config)
    if not path.exists():
        sys.exit(f"Config not found: {path}")
    if args.recompute:
        recompute(path)
    else:
        calibrate(path, args.frame_idx)


if __name__ == "__main__":
    main()
