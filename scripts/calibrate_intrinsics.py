"""
Checkerboard intrinsic calibration for the C920 cameras.

Run this ONCE per camera (top, side) to recover the true intrinsics
(focal lengths, principal point, lens distortion).  The detection/tracking
overlay then uses the real K instead of an FOV estimate, which makes the
raised-wall / corner wireframe and any Z reasoning accurate.

Workflow
--------
1. Print a checkerboard on rigid, FLAT card (foam board / clipboard).  A
   9x6 *inner-corner* board (10x7 squares) is a good default.  Measure the
   printed square size with a ruler — printers rescale.
2. With the camera at its DEPLOYMENT resolution (640x480) and autofocus OFF,
   record a short video while slowly moving/tilting the board so it appears:
   near & far, all four corners of the frame, and tilted ~30-45 deg in
   several directions.  20-40 good views beats 100 similar ones.
3. Run this script on that video (or a folder of extracted stills).

    python scripts/calibrate_intrinsics.py --video calib/top_calib.mp4 \
        --rows 6 --cols 9 --square 0.025 --camera top --config pen_config.json

    python scripts/calibrate_intrinsics.py --images "calib/side/*.jpg" \
        --rows 6 --cols 9 --square 0.025 --camera side --config pen_config.json

It prints the RMS reprojection error (aim for < 0.5 px), writes
calib/intrinsics_<camera>.json, and — if --config is given — stores K + dist
under cfg["camera_intrinsics"][camera].

Notes
-----
* --rows / --cols are INNER corners, not squares.  A board with 10x7 squares
  has 9x6 inner corners.  The detector is orientation-agnostic, but the counts
  must match exactly or no board is ever found.
* After calibrating, the cleanest path to accurate geometry is to UNDISTORT
  frames and re-run calibrate_corners.py on the undistorted video, so the floor
  homography and the K-based projection live in the same (distortion-free)
  image space.  See the README/handoff notes.
"""

from __future__ import annotations

import argparse
import glob as _glob
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

_CB_FLAGS = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_NORMALIZE_IMAGE
             | cv2.CALIB_CB_FAST_CHECK)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def _frames_from_video(path: Path, every: int) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield frame
        i += 1
    cap.release()


def _frames_from_images(pattern: str) -> Iterator[np.ndarray]:
    paths = sorted(_glob.glob(pattern))
    if not paths:
        sys.exit(f"No images match: {pattern}")
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            yield img


def calibrate(
    frames: Iterator[np.ndarray],
    rows: int,
    cols: int,
    square: float,
    max_views: int,
    min_views: int,
    show: bool,
) -> tuple[float, np.ndarray, np.ndarray, tuple[int, int], int]:
    # One object-point template for the board, scaled to metric square size.
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: Optional[tuple[int, int]] = None
    seen = 0

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        elif (gray.shape[1], gray.shape[0]) != img_size:
            print("  WARNING: frame size changed — skipping (all views must match).")
            continue

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), _CB_FLAGS)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRIT)
        obj_points.append(objp.copy())
        img_points.append(corners)
        seen += 1
        print(f"  view {seen} captured")

        if show:
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
            cv2.imshow("checkerboard (q to stop early)", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
        if seen >= max_views:
            break

    if show:
        cv2.destroyAllWindows()
    if seen < min_views:
        sys.exit(f"Only {seen} good views (need >= {min_views}). "
                 "Capture more, with the board filling different parts of the frame.")

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)
    return rms, K, dist, img_size, seen


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Checkerboard intrinsic calibration")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="Calibration video (board moved in view)")
    src.add_argument("--images", help="Glob of still images, e.g. 'calib/top/*.jpg'")
    ap.add_argument("--rows", type=int, required=True, help="INNER corners per column")
    ap.add_argument("--cols", type=int, required=True, help="INNER corners per row")
    ap.add_argument("--square", type=float, required=True, help="Square size in meters")
    ap.add_argument("--camera", required=True, help="Camera name (top / side)")
    ap.add_argument("--config", default=None, help="pen_config.json to write K+dist into")
    ap.add_argument("--every", type=int, default=15, help="Sample every Nth video frame")
    ap.add_argument("--min-views", type=int, default=12)
    ap.add_argument("--max-views", type=int, default=40)
    ap.add_argument("--output", default=None, help="Intrinsics JSON path")
    ap.add_argument("--show", action="store_true", help="Preview detections")
    args = ap.parse_args()

    if args.video:
        frames = _frames_from_video(Path(args.video), args.every)
    else:
        frames = _frames_from_images(args.images)

    print(f"Calibrating '{args.camera}' from "
          f"{'video ' + args.video if args.video else 'images ' + args.images}")
    print(f"Board: {args.cols}x{args.rows} inner corners, square={args.square} m")

    rms, K, dist, img_size, n = calibrate(
        frames, args.rows, args.cols, args.square,
        args.max_views, args.min_views, args.show)

    print(f"\nViews used: {n}")
    print(f"RMS reprojection error: {rms:.4f} px  (aim < 0.5)")
    print(f"Image size: {img_size[0]}x{img_size[1]}")
    print("K =\n", np.array2string(K, precision=2))
    print("dist =", np.array2string(dist.ravel(), precision=5))

    out = Path(args.output) if args.output else Path(f"calib/intrinsics_{args.camera}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "camera": args.camera,
        "image_size": list(img_size),
        "K": K.tolist(),
        "dist": dist.ravel().tolist(),
        "rms_px": float(rms),
        "n_views": int(n),
        "board": {"rows": args.rows, "cols": args.cols, "square_m": args.square},
    }
    out.write_text(json.dumps(record, indent=2))
    print(f"\nSaved -> {out}")

    if args.config:
        cfg_path = Path(args.config)
        cfg = json.loads(cfg_path.read_text())
        cfg.setdefault("camera_intrinsics", {})[args.camera] = {
            "K": K.tolist(),
            "dist": dist.ravel().tolist(),
            "image_size": list(img_size),
            "rms_px": float(rms),
        }
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"Wrote camera_intrinsics['{args.camera}'] -> {cfg_path}")


if __name__ == "__main__":
    main()
