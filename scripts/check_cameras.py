"""
Quick reachability check for the MediaMTX RTSP camera streams.

Grabs one frame from each configured camera (or a single role) and reports the
resolution and how long it took. Never opens /dev/video* directly — it uses the
RTSP roles in config/cameras.yaml.

    # On the Pi (localhost streams)
    python scripts/check_cameras.py --location pi

    # From another machine over Tailscale
    python scripts/check_cameras.py --location remote

    # Just one role/camera
    python scripts/check_cameras.py --role top_live --location pi
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.multiview import load_camera_specs, resolve_role  # noqa: E402


def grab(uri: str, warmup: int = 15, timeout_s: float = 12.0):
    """Open an RTSP URI and return the first decoded frame (or None) + elapsed s."""
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(uri, getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
    t0 = time.time()
    frame = None
    if cap.isOpened():
        for _ in range(warmup):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
                break
            if time.time() - t0 > timeout_s:
                break
    cap.release()
    return frame, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="Check MediaMTX RTSP camera streams")
    ap.add_argument("--cameras", default="config/cameras.yaml")
    ap.add_argument("--location", choices=["pi", "remote"], default="pi",
                    help="host_local (on the Pi) or host_remote/Tailscale")
    ap.add_argument("--role", help="check one role/camera name instead of all")
    args = ap.parse_args()

    if args.role:
        cam = resolve_role(args.cameras, args.role, args.location)
        if cam is None:
            sys.exit(f"role/camera '{args.role}' not configured in {args.cameras}")
        items = [(cam.name, cam)]
    else:
        cams, _ = load_camera_specs(args.cameras, args.location)
        items = list(cams.items())

    print(f"Checking {len(items)} stream(s), location={args.location}\n")
    ok_n = 0
    for name, cam in items:
        frame, dt = grab(str(cam.source))
        if frame is not None:
            h, w = frame.shape[:2]
            ok_n += 1
            print(f"  OK    {name:13s} {cam.source}  ->  {w}x{h}  ({dt:.1f}s)")
        else:
            print(f"  FAIL  {name:13s} {cam.source}  (no frame in {dt:.1f}s)")
    print(f"\n{ok_n}/{len(items)} stream(s) reachable.")
    sys.exit(0 if ok_n == len(items) and items else 1)


if __name__ == "__main__":
    main()
