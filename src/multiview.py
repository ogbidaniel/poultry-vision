"""
Minimal multiview camera config and ring-buffer synchronization helpers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional
import time

import cv2
import numpy as np

from .capture import CameraSourceConfig, FrameSource
from .io_utils import load_config


@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp_ms: float
    camera_name: str
    frame_id: int


@dataclass
class SyncedFramePair:
    primary: TimestampedFrame
    secondary: TimestampedFrame
    diff_ms: float
    pair_id: int


@dataclass
class SyncSettings:
    tolerance_ms: float = 75.0
    buffer_size: int = 30
    primary_camera: str = "side"


def _rtsp_uri(host: str, port: int, stream_path: str) -> str:
    return f"rtsp://{host}:{port}/{stream_path}"


def load_camera_specs(
    path: str | Path, location: str = "pi"
) -> tuple[dict[str, CameraSourceConfig], SyncSettings]:
    """Load camera specs from a MediaMTX-style config.

    All cameras are served by MediaMTX as RTSP; each entry's ``path`` is the
    stream name and the URI is built as ``rtsp://{host}:{port}/{path}`` with the
    host selected by *location* (``pi`` -> ``host_local``, ``remote`` ->
    ``host_remote``). An older ``source``-based schema is still accepted as a
    fallback. Credentials are never stored here — MediaMTX handles them.
    """
    data = load_config(path)
    mm = data.get("mediamtx") or {}
    host = mm.get("host_local", "localhost") if location == "pi" else mm.get("host_remote")
    port = int(mm.get("port", 8554) or 8554)

    cameras = {}
    for name, raw in (data.get("cameras") or {}).items():
        if mm and raw.get("path") is not None:          # new MediaMTX schema
            source: int | str = _rtsp_uri(host, port, raw["path"])
            ctype = "rtsp"
        else:                                           # legacy source-based schema
            source = raw.get("source", name)
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            ctype = raw.get("type", raw.get("camera_type", "usb"))
        cameras[name] = CameraSourceConfig(
            name=name,
            source=source,
            camera_type=ctype,
            width=int(raw.get("width", 0) or 0),
            height=int(raw.get("height", 0) or 0),
            fps=float(raw.get("fps", 30.0) or 30.0),
            rotation=int(raw.get("rotation", 0) or 0),
            buffer_size=int(raw.get("buffer_size", 1) or 1),
            backend=raw.get("backend", "ffmpeg" if ctype == "rtsp" else "v4l2"),
            fourcc=raw.get("fourcc", None if ctype == "rtsp" else "MJPG"),
            rtsp_transport=(data.get("rtsp") or {}).get("transport", "tcp"),
        )

    sync_raw = data.get("sync") or {}
    sync = SyncSettings(
        tolerance_ms=float(sync_raw.get("tolerance_ms", 75.0) or 75.0),
        buffer_size=int(sync_raw.get("buffer_size", 30) or 30),
        primary_camera=sync_raw.get("primary_camera", "side"),
    )
    return cameras, sync


def camera_roles(path: str | Path) -> dict[str, str]:
    """Logical role -> physical camera-name map (may be empty)."""
    return dict(load_config(path).get("roles") or {})


def resolve_role(
    path: str | Path, role: str, location: str = "pi"
) -> Optional[CameraSourceConfig]:
    """Resolve a logical role (or a physical camera name) to its
    ``CameraSourceConfig``, or ``None`` if that stream is not configured on this
    deployment. Lets a Pi with only a subset of cameras run what it has."""
    cam_name = camera_roles(path).get(role, role)
    cameras, _ = load_camera_specs(path, location=location)
    return cameras.get(cam_name)


class RingBufferSynchronizer:
    def __init__(self, settings: SyncSettings) -> None:
        self.settings = settings
        self._buffers: dict[str, Deque[TimestampedFrame]] = {}
        self._pair_id = 0

    def add_frame(self, frame: TimestampedFrame) -> None:
        buf = self._buffers.setdefault(
            frame.camera_name,
            deque(maxlen=self.settings.buffer_size),
        )
        buf.append(frame)

    def get_pair(self, primary_name: str, secondary_name: str) -> Optional[SyncedFramePair]:
        primary_buf = self._buffers.get(primary_name)
        secondary_buf = self._buffers.get(secondary_name)
        if not primary_buf or not secondary_buf:
            return None

        primary = primary_buf[-1]
        best_idx = None
        best_diff = None
        for idx, candidate in enumerate(secondary_buf):
            diff = abs(candidate.timestamp_ms - primary.timestamp_ms)
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = idx

        if best_idx is None or best_diff is None or best_diff > self.settings.tolerance_ms:
            return None

        secondary = secondary_buf[best_idx]
        self._pair_id += 1
        return SyncedFramePair(
            primary=primary,
            secondary=secondary,
            diff_ms=best_diff,
            pair_id=self._pair_id,
        )


def read_timestamped_frame(source: FrameSource, camera_name: str, frame_id: int) -> Optional[TimestampedFrame]:
    if source._cap is None:
        source.open()
    ok, frame = source._cap.read()
    if not ok:
        return None
    if source.rotate is not None:
        frame = cv2.rotate(frame, source.rotate)
    return TimestampedFrame(
        frame=frame,
        timestamp_ms=time.monotonic() * 1000.0,
        camera_name=camera_name,
        frame_id=frame_id,
    )
