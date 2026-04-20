"""
Camera / video input abstraction.

Supports USB cameras, RTSP streams, and local video files via OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional
import os
import shutil
import subprocess

import cv2
import numpy as np


ROTATION_TO_CV2 = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

BACKEND_TO_CV2 = {
    "any": cv2.CAP_ANY,
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
}


@dataclass
class CameraSourceConfig:
    """Resolved camera source configuration."""

    name: str
    source: int | str
    camera_type: str = "usb"
    width: int = 0
    height: int = 0
    fps: float = 30.0
    rotation: int = 0
    buffer_size: int = 1
    backend: str = "any"
    fourcc: Optional[str] = None
    rtsp_transport: str = "tcp"

    @property
    def rotate_code(self) -> Optional[int]:
        return ROTATION_TO_CV2.get(self.rotation)


class FrameSource:
    """
    Iterator that yields BGR frames from any OpenCV-compatible source.
    """

    def __init__(
        self,
        source: int | str | CameraSourceConfig,
        rotate: Optional[int] = None,
    ) -> None:
        if isinstance(source, CameraSourceConfig):
            self.config = source
        else:
            self.config = CameraSourceConfig(
                name="camera",
                source=source,
                camera_type="usb" if str(source).isdigit() else "file",
                rotation=0 if rotate is None else _rotation_from_cv2(rotate),
            )
        self.rotate = self.config.rotate_code if rotate is None else rotate
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    def open(self) -> None:
        cfg = self.config
        backend = BACKEND_TO_CV2.get(cfg.backend, cv2.CAP_ANY)

        if cfg.camera_type == "usb":
            self._prepare_usb_device()

        if cfg.camera_type == "rtsp":
            rtsp_transport = cfg.rtsp_transport or "tcp"
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{rtsp_transport}"
            )
            backend = getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY)

        src = int(cfg.source) if str(cfg.source).isdigit() else cfg.source
        self._cap = cv2.VideoCapture(src, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {cfg.source!r}")

        if cfg.width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        if cfg.height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if cfg.fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if cfg.buffer_size > 0:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)
        if cfg.fourcc:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        opened_here = self._cap is None
        if opened_here:
            self.open()
        try:
            assert self._cap is not None
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                if self.rotate is not None:
                    frame = cv2.rotate(frame, self.rotate)
                yield frame
        finally:
            if opened_here:
                self.release()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return self.config.fps
        return self._cap.get(cv2.CAP_PROP_FPS) or self.config.fps

    @property
    def frame_width(self) -> int:
        if self._cap is None:
            return self.config.width
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        if self._cap is None:
            return self.config.height
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _prepare_usb_device(self) -> None:
        cfg = self.config
        if not str(cfg.source).isdigit():
            return
        if shutil.which("v4l2-ctl") is None:
            return
        device = f"/dev/video{int(cfg.source)}"
        if not Path(device).exists():
            return
        if cfg.fps > 0:
            _run_best_effort(["v4l2-ctl", "-d", device, "--set-parm", str(int(cfg.fps))])
        if cfg.fourcc:
            _run_best_effort(["v4l2-ctl", "-d", device, "--set-fmt-video",
                              f"width={cfg.width},height={cfg.height},pixelformat={cfg.fourcc}"])


def _run_best_effort(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def _rotation_from_cv2(rotate_code: Optional[int]) -> int:
    if rotate_code == cv2.ROTATE_90_CLOCKWISE:
        return 90
    if rotate_code == cv2.ROTATE_180:
        return 180
    if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return 270
    return 0
