"""
Camera / video input abstraction.

Supports USB cameras, RTSP streams, and local video files via OpenCV.
"""

from __future__ import annotations

from typing import Generator, Optional

import cv2
import numpy as np


class FrameSource:
    """
    Iterator that yields BGR frames from any OpenCV-compatible source.

    Parameters
    ----------
    source:
        Camera device index (int), local file path (str), or RTSP URL (str).
    rotate:
        Optional OpenCV rotation code, e.g. ``cv2.ROTATE_180``.

    Usage::

        with FrameSource(source=0) as cam:
            for frame in cam:
                # process frame …

    Or as a plain generator without context manager::

        for frame in FrameSource("samplevideos/video.mp4"):
            pass
    """

    def __init__(
        self,
        source: int | str,
        rotate: Optional[int] = None,
    ) -> None:
        self.source = source
        self.rotate = rotate
        self._cap: Optional[cv2.VideoCapture] = None

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        src = int(self.source) if str(self.source).isdigit() else self.source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source!r}")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ── Iteration ─────────────────────────────────────────────────────────────

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

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
