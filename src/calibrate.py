"""
Interactive pen calibration tool.

Click 4 corners of the pen in order: top-left → top-right → bottom-right →
bottom-left.  The calibration is saved to a ``.npy`` file (numpy) plus a
human-readable ``.yaml`` sidecar.

Usage::

    python -m src.calibrate --video path/to/video.mp4 --output pen_config.npy

    # or as a module call from the repo root:
    python -c "from src.calibrate import PenCalibrator; PenCalibrator('0').calibrate()"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


class PenCalibrator:
    """
    Interactive pen calibration using OpenCV.

    Click 4 corners in order: TL → TR → BR → BL.
    Press 'r' to reset, 's' to save, 'q' to quit.
    """

    def __init__(self, video_path: str, output_path: str = "pen_config.npy") -> None:
        self.video_path  = video_path
        self.output_path = output_path
        self.points: list[tuple[int, int]] = []
        self.frame:  Optional[np.ndarray]  = None
        self._window = "Pen Calibration - Click 4 Corners"

    def _mouse_callback(self, event: int, x: int, y: int, *_) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            self._draw_overlay()

    def _draw_overlay(self) -> None:
        if self.frame is None:
            return
        display = self.frame.copy()
        for i, pt in enumerate(self.points):
            cv2.circle(display, pt, 8, (0, 0, 255), -1)
            cv2.circle(display, pt, 10, (255, 255, 255), 2)
            label = ["TL", "TR", "BR", "BL"][i] if i < 4 else str(i)
            cv2.putText(display, label, (pt[0] + 15, pt[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if len(self.points) > 1:
            pts = np.array(self.points, np.int32)
            cv2.polylines(display, [pts], len(self.points) == 4, (0, 255, 0), 2)

        for i, text in enumerate([
            "Click 4 corners: TL → TR → BR → BL",
            f"Points: {len(self.points)}/4",
            "Press 'r' reset  's' save  'q' quit",
        ]):
            cv2.putText(display, text, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(self._window, display)

    def calibrate(self) -> Optional[np.ndarray]:
        """
        Run the interactive calibration loop.

        Returns a (4, 2) float32 array of corner pixels, or None if cancelled.
        """
        src = int(self.video_path) if self.video_path.isdigit() else self.video_path
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"Error: cannot open source {self.video_path!r}")
            return None

        ret, self.frame = cap.read()
        cap.release()
        if not ret or self.frame is None:
            print("Error: could not read frame")
            return None

        cv2.namedWindow(self._window)
        cv2.setMouseCallback(self._window, self._mouse_callback)
        self._draw_overlay()

        print("\n=== Pen Calibration ===")
        print("Click 4 corners: TL → TR → BR → BL")
        print("Press 'r' reset  's' save  'q' quit\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Calibration cancelled.")
                break
            elif key == ord('r'):
                self.points = []
                print("Points reset.")
                self._draw_overlay()
            elif key == ord('s') or len(self.points) == 4:
                if len(self.points) == 4:
                    corners = np.array(self.points, dtype=np.float32)
                    self._save(corners)
                    cv2.destroyAllWindows()
                    return corners
                else:
                    print(f"Need 4 points, have {len(self.points)}")

        cv2.destroyAllWindows()
        return None

    def _save(self, corners: np.ndarray) -> None:
        np.save(self.output_path, corners)
        print(f"Calibration saved → {self.output_path}")

        yaml_path = Path(self.output_path).with_suffix(".yaml")
        labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
        cfg = {
            "pen_corners": {
                label: corners[i].tolist()
                for i, label in enumerate(labels)
            },
            "source_video": str(self.video_path),
        }
        with open(yaml_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"Human-readable config → {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive pen calibration")
    parser.add_argument("--video", "-v", default="samplevideos/poultry-vid-01.mp4",
                        help="Video file path or camera index")
    parser.add_argument("--output", "-o", default="pen_config.npy",
                        help="Output .npy calibration file")
    args = parser.parse_args()

    cal = PenCalibrator(args.video, args.output)
    result = cal.calibrate()
    if result is not None:
        print("Calibration complete!")
    else:
        print("Calibration failed or cancelled.")


if __name__ == "__main__":
    main()
