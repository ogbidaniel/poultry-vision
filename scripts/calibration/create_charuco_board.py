#!/usr/bin/env python3
"""Create a ChArUco board config and printable board image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.charuco import create_board_config, save_board_config, write_board_png
from src.calibration.common import ensure_expected_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a ChArUco board definition.")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="OpenCV ArUco dictionary name.")
    parser.add_argument("--squares-x", type=int, default=7, help="Number of board squares along X.")
    parser.add_argument("--squares-y", type=int, default=5, help="Number of board squares along Y.")
    parser.add_argument("--square-length-mm", type=float, default=40.0, help="Square size in millimeters.")
    parser.add_argument("--marker-length-mm", type=float, default=30.0, help="Marker size in millimeters.")
    parser.add_argument("--width-px", type=int, default=2000, help="Rendered board width in pixels.")
    parser.add_argument("--height-px", type=int, default=1400, help="Rendered board height in pixels.")
    parser.add_argument("--margin-px", type=int, default=32, help="Rendered board image margin in pixels.")
    parser.add_argument(
        "--config-output",
        default="artifacts/calibration/boards/charuco_board.yaml",
        help="Output YAML config path.",
    )
    parser.add_argument(
        "--image-output",
        default="artifacts/calibration/boards/charuco_board.png",
        help="Output printable PNG path.",
    )
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    config = create_board_config(
        dictionary_name=args.dictionary,
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
    )

    save_board_config(args.config_output, config)
    write_board_png(args.image_output, config, args.width_px, args.height_px, args.margin_px)

    print(f"Board config written to {Path(args.config_output)}")
    print(f"Board image written to {Path(args.image_output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
