#!/usr/bin/env python3
"""Calibrate one camera's intrinsics from offline ChArUco images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.charuco import build_board, detect_charuco, load_board_config
from src.calibration.common import ensure_expected_dirs, list_image_files, require_cv2
from src.calibration.io import save_summary, write_intrinsics_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate per-camera intrinsics from ChArUco images.")
    parser.add_argument("--camera-name", required=True, help="Camera identifier, e.g. top or side.")
    parser.add_argument("--images-dir", required=True, help="Directory containing calibration images.")
    parser.add_argument("--board-config", required=True, help="ChArUco board YAML.")
    parser.add_argument(
        "--output",
        default=None,
        help="OpenCV YAML output path. Defaults to artifacts/calibration/intrinsics/<camera>.yaml",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Plain YAML summary output path. Defaults next to the main artifact.",
    )
    parser.add_argument(
        "--debug-dir",
        default=None,
        help="Optional directory for accepted/rejected debug visualizations.",
    )
    parser.add_argument("--min-corners", type=int, default=8, help="Minimum ChArUco corners per accepted image.")
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    cv2 = require_cv2(need_aruco=True)
    aruco = cv2.aruco
    board_config = load_board_config(args.board_config)
    board = build_board(board_config)
    image_paths = list_image_files(args.images_dir)

    charuco_corners = []
    charuco_ids = []
    accepted_files = []
    rejected = []
    image_size = None
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            rejected.append({"file": image_path.name, "reason": "read_failed"})
            continue

        image_size = (image.shape[1], image.shape[0])
        result = detect_charuco(image, board_config, min_corners=args.min_corners, draw_debug=debug_dir is not None)
        if not result["success"]:
            rejected.append({"file": image_path.name, "reason": result["reason"], "charuco_count": result["charuco_count"]})
            if debug_dir is not None and result["debug_image"] is not None:
                cv2.imwrite(str(debug_dir / f"{image_path.stem}.rejected.png"), result["debug_image"])
            continue

        charuco_corners.append(result["charuco_corners"])
        charuco_ids.append(result["charuco_ids"])
        accepted_files.append(image_path.name)
        if debug_dir is not None and result["debug_image"] is not None:
            cv2.imwrite(str(debug_dir / f"{image_path.stem}.accepted.png"), result["debug_image"])

    if image_size is None or not charuco_corners:
        print("No valid calibration images were accepted.")
        return 1

    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        rms, camera_matrix, dist_coeffs, _, _, std_intrinsics, _, per_view_errors = aruco.calibrateCameraCharucoExtended(
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
    else:
        rms, camera_matrix, dist_coeffs, _, _ = aruco.calibrateCameraCharuco(
            charucoCorners=charuco_corners,
            charucoIds=charuco_ids,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        std_intrinsics = np.array([], dtype=np.float64)
        per_view_errors = np.array([], dtype=np.float64)

    output_path = Path(args.output or f"artifacts/calibration/intrinsics/{args.camera_name}.yaml")
    summary_path = Path(args.summary_output or output_path.with_suffix(".summary.yaml"))

    artifact = {
        "camera_name": args.camera_name,
        "board_config_path": str(Path(args.board_config)),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "accepted_images": len(charuco_corners),
        "total_images": len(image_paths),
        "min_corners": args.min_corners,
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "std_deviations_intrinsics": std_intrinsics,
        "per_view_errors": per_view_errors,
    }
    write_intrinsics_artifact(output_path, artifact)

    summary = {
        "camera_name": args.camera_name,
        "images_dir": str(Path(args.images_dir)),
        "board_config": str(Path(args.board_config)),
        "accepted_images": accepted_files,
        "rejected_images": rejected,
        "accepted_count": len(accepted_files),
        "total_count": len(image_paths),
        "rms": float(rms),
    }
    save_summary(summary_path, summary)

    print(f"Intrinsic artifact written to {output_path}")
    print(f"Summary written to {summary_path}")
    print(f"Accepted {len(accepted_files)} / {len(image_paths)} images. RMS={float(rms):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
