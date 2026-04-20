#!/usr/bin/env python3
"""Triangulate ChArUco corners as a sparse 3D reconstruction sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.charuco import detect_charuco, load_board_config, object_points_from_ids
from src.calibration.common import ensure_expected_dirs, pair_image_files, require_cv2, save_yaml
from src.calibration.geometry import aligned_point_error, triangulate_from_normalized_points, undistort_points
from src.calibration.io import load_intrinsics_artifact, load_stereo_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Triangulate matched ChArUco corners from paired images.")
    parser.add_argument("--images-a", required=True, help="Paired images for camera A.")
    parser.add_argument("--images-b", required=True, help="Paired images for camera B.")
    parser.add_argument("--intrinsics-a", required=True, help="Intrinsic YAML for camera A.")
    parser.add_argument("--intrinsics-b", required=True, help="Intrinsic YAML for camera B.")
    parser.add_argument("--stereo", required=True, help="Stereo YAML artifact.")
    parser.add_argument("--board-config", required=True, help="ChArUco board YAML.")
    parser.add_argument("--min-shared-corners", type=int, default=8, help="Minimum shared corners per pair.")
    parser.add_argument(
        "--output",
        default="artifacts/calibration/reports/triangulation_report.yaml",
        help="Triangulation report output path.",
    )
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    cv2 = require_cv2(need_aruco=True)
    intrinsics_a = load_intrinsics_artifact(args.intrinsics_a)
    intrinsics_b = load_intrinsics_artifact(args.intrinsics_b)
    stereo = load_stereo_artifact(args.stereo)
    board_config = load_board_config(args.board_config)

    report_pairs = []
    for path_a, path_b in pair_image_files(args.images_a, args.images_b):
        image_a = cv2.imread(str(path_a))
        image_b = cv2.imread(str(path_b))
        if image_a is None or image_b is None:
            continue

        detection_a = detect_charuco(image_a, board_config, min_corners=args.min_shared_corners)
        detection_b = detect_charuco(image_b, board_config, min_corners=args.min_shared_corners)
        if not detection_a["success"] or not detection_b["success"]:
            continue

        ids_a = detection_a["charuco_ids"].reshape(-1)
        ids_b = detection_b["charuco_ids"].reshape(-1)
        shared_ids = sorted(set(int(i) for i in ids_a) & set(int(i) for i in ids_b))
        if len(shared_ids) < args.min_shared_corners:
            continue

        index_a = {int(corner_id): idx for idx, corner_id in enumerate(ids_a)}
        index_b = {int(corner_id): idx for idx, corner_id in enumerate(ids_b)}
        ordered_ids = np.asarray(shared_ids, dtype=np.int32).reshape(-1, 1)
        object_points = object_points_from_ids(board_config, ordered_ids)
        image_points_a = np.asarray(
            [detection_a["charuco_corners"][index_a[int(corner_id)]][0] for corner_id in shared_ids],
            dtype=np.float32,
        )
        image_points_b = np.asarray(
            [detection_b["charuco_corners"][index_b[int(corner_id)]][0] for corner_id in shared_ids],
            dtype=np.float32,
        )

        normalized_a = undistort_points(image_points_a, intrinsics_a["camera_matrix"], intrinsics_a["dist_coeffs"])
        normalized_b = undistort_points(image_points_b, intrinsics_b["camera_matrix"], intrinsics_b["dist_coeffs"])
        points_3d = triangulate_from_normalized_points(normalized_a, normalized_b, stereo["R"], stereo["T"])
        geometry_error = aligned_point_error(object_points, points_3d)

        report_pairs.append(
            {
                "pair": path_a.name,
                "shared_corner_count": len(shared_ids),
                "geometry_alignment_rmse_m": float(geometry_error),
            }
        )

    report = {
        "pair_count": len(report_pairs),
        "pairs": report_pairs,
    }
    save_yaml(args.output, report)
    print(f"Triangulation report written to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
