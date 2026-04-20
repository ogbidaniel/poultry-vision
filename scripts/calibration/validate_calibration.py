#!/usr/bin/env python3
"""Validate intrinsics, stereo extrinsics, and optional static board registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.charuco import detect_charuco, load_board_config, object_points_from_ids
from src.calibration.common import ensure_expected_dirs, list_image_files, load_yaml, pair_image_files, require_cv2, save_yaml
from src.calibration.geometry import compose_object_pose_for_camera_b, reprojection_error, solve_pnp
from src.calibration.io import load_intrinsics_artifact, load_stereo_artifact


def _validate_single_camera(images_dir: Path, intrinsics: dict, board_config: dict, min_corners: int) -> list[dict]:
    cv2 = require_cv2(need_aruco=True)
    results = []
    for image_path in list_image_files(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        detection = detect_charuco(image, board_config, min_corners=min_corners)
        if not detection["success"]:
            results.append({"file": image_path.name, "status": "rejected", "reason": detection["reason"]})
            continue

        object_points = object_points_from_ids(board_config, detection["charuco_ids"])
        image_points = detection["charuco_corners"].reshape(-1, 2)
        rvec, tvec = solve_pnp(object_points, image_points, intrinsics["camera_matrix"], intrinsics["dist_coeffs"])
        error = reprojection_error(
            object_points,
            image_points,
            rvec,
            tvec,
            intrinsics["camera_matrix"],
            intrinsics["dist_coeffs"],
        )
        results.append({"file": image_path.name, "status": "accepted", "reprojection_error_px": float(error)})
    return results


def _validate_stereo_pairs(images_a: Path, images_b: Path, intrinsics_a: dict, intrinsics_b: dict, stereo: dict, board_config: dict, min_corners: int) -> list[dict]:
    cv2 = require_cv2(need_aruco=True)
    results = []
    for path_a, path_b in pair_image_files(images_a, images_b):
        image_a = cv2.imread(str(path_a))
        image_b = cv2.imread(str(path_b))
        if image_a is None or image_b is None:
            continue

        detection_a = detect_charuco(image_a, board_config, min_corners=min_corners)
        detection_b = detect_charuco(image_b, board_config, min_corners=min_corners)
        if not detection_a["success"] or not detection_b["success"]:
            results.append({"pair": path_a.name, "status": "rejected", "reason": "detection_failed"})
            continue

        ids_a = detection_a["charuco_ids"].reshape(-1)
        ids_b = detection_b["charuco_ids"].reshape(-1)
        shared_ids = sorted(set(int(i) for i in ids_a) & set(int(i) for i in ids_b))
        if len(shared_ids) < min_corners:
            results.append({"pair": path_a.name, "status": "rejected", "reason": "too_few_shared_corners"})
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

        rvec_a, tvec_a = solve_pnp(object_points, image_points_a, intrinsics_a["camera_matrix"], intrinsics_a["dist_coeffs"])
        error_a = reprojection_error(
            object_points, image_points_a, rvec_a, tvec_a, intrinsics_a["camera_matrix"], intrinsics_a["dist_coeffs"]
        )
        rvec_b_pred, tvec_b_pred = compose_object_pose_for_camera_b(rvec_a, tvec_a, stereo["R"], stereo["T"])
        error_b = reprojection_error(
            object_points, image_points_b, rvec_b_pred, tvec_b_pred, intrinsics_b["camera_matrix"], intrinsics_b["dist_coeffs"]
        )
        results.append(
            {
                "pair": path_a.name,
                "status": "accepted",
                "camera_a_error_px": float(error_a),
                "camera_b_predicted_error_px": float(error_b),
                "combined_error_px": float((error_a + error_b) / 2.0),
            }
        )
    return results


def _validate_static_registration(static_registration_path: Path) -> dict:
    artifact = load_yaml(static_registration_path)
    boards = artifact.get("boards", [])
    return {
        "registered_board_count": len(boards),
        "board_names": [board["name"] for board in boards],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate calibration artifacts with offline images.")
    parser.add_argument("--intrinsics-a", required=True, help="Intrinsic YAML for camera A.")
    parser.add_argument("--intrinsics-b", required=False, help="Intrinsic YAML for camera B.")
    parser.add_argument("--stereo", required=False, help="Stereo YAML artifact.")
    parser.add_argument("--board-config", required=True, help="ChArUco board YAML.")
    parser.add_argument("--images-a", required=True, help="Validation images for camera A.")
    parser.add_argument("--images-b", required=False, help="Validation images for camera B.")
    parser.add_argument("--static-registration", required=False, help="Static board registration YAML.")
    parser.add_argument("--min-corners", type=int, default=8, help="Minimum accepted ChArUco corners.")
    parser.add_argument(
        "--output",
        default="artifacts/calibration/reports/validation_report.yaml",
        help="Validation report output path.",
    )
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    board_config = load_board_config(args.board_config)
    intrinsics_a = load_intrinsics_artifact(args.intrinsics_a)
    report = {
        "board_config": str(Path(args.board_config)),
        "camera_a": _validate_single_camera(Path(args.images_a), intrinsics_a, board_config, args.min_corners),
    }

    if args.intrinsics_b and args.images_b:
        intrinsics_b = load_intrinsics_artifact(args.intrinsics_b)
        report["camera_b"] = _validate_single_camera(Path(args.images_b), intrinsics_b, board_config, args.min_corners)
    else:
        intrinsics_b = None

    if args.stereo and intrinsics_b is not None and args.images_b:
        stereo = load_stereo_artifact(args.stereo)
        report["stereo"] = _validate_stereo_pairs(
            Path(args.images_a),
            Path(args.images_b),
            intrinsics_a,
            intrinsics_b,
            stereo,
            board_config,
            args.min_corners,
        )

    if args.static_registration:
        report["static_registration"] = _validate_static_registration(Path(args.static_registration))

    save_yaml(args.output, report)
    print(f"Validation report written to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
