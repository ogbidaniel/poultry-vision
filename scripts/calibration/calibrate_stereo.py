#!/usr/bin/env python3
"""Calibrate stereo extrinsics from synchronized paired ChArUco images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.charuco import detect_charuco, load_board_config, object_points_from_ids
from src.calibration.common import ensure_expected_dirs, pair_image_files, require_cv2
from src.calibration.geometry import compose_object_pose_for_camera_b, reprojection_error, solve_pnp
from src.calibration.io import load_intrinsics_artifact, save_summary, write_stereo_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate stereo extrinsics from paired ChArUco images.")
    parser.add_argument("--camera-a", default="top", help="First camera name.")
    parser.add_argument("--camera-b", default="side", help="Second camera name.")
    parser.add_argument("--images-a", required=True, help="Directory of paired images for camera A.")
    parser.add_argument("--images-b", required=True, help="Directory of paired images for camera B.")
    parser.add_argument("--intrinsics-a", required=True, help="Intrinsic YAML for camera A.")
    parser.add_argument("--intrinsics-b", required=True, help="Intrinsic YAML for camera B.")
    parser.add_argument("--board-config", required=True, help="ChArUco board YAML.")
    parser.add_argument("--min-shared-corners", type=int, default=8, help="Minimum shared ChArUco corners per pair.")
    parser.add_argument(
        "--output",
        default=None,
        help="OpenCV YAML output path. Defaults to artifacts/calibration/stereo/<a>_<b>.yaml",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Plain YAML summary output path. Defaults next to the main artifact.",
    )
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    cv2 = require_cv2(need_aruco=True)
    board_config = load_board_config(args.board_config)
    intrinsics_a = load_intrinsics_artifact(args.intrinsics_a)
    intrinsics_b = load_intrinsics_artifact(args.intrinsics_b)
    pairs = pair_image_files(args.images_a, args.images_b)

    object_points = []
    image_points_a = []
    image_points_b = []
    accepted_pairs = []
    rejected_pairs = []
    image_size = None

    for path_a, path_b in pairs:
        image_a = cv2.imread(str(path_a))
        image_b = cv2.imread(str(path_b))
        if image_a is None or image_b is None:
            rejected_pairs.append({"file": path_a.name, "reason": "read_failed"})
            continue

        image_size = (image_a.shape[1], image_a.shape[0])
        result_a = detect_charuco(image_a, board_config, min_corners=args.min_shared_corners)
        result_b = detect_charuco(image_b, board_config, min_corners=args.min_shared_corners)
        if not result_a["success"] or not result_b["success"]:
            rejected_pairs.append({"file": path_a.name, "reason": "detection_failed"})
            continue

        ids_a = result_a["charuco_ids"].reshape(-1)
        ids_b = result_b["charuco_ids"].reshape(-1)
        shared_ids = sorted(set(int(i) for i in ids_a) & set(int(i) for i in ids_b))
        if len(shared_ids) < args.min_shared_corners:
            rejected_pairs.append({"file": path_a.name, "reason": "too_few_shared_corners", "shared": len(shared_ids)})
            continue

        index_a = {int(corner_id): idx for idx, corner_id in enumerate(ids_a)}
        index_b = {int(corner_id): idx for idx, corner_id in enumerate(ids_b)}
        ordered_ids = np.asarray(shared_ids, dtype=np.int32).reshape(-1, 1)
        shared_a = np.asarray(
            [result_a["charuco_corners"][index_a[int(corner_id)]][0] for corner_id in shared_ids],
            dtype=np.float32,
        )
        shared_b = np.asarray(
            [result_b["charuco_corners"][index_b[int(corner_id)]][0] for corner_id in shared_ids],
            dtype=np.float32,
        )
        object_points.append(object_points_from_ids(board_config, ordered_ids))
        image_points_a.append(shared_a)
        image_points_b.append(shared_b)
        accepted_pairs.append(path_a.name)

    if not object_points or image_size is None:
        print("No valid stereo pairs were accepted.")
        return 1

    rms, _, _, _, _, rotation, translation, essential, fundamental = cv2.stereoCalibrate(
        object_points,
        image_points_a,
        image_points_b,
        intrinsics_a["camera_matrix"],
        intrinsics_a["dist_coeffs"],
        intrinsics_b["camera_matrix"],
        intrinsics_b["dist_coeffs"],
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )

    per_pair_errors = []
    for obj_pts, img_a, img_b, pair_name in zip(object_points, image_points_a, image_points_b, accepted_pairs):
        rvec_a, tvec_a = solve_pnp(obj_pts, img_a, intrinsics_a["camera_matrix"], intrinsics_a["dist_coeffs"])
        err_a = reprojection_error(obj_pts, img_a, rvec_a, tvec_a, intrinsics_a["camera_matrix"], intrinsics_a["dist_coeffs"])
        rvec_b_pred, tvec_b_pred = compose_object_pose_for_camera_b(rvec_a, tvec_a, rotation, translation)
        err_b = reprojection_error(
            obj_pts,
            img_b,
            rvec_b_pred,
            tvec_b_pred,
            intrinsics_b["camera_matrix"],
            intrinsics_b["dist_coeffs"],
        )
        per_pair_errors.append(
            {
                "pair": pair_name,
                "camera_a_reprojection_error_px": float(err_a),
                "camera_b_predicted_reprojection_error_px": float(err_b),
                "combined_error_px": float((err_a + err_b) / 2.0),
            }
        )

    output_path = Path(args.output or f"artifacts/calibration/stereo/{args.camera_a}_{args.camera_b}.yaml")
    summary_path = Path(args.summary_output or output_path.with_suffix(".summary.yaml"))
    write_stereo_artifact(
        output_path,
        {
            "camera_a": args.camera_a,
            "camera_b": args.camera_b,
            "board_config_path": str(Path(args.board_config)),
            "accepted_pairs": len(accepted_pairs),
            "total_pairs": len(pairs),
            "rms": float(rms),
            "camera_matrix_a": intrinsics_a["camera_matrix"],
            "dist_coeffs_a": intrinsics_a["dist_coeffs"],
            "camera_matrix_b": intrinsics_b["camera_matrix"],
            "dist_coeffs_b": intrinsics_b["dist_coeffs"],
            "R": rotation,
            "T": translation,
            "E": essential,
            "F": fundamental,
            "per_pair_errors": np.asarray([item["combined_error_px"] for item in per_pair_errors], dtype=np.float64),
        },
    )

    save_summary(
        summary_path,
        {
            "camera_a": args.camera_a,
            "camera_b": args.camera_b,
            "accepted_pairs": accepted_pairs,
            "rejected_pairs": rejected_pairs,
            "accepted_count": len(accepted_pairs),
            "total_count": len(pairs),
            "rms": float(rms),
            "per_pair_errors": per_pair_errors,
        },
    )

    print(f"Stereo artifact written to {output_path}")
    print(f"Summary written to {summary_path}")
    print(f"Accepted {len(accepted_pairs)} / {len(pairs)} pairs. RMS={float(rms):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
