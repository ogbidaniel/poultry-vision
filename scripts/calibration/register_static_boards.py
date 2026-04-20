#!/usr/bin/env python3
"""Register static ChArUco boards into the pen/world frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.calibration.common import ensure_expected_dirs, load_yaml, save_yaml


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Axis vectors must be non-zero")
    return vector / norm


def _pose_from_axes(origin: np.ndarray, x_axis_end: np.ndarray, y_axis_end: np.ndarray) -> dict:
    x_axis = _normalize(x_axis_end - origin)
    y_axis_raw = y_axis_end - origin
    z_axis = _normalize(np.cross(x_axis, y_axis_raw))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return {
        "origin_cm": origin.tolist(),
        "rotation_matrix": rotation.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Register permanent boards in the pen/world frame.")
    parser.add_argument(
        "--measurements",
        required=True,
        help=(
            "YAML measurements file. Expected schema: "
            "boards: [{name, board_config, origin_cm, x_axis_end_cm, y_axis_end_cm}]"
        ),
    )
    parser.add_argument(
        "--output",
        default="artifacts/calibration/static/static_boards.yaml",
        help="Output registration YAML path.",
    )
    args = parser.parse_args()

    ensure_expected_dirs(create=True)
    measurements = load_yaml(args.measurements)
    boards = measurements.get("boards", [])
    if not boards:
        raise ValueError("No `boards` entries found in the measurements YAML.")

    registered = []
    for item in boards:
        origin = np.asarray(item["origin_cm"], dtype=np.float64)
        x_axis_end = np.asarray(item["x_axis_end_cm"], dtype=np.float64)
        y_axis_end = np.asarray(item["y_axis_end_cm"], dtype=np.float64)
        pose = _pose_from_axes(origin, x_axis_end, y_axis_end)
        registered.append(
            {
                "name": item["name"],
                "board_config": item["board_config"],
                "pose_world": pose,
                "notes": item.get("notes", ""),
            }
        )

    output = {
        "schema_version": 1,
        "artifact_type": "static_board_registration",
        "source_measurements": str(Path(args.measurements)),
        "boards": registered,
    }
    save_yaml(args.output, output)
    print(f"Static board registration written to {Path(args.output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
