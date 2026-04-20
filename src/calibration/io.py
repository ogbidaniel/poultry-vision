"""Artifact readers and writers for calibration outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import load_yaml, require_cv2, save_yaml


def _write_scalar(fs, key: str, value: Any) -> None:
    fs.write(key, value)


def write_intrinsics_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    """Write an OpenCV YAML intrinsic artifact."""
    cv2 = require_cv2(need_aruco=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open artifact for writing: {path}")

    _write_scalar(fs, "schema_version", 1)
    _write_scalar(fs, "artifact_type", "camera_intrinsics")
    _write_scalar(fs, "camera_name", artifact["camera_name"])
    _write_scalar(fs, "board_config_path", artifact["board_config_path"])
    _write_scalar(fs, "image_width", int(artifact["image_width"]))
    _write_scalar(fs, "image_height", int(artifact["image_height"]))
    _write_scalar(fs, "accepted_images", int(artifact["accepted_images"]))
    _write_scalar(fs, "total_images", int(artifact["total_images"]))
    _write_scalar(fs, "min_corners", int(artifact["min_corners"]))
    _write_scalar(fs, "rms", float(artifact["rms"]))
    fs.write("camera_matrix", np.asarray(artifact["camera_matrix"], dtype=np.float64))
    fs.write("dist_coeffs", np.asarray(artifact["dist_coeffs"], dtype=np.float64))
    fs.write(
        "std_deviations_intrinsics",
        np.asarray(artifact.get("std_deviations_intrinsics", []), dtype=np.float64),
    )
    fs.write("per_view_errors", np.asarray(artifact.get("per_view_errors", []), dtype=np.float64))
    fs.release()


def load_intrinsics_artifact(path: str | Path) -> dict[str, Any]:
    """Read an intrinsic artifact."""
    cv2 = require_cv2(need_aruco=False)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open intrinsic artifact: {path}")

    artifact = {
        "schema_version": int(fs.getNode("schema_version").real()),
        "artifact_type": fs.getNode("artifact_type").string(),
        "camera_name": fs.getNode("camera_name").string(),
        "board_config_path": fs.getNode("board_config_path").string(),
        "image_width": int(fs.getNode("image_width").real()),
        "image_height": int(fs.getNode("image_height").real()),
        "accepted_images": int(fs.getNode("accepted_images").real()),
        "total_images": int(fs.getNode("total_images").real()),
        "min_corners": int(fs.getNode("min_corners").real()),
        "rms": float(fs.getNode("rms").real()),
        "camera_matrix": fs.getNode("camera_matrix").mat(),
        "dist_coeffs": fs.getNode("dist_coeffs").mat(),
        "std_deviations_intrinsics": fs.getNode("std_deviations_intrinsics").mat(),
        "per_view_errors": fs.getNode("per_view_errors").mat(),
    }
    fs.release()
    return artifact


def write_stereo_artifact(path: str | Path, artifact: dict[str, Any]) -> None:
    """Write an OpenCV YAML stereo artifact."""
    cv2 = require_cv2(need_aruco=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open artifact for writing: {path}")

    _write_scalar(fs, "schema_version", 1)
    _write_scalar(fs, "artifact_type", "stereo_extrinsics")
    _write_scalar(fs, "camera_a", artifact["camera_a"])
    _write_scalar(fs, "camera_b", artifact["camera_b"])
    _write_scalar(fs, "board_config_path", artifact["board_config_path"])
    _write_scalar(fs, "accepted_pairs", int(artifact["accepted_pairs"]))
    _write_scalar(fs, "total_pairs", int(artifact["total_pairs"]))
    _write_scalar(fs, "rms", float(artifact["rms"]))
    fs.write("camera_matrix_a", np.asarray(artifact["camera_matrix_a"], dtype=np.float64))
    fs.write("dist_coeffs_a", np.asarray(artifact["dist_coeffs_a"], dtype=np.float64))
    fs.write("camera_matrix_b", np.asarray(artifact["camera_matrix_b"], dtype=np.float64))
    fs.write("dist_coeffs_b", np.asarray(artifact["dist_coeffs_b"], dtype=np.float64))
    fs.write("R", np.asarray(artifact["R"], dtype=np.float64))
    fs.write("T", np.asarray(artifact["T"], dtype=np.float64))
    fs.write("E", np.asarray(artifact["E"], dtype=np.float64))
    fs.write("F", np.asarray(artifact["F"], dtype=np.float64))
    fs.write("per_pair_errors", np.asarray(artifact.get("per_pair_errors", []), dtype=np.float64))
    fs.release()


def load_stereo_artifact(path: str | Path) -> dict[str, Any]:
    """Read a stereo artifact."""
    cv2 = require_cv2(need_aruco=False)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open stereo artifact: {path}")

    artifact = {
        "schema_version": int(fs.getNode("schema_version").real()),
        "artifact_type": fs.getNode("artifact_type").string(),
        "camera_a": fs.getNode("camera_a").string(),
        "camera_b": fs.getNode("camera_b").string(),
        "board_config_path": fs.getNode("board_config_path").string(),
        "accepted_pairs": int(fs.getNode("accepted_pairs").real()),
        "total_pairs": int(fs.getNode("total_pairs").real()),
        "rms": float(fs.getNode("rms").real()),
        "camera_matrix_a": fs.getNode("camera_matrix_a").mat(),
        "dist_coeffs_a": fs.getNode("dist_coeffs_a").mat(),
        "camera_matrix_b": fs.getNode("camera_matrix_b").mat(),
        "dist_coeffs_b": fs.getNode("dist_coeffs_b").mat(),
        "R": fs.getNode("R").mat(),
        "T": fs.getNode("T").mat(),
        "E": fs.getNode("E").mat(),
        "F": fs.getNode("F").mat(),
        "per_pair_errors": fs.getNode("per_pair_errors").mat(),
    }
    fs.release()
    return artifact


def save_summary(path: str | Path, summary: dict[str, Any]) -> None:
    """Write a plain YAML summary next to OpenCV artifacts."""
    save_yaml(path, summary)


def load_summary(path: str | Path) -> dict[str, Any]:
    """Read a plain YAML summary."""
    return load_yaml(path)

