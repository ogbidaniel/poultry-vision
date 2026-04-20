"""Geometry helpers for validation and triangulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .common import require_cv2


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    """Compute mean reprojection error in pixels."""
    cv2 = require_cv2(need_aruco=False)
    projected, _ = cv2.projectPoints(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    projected = projected.reshape(-1, 2)
    observed = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    return float(np.linalg.norm(observed - projected, axis=1).mean())


def compose_object_pose_for_camera_b(
    rvec_a: np.ndarray,
    tvec_a: np.ndarray,
    rotation_a_to_b: np.ndarray,
    translation_a_to_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform an object pose from camera A coordinates into camera B coordinates."""
    cv2 = require_cv2(need_aruco=False)
    rotation_obj_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64))
    rotation_obj_b = np.asarray(rotation_a_to_b, dtype=np.float64) @ rotation_obj_a
    translation_obj_b = (
        np.asarray(rotation_a_to_b, dtype=np.float64) @ np.asarray(tvec_a, dtype=np.float64).reshape(3, 1)
        + np.asarray(translation_a_to_b, dtype=np.float64).reshape(3, 1)
    )
    rvec_b, _ = cv2.Rodrigues(rotation_obj_b)
    return rvec_b, translation_obj_b


def triangulate_from_normalized_points(
    points_a: np.ndarray,
    points_b: np.ndarray,
    rotation_a_to_b: np.ndarray,
    translation_a_to_b: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points in camera A coordinates from normalized points."""
    cv2 = require_cv2(need_aruco=False)
    proj_a = np.hstack([np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)])
    proj_b = np.hstack([
        np.asarray(rotation_a_to_b, dtype=np.float64),
        np.asarray(translation_a_to_b, dtype=np.float64).reshape(3, 1),
    ])
    points4d = cv2.triangulatePoints(
        proj_a.astype(np.float32),
        proj_b.astype(np.float32),
        np.asarray(points_a, dtype=np.float32).T,
        np.asarray(points_b, dtype=np.float32).T,
    )
    points3d = (points4d[:3] / points4d[3]).T
    return np.asarray(points3d, dtype=np.float64)


def rigid_align(reference: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the best-fit rigid transform mapping reference points to observed points."""
    ref = np.asarray(reference, dtype=np.float64)
    obs = np.asarray(observed, dtype=np.float64)
    if ref.shape != obs.shape:
        raise ValueError("reference and observed point sets must share a shape")

    ref_mean = ref.mean(axis=0)
    obs_mean = obs.mean(axis=0)
    ref_centered = ref - ref_mean
    obs_centered = obs - obs_mean

    h_mat = ref_centered.T @ obs_centered
    u_mat, _, v_t = np.linalg.svd(h_mat)
    rotation = v_t.T @ u_mat.T
    if np.linalg.det(rotation) < 0:
        v_t[-1, :] *= -1
        rotation = v_t.T @ u_mat.T

    translation = obs_mean - rotation @ ref_mean
    return rotation, translation


def aligned_point_error(reference: np.ndarray, observed: np.ndarray) -> float:
    """Compute RMS error after rigid alignment."""
    rotation, translation = rigid_align(reference, observed)
    aligned = (rotation @ np.asarray(reference).T).T + translation
    diff = aligned - np.asarray(observed, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def undistort_points(
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """Convert distorted pixel points into normalized camera coordinates."""
    cv2 = require_cv2(need_aruco=False)
    result = cv2.undistortPoints(
        np.asarray(image_points, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
    )
    return result.reshape(-1, 2)


def solve_pnp(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate object pose using OpenCV's iterative PnP solver."""
    cv2 = require_cv2(need_aruco=False)
    success, rvec, tvec = cv2.solvePnP(
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed to find a pose")
    return rvec, tvec

