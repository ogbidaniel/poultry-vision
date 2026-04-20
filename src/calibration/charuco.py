"""ChArUco board helpers and detection routines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import load_yaml, require_cv2, save_yaml


def create_board_config(
    *,
    dictionary_name: str,
    squares_x: int,
    squares_y: int,
    square_length_mm: float,
    marker_length_mm: float,
) -> dict[str, Any]:
    """Create a normalized board configuration mapping."""
    if marker_length_mm >= square_length_mm:
        raise ValueError("marker_length_mm must be smaller than square_length_mm")

    return {
        "board_type": "charuco",
        "dictionary": dictionary_name,
        "squares_x": int(squares_x),
        "squares_y": int(squares_y),
        "square_length_mm": float(square_length_mm),
        "marker_length_mm": float(marker_length_mm),
        "square_length_m": float(square_length_mm) / 1000.0,
        "marker_length_m": float(marker_length_mm) / 1000.0,
    }


def save_board_config(path: str | Path, config: dict[str, Any]) -> None:
    """Write the board definition YAML."""
    save_yaml(path, config)


def load_board_config(path: str | Path) -> dict[str, Any]:
    """Read a board definition YAML."""
    return load_yaml(path)


def _dictionary_constant(aruco_module, dictionary_name: str):
    if not hasattr(aruco_module, dictionary_name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")
    return getattr(aruco_module, dictionary_name)


def build_dictionary(config: dict[str, Any]):
    """Construct an OpenCV predefined dictionary from board config."""
    cv2 = require_cv2(need_aruco=True)
    aruco = cv2.aruco
    dictionary_id = _dictionary_constant(aruco, str(config["dictionary"]))
    return aruco.getPredefinedDictionary(dictionary_id)


def build_board(config: dict[str, Any]):
    """Construct a ChArUco board object."""
    cv2 = require_cv2(need_aruco=True)
    aruco = cv2.aruco
    dictionary = build_dictionary(config)
    size = (int(config["squares_x"]), int(config["squares_y"]))
    square_length = float(config["square_length_m"])
    marker_length = float(config["marker_length_m"])

    # OpenCV 4.7+ exposes the constructor as a class.
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(size, square_length, marker_length, dictionary)
    return aruco.CharucoBoard_create(size[0], size[1], square_length, marker_length, dictionary)


def render_board_image(
    config: dict[str, Any],
    width_px: int,
    height_px: int,
    margin_px: int = 32,
) -> np.ndarray:
    """Render a printable ChArUco board image."""
    board = build_board(config)
    size = (int(width_px), int(height_px))
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=int(margin_px))

    cv2 = require_cv2(need_aruco=True)
    return cv2.aruco.drawPlanarBoard(board, size, marginSize=int(margin_px), borderBits=1)


def get_board_corner_positions(config: dict[str, Any]) -> np.ndarray:
    """Return the board corner locations indexed by ChArUco corner ID."""
    board = build_board(config)
    corners = board.getChessboardCorners()
    return np.asarray(corners, dtype=np.float32)


def detect_charuco(
    image: np.ndarray,
    config: dict[str, Any],
    *,
    min_corners: int = 6,
    draw_debug: bool = False,
) -> dict[str, Any]:
    """Detect ChArUco corners in one image."""
    cv2 = require_cv2(need_aruco=True)
    aruco = cv2.aruco
    dictionary = build_dictionary(config)
    board = build_board(config)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    parameters = aruco.DetectorParameters() if hasattr(aruco, "DetectorParameters") else aruco.DetectorParameters_create()
    marker_corners, marker_ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)

    debug_image = None
    if draw_debug:
        debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if marker_ids is not None and len(marker_ids) > 0:
            aruco.drawDetectedMarkers(debug_image, marker_corners, marker_ids)

    if marker_ids is None or len(marker_ids) == 0:
        return {
            "success": False,
            "reason": "no_markers",
            "marker_count": 0,
            "charuco_count": 0,
            "debug_image": debug_image,
        }

    if hasattr(aruco, "CharucoDetector"):
        detector = aruco.CharucoDetector(board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    else:
        _, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )

    charuco_count = 0 if charuco_ids is None else int(len(charuco_ids))
    if charuco_ids is None or charuco_count < min_corners:
        return {
            "success": False,
            "reason": "too_few_charuco_corners",
            "marker_count": int(len(marker_ids)),
            "charuco_count": charuco_count,
            "debug_image": debug_image,
        }

    if draw_debug and debug_image is not None:
        aruco.drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids)

    return {
        "success": True,
        "marker_count": int(len(marker_ids)),
        "charuco_count": charuco_count,
        "marker_ids": marker_ids,
        "charuco_corners": np.asarray(charuco_corners, dtype=np.float32),
        "charuco_ids": np.asarray(charuco_ids, dtype=np.int32),
        "debug_image": debug_image,
    }


def object_points_from_ids(config: dict[str, Any], charuco_ids: np.ndarray) -> np.ndarray:
    """Map ChArUco corner IDs to their 3D board coordinates."""
    board_corners = get_board_corner_positions(config)
    flat_ids = np.asarray(charuco_ids).reshape(-1)
    return np.asarray([board_corners[int(idx)] for idx in flat_ids], dtype=np.float32)


def write_board_png(path: str | Path, config: dict[str, Any], width_px: int, height_px: int, margin_px: int) -> None:
    """Render and save the board image."""
    cv2 = require_cv2(need_aruco=True)
    image = render_board_image(config, width_px, height_px, margin_px)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
