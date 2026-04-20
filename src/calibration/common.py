"""Shared helpers for calibration scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


ARTIFACT_ROOT = Path("artifacts/calibration")
EXPECTED_DIRS = [
    ARTIFACT_ROOT,
    ARTIFACT_ROOT / "boards",
    ARTIFACT_ROOT / "intrinsics",
    ARTIFACT_ROOT / "stereo",
    ARTIFACT_ROOT / "static",
    ARTIFACT_ROOT / "reports",
    ARTIFACT_ROOT / "debug",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def require_cv2(need_aruco: bool = False):
    """Import cv2 lazily so scripts can fail with actionable messages."""
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "OpenCV is not installed in this environment. "
            "Install calibration dependencies with `opencv-contrib-python`."
        ) from exc

    if need_aruco and not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not expose `cv2.aruco`. "
            "Replace `opencv-python` with `opencv-contrib-python`."
        )
    return cv2


def ensure_expected_dirs(create: bool = False) -> list[Path]:
    """Ensure the calibration artifact layout exists."""
    missing = [path for path in EXPECTED_DIRS if not path.exists()]
    if create:
        for path in missing:
            path.mkdir(parents=True, exist_ok=True)
    return missing


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file."""
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {path}")
    return data


def save_yaml(path: str | Path, data: dict) -> None:
    """Write a plain YAML file."""
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def list_image_files(directory: str | Path) -> list[Path]:
    """Return sorted image files from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    files = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {directory}")
    return files


def pair_image_files(directory_a: str | Path, directory_b: str | Path) -> list[tuple[Path, Path]]:
    """Pair images across two directories by exact filename."""
    files_a = {path.name: path for path in list_image_files(directory_a)}
    files_b = {path.name: path for path in list_image_files(directory_b)}
    shared = sorted(set(files_a) & set(files_b))
    if not shared:
        raise FileNotFoundError(
            "No shared filenames found between "
            f"{directory_a} and {directory_b}. Pairing is filename-based."
        )
    return [(files_a[name], files_b[name]) for name in shared]


def as_float_list(values: Iterable[float]) -> list[float]:
    """Convert iterables to plain float lists for YAML output."""
    return [float(value) for value in values]
