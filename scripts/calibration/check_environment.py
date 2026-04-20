#!/usr/bin/env python3
"""Validate the local calibration environment and artifact layout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.common import EXPECTED_DIRS, ensure_expected_dirs, require_cv2


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the calibration environment.")
    parser.add_argument(
        "--create-dirs",
        action="store_true",
        help="Create the expected calibration artifact directories if they do not exist.",
    )
    args = parser.parse_args()

    missing = ensure_expected_dirs(create=args.create_dirs)
    print("Expected artifact directories:")
    for path in EXPECTED_DIRS:
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")

    try:
        cv2 = require_cv2(need_aruco=True)
    except RuntimeError as exc:
        print(f"\nEnvironment check failed: {exc}")
        print("\nInstall guidance:")
        print("  pip uninstall opencv-python")
        print("  pip install opencv-contrib-python")
        return 1

    print("\nOpenCV:")
    print(f"  version: {cv2.__version__}")
    print(f"  has aruco: {hasattr(cv2, 'aruco')}")
    print(f"  has CharucoBoard: {hasattr(cv2.aruco, 'CharucoBoard')}")
    print(f"  has CharucoDetector: {hasattr(cv2.aruco, 'CharucoDetector')}")

    if missing and not args.create_dirs:
        print("\nSome expected directories are missing. Re-run with `--create-dirs` to create them.")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
