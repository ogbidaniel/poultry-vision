#!/usr/bin/env python3
"""
Capture synchronized frame pairs from two cameras using config/cameras.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2

from src.capture import FrameSource
from src.multiview import (
    RingBufferSynchronizer,
    load_camera_specs,
    read_timestamped_frame,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture synchronized multiview sample frames.")
    parser.add_argument("--camera-config", default="config/cameras.yaml")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument("--output-dir", default="samples/sampleimages/sync")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cameras, sync = load_camera_specs(args.camera_config)
    primary_name = sync.primary_camera
    secondary_name = next(name for name in cameras if name != primary_name)
    primary = FrameSource(cameras[primary_name])
    secondary = FrameSource(cameras[secondary_name])
    synchronizer = RingBufferSynchronizer(sync)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_diffs: list[float] = []
    saved = 0
    started = time.monotonic()
    primary_id = 0
    secondary_id = 0

    primary.open()
    secondary.open()
    try:
        while time.monotonic() - started < args.duration:
            frame_primary = read_timestamped_frame(primary, primary_name, primary_id)
            primary_id += 1
            if frame_primary is not None:
                synchronizer.add_frame(frame_primary)

            frame_secondary = read_timestamped_frame(secondary, secondary_name, secondary_id)
            secondary_id += 1
            if frame_secondary is not None:
                synchronizer.add_frame(frame_secondary)

            pair = synchronizer.get_pair(primary_name, secondary_name)
            if pair is None:
                continue

            pair_diffs.append(pair.diff_ms)
            if saved < args.sample_count:
                cv2.imwrite(str(output_dir / f"{saved:02d}_{primary_name}.jpg"), pair.primary.frame)
                cv2.imwrite(str(output_dir / f"{saved:02d}_{secondary_name}.jpg"), pair.secondary.frame)
                with open(output_dir / f"{saved:02d}_meta.json", "w") as handle:
                    json.dump(
                        {
                            "pair_id": pair.pair_id,
                            "diff_ms": pair.diff_ms,
                            "primary_camera": pair.primary.camera_name,
                            "secondary_camera": pair.secondary.camera_name,
                            "primary_frame_id": pair.primary.frame_id,
                            "secondary_frame_id": pair.secondary.frame_id,
                        },
                        handle,
                        indent=2,
                    )
                saved += 1

        summary = {
            "primary_camera": primary_name,
            "secondary_camera": secondary_name,
            "pairs": len(pair_diffs),
            "saved_pairs": saved,
            "mean_diff_ms": round(statistics.mean(pair_diffs), 2) if pair_diffs else None,
            "median_diff_ms": round(statistics.median(pair_diffs), 2) if pair_diffs else None,
            "max_diff_ms": round(max(pair_diffs), 2) if pair_diffs else None,
        }
        with open(output_dir / "summary.json", "w") as handle:
            json.dump(summary, handle, indent=2)
        print(json.dumps(summary, indent=2))
    finally:
        primary.release()
        secondary.release()


if __name__ == "__main__":
    main()
