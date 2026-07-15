"""
Train the YOLOv12s pose model on the augmented pose dataset.

Reproduces the configuration of the prior best pose run (from-scratch YOLOv12s,
AdamW, keypoint-aware flips via data.yaml flip_idx) on the freshly augmented
dataset.

Usage:
    python scripts/train_pose.py
"""
from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="config/yolov12s-pose-hen.yaml")
    ap.add_argument("--data", default="dataset/pose/pose-dataset-merged-aug-split/data.yaml")
    ap.add_argument("--name", default="hen_pose_yolo12s_aug_v3")
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)  # 2 is stable on Windows
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        patience=50,
        close_mosaic=10,
        cos_lr=False,
        seed=42,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=8.0, translate=0.05, scale=0.25,
        fliplr=0.5, mosaic=0.25,
        device=0,
        workers=args.workers,
        cache=True,
        project="runs/pose",
        name=args.name,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
