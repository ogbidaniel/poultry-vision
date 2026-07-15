"""
Train a YOLOv12 instance segmentation model on the poultry dataset
(feeder / hen / waterer), matching the configuration of the prior YOLOv8-seg run.

Usage:
    python scripts/train_seg.py
"""
from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="config/yolov12s-seg.yaml")  # no v12-seg pretrained; train from scratch
    ap.add_argument("--data", default="dataset/segment/merged_poultry_dataset/data.yaml")
    ap.add_argument("--name", default="seg_yolo12s")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,
        optimizer="auto",
        patience=50,
        close_mosaic=10,
        seed=0,
        device=0,
        workers=args.workers,
        cache=True,
        project="runs/segment",
        name=args.name,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
