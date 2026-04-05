# Poultry Vision

A computer vision system for automated monitoring and behavior analysis of
poultry in cage-free environments.  Built on **YOLOv12** with colour-based
re-identification, supporting USB cameras and RTSP streams.

The system detects individual hens in real time, classifies their behaviors
(feeding, drinking, idle) and maintains a per-bird SQLite history.

---

## Quick start

```bash
# Run on webcam
python -m src.run

# Run on a video file
python -m src.run --source samplevideos/video.mp4

# Headless + save output
python -m src.run --source video.mp4 --no-display --save-video output.mp4

# Gradio web dashboard
python -m src.run --mode dashboard --source 0
```

See [setup.md](setup.md) for installation instructions.

---

## Repository layout

```
poultry-vision/
├── src/                   # runtime package
│   ├── run.py             # ← single entry point
│   ├── pipeline.py        # core orchestration
│   ├── capture.py         # camera / video input
│   ├── detect.py          # YOLO detection wrapper
│   ├── pose.py            # pose wrapper
│   ├── track.py           # ReID / tracking
│   ├── calibrate.py       # pen calibration tool
│   ├── geometry.py        # homography helpers
│   ├── behavior.py        # behavior classification
│   ├── render.py          # frame overlays
│   ├── io_utils.py        # DB + config I/O
│   └── types.py           # shared dataclasses
│
├── tools/
│   └── pose_labeler/      # PyQt6 pose annotation tool
│
├── config/
│   ├── system.yaml
│   ├── cameras.yaml
│   └── labels.yaml
│
├── models/                # .pt weight files
├── docs/                  # this folder
└── ultralytics/           # custom YOLOv12 fork
```

---

## Documentation

| File | Contents |
|------|----------|
| [setup.md](setup.md) | Installation and environment setup |
| [calibration.md](calibration.md) | Pen calibration walkthrough |
| [cameras.md](cameras.md) | Camera hardware and config |
| [physical_environment.md](physical_environment.md) | Pen setup, lighting, mounting |
| [training.md](training.md) | Model training with custom data |
| [abstraction_notes.md](abstraction_notes.md) | Architecture decisions |
