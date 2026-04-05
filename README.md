# Poultry Vision

A computer vision system for automated monitoring and behavior analysis of
poultry in cage-free environments.  Built on **YOLOv12** with colour-based
re-identification.

![Inference Preview](assets/inference_preview.gif)

## Quick start

```bash
# Install
pip install -r requirements.txt && pip install -e .

# Run on webcam
python -m src.run

# Run on a video file
python -m src.run --source samplevideos/video.mp4

# Gradio web dashboard
python -m src.run --mode dashboard
```

## Full documentation

See [docs/README.md](docs/README.md) for the complete documentation index.

| Doc | Contents |
|-----|----------|
| [docs/setup.md](docs/setup.md) | Installation and CLI reference |
| [docs/calibration.md](docs/calibration.md) | Pen calibration |
| [docs/cameras.md](docs/cameras.md) | Camera hardware and config |
| [docs/physical_environment.md](docs/physical_environment.md) | Pen setup, lighting, paint |
| [docs/training.md](docs/training.md) | Custom model training |
| [docs/abstraction_notes.md](docs/abstraction_notes.md) | Architecture decisions |

## Calibration

```bash
python -m src.calibrate --video samplevideos/video.mp4 --output pen_config.npy
```

## Pose labeler

```bash
python tools/pose_labeler/main.py
```
