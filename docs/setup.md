# Setup & Installation

## Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional — for GPU inference)
- Conda (recommended)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ogbidaniel/poultry-vision.git
cd poultry-vision
```

### 2. Create conda environment

```bash
conda create --name poultry-vision python=3.12
conda activate poultry-vision
```

### 3. Install dependencies

```bash
# Runtime dependencies
pip install -r requirements.txt

# Install the custom Ultralytics fork (required for YOLOv12)
pip install -e .
```

## Running inference

```bash
# USB camera (device 0)
python -m src.run --source 0

# Video file
python -m src.run --source samplevideos/video.mp4

# RTSP stream
python -m src.run --source "rtsp://ip:port/stream"

# Headless + save annotated video
python -m src.run --source video.mp4 --no-display --save-video out.mp4

# Gradio web dashboard
python -m src.run --mode dashboard
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` / `-s` | `0` | Camera index, file path, or RTSP URL |
| `--model` / `-m` | `models/poultry-yolov12n-v1.pt` | Model weights |
| `--config` / `-c` | `config/system.yaml` | Config file |
| `--mode` | `terminal` | `terminal` or `dashboard` |
| `--no-display` | false | Disable OpenCV window |
| `--save-video` | — | Output video file path |
| `--port` | `7860` | Gradio port (dashboard mode) |

## Keyboard shortcuts (terminal mode)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |

## Pose labeler

```bash
python tools/pose_labeler/main.py
```
