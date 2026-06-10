# Setup & Installation

## Goals

This repository should run on:

- macOS
- Linux
- Windows
- CPU-only systems
- NVIDIA GPU systems when CUDA is available

The bundled `ultralytics/` fork already falls back when FlashAttention is not
available, so **FlashAttention is optional** and should not be part of the
default install path.

## Prerequisites

- Python 3.10 to 3.12
- `pip`
- Conda or venv (recommended)
- Optional: NVIDIA drivers + CUDA-compatible PyTorch wheel for GPU acceleration

## 1. Clone the repository

```bash
git clone https://github.com/ogbidaniel/poultry-vision.git
cd poultry-vision
```

## 2. Create and activate an environment

### Conda

```bash
conda create --name poultry-vision python=3.11
conda activate poultry-vision
```

### venv

```bash
python -m venv .venv
```

Activate with:

- macOS / Linux: `source .venv/bin/activate`
- Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
- Windows cmd: `.venv\\Scripts\\activate.bat`

## 3. Install base dependencies

Install the shared cross-platform dependencies first:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 4. Install PyTorch for your system

Install `torch` and `torchvision` separately so you can choose CPU or GPU builds
that match your OS and hardware.

### Option A: macOS (Intel or Apple Silicon)

```bash
pip install torch torchvision
```

Notes:
- PyTorch on Apple Silicon can use Metal acceleration automatically when supported.
- FlashAttention is not required on macOS.

### Option B: Linux CPU-only

```bash
pip install torch torchvision
```

### Option C: Linux with NVIDIA GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If your system requires a different CUDA wheel, install the matching PyTorch
build for your driver/runtime instead.

### Option D: Windows CPU-only

```bash
pip install torch torchvision
```

### Option E: Windows with NVIDIA GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If needed, replace `cu121` with the CUDA build that matches your environment.

## 5. Install the bundled YOLOv12 fork

```bash
pip install -e .
```

## 6. Optional acceleration packages

### FlashAttention

`flash-attn` is optional and primarily useful on supported NVIDIA Linux setups.
Do **not** install it by default on macOS, Windows, or CPU-only systems.

The code already falls back to scaled dot-product attention when FlashAttention
is unavailable.

### ONNX Runtime

For ONNX export or inference:

#### CPU version

```bash
pip install onnx onnxruntime onnxslim
```

#### NVIDIA GPU version

```bash
pip install onnx onnxruntime-gpu onnxslim
```

The runtime will fall back to CPU if CUDA execution providers are unavailable.

## 7. Verify your installation

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
python -c "from ultralytics import YOLO; print('ultralytics import ok')"
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
