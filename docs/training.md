# Model Training

## Dataset preparation

### 1. Extract frames for annotation

```bash
python -c "
from src.tools.extract_frames import extract_frames
extract_frames(
    video_path='samplevideos/video.mp4',
    output_dir='dataset/frames',
    interval=30,        # 1 frame/second at 30 fps
)
"
```

### 2. Annotate with PoseLabeler

```bash
python tools/pose_labeler/main.py
```

Create a new project, point it at the extracted frames directory, and
annotate each hen with:
- Bounding box
- 10 keypoints per the poultry pose schema (`config/labels.yaml`)

Export to YOLO format using **File → Export**.

## Training a pose model

```bash
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")   # start from a pretrained pose checkpoint
model.train(
    data="dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,         # GPU
)
```

The `data.yaml` is produced by the PoseLabeler export and looks like:

```yaml
path: dataset/
train: images/train
val:   images/val
kpt_shape: [10, 3]
names:
  0: hen
```

## Model files

| Model | File | Classes |
|-------|------|---------|
| Pose detection | `models/poultry-yolov12n-v1.pt` | hen (10 kp) |

Pass the trained weights with:

```bash
python -m src.run --model models/my-trained-model.pt
```

## Evaluation

See `docs/notebooks/01_model_evaluation.ipynb` for mAP, precision, recall,
and speed benchmarks.
