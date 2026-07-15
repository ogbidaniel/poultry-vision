# Poultry Vision — multi-view laying-hen monitoring

A low-cost, multi-view computer-vision system for continuous behavior monitoring
of laying hens, targeting a Raspberry Pi 5 + Hailo edge deployment. Two
synchronized cameras (overhead + lateral) are fused into a shared metric floor
frame; the five color-painted hens are tracked by **color identity**, and posture
is recovered with a **YOLOv12 pose** model run on per-hen crops. Built on a local
**YOLOv12** fork (`ultralytics/`).

## Pipeline at a glance
```
cam0 (top)  ─┐                              ┌─ color identity (5 fixed slots)
             ├─ detect + track per camera ──┤
cam1 (side) ─┘     │                         └─ floor-gating (reject off-floor)
                   ▼
        homography borrow-solve  ──► shared metric floor frame
                   │
                   └─ pose on hen crops ──► keypoints ──► behavior (planned)
```

## Repository structure
| Path | Contents |
|------|----------|
| `src/` | Runtime modules: `floor.py` (calibration projection + gating), `identity.py` (color-as-identity tracker), `color_marker.py`, `multiview.py` (sync), `geometry.py`, `pen_overlay.py`, detect/track/pose helpers |
| `scripts/track_pen.py` | Dual-camera detect + track + color identity + live floor map |
| `calibrate_corners.py` | Interactive floor calibration (overlap-tie-point **borrow-solve**); `--recompute` rebuilds from saved clicks |
| `scripts/` | Data + training tooling: `build_hen_crops.py`, `extract_pose_crops.py`, `extract_crops_from_regions.py`, `merge_pose_crops.py`, `augment_pose_dataset.py`, `train_pose.py`, `train_seg.py`, `calibrate_intrinsics.py`, figure generators |
| `config/` | Model configs: `yolov12s-pose-hen.yaml`, `yolov12s-seg.yaml` |
| `experiments/` | Notebooks the researcher runs: `crop_hens/`, `eval_pose_crops/`, `mine_hard_poses/` (see its README) |
| `MDPI_Poultry_Multi_View_Camera_2026/` | The paper (LaTeX source: `main.tex`, `assets/`, `figures/`, `Definitions/`, `ref.bib`) |
| `research/` | Reference papers + notes for the literature review (gitignored; see its README) |
| `pen_config.json` | Pen geometry, camera positions, homographies, coordinate convention |
| `dataset/`, `runs/`, `models/` | Datasets, training runs, weights (gitignored) |

## Datasets
- **Segmentation/detection** (`dataset/segment/merged_poultry_dataset`): 947 images,
  3 classes (feeder/hen/waterer), labeled in Roboflow.
- **Pose** (`dataset/pose/...`): hand-taken photos + extracted lateral frames,
  hen-only with a 10-keypoint schema; extended with per-hen **crops** that match
  the crop-based inference path.

## Common workflows
```bash
# Floor calibration (click points once per pen), then recompute if needed
python calibrate_corners.py --config pen_config.json --frame-idx 30
python calibrate_corners.py --recompute

# Live dual-camera detection + tracking + color identity
python scripts/track_pen.py --config pen_config.json --pair 20260605_141702 --walls

# Pose crop pipeline: extract -> merge -> augment -> train
python scripts/extract_pose_crops.py
python scripts/build_pose_v5.py
python scripts/augment_pose_dataset.py --src <merged> --dst <merged-aug>
python scripts/train_pose.py --data <merged-aug>/data.yaml --name hen_pose_yolo12s_aug_v5
python scripts/train_seg.py                      # YOLOv12 segmentation
```

## Installation
See `docs/setup.md` for per-platform commands. In brief:
```bash
pip install -r requirements.txt
pip install torch torchvision
pip install -e .        # the bundled YOLOv12 fork
```

## Models
The deployed perception models are **YOLOv12** throughout: instance segmentation
(top-down) and pose (lateral, run on crops). Calibration of the C920 intrinsics
for accurate Z/wall overlay is documented in `docs/calibration_checkerboard.md`.
