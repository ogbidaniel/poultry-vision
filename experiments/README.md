# Experiments

Self-contained **notebooks** that validate design decisions for the multi-view
poultry monitoring system before they are committed to the runtime pipeline or
the paper. Each lives in its own folder and is run interactively by the lead
researcher (not auto-executed). Each notebook resolves the repo root by walking
up to `pen_config.json`, so it runs correctly from anywhere.

## Research outcomes this work targets
1. **Accurate per-bird posture** in the lateral view, robust to small apparent
   size and occlusion.
2. **Reliable individual identity** for the five painted hens, without a heavy
   re-identification network.
3. **A lightweight, real-time edge pipeline** (Raspberry Pi 5 + Hailo) running at
   most two perception models.
4. **Behavior inference** (feeding, drinking, locomotion, resting, laying) from
   the fused multi-view state.

## Experiments

### `crop_hens/` — detect → crop → pose on real video frames
Samples corresponding cam0/cam1 frames, detects hens, crops each confident
detection (0.10 padding), and runs the pose model on the crops. Lets us see, on
real 640×480 video, how detection and crop-based pose behave end to end.
Advances outcomes (1) and (3).

### `eval_pose_crops/` — did the labeled crops improve crop-pose?
Compares pose models (v3/v4/v5…) on the held-out crop validation set: detection
recall, PCK@0.05/0.10, and keypoint confidence, split by camera. The
apples-to-apples test of whether crop labeling improved crop-pose. Advances (1).

### `mine_hard_poses/` — hard-example mining for labeling
Scores a candidate crop pool with the current best pose model and selects the
hardest examples to label next: Method 1 (low mean keypoint confidence) and
Method 2 (limb-geometry outliers vs the GT mean). Curates ~300 crops to
`dataset/pose/pose-crops-to-label/`. Includes `candidate_cam1_videos.csv`, the
spread-out source-video list. Advances (1).
