# Pen Calibration

Before running inference you must define the pen boundaries so that detections
can be filtered to the area of interest and world coordinates computed.

## Interactive calibration

```bash
python -m src.calibrate --video samplevideos/poultry-vid-01.mp4 --output pen_config.npy
```

This opens an OpenCV window showing the first frame of the video.

**Click the 4 corners of the pen in order:**

1. Top-left (TL)
2. Top-right (TR)
3. Bottom-right (BR)
4. Bottom-left (BL)

### Controls

| Key | Action |
|-----|--------|
| Left click | Place corner point |
| `r` | Reset all points |
| `s` | Save and exit |
| `q` | Cancel |

The tool auto-saves when the 4th point is placed.

## Output files

| File | Contents |
|------|----------|
| `pen_config.npy` | 4 × 2 float32 numpy array (pixel corners) |
| `pen_config.yaml` | Same corners in human-readable YAML |

## Using the calibration in the pipeline

The pipeline reads `pen_config.npy` automatically when the path is set in
`config/system.yaml`:

```yaml
pen:
  calibration_file: pen_config.npy
  width_cm: 120
  height_cm: 200
```

## Homography

The pen corners define a projective transform from pixel space to world space
(centimetres).  This is computed with `src.geometry.get_homography_matrix`.

```python
from src.geometry import get_homography_matrix, transform_point
import numpy as np

corners = np.load("pen_config.npy")
H = get_homography_matrix(corners, pen_width_cm=120, pen_height_cm=200)

# Convert a pixel centre to world cm
world_pos = transform_point((cx, cy), H)
```
