# Physical Environment

## Recommended pen layout

```
┌─────────────────────────────┐
│   Top-down camera           │  ← mounted overhead, centred on pen
│                             │
│  [feeder]       [waterer]   │  ← clearly visible from above
│                             │
│   ··  hen  ··   hen  ··     │
│      painted back-neck ★    │
│                             │
└─────────────────────────────┘
```

## Camera mounting

- **Height**: 1.5–2 m above pen floor for full pen visibility.
- **Angle**: Straight down (±10°) for accurate homography.
- **Lens**: Wide-angle (70–90° FOV) to cover the full pen without distortion.
- **Minimum resolution**: 720p; 1080p recommended.

## Lighting

- **Consistent, diffuse lighting** prevents shadow hotspots that confuse
  the detection model.
- Avoid direct sunlight through windows (creates moving shadows).
- 4000–5000 K fluorescent / LED panels work well.
- ReID depends on painted colour patches — ensure lights render colours
  accurately (CRI ≥ 80).

## Bird identification paint

Each hen has a small patch of food-safe coloured paint applied to the
**back of the neck** (keypoint index 2 — `back_neck`).

- Use different hue groups per bird (reds, blues, greens, etc.).
- Avoid yellow-orange tones — they are similar to the litter/wood background.
- Reapply every 4–6 weeks as feathers regrow.

The ReID algorithm (`src/track.py`) crops a 12-pixel radius patch around
the `back_neck` keypoint, computes a hue-saturation histogram, and matches
cosine similarity against stored signatures.

## Pen dimensions

Record the physical pen dimensions in `config/system.yaml`:

```yaml
pen:
  width_cm: 120
  height_cm: 200
```

Then run calibration to map pixel space → world space:

```bash
python -m src.calibrate --video samplevideos/video.mp4
```
