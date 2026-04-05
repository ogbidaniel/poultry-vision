# Architecture & Abstraction Notes

## Module responsibilities

```
src/
├── run.py        Entry point — CLI arg parsing, mode dispatch (terminal / dashboard)
├── pipeline.py   Orchestrator — calls each module in order per frame
├── capture.py    FrameSource — wraps cv2.VideoCapture, yields BGR frames
├── detect.py     Detector — wraps YOLO.predict(), returns Detection objects
├── pose.py       PoseDetector — subclass that filters by back_neck kp confidence
├── track.py      BirdRegistry — colour ReID via HSV histogram cosine match
├── behavior.py   HenBehaviorMonitor — overlap-based FEEDING / DRINKING / IDLE
├── render.py     draw_birds() — skeleton + keypoint + label overlays
├── io_utils.py   Database (SQLite), load_config / save_config
├── calibrate.py  PenCalibrator — interactive 4-corner homography setup
├── geometry.py   Pure functions: homography, box ops, distance
└── types.py      Shared dataclasses: Detection, TrackedBird, FrameResult
```

## Data flow (per frame)

```
FrameSource         yields BGR frame
  └─ PoseDetector   predict(frame) → list[Detection]   (kp-filtered)
      └─ BirdRegistry  assign(frame, det) → bird_id
          └─ Database  write_detection(...)
          └─ BehaviorMonitor  update(hens, feeders, waterers)
              └─ draw_birds(frame, birds) → annotated
                  └─ yield FrameResult
```

## Design principles

**Flat modules, no deep inheritance.**
Each module is a small, direct implementation.  No abstract base classes,
no plugin registries, no dependency injection containers.

**Single source of truth for types.**
`src/types.py` contains all shared dataclasses.  Modules import from there;
they do not re-define their own versions.

**Config is just a dict.**
`load_config(path)` returns a plain Python dict.  Modules read keys with
`.get("key", default)`.  No Pydantic models, no nested config objects.

**DB is write-ahead only.**
The SQLite database is append-only from the hot path.  Reads happen only
for the dashboard stats table, which is off the critical path.

## ReID strategy

Each hen has paint on its `back_neck` (keypoint index 2).  The ReID crops
a 24×24 px patch in HSV space, computes an 18-bin Hue × 8-bin Saturation
histogram, normalises it to a unit vector, and matches cosine similarity
against stored signatures.

- `match_threshold = 0.82` — raise to reduce cross-bird confusion
- `min_saturation = 40` — background wood/litter has low saturation; below
  this threshold the detection is discarded as a false positive
- `smooth_alpha = 0.15` — EMA weight; higher value adapts faster to lighting
  changes but risks drifting across birds

## Behavior classification

Overlap-based only (no trajectory, no temporal smoothing beyond per-frame
assignment):

```
FEEDING  ← hen box overlaps any feeder box
DRINKING ← hen box overlaps any waterer box  (only if not FEEDING)
IDLE     ← no overlap
```

Feeding takes priority when a hen sits on both a feeder and a waterer.

To use behavior classification with a multi-class model (hen + feeder +
waterer), wire feeder/waterer detections into `Pipeline._process_frame`.
