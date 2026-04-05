# Camera Configuration

## Supported input types

| Type | Example `source` value |
|------|------------------------|
| USB camera | `0`, `1`, `2` … (device index) |
| RTSP stream | `"rtsp://192.168.1.100:8554/stream"` |
| Video file | `"samplevideos/video.mp4"` |

## Configuration file

`config/cameras.yaml` defines named camera profiles used when operating in
multi-camera mode or when overriding defaults:

```yaml
cameras:
  top:
    name: top
    type: usb
    source: "0"
    width: 1920
    height: 1080
    fps: 30
    rotation: 0          # degrees: 0, 90, 180, 270

  side:
    name: side
    type: rtsp
    source: "rtsp://100.68.7.57:8554/low_cam"
    fps: 30
    rotation: 180        # camera mounted upside-down

sync:
  tolerance_ms: 50       # max timestamp diff for frame pairing
  primary_camera: top
```

## Rotation

The `rotation` field maps to OpenCV rotation codes:

| Degrees | OpenCV code |
|---------|-------------|
| 90      | `cv2.ROTATE_90_CLOCKWISE` |
| 180     | `cv2.ROTATE_180` |
| 270     | `cv2.ROTATE_90_COUNTERCLOCKWISE` |

Pass the appropriate `cv2.ROTATE_*` constant to `FrameSource(source, rotate=...)`.

## RTSP tips

- Use TCP transport when on a lossy network: add `?tcp` to the URL or set
  `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` in the environment.
- Set `buffer_size: 1` in OpenCV to minimise latency at the cost of some
  frames.

## Testing camera access

```bash
# Quick sanity check — opens camera 0 for 5 seconds
python -c "
import cv2, time
cap = cv2.VideoCapture(0)
t = time.time()
while time.time() - t < 5:
    ok, frame = cap.read()
    if ok:
        cv2.imshow('test', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
"
```
