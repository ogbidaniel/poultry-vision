"""
Poultry detection viewer.

Runs YOLO detection + ByteTrack on one or two time-aligned videos and
displays them side by side in a single window.

Controls
--------
Space  : pause / resume
q      : quit

Usage — single video
--------------------
    python scripts/detect.py --source dataset/treatment1/cam0/20260605_141702.mp4

Usage — dual camera (side by side, time-aligned)
-------------------------------------------------
    python scripts/detect.py \
        --source  dataset/treatment1/cam0/20260605_141702.mp4 \
        --source2 dataset/treatment1/cam1/20260605_141702.mp4

Specify a different model
-------------------------
    python scripts/detect.py --source ... --model models/det0-yolo12s.pt
"""

import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO

_DISPLAY_H = 720


def _resize_h(img: np.ndarray, h: int) -> np.ndarray:
    scale = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), h))


def _make_annotators():
    return (
        sv.TraceAnnotator(thickness=2, trace_length=60),
        sv.BoxAnnotator(thickness=2),
        sv.LabelAnnotator(text_scale=0.5, text_thickness=1),
    )


def _annotate(frame, results, model, tracker, trace_ann, box_ann, label_ann):
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    labels = [
        f"{model.names[cid]} #{tid}"
        for cid, tid in zip(detections.class_id, detections.tracker_id)
    ]
    ann = trace_ann.annotate(scene=frame.copy(), detections=detections)
    ann = box_ann.annotate(scene=ann, detections=detections)
    ann = label_ann.annotate(scene=ann, detections=detections, labels=labels)
    return ann, len(detections)


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 0, 0), 1, cv2.LINE_AA)
    return img


def main(source1: str, source2: str | None, model_path: str, classes) -> None:
    model = YOLO(model_path)

    tracker1 = sv.ByteTrack()
    trace1, box1, label1 = _make_annotators()

    tracker2 = sv.ByteTrack() if source2 else None
    trace2, box2, label2 = (_make_annotators() if source2 else (None, None, None))

    def _open(src: str):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {src}")
        return cap

    cap1 = _open(source1)
    cap2 = _open(source2) if source2 else None

    dual = cap2 is not None
    win  = "Poultry Detection" + (" — dual view" if dual else "")
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused    = False
    last_display = None

    print(f"Model : {model_path}")
    print(f"Cam 0 : {source1}")
    if dual:
        print(f"Cam 1 : {source2}")
    print("Controls: [space]=pause  [q]=quit")

    while True:
        if not paused:
            ret1, frame1 = cap1.read()
            if not ret1:
                print("End of cam0 video.")
                break

            res1 = model(frame1, classes=classes, verbose=False)[0]
            ann1, n1 = _annotate(frame1, res1, model, tracker1, trace1, box1, label1)
            _add_label(ann1, f"CAM0  f={frame_idx}  n={n1}")

            if dual:
                ret2, frame2 = cap2.read()
                if not ret2:
                    print("End of cam1 video.")
                    break
                res2 = model(frame2, classes=classes, verbose=False)[0]
                ann2, n2 = _annotate(frame2, res2, model, tracker2, trace2, box2, label2)
                _add_label(ann2, f"CAM1  f={frame_idx}  n={n2}")
                display = np.hstack([_resize_h(ann1, _DISPLAY_H),
                                     _resize_h(ann2, _DISPLAY_H)])
                print(f"frame {frame_idx:5d}  |  cam0: {n1}  cam1: {n2}")
            else:
                display = _resize_h(ann1, _DISPLAY_H)
                print(f"frame {frame_idx:5d}  |  cam0: {n1}")

            last_display = display
            frame_idx += 1

        if last_display is not None:
            cv2.imshow(win, last_display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap1.release()
    if cap2:
        cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poultry detection viewer")
    parser.add_argument("--source",  required=True,
                        help="Cam0 video path, file or USB index.")
    parser.add_argument("--source2", default=None,
                        help="Cam1 video path for dual-view (optional).")
    parser.add_argument("--model",   default="models/det0-yolo12s.pt",
                        help="YOLO model path.")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="Class IDs to detect (default: all).")
    args = parser.parse_args()
    main(args.source, args.source2, args.model, args.classes)
