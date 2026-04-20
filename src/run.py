"""
Single entry point for realtime and offline pipeline runs.

Modes
-----
terminal (default):
    OpenCV window showing the annotated live feed.
    Press 'q' to quit, 's' to save a screenshot.

dashboard:
    Gradio web dashboard with live video stream and bird stats table.
    Accessible at http://localhost:7860 by default.

Usage::

    # terminal mode (webcam)
    python -m src.run

    # terminal mode (video file)
    python -m src.run --source samplevideos/video.mp4

    # headless (no display window), save annotated video
    python -m src.run --source video.mp4 --no-display --save-video output.mp4

    # Gradio dashboard
    python -m src.run --mode dashboard --source 0

    # Custom config
    python -m src.run --config config/system.yaml --model models/poultry-yolov12n-v1.pt
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

from .capture import FrameSource
from .io_utils import load_config, load_merged_config
from .multiview import load_camera_specs
from .pipeline import Pipeline


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.run",
        description="Poultry Vision — realtime / offline inference pipeline",
    )
    p.add_argument("--source", "-s", default="0",
                   help="Camera index, video file path, or RTSP URL  (default: 0)")
    p.add_argument("--camera", choices=["top", "side"],
                   help="Named camera from config/cameras.yaml")
    p.add_argument("--camera-config", default="config/cameras.yaml",
                   help="Camera YAML config file  (default: config/cameras.yaml)")
    p.add_argument("--model", "-m", default="models/poultry-yolov12n-v1.pt",
                   help="Path to YOLO model weights")
    p.add_argument("--config", "-c", default="config/system.yaml",
                   help="YAML config file  (default: config/system.yaml)")
    p.add_argument("--mode", choices=["terminal", "dashboard"],
                   default="terminal",
                   help="Run mode: terminal (OpenCV) or dashboard (Gradio)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable OpenCV display window (terminal mode only)")
    p.add_argument("--save-video", metavar="PATH",
                   help="Save annotated output to a video file")
    p.add_argument("--port", type=int, default=7860,
                   help="Gradio dashboard port  (default: 7860)")
    return p


def _resolve_config(args: argparse.Namespace) -> dict:
    """Load config file and override with CLI arguments."""
    config = load_merged_config(args.config, args.camera_config)

    # CLI arguments take precedence
    if args.model:
        config["model_path"] = args.model
    if "model_path" not in config:
        config["model_path"] = "models/poultry-yolov12n-v1.pt"

    # source is resolved separately in each mode runner
    return config


# ── Terminal mode ─────────────────────────────────────────────────────────────

def _run_terminal(source: int | str | FrameSource, config: dict, args: argparse.Namespace) -> None:
    """
    OpenCV display loop.  Press 'q' to quit, 's' to save a screenshot.
    """
    pipe   = Pipeline(config)
    writer = None

    try:
        for result in pipe.run(source):
            frame = result.annotated

            if args.save_video and writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, 30, (w, h))

            if writer is not None:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Poultry Vision", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    ts  = time.strftime("%Y%m%d_%H%M%S")
                    out = f"screenshot_{ts}.jpg"
                    cv2.imwrite(out, frame)
                    print(f"Screenshot saved → {out}")

    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


# ── Dashboard mode ────────────────────────────────────────────────────────────

def _run_dashboard(source: int | str | FrameSource, config: dict, port: int) -> None:
    """
    Gradio web dashboard with live video stream and per-bird stats table.
    """
    try:
        import gradio as gr
    except ImportError:
        print("ERROR: gradio is not installed.  Run: pip install gradio")
        sys.exit(1)

    pipe    = Pipeline(config)
    _latest: dict = {"frame": None}
    _lock   = threading.Lock()

    def _pipeline_thread() -> None:
        import cv2 as _cv2
        for result in pipe.run(source):
            rgb = _cv2.cvtColor(result.annotated, _cv2.COLOR_BGR2RGB)
            with _lock:
                _latest["frame"] = rgb

    thread = threading.Thread(target=_pipeline_thread, daemon=True)
    thread.start()

    def stream_frame():
        while True:
            with _lock:
                frame = _latest["frame"]
            if frame is not None:
                yield frame
            time.sleep(0.033)

    def get_stats():
        rows = pipe.db.get_bird_stats()
        if not rows:
            return []
        now = time.time()
        return [
            [
                r["id"],
                r["detection_count"],
                f"{r['avg_conf']:.2f}" if r["avg_conf"] else "—",
                r["frame_count"] or 0,
                f"({r['center_x']:.0f}, {r['center_y']:.0f})" if r["center_x"] else "—",
                f"{now - r['last_seen']:.1f}s ago" if r["last_seen"] else "—",
            ]
            for r in rows
        ]

    with gr.Blocks(title="Poultry Vision", theme=gr.themes.Monochrome()) as app:
        gr.Markdown("## Poultry Vision — Live Monitor")

        with gr.Row():
            with gr.Column(scale=3):
                live_feed = gr.Image(label="Live Feed", streaming=True, height=480)
            with gr.Column(scale=2):
                stats_table = gr.Dataframe(
                    headers=["ID", "Detections", "Avg Conf", "Frames", "Position", "Last Seen"],
                    label="Bird Stats",
                    interactive=False,
                )
                gr.Button("Refresh").click(fn=get_stats, outputs=stats_table)

        app.load(fn=get_stats, outputs=stats_table, every=3)
        live_feed.stream(fn=stream_frame, inputs=[], outputs=live_feed)

    app.queue()
    app.launch(server_name="0.0.0.0", server_port=port)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    config = _resolve_config(args)

    # Resolve source type
    if args.camera:
        cameras, _ = load_camera_specs(args.camera_config)
        source: int | str | FrameSource = FrameSource(cameras[args.camera])
    else:
        raw = args.source
        source = int(raw) if raw.isdigit() else raw

    if args.mode == "dashboard":
        _run_dashboard(source, config, args.port)
    else:
        _run_terminal(source, config, args)


if __name__ == "__main__":
    main()
