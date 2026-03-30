"""
Gradio admin dashboard.

Streams live annotated video from the pipeline and shows per-bird stats.

Run:
    python -m pipeline.app
    python -m pipeline.app --config config/pipeline.yaml
    python -m pipeline.app --source 0 --model models/poultry-yolov12n-v1.pt
"""

import argparse
import threading
import time

import cv2
import numpy as np
import gradio as gr
import yaml

from .pipeline import Pipeline


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_app(config: dict) -> gr.Blocks:
    pipe   = Pipeline(config)
    source = config.get("source", 0)

    # Latest annotated frame shared between pipeline thread and Gradio
    _latest: dict = {"frame": None}
    _lock = threading.Lock()

    def _pipeline_thread():
        for annotated, _ in pipe.run(source):
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            with _lock:
                _latest["frame"] = rgb

    thread = threading.Thread(target=_pipeline_thread, daemon=True)
    thread.start()

    # ── Gradio callbacks ─────────────────────────────────────────────────────

    def stream_frame():
        """Generator — yields the latest RGB frame. Consumed by gr.Image stream."""
        while True:
            with _lock:
                frame = _latest["frame"]
            if frame is not None:
                yield frame
            time.sleep(0.033)  # ~30 fps cap

    def get_stats():
        """Return bird stats as a list-of-lists for gr.Dataframe."""
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
                f"({r['center_x']:.0f}, {r['center_y']:.0f})"
                    if r["center_x"] else "—",
                f"{now - r['last_seen']:.1f}s ago"
                    if r["last_seen"] else "—",
            ]
            for r in rows
        ]

    # ── Layout ───────────────────────────────────────────────────────────────

    with gr.Blocks(title="Poultry Vision", theme=gr.themes.Monochrome()) as app:
        gr.Markdown("## Poultry Vision — Live Monitor")

        with gr.Row():
            with gr.Column(scale=3):
                live_feed = gr.Image(
                    label="Live Feed",
                    streaming=True,
                    height=480,
                )

            with gr.Column(scale=2):
                stats_table = gr.Dataframe(
                    headers=["ID", "Detections", "Avg Conf",
                              "Frames", "Position", "Last Seen"],
                    label="Bird Stats",
                    interactive=False,
                )
                gr.Button("Refresh").click(fn=get_stats, outputs=stats_table)

        # Auto-refresh stats every 3 s
        app.load(fn=get_stats, outputs=stats_table, every=3)
        # Stream live frames
        live_feed.stream(fn=stream_frame, inputs=[], outputs=live_feed)

    return app


def main():
    parser = argparse.ArgumentParser(description="Poultry Vision admin dashboard")
    parser.add_argument("--config", default="config/pipeline.yaml")
    parser.add_argument("--source", default=None,
                        help="Camera index, file path, or RTSP URL")
    parser.add_argument("--model",  default=None,
                        help="Path to YOLO model weights")
    parser.add_argument("--port",   type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    if args.source is not None:
        config["source"] = (
            int(args.source) if args.source.isdigit() else args.source
        )
    if args.model is not None:
        config["model_path"] = args.model
    if args.port is not None:
        config["port"] = args.port

    app = build_app(config)
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=config.get("port", 7860),
    )


if __name__ == "__main__":
    main()
