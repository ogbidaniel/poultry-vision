"""
I/O utilities: configuration loading, session management, and the SQLite
live database for per-bird detection history.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Config helpers ────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_merged_config(*paths: str | Path) -> dict:
    """Load multiple YAML files and merge them shallowly from left to right."""
    merged: dict = {}
    for path in paths:
        file_path = Path(path)
        if not file_path.exists():
            continue
        data = load_config(file_path)
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
    return merged


def save_config(config: dict, path: str | Path) -> None:
    """Write a dict to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ── SQLite live database ──────────────────────────────────────────────────────

class Database:
    """
    SQLite database for per-bird detection history and rolling stats.

    Tables
    ------
    birds       id, registered_at, detection_count
    detections  per-frame records (ts, bird_id, conf, bbox, keypoints JSON)
    scores      rolling stats per bird (avg_conf, frame_count, mean position)
    """

    def __init__(self, path: str = "poultry.db") -> None:
        self.path  = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS birds (
                id               INTEGER PRIMARY KEY,
                registered_at    REAL    NOT NULL,
                detection_count  INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS detections (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                ts       REAL    NOT NULL,
                bird_id  INTEGER NOT NULL,
                conf     REAL    NOT NULL,
                box_x1   REAL, box_y1 REAL, box_x2 REAL, box_y2 REAL,
                keypoints TEXT,
                FOREIGN KEY (bird_id) REFERENCES birds(id)
            );

            CREATE TABLE IF NOT EXISTS scores (
                bird_id      INTEGER PRIMARY KEY,
                last_seen    REAL,
                avg_conf     REAL    NOT NULL DEFAULT 0.0,
                frame_count  INTEGER NOT NULL DEFAULT 0,
                center_x     REAL    NOT NULL DEFAULT 0.0,
                center_y     REAL    NOT NULL DEFAULT 0.0,
                FOREIGN KEY (bird_id) REFERENCES birds(id)
            );

            CREATE INDEX IF NOT EXISTS idx_det_bird ON detections(bird_id);
            CREATE INDEX IF NOT EXISTS idx_det_ts   ON detections(ts);
        """)
        self._conn.commit()

    # ── Writes ────────────────────────────────────────────────────────────────

    def write_detection(
        self,
        ts: float,
        bird_id: int,
        box: np.ndarray,
        conf: float,
        kps: np.ndarray | None = None,
    ) -> None:
        """Record a single detection and update rolling scores."""
        self._conn.execute(
            "INSERT OR IGNORE INTO birds (id, registered_at) VALUES (?, ?)",
            (bird_id, ts),
        )
        self._conn.execute(
            "UPDATE birds SET detection_count = detection_count + 1 WHERE id = ?",
            (bird_id,),
        )

        kps_json = json.dumps(kps.tolist()) if kps is not None else None
        self._conn.execute(
            """INSERT INTO detections
               (ts, bird_id, conf, box_x1, box_y1, box_x2, box_y2, keypoints)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, bird_id, conf, *map(float, box[:4]), kps_json),
        )

        cx = float((box[0] + box[2]) / 2)
        cy = float((box[1] + box[3]) / 2)
        self._conn.execute(
            """INSERT INTO scores (bird_id, last_seen, avg_conf, frame_count, center_x, center_y)
               VALUES (?, ?, ?, 1, ?, ?)
               ON CONFLICT(bird_id) DO UPDATE SET
                   last_seen   = excluded.last_seen,
                   avg_conf    = (avg_conf * frame_count + excluded.avg_conf)
                                 / (frame_count + 1),
                   frame_count = frame_count + 1,
                   center_x    = (center_x * frame_count + excluded.center_x)
                                 / (frame_count + 1),
                   center_y    = (center_y * frame_count + excluded.center_y)
                                 / (frame_count + 1)""",
            (bird_id, ts, conf, cx, cy),
        )
        self._conn.commit()

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get_bird_stats(self) -> list[dict[str, Any]]:
        """Return live stats for all birds, ordered by ID."""
        rows = self._conn.execute("""
            SELECT
                b.id,
                b.registered_at,
                b.detection_count,
                s.last_seen,
                s.avg_conf,
                s.frame_count,
                s.center_x,
                s.center_y
            FROM birds b
            LEFT JOIN scores s ON b.id = s.bird_id
            ORDER BY b.id
        """).fetchall()
        return [dict(r) for r in rows]

    def get_recent_detections(self, bird_id: int, limit: int = 50) -> list[dict]:
        """Return the most recent detections for a specific bird."""
        rows = self._conn.execute(
            """SELECT ts, conf, box_x1, box_y1, box_x2, box_y2
               FROM detections WHERE bird_id = ?
               ORDER BY ts DESC LIMIT ?""",
            (bird_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()
