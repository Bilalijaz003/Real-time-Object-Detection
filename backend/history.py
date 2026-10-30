"""
Detection history, stored in a local SQLite database (backend/history.db).

A "scan" is one detection job:
  - image   : one uploaded photo
  - camera  : one live webcam session (Start -> Stop)
  - video   : one video file session

For every scan we store, per object class:
  - objects      : how many of that object were found. For an image this is
                   the count in the photo; for camera/video it's the most
                   seen at the same time, so one person walking past for a
                   minute counts as 1, not as hundreds of frames.
  - frames_seen  : in how many frames the object appeared

Totals across all scans give "how many times" each object was detected.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "history.db"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source        TEXT NOT NULL CHECK (source IN ('image', 'camera', 'video')),
    name          TEXT,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    frames        INTEGER NOT NULL DEFAULT 0,
    total_objects INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS scan_objects (
    scan_id     INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    class       TEXT NOT NULL,
    objects     INTEGER NOT NULL DEFAULT 0,
    frames_seen INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scan_id, class)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _lock, _connect() as conn:
        conn.executescript(SCHEMA)


def start_scan(source: str, name: str | None = None) -> int:
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO scans (source, name, started_at) VALUES (?, ?, ?)",
            (source, name, _now()))
        return cur.lastrowid


def scan_exists(scan_id: int) -> bool:
    with _lock, _connect() as conn:
        row = conn.execute("SELECT ended_at FROM scans WHERE id = ?",
                           (scan_id,)).fetchone()
        return row is not None and row["ended_at"] is None


def add_frame(scan_id: int, counts: dict[str, int]) -> None:
    """Record one processed frame's per-class counts into a scan."""
    with _lock, _connect() as conn:
        conn.execute("UPDATE scans SET frames = frames + 1 WHERE id = ?",
                     (scan_id,))
        for cls, n in counts.items():
            conn.execute(
                """INSERT INTO scan_objects (scan_id, class, objects, frames_seen)
                   VALUES (?, ?, ?, 1)
                   ON CONFLICT(scan_id, class) DO UPDATE SET
                       objects = MAX(objects, excluded.objects),
                       frames_seen = frames_seen + 1""",
                (scan_id, cls, n))
        conn.execute(
            """UPDATE scans SET total_objects =
                   (SELECT COALESCE(SUM(objects), 0) FROM scan_objects
                    WHERE scan_id = ?)
               WHERE id = ?""", (scan_id, scan_id))


def end_scan(scan_id: int) -> None:
    with _lock, _connect() as conn:
        conn.execute("UPDATE scans SET ended_at = ? WHERE id = ? "
                     "AND ended_at IS NULL", (_now(), scan_id))
        # Drop sessions where nothing was processed.
        conn.execute("DELETE FROM scans WHERE id = ? AND frames = 0",
                     (scan_id,))


def record_image(name: str | None, counts: dict[str, int]) -> int:
    scan_id = start_scan("image", name)
    add_frame(scan_id, counts)
    end_scan(scan_id)
    return scan_id


def list_scans(limit: int = 20) -> list[dict]:
    with _lock, _connect() as conn:
        scans = conn.execute(
            "SELECT * FROM scans WHERE frames > 0 "
            "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for s in scans:
            objs = conn.execute(
                "SELECT class, objects, frames_seen FROM scan_objects "
                "WHERE scan_id = ? ORDER BY objects DESC, class", (s["id"],)
            ).fetchall()
            out.append({**dict(s), "objects": [dict(o) for o in objs]})
        return out


def stats() -> dict:
    with _lock, _connect() as conn:
        per_class = conn.execute(
            """SELECT class,
                      COUNT(*)          AS times_detected,
                      SUM(objects)      AS total_objects,
                      SUM(frames_seen)  AS frames_seen
               FROM scan_objects
               GROUP BY class
               ORDER BY times_detected DESC, total_objects DESC, class"""
        ).fetchall()
        totals = conn.execute(
            """SELECT COUNT(*) AS scans,
                      COALESCE(SUM(frames), 0) AS frames,
                      COALESCE(SUM(total_objects), 0) AS objects,
                      COALESCE(SUM(source = 'image'), 0)  AS images,
                      COALESCE(SUM(source = 'camera'), 0) AS camera_sessions,
                      COALESCE(SUM(source = 'video'), 0)  AS video_sessions
               FROM scans WHERE frames > 0"""
        ).fetchone()
        return {"totals": dict(totals), "classes": [dict(r) for r in per_class]}


def clear() -> None:
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM scan_objects")
        conn.execute("DELETE FROM scans")
