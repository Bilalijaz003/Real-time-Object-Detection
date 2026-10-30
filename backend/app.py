"""
Flask backend for Real-time Object Detection.

Runs YOLOv8 on images sent from the browser and returns the detections as
JSON. The web interface (frontend/) captures frames from the webcam, a video
file or an uploaded image, sends them to /api/detect, and draws the boxes.

Every detection is saved to a local history database (see history.py), so
you can see what was detected and how many times.

Run:
    python backend/app.py
Then open http://localhost:5000
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO

import history

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")
DEVICE = os.environ.get("YOLO_DEVICE") or None  # e.g. "cpu", "cuda", "mps"
MAX_UPLOAD_MB = 10
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

print(f"[info] Loading model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
CLASS_NAMES: dict[int, str] = model.names
NAME_TO_ID = {v.lower(): k for k, v in CLASS_NAMES.items()}
# One model instance shared by all requests; YOLO isn't thread-safe.
model_lock = threading.Lock()
print(f"[info] Model ready: {len(CLASS_NAMES)} classes")
history.init_db()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def parse_float(value, default, low=0.0, high=1.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return min(max(v, low), high)


def parse_classes(raw: str | None):
    """'person,car' -> [0, 2]. Unknown names are ignored."""
    if not raw:
        return None
    ids = [NAME_TO_ID[n.strip().lower()] for n in raw.split(",")
           if n.strip().lower() in NAME_TO_ID]
    return ids or None


def decode_image(file_storage):
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME,
                    "classes": len(CLASS_NAMES)})


@app.get("/api/classes")
def classes():
    return jsonify({"classes": [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]})


@app.post("/api/detect")
def detect():
    """
    Form fields:
        image    (file, required)  JPEG/PNG/WebP/BMP frame or photo
        conf     (float, 0-1)      confidence threshold, default 0.4
        iou      (float, 0-1)      NMS IoU threshold, default 0.5
        classes  (str)             comma-separated class names to keep
        scan_id  (int)             live session id from POST /api/scans;
                                   omit for a single photo
        save     ("0" to skip)     photos are saved to history by default
    Returns detections with pixel boxes [x1, y1, x2, y2] in the image's
    own coordinate space.
    """
    file = request.files.get("image")
    if file is None:
        return error("Send an image in the 'image' form field.")
    if file.mimetype and file.mimetype not in ALLOWED_TYPES:
        return error(f"Unsupported file type '{file.mimetype}'. "
                     "Use JPEG, PNG, WebP or BMP.", 415)

    frame = decode_image(file)
    if frame is None:
        return error("Could not read the image. Check that the file isn't "
                     "damaged.")

    conf = parse_float(request.form.get("conf"), 0.4)
    iou = parse_float(request.form.get("iou"), 0.5)
    class_ids = parse_classes(request.form.get("classes"))

    t0 = time.perf_counter()
    with model_lock:
        result = model.predict(frame, conf=conf, iou=iou, classes=class_ids,
                               device=DEVICE, verbose=False)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    detections, counts = [], Counter()
    boxes = result.boxes
    if boxes is not None and len(boxes):
        for (x1, y1, x2, y2), c, k in zip(boxes.xyxy.cpu().numpy(),
                                          boxes.conf.cpu().numpy(),
                                          boxes.cls.cpu().numpy().astype(int)):
            name = CLASS_NAMES[int(k)]
            counts[name] += 1
            detections.append({
                "class": name,
                "class_id": int(k),
                "confidence": round(float(c), 3),
                "box": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            })

    counts_dict = dict(counts.most_common())
    scan_id = request.form.get("scan_id", type=int)
    if scan_id is not None:
        if not history.scan_exists(scan_id):
            return error("That scan has ended or doesn't exist. Start a new "
                         "one with POST /api/scans.", 404)
        history.add_frame(scan_id, counts_dict)
    elif request.form.get("save", "1") != "0":
        scan_id = history.record_image(file.filename or None, counts_dict)

    h, w = frame.shape[:2]
    return jsonify({
        "scan_id": scan_id,
        "width": w,
        "height": h,
        "inference_ms": round(inference_ms, 1),
        "count": len(detections),
        "counts": counts_dict,
        "detections": detections,
    })


# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------
@app.post("/api/scans")
def start_scan():
    """Start a camera or video session. Body: {"source": "camera"|"video",
    "name": optional label such as the video file name}."""
    body = request.get_json(silent=True) or {}
    source = body.get("source")
    if source not in ("camera", "video"):
        return error("source must be 'camera' or 'video'.")
    scan_id = history.start_scan(source, (body.get("name") or None))
    return jsonify({"scan_id": scan_id}), 201


@app.post("/api/scans/<int:scan_id>/end")
def end_scan(scan_id):
    history.end_scan(scan_id)
    return jsonify({"scan_id": scan_id, "status": "ended"})


@app.get("/api/history")
def get_history():
    limit = min(max(request.args.get("limit", 20, type=int), 1), 200)
    return jsonify({"scans": history.list_scans(limit)})


@app.get("/api/stats")
def get_stats():
    return jsonify(history.stats())


@app.delete("/api/history")
def clear_history():
    history.clear()
    return jsonify({"status": "cleared"})


@app.errorhandler(413)
def too_large(_):
    return error(f"File is larger than {MAX_UPLOAD_MB} MB.", 413)


# ----------------------------------------------------------------------------
# Frontend
# ----------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[info] Open http://localhost:{port} in your browser")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
