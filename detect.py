"""
Real-time Object Detection with YOLOv8 + OpenCV.

Detects objects live from a webcam, a video file, an image, or a stream URL,
draws bounding boxes with labels and confidence, shows FPS and per-class
counts, and can save the annotated result.

Examples:
    python detect.py                          # default webcam
    python detect.py --source video.mp4 --save
    python detect.py --source photo.jpg --save
    python detect.py --classes person car --conf 0.5
    python detect.py --model yolov8s.pt --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW_NAME = "Real-time Object Detection  |  q: quit   s: snapshot   p: pause"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def color_for(class_id: int) -> tuple[int, int, int]:
    """Stable, distinct BGR color per class id."""
    rng = np.random.default_rng(class_id * 7919 + 17)
    r, g, b = rng.integers(60, 255, size=3)
    return int(b), int(g), int(r)


def resolve_class_ids(model: YOLO, names: list[str] | None) -> list[int] | None:
    """Turn class names like ['person', 'car'] into model class ids."""
    if not names:
        return None
    lookup = {v.lower(): k for k, v in model.names.items()}
    ids, unknown = [], []
    for n in names:
        key = n.lower().replace("_", " ")
        (ids.append(lookup[key]) if key in lookup else unknown.append(n))
    if unknown:
        print(f"[warn] Unknown class name(s) ignored: {', '.join(unknown)}")
        print(f"       Available: {', '.join(sorted(lookup))}")
    return ids or None


def draw_label(frame, text, x, y, color):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    y = max(y, th + base + 4)
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - base - 2), font, scale,
                (0, 0, 0), thick, cv2.LINE_AA)


def annotate(frame, result, model: YOLO) -> Counter:
    """Draw boxes/labels on the frame and return per-class counts."""
    counts: Counter = Counter()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return counts

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    for i, ((x1, y1, x2, y2), conf, cls) in enumerate(zip(xyxy, confs, clss)):
        name = model.names[cls]
        counts[name] += 1
        color = color_for(cls)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.2f}"
        if ids is not None:
            label = f"#{ids[i]} {label}"
        draw_label(frame, label, x1, y1, color)
    return counts


def draw_hud(frame, fps: float | None, counts: Counter):
    """Semi-transparent info panel: FPS + object counts."""
    lines = []
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    total = sum(counts.values())
    lines.append(f"Objects: {total}")
    for name, n in counts.most_common(6):
        lines.append(f"  {name}: {n}")

    pad, lh = 8, 20
    w = 180
    h = pad * 2 + lh * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + w, 8 + h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (8 + pad, 8 + pad + 14 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)


def open_source(source: str):
    """Webcam index ('0', '1', ...) or path / URL."""
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"[error] Could not open source: {source}")
    return cap


# ----------------------------------------------------------------------------
# Modes
# ----------------------------------------------------------------------------
def run_image(model, args, class_ids, out_dir: Path):
    frame = cv2.imread(args.source)
    if frame is None:
        sys.exit(f"[error] Could not read image: {args.source}")
    result = model.predict(frame, conf=args.conf, iou=args.iou,
                           classes=class_ids, imgsz=args.imgsz,
                           device=args.device, verbose=False)[0]
    counts = annotate(frame, result, model)
    draw_hud(frame, None, counts)

    print("Detected:", dict(counts) if counts else "nothing")
    if args.save:
        out = out_dir / f"{Path(args.source).stem}_detected.jpg"
        cv2.imwrite(str(out), frame)
        print(f"[saved] {out}")
    if not args.no_show:
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_stream(model, args, class_ids, out_dir: Path):
    cap = open_source(args.source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if args.save:
        name = "webcam" if args.source.isdigit() else Path(args.source).stem
        out_path = out_dir / f"{name}_detected_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                 src_fps, (width, height))
        print(f"[recording] {out_path}")

    frame_times: deque = deque(maxlen=30)
    paused, frame = False, None
    print("Running... press 'q' to quit, 's' for snapshot, 'p' to pause.")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break

                t0 = time.perf_counter()
                if args.track:
                    result = model.track(frame, persist=True, conf=args.conf,
                                         iou=args.iou, classes=class_ids,
                                         imgsz=args.imgsz, device=args.device,
                                         verbose=False)[0]
                else:
                    result = model.predict(frame, conf=args.conf, iou=args.iou,
                                           classes=class_ids, imgsz=args.imgsz,
                                           device=args.device, verbose=False)[0]
                frame_times.append(time.perf_counter() - t0)

                counts = annotate(frame, result, model)
                fps = len(frame_times) / max(sum(frame_times), 1e-6)
                draw_hud(frame, fps, counts)

                if writer is not None:
                    writer.write(frame)

            if args.no_show:
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("p"):
                paused = not paused
            if key == ord("s") and frame is not None:
                snap = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(snap), frame)
                print(f"[snapshot] {snap}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time object detection with YOLOv8 + OpenCV")
    p.add_argument("--source", default="0",
                   help="Webcam index (0, 1...), video/image path, or stream URL")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLOv8 weights: yolov8n/s/m/l/x.pt or a custom .pt")
    p.add_argument("--conf", type=float, default=0.4,
                   help="Confidence threshold (0-1)")
    p.add_argument("--iou", type=float, default=0.5,
                   help="NMS IoU threshold (0-1)")
    p.add_argument("--imgsz", type=int, default=640, help="Inference size")
    p.add_argument("--classes", nargs="+",
                   help="Only detect these classes, e.g. --classes person car")
    p.add_argument("--device", default=None,
                   help="cpu, cuda, cuda:0, or mps (auto if omitted)")
    p.add_argument("--track", action="store_true",
                   help="Enable object tracking (gives each object an ID)")
    p.add_argument("--save", action="store_true",
                   help="Save annotated output to the output folder")
    p.add_argument("--output", default="outputs", help="Output folder")
    p.add_argument("--no-show", action="store_true",
                   help="Don't open a display window (headless)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading model: {args.model}")
    model = YOLO(args.model)
    class_ids = resolve_class_ids(model, args.classes)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(model, args, class_ids, out_dir)
    else:
        run_stream(model, args, class_ids, out_dir)


if __name__ == "__main__":
    main()
