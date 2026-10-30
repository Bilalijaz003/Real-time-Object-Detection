# Real-time Object Detection

Detect and track objects live from a webcam, video, image or stream using **YOLOv8** and **OpenCV**. The app draws bounding boxes with class names and confidence scores, shows live FPS and per-class object counts, and can record the annotated output.

It comes in two versions:

- **Web app**: a Flask backend with a browser interface for live camera, video and photo detection, plus a saved history of everything detected
- **Desktop app**: a single Python script (`detect.py`) that opens an OpenCV window

It recognizes the **80 COCO object classes** out of the box (people, cars, buses, animals, phones, bottles and more) and works with custom-trained YOLOv8 models too.

## Features

- **Live detection** from webcam, video files, images, or RTSP/HTTP streams
- **Object tracking** with persistent IDs (`--track`)
- **Class filtering**, e.g. detect only `person` and `car`
- **On-screen HUD** with FPS, total objects and per-class counts
- **Save results**: annotated video recordings, images, and one-key snapshots
- **Model choice**: swap between YOLOv8 nano → extra-large, or load your own weights
- **GPU support** (CUDA / Apple MPS) with automatic CPU fallback
- **Detection history** (web app): every photo and camera/video session is saved, with how many times each object was detected

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.9+ | Language |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection & tracking |
| OpenCV | Video capture, drawing, display, recording |
| Flask | Backend REST API and web server |
| SQLite | Detection history database |
| HTML, CSS, JavaScript | Web interface (camera capture, drawing boxes) |
| NumPy | Array handling |

## Installation

```bash
git clone https://github.com/Bilalijaz003/Real-time-Object-Detection.git
cd Real-time-Object-Detection

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

The YOLOv8 weights (`yolov8n.pt`, ~6 MB) download automatically on first run.

## Usage: Web App

```bash
python backend/app.py
```

Then open **http://localhost:5000** in your browser. It has three tabs:

| Tab | What it does |
|-----|--------------|
| Camera | Live detection from your webcam (the browser asks for camera permission) |
| Video file | Runs detection on a video as it plays |
| Photo | Detects the objects in an uploaded image |

Adjust the confidence and choose which objects to detect in the side panel. The **History** section shows every scan and a table of how many times each object has been detected. History is stored in `backend/history.db`.

### How history is counted

- **Photo**: each upload is one scan; objects are counted as found in the photo.
- **Camera / video**: each session (Start → Stop, or one video) is one scan. An object's count is the most seen *at the same time*, so one person in view for a minute counts as 1, not as hundreds of frames.
- **Times detected**: the number of scans an object appeared in.

### API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Backend and model status |
| GET | `/api/classes` | The 80 object classes the model knows |
| POST | `/api/detect` | Detect objects in an image (`image` file; optional `conf`, `iou`, `classes`, `scan_id`) |
| POST | `/api/scans` | Start a camera/video session: `{"source": "camera"}` |
| POST | `/api/scans/<id>/end` | End a session |
| GET | `/api/history` | Recent scans with the objects found |
| GET | `/api/stats` | Times each object was detected, and totals |
| DELETE | `/api/history` | Clear all history |

Example:

```bash
curl -F "image=@photo.jpg" -F "conf=0.5" http://localhost:5000/api/detect
```

```json
{
  "count": 2,
  "counts": {"person": 1, "car": 1},
  "detections": [
    {"class": "person", "confidence": 0.91, "box": [48.0, 120.5, 210.3, 470.0]}
  ],
  "inference_ms": 42.7
}
```

Settings via environment variables: `YOLO_MODEL` (default `yolov8n.pt`), `YOLO_DEVICE` (`cpu`, `cuda`, `mps`), `PORT` (default `5000`).

## Usage: Desktop App

```bash
# Live webcam (default camera)
python detect.py

# Video file, saving the annotated result
python detect.py --source video.mp4 --save

# Single image
python detect.py --source photo.jpg --save

# Only detect people and cars, higher confidence
python detect.py --classes person car --conf 0.5

# Track objects with IDs
python detect.py --track

# Bigger, more accurate model on GPU
python detect.py --model yolov8s.pt --device cuda
```

### Keyboard controls

| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `p` | Pause / resume |
| `s` | Save a snapshot |

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index, video/image path, or stream URL |
| `--model` | `yolov8n.pt` | Weights: `yolov8n/s/m/l/x.pt` or custom `.pt` |
| `--conf` | `0.4` | Confidence threshold |
| `--iou` | `0.5` | NMS IoU threshold |
| `--imgsz` | `640` | Inference image size |
| `--classes` | all | Class names to keep, e.g. `person car dog` |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, `mps` |
| `--track` | off | Enable tracking with object IDs |
| `--save` | off | Save annotated output |
| `--output` | `outputs` | Output folder |
| `--no-show` | off | Run without a display window |

## Choosing a Model

| Model | Size | Speed | Accuracy | Best for |
|-------|------|-------|----------|----------|
| `yolov8n.pt` | 6 MB | Fastest | Good | CPU, laptops, real-time webcam |
| `yolov8s.pt` | 22 MB | Fast | Better | Balanced |
| `yolov8m.pt` | 52 MB | Medium | High | GPU |
| `yolov8l/x.pt` | 87–136 MB | Slower | Highest | Powerful GPU, offline video |

## How It Works

**Web app:** the browser captures frames from the camera or video, resizes them, and sends each one to the Flask backend. The backend runs YOLOv8, records the results in the history database, and returns the boxes as JSON, which the browser draws over the video. The next frame is sent as soon as the answer comes back, so it runs as fast as your computer allows.

**Desktop app:**

1. **Capture**: OpenCV reads frames from the camera, video or image.
2. **Detect**: each frame goes through YOLOv8, which predicts bounding boxes, class labels and confidence scores in a single pass.
3. **Filter**: low-confidence boxes are dropped and overlapping boxes are merged with Non-Maximum Suppression (NMS).
4. **Track** (optional): ByteTrack links detections across frames so each object keeps the same ID.
5. **Render**: boxes, labels, FPS and counts are drawn on the frame, which is shown and optionally recorded.

## Project Structure

```
Real-time-Object-Detection/
├── backend/
│   ├── app.py         # Flask API: detection, sessions, history
│   └── history.py     # SQLite detection history
├── frontend/
│   ├── index.html     # Web interface
│   ├── style.css
│   └── app.js         # Camera capture, API calls, drawing boxes
├── detect.py          # Desktop app (OpenCV window)
├── requirements.txt   # Dependencies
├── README.md
└── outputs/           # Saved results (created automatically)
```

## Future Improvements

- Export history to CSV
- Train a custom model on a specific dataset
- Zone-based counting (e.g. people entering an area)
- Export to ONNX / TensorRT for faster inference

## Author

**Muhammad Bilal**
GitHub: [@Bilalijaz003](https://github.com/Bilalijaz003)

## License

Released under the MIT License for this project's own code. Note that Ultralytics YOLOv8 is licensed under AGPL-3.0.
