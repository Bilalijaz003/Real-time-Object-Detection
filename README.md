# Real-time Object Detection

Detect and track objects live from a webcam, video, image or stream using **YOLOv8** and **OpenCV**. The app draws bounding boxes with class names and confidence scores, shows live FPS and per-class object counts, and can record the annotated output.

It recognizes the **80 COCO object classes** out of the box (people, cars, buses, animals, phones, bottles and more) and works with custom-trained YOLOv8 models too.

## Features

- **Live detection** from webcam, video files, images, or RTSP/HTTP streams
- **Object tracking** with persistent IDs (`--track`)
- **Class filtering**, e.g. detect only `person` and `car`
- **On-screen HUD** with FPS, total objects and per-class counts
- **Save results**: annotated video recordings, images, and one-key snapshots
- **Model choice**: swap between YOLOv8 nano → extra-large, or load your own weights
- **GPU support** (CUDA / Apple MPS) with automatic CPU fallback

## Tech Stack

| Component | Purpose |
|-----------|---------|
| Python 3.9+ | Language |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection & tracking |
| OpenCV | Video capture, drawing, display, recording |
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

## Usage

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

1. **Capture**: OpenCV reads frames from the camera, video or image.
2. **Detect**: each frame goes through YOLOv8, which predicts bounding boxes, class labels and confidence scores in a single pass.
3. **Filter**: low-confidence boxes are dropped and overlapping boxes are merged with Non-Maximum Suppression (NMS).
4. **Track** (optional): ByteTrack links detections across frames so each object keeps the same ID.
5. **Render**: boxes, labels, FPS and counts are drawn on the frame, which is shown and optionally recorded.

## Project Structure

```
Real-time-Object-Detection/
├── detect.py          # Main application
├── requirements.txt   # Dependencies
├── README.md
└── outputs/           # Saved results (created automatically)
```

## Future Improvements

- Web interface with Streamlit
- Train a custom model on a specific dataset
- Zone-based counting (e.g. people entering an area)
- Export to ONNX / TensorRT for faster inference

## Author

**Bilal Ijaz**
GitHub: [@Bilalijaz003](https://github.com/Bilalijaz003)

## License

Released under the MIT License for this project's own code. Note that Ultralytics YOLOv8 is licensed under AGPL-3.0.
