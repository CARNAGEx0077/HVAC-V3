# HVEAC — Environmental AI Control Center & Real Occupancy Module

A desktop-first engineering operations console for the **HVEAC (Heating, Ventilation, Environmental AI Control)** system featuring a **real-time computer vision occupancy detection module** powered by OpenCV, Ultralytics YOLO, ByteTrack, and FastAPI.

---

## 📋 Overview

The HVEAC Control Center provides real-time situational awareness across occupancy detection, compute cluster thermal load, and ambient meteorological telemetry.

- **Real-Time Vision Pipeline**: Captures video frames from physical webcams, runs real YOLOv11 person detection, maintains stable track identities via ByteTrack, applies temporal smoothing, and streams annotated MJPEG video.
- **Zero Fake Data**: Strictly reports verified physical camera states, actual detection metrics, and real inference latencies. If the camera is offline, the system reports `DISCONNECTED` / `ERROR` rather than fabricating synthetic counts.
- **Desktop-First Console**: Full viewport height (`100vh`) with a fixed 236px sidebar, dynamic main viewport, and dark charcoal aesthetic matching industrial engineering workstations.
- **Unified Central State**: A single camera worker, vision inference worker, and centralized state store consumed synchronously across all WebSocket clients and dashboard pages.
- **Non-Blocking Control Layer**: Real hardware/subsystem lifecycle controls (`START`, `STOP`, `RESTART`, `RESET STATE`) with real-time audit logging in **Command History**.

---

## 📁 Project Structure

```text
HVAC-V2/
│
├── backend/
│   ├── main.py             # FastAPI entry point, lifespan, static mounting, router wiring
│   ├── config.py           # Environment-based configuration (camera, model, thresholds, FPS)
│   ├── api/
│   │   ├── routes.py       # REST API endpoints & MJPEG /video_feed stream
│   │   └── websocket.py    # WebSocket /ws/occupancy connection manager & telemetry loop
│   └── occupancy/
│       ├── camera.py       # Thread-safe camera capture abstraction with reconnect handling
│       ├── detector.py     # Ultralytics YOLO person detector (strictly class 0: person)
│       ├── tracker.py      # ByteTrack identity persistence and track purging
│       ├── engine.py       # Occupancy aggregation, temporal smoothing, HUD annotations
│       └── state.py        # Centralized thread-safe runtime state store
│
├── frontend/
│   ├── index.html          # Application layout shell (Sidebar, Header, Main viewport)
│   ├── css/
│   │   └── style.css       # Dark industrial engineering stylesheet (CSS variables, UTF-8)
│   └── js/
│       ├── app.js          # SPA router, WebSocket telemetry listener, targeted DOM updater
│       ├── state.js        # Reactive state store & REST command dispatcher
│       ├── components.js   # Reusable functional UI components (Cards, Tables, Pipeline)
│       └── pages.js        # View renderers for all 8 application routes
│
├── models/
│   └── yolo11n.pt          # Ultralytics YOLOv11 nano model weights
│
├── requirements.txt        # Python dependency declarations
└── README.md               # Engineering documentation and runbook
```

---

## 🛠 Technology Stack

### Backend
- **Language**: Python 3.10+ (Tested on Python 3.14.3)
- **Web Framework**: FastAPI & Starlette
- **ASGI Server**: Uvicorn
- **Computer Vision**: OpenCV (`opencv-python`)
- **Deep Learning**: Ultralytics YOLO (`yolo11n.pt`, PyTorch, Torchvision)
- **Real-Time Networking**: WebSockets (`websockets`)

### Frontend
- **Structure**: Semantic HTML5 with desktop viewport optimization (`100vh`).
- **Styling**: Pure Vanilla CSS3 with CSS custom properties (`:root` variables), zero dependencies (no Tailwind, Bootstrap, or jQuery).
- **Client Logic**: Vanilla JavaScript ES6+ with client-side hash routing (`#/`) and targeted DOM mutations on incoming telemetry frames.

---

## 🔍 Occupancy Vision Pipeline

```text
CAMERA (OpenCV)
      ↓
FRAME CAPTURE (Background Worker Thread)
      ↓
YOLOv11 INFERENCE (Strictly Person Class 0, Conf >= 0.45)
      ↓
PERSON DETECTIONS (x1, y1, x2, y2, Confidence)
      ↓
BYTETRACK TRACKER (Unique IDs, Track Retention, Stale Track Purging)
      ↓
TEMPORAL SMOOTHING (Sliding window of 5 frames, Rounded Mean)
      ↓
CURRENT OCCUPANCY & AGGREGATE CONFIDENCE CALCULATION
      ↓
MJPEG ANNOTATION STREAM (/video_feed) + WEBSOCKET BROADCAST (/ws/occupancy)
      ↓
CONTROL CENTER DASHBOARD
```

### Detection & Tracking Specifications
- **Target Class**: Strictly class `0` (`person`). Cars, chairs, and non-human objects are discarded before tracking.
- **Model Loading**: The YOLO model is loaded **once** at server startup and cached in memory. Models are never reloaded per frame.
- **Device Support**: Configurable via `DEVICE=cpu` or `DEVICE=cuda`. If CUDA is requested but unavailable, the detector safely falls back to CPU without server interruption.
- **Identity Retention**: ByteTrack maintains consistent IDs across consecutive frames, handling brief occlusions and personnel moving through the camera field.
- **Temporal Smoothing**: Raw detection counts pass through a configurable sliding window (`OCCUPANCY_SMOOTHING_WINDOW=5`) to eliminate single-frame flicker while preserving rapid reaction to real room entries/exits.

### Confidence Calculation
Aggregate confidence represents the **arithmetic mean** of detection confidences for all active persons detected in the current frame:
$$\text{Confidence} = \frac{1}{N} \sum_{i=1}^{N} \text{conf}_i \quad (N > 0)$$
If no persons are detected ($N = 0$):
$$\text{Confidence} = \text{None (rendered as "N/A" in the UI)}$$
The system explicitly avoids fabricating synthetic non-zero confidence when the room is empty.

---

## 🌐 API & Protocol Reference

### REST Endpoints

#### 1. Live Occupancy Metrics
`GET /api/occupancy`
```json
{
  "occupancy": 3,
  "confidence": 0.84,
  "confidence_display": "84%",
  "tracked_persons": 3,
  "tracked_ids": [1, 39, 152],
  "status": "RUNNING",
  "camera_status": "CONNECTED",
  "processing_fps": 14.9,
  "inference_latency_ms": 62.4,
  "timestamp": "2026-09-25T13:45:00.123456+00:00"
}
```

#### 2. Detailed Subsystem Status
`GET /api/occupancy/status`
Returns complete hardware telemetry including frames received, frames processed, frames dropped, and subsystem errors.

#### 3. Subsystem Hardware Lifecycle
- `POST /api/control/camera`: `{"action": "START" | "STOP" | "RESTART"}`
- `POST /api/control/vision`: `{"action": "START" | "STOP" | "RESTART"}`
- `POST /api/control/occupancy`: `{"action": "RESET"}`

#### 4. Live Annotated Video Stream
`GET /video_feed`
Streams multipart MJPEG (`multipart/x-mixed-replace; boundary=frame`) rendered with:
- Top engineering telemetry HUD (`HVEAC VISION`, `PEOPLE: <count>`, `FPS: <val>`, `LATENCY: <val>`, `TRACKING: ACTIVE`).
- Bounding boxes and identity badges (`ID:<track_id> <conf>%`) for each detected person.

---

### WebSocket Protocol

`WS /ws/occupancy`

The backend broadcaster pushes real-time telemetry updates at ~5 Hz:

```json
{
  "type": "occupancy_update",
  "occupancy": 3,
  "raw_count": 3,
  "confidence": 0.84,
  "confidence_display": "84%",
  "tracked_persons": 3,
  "camera_status": "CONNECTED",
  "vision_status": "RUNNING",
  "processing_fps": 14.9,
  "inference_latency_ms": 62.4,
  "timestamp": "2026-09-25T13:45:00.123456+00:00"
}
```

Connected clients receive immediate snapshots upon connection. Incoming messages update only the affected DOM elements without re-rendering the outer application shell.

---

## ⚙️ Configuration & Environment

Environment variables can be specified in a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Camera device index (`0`, `1`) or RTSP/file URI |
| `MODEL_PATH` | `models/yolo11n.pt` | Path to YOLO weights file |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `PROCESS_FPS` | `15` | Target vision processing rate |
| `DEVICE` | `cpu` | Inference accelerator (`cpu`, `cuda`, or `auto`) |
| `OCCUPANCY_SMOOTHING_WINDOW` | `5` | Frame window length for temporal smoothing |
| `TRACKER_CONFIG` | `bytetrack.yaml` | Tracker configuration file |
| `TRACKER_MAX_AGE` | `30` | Max frames before purging unseen tracks |
| `HOST` | `0.0.0.0` | Bind IP address |
| `PORT` | `8000` | HTTP and WebSocket port |

---

## 🚀 Installation & Running

### 1. Clone & Navigate
```bash
cd d:\HVAC-V2
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Placement
Ensure `yolo11n.pt` exists in the `models/` directory:
```text
models/
└── yolo11n.pt
```
*(If missing, copy or download `yolo11n.pt` directly into `models/`)*

### 4. Start the Application
```bash
python -m backend.main
```
Or with Uvicorn directly:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Dashboard
Open your browser to:
```text
http://localhost:8000/
```

---

## 🧪 Verification & Failure Testing

| Test Case | Description | Expected Result | Status |
|---|---|---|:---:|
| **Server Startup** | FastAPI initialization, model loading, static file mount | Port 8000 active, frontend served at root `/` | **PASS** |
| **YOLO Initialization** | Load `yolo11n.pt` once on CPU | Weights initialized, zero per-frame reloads | **PASS** |
| **Webcam Acquisition** | Connect to physical webcam device `0` | Frames acquired, status transitions to `CONNECTED` | **PASS** |
| **Real Inference** | Live inference on video stream | Persons detected with bounding boxes and IDs | **PASS** |
| **Tracking Identity** | ByteTrack identity persistence | Unique IDs assigned, no exploding track counts | **PASS** |
| **Temporal Smoothing** | 5-frame moving window | Stabilizes count against one-frame occlusions | **PASS** |
| **MJPEG Stream** | `GET /video_feed` | Live video stream with engineering HUD | **PASS** |
| **WebSocket Updates** | `WS /ws/occupancy` | Real-time messages broadcast to dashboard | **PASS** |
| **REST API** | `GET /api/occupancy` | JSON payload returned with live metrics | **PASS** |
| **Camera Lifecycle** | `STOP`, `START`, `RESTART` | Hardware resource released and reacquired cleanly | **PASS** |
| **Vision Lifecycle** | `STOP`, `START`, `RESTART` | Inference loop halted and restarted without camera destruction | **PASS** |
| **Occupancy Reset** | `RESET STATE` | Clears tracking history and smoothing buffer | **PASS** |
| **Camera Unavailable** | Probed with `CAMERA_SOURCE=99` | Status reports `ERROR`, vision refuses `RUNNING` | **PASS** |
| **Missing Model File** | Probed with invalid path | Raises `YOLO MODEL NOT AVAILABLE`, reports failure | **PASS** |

---

## 🔧 Troubleshooting

### 1. Camera Status shows `ERROR` or Initial Frame Capture Fails
- **Exclusive Device Locks**: On Windows, only one application can access a webcam at a time. If the Windows Camera app, Teams, Zoom, or OBS is open, close them completely.
- **Device Index**: If device index `0` is not your desired webcam, check your available cameras in Windows Device Manager and adjust `CAMERA_SOURCE=1` in `.env`.
- **Privacy Settings**: Ensure Windows Camera Privacy Settings allow desktop applications to access the camera (*Settings > Privacy & Security > Camera*).

### 2. YOLO Model Missing Alert
- If the server reports `YOLO MODEL NOT AVAILABLE`, ensure `yolo11n.pt` is located in `models/yolo11n.pt` relative to the project root.

### 3. High CPU Utilization
- Ensure `PROCESS_FPS` is set sensibly (default: `15`).
- If an NVIDIA GPU is available, set `DEVICE=cuda` in `.env` to enable CUDA acceleration.
