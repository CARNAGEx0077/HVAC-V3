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
│   ├── occupancy/
│   │   ├── camera.py       # Thread-safe camera capture abstraction with reconnect handling
│   │   ├── detector.py     # Ultralytics YOLO person detector (strictly class 0: person)
│   │   ├── tracker.py      # ByteTrack identity persistence and track purging
│   │   ├── engine.py       # Occupancy aggregation, temporal smoothing, HUD annotations
│   │   └── state.py        # Centralized thread-safe runtime state store
│   └── simulation/
│       ├── manager.py      # Simulation manager: state machine, scenario datasets, ~10Hz tick loop
│       └── router.py       # Simulation REST API (/api/simulation/*) & WebSocket (/ws/simulation)
│
├── frontend/
│   ├── index.html          # Application layout shell (Sidebar with SIMULATION Lab, Main viewport)
│   ├── css/
│   │   └── style.css       # Dark industrial engineering stylesheet (Simulation canvas, Pitch mode)
│   └── js/
│       ├── app.js          # SPA router (#/simulation), WebSocket telemetry, targeted updates
│       ├── state.js        # Reactive state store & Simulation command dispatcher
│       ├── components.js   # Reusable functional UI components (Cards, Tables, Pipeline)
│       ├── pages.js        # View renderers including renderSimulation() & updateSimulationDom()
│       └── sim_visuals.js  # Canvas 2D thermal heatmap renderer & rolling 5-min history chart
│
├── simulator/              # Deterministic lumped-capacitance thermal simulation engine
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

## 🧪 HVEAC V3 Simulation Lab

The **Simulation Lab** is a dedicated first-class dashboard view (`#/simulation`) for exploring, replaying, and demonstrating the synthetic thermal dynamics of an enclosed commercial space ($12.0\text{ m} \times 10.0\text{ m}$) under five deterministic operational scenarios.

> [!NOTE]
> **Strict System Separation & Synthetic Semantics**:
> The Simulation Lab operates as an independent synthetic data and scenario demonstration engine. All simulation states, temperatures, workloads, and AC interactions are **100% SYNTHETIC DATA** generated by the physics engine. It is strictly separated from physical hardware agents (`/api/nodes`), camera occupancy detection (`/ws/occupancy`), and production HVAC actuators.

### Architecture & Data Flow

```text
SCENARIO ENGINE (simulator/)
      ↓ (Pre-computed / In-Memory Replay Cache)
SIMULATION MANAGER (backend/simulation/manager.py)
      ↓ (REST: /api/simulation/* & Streaming WS: /ws/simulation @ ~10Hz)
SIMULATION DASHBOARD (frontend/#/simulation)
      ├── 2D Thermal Heatmap Canvas (Bilinear quadrant interpolation + hot/cool auras)
      ├── Real-Time Rolling 5-Min Multi-Line Chart (Avg, Z1, Z2, Z3, Z4, Setpoint)
      ├── 4 Perimeter AC Indicators (Airflow vectors, setpoints, duty cycle)
      ├── 10 Fixed Workstations (Interactive map pins synchronized with table)
      ├── Simulated Computer Node Telemetry Table (C1–C10 live telemetry, micro-bars, trends)
      ├── Selected Node Inspector Banner (Live machine focus with CPU, GPU, Workload, Heat)
      ├── Summary Statistics Bar (Computers 10/10, Heavy Count, Heat Watts, Avg CPU, Avg GPU)
      ├── Scenario Selector (5 Scenarios with immediate hot-swap & zero stale state)
      ├── Presentation Pitch Mode (Enlarged viewport, glowing hotspots, high contrast)
      └── Scenario Comparison Drawer (Empirical thermal & energy metrics matrix)
```

### The 5 Demonstration Scenarios

1. **Scenario 1: Localized Heavy Compute**
   - Four physically clustered computers (`C1`–`C4`) in Zone 1 ramp smoothly from baseline into heavy workloads (80–95% CPU/GPU, ~280W heat). `C5`–`C10` maintain light load (~15–18% CPU, ~70W). Uniform occupancy.
   - *Visual Outcome*: Sharp localized computational hotspot forms in Zone 1 (NW quadrant).
2. **Scenario 2: Occupancy Concentration**
   - All 10 workstations maintain light, uniform compute (18–24% CPU). Occupancy surges to 16 people in Zone 3 (SE quadrant).
   - *Visual Outcome*: Occupancy metabolic heat creates a distinctive thermal hotspot in Zone 3 without compute spikes.
3. **Scenario 3: Distributed Heavy Compute**
   - All 10 computers across all 4 zones ramp into heavy batch workloads (~88–94% CPU, ~275–290W). Uniform occupancy.
   - *Visual Outcome*: Wide thermal elevation across the entire room requiring coordinated multi-unit cooling.
4. **Scenario 4: High Occupancy / Low Computer Load**
   - All computers operate in low-power idle (8–14% CPU, ~65–75W). High occupancy (34 occupants) distributed throughout the room.
   - *Visual Outcome*: Human metabolic heat (~2900 W) completely dominates compute heat (~650 W) by > 4.5:1.
5. **Scenario 5: Opposing Thermal Zones**
   - West zones (Computers 1–4 in Zone 1 and 9–10 in Zone 4) ramp to heavy compute (~88–92% CPU). East zones (Computers 5–8) host 20+ occupants under light compute.
   - *Visual Outcome*: Two contrasting thermal regions driven by fundamentally different heat sources.

### Simulation REST & WebSocket API

| Endpoint | Method | Description |
|:---|:---|:---|
| `/api/simulation/scenarios` | `GET` | Returns metadata for all 5 scenarios with descriptions and dominant sources |
| `/api/simulation/state` | `GET` | Returns full current snapshot of active simulation step conforming to V3 schema |
| `/api/simulation/select` | `POST` | `{"scenario_id": 1..5}`: Stops active run, resets clocks, and loads new scenario |
| `/api/simulation/start` | `POST` | Starts or resumes real-time simulation clock |
| `/api/simulation/pause` | `POST` | Pauses real-time simulation clock |
| `/api/simulation/reset` | `POST` | Resets active scenario to timestep 0 (initial conditions) |
| `/api/simulation/speed` | `POST` | `{"speed": 1 \| 5 \| 10 \| 30}`: Configures playback time acceleration multiplier |
| `/api/simulation/comparison` | `GET` | Returns summary comparison matrix computed from complete simulation runs |
| `/ws/simulation` | `WS` | Broadcasts simulation telemetry updates at ~10 Hz and handles bi-directional client commands |

### Verification Status

| Feature | Tested Behavior | Status |
|:---|:---|:---:|
| **Route Navigation** | Sidebar navigation to `#/simulation` without page reload | **PASS** |
| **Room Visualization** | 4 perimeter ACs, 10 fixed workstations, quadrant boundaries | **PASS** |
| **Bilinear Heatmap** | Dynamic continuous color gradient with localized compute & occupancy auras | **PASS** |
| **Multi-Line History Chart** | Rolling 5-minute chart with Avg, Zone 1–4, and 22.5°C setpoint target | **PASS** |
| **Speed Multipliers** | 1x, 5x, 10x, 30x acceleration tested and verified | **PASS** |
| **Scenario Switching** | Instant stop, clock reset, thermal field reinitialization, zero stale data | **PASS** |
| **Pitch Mode** | High-contrast presentation layout with emphasized thermal gradients | **PASS** |
| **Comparison Matrix** | Actual run statistics (Initial, Final, Min, Max, Peak Watts) loaded in modal | **PASS** |
| **Computer Telemetry Table** | All 10 rows update smoothly in real time without page re-renders | **PASS** |
| **Gradual Workload Ramping** | C1–C4 ramp smoothly from 20% to 92% CPU and 86W to 288W in Scenario 1 | **PASS** |
| **Summary Bar Metrics** | Live indicators for total, heavy count, total heat, average CPU, average GPU | **PASS** |
| **Bidirectional Selection** | Clicking row or pin highlights both map and table, opens Inspector banner | **PASS** |
| **Hover Synchronization** | Hovering pin highlights table row (`.row-hover`); hovering row highlights pin | **PASS** |
| **Hardware Isolation** | Zero cross-contamination between simulation and `/api/nodes` physical telemetry | **PASS** |

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
