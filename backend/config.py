"""
HVEAC Control Center - Backend Configuration
Handles environment-based settings with sensible, safe defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Camera Configuration
raw_camera_source = os.getenv("CAMERA_SOURCE", "0").strip()
if raw_camera_source.isdigit():
    CAMERA_SOURCE = int(raw_camera_source)
else:
    CAMERA_SOURCE = raw_camera_source

# YOLO Model Configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "yolo11n.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
DEVICE = os.getenv("DEVICE", "cpu").strip().lower()

# Pipeline & Inference Performance
PROCESS_FPS = int(os.getenv("PROCESS_FPS", "15"))
OCCUPANCY_SMOOTHING_WINDOW = int(os.getenv("OCCUPANCY_SMOOTHING_WINDOW", "5"))
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "bytetrack.yaml")
TRACKER_MAX_AGE = int(os.getenv("TRACKER_MAX_AGE", "30"))

# Network & Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Frontend directory
FRONTEND_DIR = BASE_DIR / "frontend"
