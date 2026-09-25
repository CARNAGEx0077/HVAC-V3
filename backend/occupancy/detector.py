"""
HVEAC Control Center - YOLO Person Detector
Integrates Ultralytics YOLO with strictly person-only filtering and device fallback.
"""

import os
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("hveac.detector")


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.45, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.device = self._resolve_device(device)
        self._lock = threading.Lock()
        self.model = None

        self._load_model()

    def _resolve_device(self, req_device: str) -> str:
        req = req_device.strip().lower()
        if req in ("cuda", "auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info(f"[YOLO DEVICE] CUDA accelerated device detected: {torch.cuda.get_device_name(0)}")
                    return "cuda"
            except Exception as e:
                logger.warning(f"[YOLO DEVICE] Error checking CUDA: {e}")
            logger.info("[YOLO DEVICE] CUDA unavailable; falling back safely to CPU")
            return "cpu"
        return "cpu"

    def _load_model(self):
        if not self.model_path.exists():
            error_msg = f"YOLO MODEL NOT AVAILABLE: File not found at '{self.model_path.resolve()}'"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"[YOLO LOADING] Loading YOLO weights from {self.model_path} on {self.device}")
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
            logger.info(f"[YOLO LOADED] Model {self.model_path.name} initialized successfully.")
        except Exception as e:
            error_msg = f"Failed to initialize YOLO model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def track(self, frame: np.ndarray, tracker_config: str = "bytetrack.yaml") -> List[Dict[str, Any]]:
        """
        Runs tracking and person detection in a single pass.
        Strictly filters for class 0 (person).
        """
        if self.model is None or frame is None:
            return []

        with self._lock:
            try:
                results = self.model.track(
                    source=frame,
                    persist=True,
                    classes=[0],  # Strictly person
                    conf=self.conf_threshold,
                    device=self.device,
                    tracker=tracker_config,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"[DETECTOR ERROR] Inference failed: {e}")
                return []

        if not results or len(results) == 0:
            return []

        res = results[0]
        detections = []

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes
            for i in range(len(boxes)):
                box = boxes[i]
                # Class check
                cls_id = int(box.cls[0].item())
                if cls_id != 0:
                    continue

                conf = float(box.conf[0].item())
                if conf < self.conf_threshold:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().tolist()
                track_id = int(box.id[0].item()) if (box.id is not None and len(box.id) > 0) else None

                detections.append({
                    "track_id": track_id,
                    "x1": int(xyxy[0]),
                    "y1": int(xyxy[1]),
                    "x2": int(xyxy[2]),
                    "y2": int(xyxy[3]),
                    "confidence": conf,
                    "class_id": 0,
                    "class_name": "person"
                })

        return detections
