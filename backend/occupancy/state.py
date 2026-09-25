"""
HVEAC Control Center - Runtime State Store
Provides thread-safe centralized state for camera, vision engine, and occupancy metrics.
"""

import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class RuntimeState:
    def __init__(self):
        self._lock = threading.Lock()

        # Subsystem statuses
        self.system_status: str = "ONLINE"
        self.camera_status: str = "DISCONNECTED"
        self.camera_source: Any = None
        self.camera_error: Optional[str] = None

        self.vision_status: str = "STOPPED"
        self.vision_error: Optional[str] = None

        # Occupancy metrics
        self.occupancy: int = 0
        self.raw_count: int = 0
        self.confidence: Optional[float] = None
        self.tracked_persons: int = 0
        self.tracked_ids: List[int] = []

        # Performance & telemetry
        self.processing_fps: float = 0.0
        self.inference_latency_ms: float = 0.0
        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.frames_dropped: int = 0
        self.last_frame_timestamp: Optional[str] = None
        self.timestamp: str = datetime.now(timezone.utc).isoformat()

    def update_camera(self, status: str, source: Any = None, error: Optional[str] = None):
        with self._lock:
            self.camera_status = status
            if source is not None:
                self.camera_source = source
            self.camera_error = error
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def update_vision(self, status: str, error: Optional[str] = None):
        with self._lock:
            self.vision_status = status
            self.vision_error = error
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def update_occupancy(
        self,
        occupancy: int,
        raw_count: int,
        confidence: Optional[float],
        tracked_persons: int,
        tracked_ids: List[int],
        processing_fps: float,
        inference_latency_ms: float,
        frames_received: int,
        frames_processed: int,
        frames_dropped: int
    ):
        with self._lock:
            self.occupancy = occupancy
            self.raw_count = raw_count
            self.confidence = confidence
            self.tracked_persons = tracked_persons
            self.tracked_ids = tracked_ids
            self.processing_fps = round(processing_fps, 1)
            self.inference_latency_ms = round(inference_latency_ms, 1)
            self.frames_received = frames_received
            self.frames_processed = frames_processed
            self.frames_dropped = frames_dropped
            now_iso = datetime.now(timezone.utc).isoformat()
            self.last_frame_timestamp = now_iso
            self.timestamp = now_iso

    def reset_occupancy(self):
        with self._lock:
            self.occupancy = 0
            self.raw_count = 0
            self.confidence = None
            self.tracked_persons = 0
            self.tracked_ids = []
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def get_occupancy_api_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "occupancy": self.occupancy,
                "confidence": round(self.confidence, 2) if self.confidence is not None else None,
                "confidence_display": f"{round(self.confidence * 100)}%" if self.confidence is not None else "N/A",
                "tracked_persons": self.tracked_persons,
                "tracked_ids": list(self.tracked_ids),
                "status": self.vision_status,
                "camera_status": self.camera_status,
                "processing_fps": self.processing_fps,
                "inference_latency_ms": self.inference_latency_ms,
                "timestamp": self.timestamp
            }

    def get_status_api_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "system": {
                    "status": self.system_status,
                    "timestamp": self.timestamp
                },
                "camera": {
                    "status": self.camera_status,
                    "source": str(self.camera_source),
                    "error": self.camera_error,
                    "frames_received": self.frames_received
                },
                "vision": {
                    "status": self.vision_status,
                    "error": self.vision_error,
                    "processing_fps": self.processing_fps,
                    "inference_latency_ms": self.inference_latency_ms,
                    "frames_processed": self.frames_processed,
                    "frames_dropped": self.frames_dropped
                },
                "occupancy": {
                    "occupancy": self.occupancy,
                    "raw_count": self.raw_count,
                    "confidence": self.confidence,
                    "tracked_persons": self.tracked_persons,
                    "tracked_ids": list(self.tracked_ids),
                    "last_frame_timestamp": self.last_frame_timestamp
                }
            }

    def get_websocket_payload(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "type": "occupancy_update",
                "occupancy": self.occupancy,
                "raw_count": self.raw_count,
                "confidence": round(self.confidence, 2) if self.confidence is not None else None,
                "confidence_display": f"{round(self.confidence * 100)}%" if self.confidence is not None else "N/A",
                "tracked_persons": self.tracked_persons,
                "camera_status": self.camera_status,
                "vision_status": self.vision_status,
                "processing_fps": self.processing_fps,
                "inference_latency_ms": self.inference_latency_ms,
                "timestamp": self.timestamp
            }


# Global shared runtime state instance
global_state = RuntimeState()
