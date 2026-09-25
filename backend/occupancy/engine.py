"""
HVEAC Control Center - Occupancy Engine
Orchestrates frame ingestion, YOLO inference, ByteTrack, temporal smoothing,
confidence aggregation, and annotated MJPEG streaming.
"""

import cv2
import time
import logging
import threading
from collections import deque
from typing import Optional, List, Dict, Any
import numpy as np

from backend.occupancy.camera import Camera
from backend.occupancy.detector import YoloDetector
from backend.occupancy.tracker import PersonTracker
from backend.occupancy.state import global_state

logger = logging.getLogger("hveac.engine")


class OccupancyEngine:
    def __init__(
        self,
        camera: Camera,
        detector: YoloDetector,
        tracker: PersonTracker,
        target_fps: int = 15,
        smoothing_window: int = 5,
        tracker_config: str = "bytetrack.yaml"
    ):
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.target_fps = target_fps
        self.frame_interval = 1.0 / max(1, target_fps)
        self.smoothing_window = smoothing_window
        self.tracker_config = tracker_config

        self._lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False

        # Temporal smoothing buffer
        self.smoothing_history: deque = deque(maxlen=smoothing_window)

        # Performance counters
        self.frames_processed = 0
        self.frames_dropped = 0
        self._fps_history: deque = deque(maxlen=15)
        self.current_fps = 0.0

        # Latest annotated JPEG frame buffer for MJPEG streaming
        self._latest_jpeg: Optional[bytes] = None

    @property
    def status(self) -> str:
        return global_state.vision_status

    def start(self) -> bool:
        """
        Starts the vision processing loop.
        Fails safely if camera is unavailable.
        """
        with self._lock:
            if self._running and self._worker_thread and self._worker_thread.is_alive():
                logger.info(f"[VISION] Already running in state: {self.status}")
                return True

            # Pre-flight check: Verify camera connectivity
            if self.camera.status != "CONNECTED":
                logger.info("[VISION START] Camera is not CONNECTED. Attempting camera start...")
                cam_ok = self.camera.start()
                if not cam_ok or self.camera.status != "CONNECTED":
                    err = "Cannot start Vision Engine: Camera unavailable or disconnected."
                    logger.error(f"[VISION ERROR] {err}")
                    global_state.update_vision("ERROR", error=err)
                    return False

            logger.info("[VISION START] Initializing vision worker loop")
            global_state.update_vision("STARTING")

            self._running = True
            self._worker_thread = threading.Thread(
                target=self._processing_loop,
                name="VisionEngineWorkerThread",
                daemon=True
            )
            self._worker_thread.start()

        time.sleep(0.2)
        return global_state.vision_status == "RUNNING"

    def stop(self) -> bool:
        """
        Stops vision processing without destroying camera hardware.
        """
        logger.info("[VISION STOP] Stopping vision worker")
        with self._lock:
            self._running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)

        global_state.update_vision("STOPPED")
        logger.info("[VISION STOPPED] Processing safely terminated")
        return True

    def restart(self) -> bool:
        """
        Stop vision -> clear processing state -> restart vision.
        """
        logger.info("[VISION RESTART] Restarting vision engine")
        self.stop()
        self.reset_occupancy()
        time.sleep(0.3)
        return self.start()

    def reset_occupancy(self):
        """
        Clears tracking state and temporal smoothing without stopping camera or unloading YOLO.
        """
        with self._lock:
            self.tracker.reset()
            self.smoothing_history.clear()
            global_state.reset_occupancy()
        logger.info("[OCCUPANCY RESET] State, tracking identities, and smoothing history cleared.")

    def _processing_loop(self):
        global_state.update_vision("RUNNING")
        logger.info("[VISION RUNNING] Inference loop active")

        last_loop_time = time.perf_counter()

        while self._running:
            loop_start = time.perf_counter()

            # Ensure camera remains connected
            if self.camera.status != "CONNECTED":
                logger.warning(f"[VISION PAUSED] Camera status: {self.camera.status}")
                time.sleep(0.1)
                continue

            has_new, frame = self.camera.get_frame()
            if not has_new or frame is None:
                self.frames_dropped += 1
                time.sleep(0.01)
                continue

            # Run inference & tracking
            t_infer_start = time.perf_counter()
            try:
                detections = self.detector.track(frame, tracker_config=self.tracker_config)
                active_ids, raw_count = self.tracker.update(detections)
                latency_ms = (time.perf_counter() - t_infer_start) * 1000
            except Exception as e:
                logger.error(f"[VISION ERROR] Exception during inference/tracking: {e}")
                global_state.update_vision("ERROR", error=str(e))
                break

            self.frames_processed += 1

            # Temporal smoothing (window median/mean)
            self.smoothing_history.append(raw_count)
            smoothed_count = int(round(float(np.mean(self.smoothing_history))))

            # Confidence calculation:
            # Arithmetic mean of active person detection confidences in current frame.
            # If 0 detections, confidence is None (rendered as "N/A" in UI).
            if len(detections) > 0:
                conf_values = [d["confidence"] for d in detections]
                aggregate_conf = float(np.mean(conf_values))
            else:
                aggregate_conf = None

            # Calculate FPS
            now = time.perf_counter()
            dt = now - last_loop_time
            last_loop_time = now
            if dt > 0:
                self._fps_history.append(1.0 / dt)
            self.current_fps = float(np.mean(self._fps_history)) if self._fps_history else 0.0

            # Annotate frame
            annotated_frame = self._annotate_frame(
                frame, detections, smoothed_count, self.current_fps, latency_ms
            )

            # Encode to JPEG for video feed
            try:
                ret, jpeg = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    with self._lock:
                        self._latest_jpeg = jpeg.tobytes()
            except Exception as e:
                logger.debug(f"[FRAME ENCODE ERROR] {e}")

            # Update Central Runtime State
            global_state.update_occupancy(
                occupancy=smoothed_count,
                raw_count=raw_count,
                confidence=aggregate_conf,
                tracked_persons=len(active_ids),
                tracked_ids=active_ids,
                processing_fps=self.current_fps,
                inference_latency_ms=latency_ms,
                frames_received=self.camera.frames_received,
                frames_processed=self.frames_processed,
                frames_dropped=self.frames_dropped
            )

            # Throttle to target FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, self.frame_interval - elapsed)
            time.sleep(sleep_time)

        if global_state.vision_status != "ERROR":
            global_state.update_vision("STOPPED")

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        occupancy: int,
        fps: float,
        latency_ms: float
    ) -> np.ndarray:
        """
        Overlays bounding boxes, identities, and engineering telemetry bar on the frame.
        """
        img = frame.copy()
        h, w = img.shape[:2]

        # Draw detected person boxes
        for det in detections:
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            track_id = det.get("track_id", "N/A")
            conf = det.get("confidence", 0.0)

            # Box color (subtle cyan/green #10b981 in BGR: 129, 185, 16)
            cv2.rectangle(img, (x1, y1), (x2, y2), (129, 185, 16), 2)

            label = f"ID:{track_id} {int(conf * 100)}%"
            # Background badge for label
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(
                img,
                (x1, max(0, y1 - 18)),
                (x1 + label_size[0] + 6, max(18, y1)),
                (20, 26, 36),
                -1
            )
            cv2.putText(
                img,
                label,
                (x1 + 3, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (240, 244, 248),
                1,
                cv2.LINE_AA
            )

        # Top Engineering HUD Overlay (dark translucent band)
        cv2.rectangle(img, (0, 0), (w, 36), (10, 14, 20), -1)
        cv2.line(img, (0, 36), (w, 36), (30, 42, 58), 1)

        # Left Branding
        cv2.putText(
            img,
            "HVEAC VISION",
            (14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 244, 248),
            1,
            cv2.LINE_AA
        )

        # Center Occupancy
        people_text = f"PEOPLE: {occupancy}"
        cv2.putText(
            img,
            people_text,
            (w // 2 - 45, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (16, 185, 129) if occupancy > 0 else (126, 140, 159),
            1,
            cv2.LINE_AA
        )

        # Right Telemetry
        telemetry_text = f"FPS: {fps:.1f} | {latency_ms:.0f}ms | TRACKING: ACTIVE"
        text_size, _ = cv2.getTextSize(telemetry_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(
            img,
            telemetry_text,
            (w - text_size[0] - 14, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (126, 140, 159),
            1,
            cv2.LINE_AA
        )

        return img

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg
