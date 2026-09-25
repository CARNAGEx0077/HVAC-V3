"""
HVEAC Control Center - Camera Subsystem
Provides robust, thread-safe frame acquisition from webcams or video streams.
"""

import cv2
import time
import logging
import threading
from typing import Tuple, Optional, Any
import numpy as np

from backend.occupancy.state import global_state

logger = logging.getLogger("hveac.camera")


class Camera:
    def __init__(self, source: Any = 0):
        self.source = source
        self._lock = threading.Lock()
        self._cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._new_frame_available = False
        self.frames_received = 0

    @property
    def status(self) -> str:
        return global_state.camera_status

    def start(self) -> bool:
        """
        Starts frame capture safely. Prevents duplicate capture workers.
        """
        with self._lock:
            if self._running and self._capture_thread and self._capture_thread.is_alive():
                logger.info(f"[CAMERA] Already active in state: {self.status}")
                return True

            logger.info(f"[CAMERA CONNECTING] Initiating connection to source: {self.source}")
            global_state.update_camera("CONNECTING", source=self.source)

            self._running = True
            self._capture_thread = threading.Thread(
                target=self._worker_loop,
                name="CameraWorkerThread",
                daemon=True
            )
            self._capture_thread.start()

        # Wait for initial frame verification (Windows USB drivers may take 3-5s)
        start_wait = time.time()
        while time.time() - start_wait < 6.0:
            if global_state.camera_status in ("CONNECTED", "ERROR"):
                break
            time.sleep(0.1)

        return global_state.camera_status == "CONNECTED"

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        Safely attempts opening the camera using appropriate OS backends.
        """
        cap = None
        if isinstance(self.source, int):
            # Attempt default API (MSMF on modern Windows)
            try:
                cap = cv2.VideoCapture(self.source)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception as e:
                logger.debug(f"[CAMERA] Default open failed: {e}")

            # Fallback to DirectShow
            try:
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if cap.isOpened():
                    return cap
                cap.release()
            except Exception as e:
                logger.debug(f"[CAMERA] DSHOW backend open failed: {e}")
        else:
            # String stream / file source
            cap = cv2.VideoCapture(str(self.source))
            if cap.isOpened():
                return cap

        return None

    def _worker_loop(self):
        """
        Dedicated single-worker capture loop.
        """
        cap = self._open_capture()
        if not cap or not cap.isOpened():
            err_msg = f"Failed to open video source: {self.source}"
            logger.error(f"[CAMERA ERROR] {err_msg}")
            global_state.update_camera("ERROR", source=self.source, error=err_msg)
            self._running = False
            return

        # Attempt to read the very first frame to confirm connection
        ret, frame = cap.read()
        if not ret or frame is None:
            err_msg = "Camera opened but failed to capture initial frame"
            logger.error(f"[CAMERA ERROR] {err_msg}")
            cap.release()
            global_state.update_camera("ERROR", source=self.source, error=err_msg)
            self._running = False
            return

        with self._lock:
            self._cap = cap
            self._latest_frame = frame.copy()
            self._new_frame_available = True
            self.frames_received += 1

        logger.info(f"[CAMERA CONNECTED] Video capture live ({frame.shape[1]}x{frame.shape[0]})")
        global_state.update_camera("CONNECTED", source=self.source)

        consecutive_read_errors = 0

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_errors += 1
                if consecutive_read_errors > 15:
                    err_msg = "Repeated frame read failures; camera stream lost"
                    logger.error(f"[CAMERA ERROR] {err_msg}")
                    global_state.update_camera("ERROR", source=self.source, error=err_msg)
                    break
                time.sleep(0.05)
                continue

            consecutive_read_errors = 0
            with self._lock:
                self._latest_frame = frame
                self._new_frame_available = True
                self.frames_received += 1

            time.sleep(0.005)

        # Cleanup
        logger.info("[CAMERA DISCONNECTED] Releasing capture resource")
        try:
            cap.release()
        except Exception as e:
            logger.warning(f"[CAMERA] Exception releasing capture: {e}")

        with self._lock:
            self._cap = None
            self._latest_frame = None
            self._running = False

        if global_state.camera_status != "ERROR":
            global_state.update_camera("DISCONNECTED", source=self.source)

    def stop(self) -> bool:
        """
        Safely stops capture and releases camera resource.
        """
        logger.info("[CAMERA STOP] Initiating stop sequence")
        global_state.update_camera("STOPPING", source=self.source)

        with self._lock:
            self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        with self._lock:
            if self._cap and self._cap.isOpened():
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._latest_frame = None

        global_state.update_camera("STOPPED", source=self.source)
        logger.info("[CAMERA STOPPED] Hardware resource safely released")
        return True

    def restart(self) -> bool:
        """
        STOP -> release -> reopen -> verify frame -> resume.
        """
        logger.info("[CAMERA RESTART] Performing clean camera restart")
        self.stop()
        time.sleep(0.5)
        return self.start()

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns (has_new_frame, frame_copy).
        Thread-safe copy prevents race conditions.
        """
        with self._lock:
            if not self._running or self._latest_frame is None:
                return False, None
            has_new = self._new_frame_available
            self._new_frame_available = False
            return has_new, self._latest_frame.copy()
