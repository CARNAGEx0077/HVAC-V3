"""
HVEAC Control Center - REST API & Streaming Endpoints
"""

import cv2
import time
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import numpy as np

from backend.occupancy.state import global_state

logger = logging.getLogger("hveac.api")

router = APIRouter()

# Global engine and camera references injected at app startup
_app_engine = None
_app_camera = None


def set_engine_and_camera(engine, camera):
    global _app_engine, _app_camera
    _app_engine = engine
    _app_camera = camera


class ControlActionRequest(BaseModel):
    action: str = Field(..., description="Action: START, STOP, RESTART, RESET")


@router.get("/api/occupancy")
async def get_occupancy():
    """
    Returns live occupancy count, aggregate confidence, and processing stats.
    """
    return JSONResponse(content=global_state.get_occupancy_api_dict())


@router.get("/api/occupancy/status")
async def get_occupancy_status():
    """
    Returns full subsystem telemetry for camera, vision engine, and detector.
    """
    return JSONResponse(content=global_state.get_status_api_dict())


@router.post("/api/control/camera")
async def control_camera(req: ControlActionRequest):
    """
    Controls Camera lifecycle: START, STOP, RESTART.
    """
    if not _app_camera:
        raise HTTPException(status_code=500, detail="Camera subsystem not initialized")

    action = req.action.strip().upper()
    logger.info(f"[API COMMAND] Camera action: {action}")

    success = False
    message = ""

    if action == "START":
        success = _app_camera.start()
        message = "Camera started" if success else "Camera start failed"
    elif action == "STOP":
        success = _app_camera.stop()
        message = "Camera stopped"
    elif action == "RESTART":
        success = _app_camera.restart()
        message = "Camera restarted" if success else "Camera restart failed"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Expected START, STOP, or RESTART")

    return {
        "success": success,
        "target": "CAMERA",
        "action": action,
        "camera_status": global_state.camera_status,
        "message": message,
        "timestamp": global_state.timestamp
    }


@router.post("/api/control/vision")
async def control_vision(req: ControlActionRequest):
    """
    Controls Vision Engine lifecycle: START, STOP, RESTART.
    """
    if not _app_engine:
        raise HTTPException(status_code=500, detail="Vision Engine subsystem not initialized")

    action = req.action.strip().upper()
    logger.info(f"[API COMMAND] Vision action: {action}")

    success = False
    message = ""

    if action == "START":
        success = _app_engine.start()
        message = "Vision engine started" if success else "Vision start failed (check camera)"
    elif action == "STOP":
        success = _app_engine.stop()
        message = "Vision engine stopped"
    elif action == "RESTART":
        success = _app_engine.restart()
        message = "Vision engine restarted" if success else "Vision restart failed"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Expected START, STOP, or RESTART")

    return {
        "success": success,
        "target": "VISION ENGINE",
        "action": action,
        "vision_status": global_state.vision_status,
        "message": message,
        "timestamp": global_state.timestamp
    }


@router.post("/api/control/occupancy")
async def control_occupancy(req: ControlActionRequest):
    """
    Resets occupancy tracking state and temporal smoothing without stopping camera.
    """
    if not _app_engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")

    action = req.action.strip().upper()
    if action in ("RESET", "RESET STATE"):
        _app_engine.reset_occupancy()
        return {
            "success": True,
            "target": "OCCUPANCY",
            "action": action,
            "message": "Occupancy state and tracking reset successfully",
            "timestamp": global_state.timestamp
        }

    raise HTTPException(status_code=400, detail=f"Invalid action: {action}. Expected RESET")


def _generate_placeholder_frame(status_text: str = "CAMERA OFFLINE") -> bytes:
    """
    Generates a dark engineering placeholder frame when stream is inactive.
    """
    h, w = 480, 640
    img = np.full((h, w, 3), (12, 16, 23), dtype=np.uint8)

    # Brackets
    cv2.rectangle(img, (20, 20), (w - 20, h - 20), (28, 38, 54), 1)

    # Status text
    cv2.putText(
        img,
        "HVEAC VISION",
        (w // 2 - 80, h // 2 - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (126, 140, 159),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        img,
        status_text,
        (w // 2 - 95, h // 2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (239, 68, 68) if "OFFLINE" in status_text or "ERROR" in status_text else (126, 140, 159),
        1,
        cv2.LINE_AA
    )

    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes()


@router.get("/video_feed")
async def video_feed():
    """
    Streams live annotated MJPEG video from the Occupancy Engine.
    """
    def frame_generator():
        while True:
            frame_bytes = None
            if _app_engine and global_state.vision_status == "RUNNING":
                frame_bytes = _app_engine.get_latest_jpeg()

            if frame_bytes is None:
                # Render clean dark standby frame
                status_msg = f"CAMERA {global_state.camera_status}"
                frame_bytes = _generate_placeholder_frame(status_msg)
                time.sleep(0.2)
            else:
                time.sleep(0.04)  # ~25 FPS stream pacing

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
