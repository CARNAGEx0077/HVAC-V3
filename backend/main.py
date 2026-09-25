"""
HVEAC Control Center - Backend Application Server
FastAPI + Uvicorn + OpenCV + Ultralytics YOLO + WebSockets
"""

import sys
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend import config
from backend.occupancy.state import global_state
from backend.occupancy.camera import Camera
from backend.occupancy.detector import YoloDetector
from backend.occupancy.tracker import PersonTracker
from backend.occupancy.engine import OccupancyEngine
from backend.api import routes, websocket
from backend.simulation import router as simulation_router
from ml.brain_api import router as brain_router

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("hveac.main")

# Subsystem instances
camera_instance: Camera = None
detector_instance: YoloDetector = None
tracker_instance: PersonTracker = None
engine_instance: OccupancyEngine = None
broadcaster_task: asyncio.Task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global camera_instance, detector_instance, tracker_instance, engine_instance, broadcaster_task

    logger.info("==================================================")
    logger.info("  HVEAC — Environmental AI Control Center Server  ")
    logger.info("==================================================")
    logger.info(f"[SYSTEM START] Camera Source: {config.CAMERA_SOURCE} | Model: {config.MODEL_PATH}")

    # 1. Initialize Camera
    camera_instance = Camera(source=config.CAMERA_SOURCE)

    # 2. Initialize YOLO Detector
    try:
        detector_instance = YoloDetector(
            model_path=config.MODEL_PATH,
            conf_threshold=config.CONFIDENCE_THRESHOLD,
            device=config.DEVICE
        )
    except FileNotFoundError as fnf:
        logger.error(f"[SYSTEM ALERT] {fnf}")
        global_state.update_vision("ERROR", error="YOLO MODEL NOT AVAILABLE")
    except Exception as e:
        logger.error(f"[SYSTEM ALERT] YOLO initialization failed: {e}")
        global_state.update_vision("ERROR", error=str(e))

    # 3. Initialize Tracker
    tracker_instance = PersonTracker(max_stale_seconds=config.TRACKER_MAX_AGE / max(1, config.PROCESS_FPS))

    # 4. Initialize Engine if detector successfully loaded
    if detector_instance:
        engine_instance = OccupancyEngine(
            camera=camera_instance,
            detector=detector_instance,
            tracker=tracker_instance,
            target_fps=config.PROCESS_FPS,
            smoothing_window=config.OCCUPANCY_SMOOTHING_WINDOW,
            tracker_config=config.TRACKER_CONFIG
        )
        routes.set_engine_and_camera(engine_instance, camera_instance)

        # Attempt initial hardware start
        cam_started = camera_instance.start()
        if cam_started:
            engine_instance.start()
    else:
        routes.set_engine_and_camera(None, camera_instance)

    # 5. Start background WebSocket telemetry broadcast loop
    broadcaster_task = asyncio.create_task(websocket.broadcast_telemetry_loop())

    yield

    # Shutdown sequence
    logger.info("[SYSTEM SHUTDOWN] Stopping all vision and camera subsystems...")
    if broadcaster_task:
        broadcaster_task.cancel()
    if engine_instance:
        engine_instance.stop()
    if camera_instance:
        camera_instance.stop()
    logger.info("[SYSTEM SHUTDOWN] Complete.")


app = FastAPI(
    title="HVEAC Control Center",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register REST and WebSocket routers
app.include_router(routes.router)
app.include_router(websocket.router)
app.include_router(simulation_router.router)
app.include_router(brain_router)

# Mount frontend static files
frontend_path = config.FRONTEND_DIR
if frontend_path.exists():
    app.mount("/css", StaticFiles(directory=str(frontend_path / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend_path / "js")), name="js")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(frontend_path / "index.html"))

    @app.get("/{catchall:path}")
    async def catch_all(catchall: str):
        # Allow SPA hash routing
        requested = frontend_path / catchall
        if requested.exists() and requested.is_file():
            return FileResponse(str(requested))
        return FileResponse(str(frontend_path / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=False
    )
