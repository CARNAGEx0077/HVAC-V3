"""
FastAPI REST and WebSocket Router for HVEAC V3 Simulation Lab.

Endpoints:
- GET  /api/simulation/scenarios
- GET  /api/simulation/state
- POST /api/simulation/select
- POST /api/simulation/start
- POST /api/simulation/pause
- POST /api/simulation/reset
- POST /api/simulation/speed
- GET  /api/simulation/comparison
- WS   /ws/simulation
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from backend.simulation.manager import global_simulation_manager

router = APIRouter(tags=["simulation"])


class ScenarioSelectRequest(BaseModel):
    scenario_id: int = Field(..., ge=1, le=5, description="Scenario ID between 1 and 5")


class SimulationSpeedRequest(BaseModel):
    speed: int = Field(..., description="Playback speed multiplier (1, 5, 10, 30)")


@router.get("/api/simulation/scenarios")
async def get_simulation_scenarios():
    """Returns metadata for all 5 simulation scenarios."""
    return {
        "scenarios": global_simulation_manager.get_scenarios(),
        "active_scenario_id": global_simulation_manager.scenario_id,
        "is_synthetic": True,
    }


@router.get("/api/simulation/state")
async def get_simulation_state():
    """Returns the current snapshot of the active simulation."""
    return global_simulation_manager.get_state()


@router.post("/api/simulation/select")
async def select_simulation_scenario(req: ScenarioSelectRequest):
    """Switches active scenario, stopping current playback and resetting state."""
    try:
        new_state = await global_simulation_manager.select_scenario(req.scenario_id)
        return {
            "status": "success",
            "message": f"Scenario {req.scenario_id} selected and loaded.",
            "state": new_state,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/simulation/start")
async def start_simulation():
    """Starts or resumes simulation playback."""
    state = await global_simulation_manager.start()
    return {
        "status": "success",
        "action": "start",
        "state": state,
    }


@router.post("/api/simulation/pause")
async def pause_simulation():
    """Pauses simulation playback."""
    state = await global_simulation_manager.pause()
    return {
        "status": "success",
        "action": "pause",
        "state": state,
    }


@router.post("/api/simulation/reset")
async def reset_simulation():
    """Resets simulation time to 0 while keeping current scenario."""
    state = await global_simulation_manager.reset()
    return {
        "status": "success",
        "action": "reset",
        "state": state,
    }


@router.post("/api/simulation/speed")
async def set_simulation_speed(req: SimulationSpeedRequest):
    """Sets playback acceleration factor (1x, 5x, 10x, 30x)."""
    try:
        state = await global_simulation_manager.set_speed(req.speed)
        return {
            "status": "success",
            "speed": req.speed,
            "state": state,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.get("/api/simulation/comparison")
async def get_simulation_comparison():
    """Returns thermal summary comparison metrics across all 5 scenarios."""
    return {
        "is_synthetic": True,
        "comparisons": global_simulation_manager.get_comparison_summary(),
    }


@router.websocket("/ws/simulation")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live simulation updates (~10 Hz).
    Allows client to receive real-time thermal evolution and send play/pause commands.
    """
    await global_simulation_manager.register_client(websocket)
    try:
        while True:
            # Listen for optional incoming client controls over WebSocket
            data = await websocket.receive_text()
            try:
                msg = pydantic_parse = None
                import json
                msg = json.loads(data)
                cmd = msg.get("action")
                if cmd == "start":
                    await global_simulation_manager.start()
                elif cmd == "pause":
                    await global_simulation_manager.pause()
                elif cmd == "reset":
                    await global_simulation_manager.reset()
                elif cmd == "select":
                    sid = msg.get("scenario_id", 1)
                    await global_simulation_manager.select_scenario(sid)
                elif cmd == "speed":
                    spd = msg.get("speed", 10)
                    await global_simulation_manager.set_speed(spd)
            except Exception:
                pass
    except WebSocketDisconnect:
        global_simulation_manager.unregister_client(websocket)
    except Exception:
        global_simulation_manager.unregister_client(websocket)
