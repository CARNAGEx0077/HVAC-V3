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


@router.get("/api/simulation/ai-shadow")
async def get_ai_shadow_state():
    """
    Returns the full HVEAC Brain shadow prediction state.

    This is SHADOW MODE ONLY:
    - The prediction is returned for UI display and comparison.
    - There is NO automated action path.
    - The simulation continues using its own rule-based optimizer.
    """
    from backend.simulation.manager import _get_shadow_predictor

    shadow = _get_shadow_predictor()
    if shadow is None:
        return {
            "enabled": False,
            "mode": "SHADOW",
            "status": "UNAVAILABLE",
            "model_version": "hveac_brain_v1",
            "control_path": False,
            "error": "Shadow predictor not initialized",
        }

    return {
        "shadow_state": shadow.get_full_shadow_state(),
        "metrics": shadow.get_metrics(),
        "history": shadow.get_history()[-60:],  # Last 60 data points for chart
        "feature_snapshot": shadow.get_grouped_features(),
        "feature_warnings": shadow.get_feature_warnings(),
        "mode": "SHADOW",
        "control_path": False,
    }


class ControlModeRequest(BaseModel):
    control_mode: str = Field(..., description="Control mode: BASELINE, SHADOW, or AI_CONTROL")


@router.get("/api/simulation/control-mode")
async def get_simulation_control_mode():
    """Returns active simulation control mode (BASELINE, SHADOW; AI_CONTROL disabled)."""
    return {
        "control_mode": global_simulation_manager.control_mode,
        "allowed_modes": global_simulation_manager.allowed_control_modes,
        "disabled_modes": getattr(global_simulation_manager, "disabled_control_modes", {}),
        "control_path": "SIMULATOR_ONLY",
        "is_synthetic": True,
    }


@router.post("/api/simulation/control-mode")
async def set_simulation_control_mode(req: ControlModeRequest):
    """Sets simulation control mode (BASELINE, SHADOW, AI_CONTROL)."""
    try:
        result = await global_simulation_manager.set_control_mode(req.control_mode)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


@router.get("/api/simulation/ai-control")
async def get_ai_control_state():
    """Returns AI control closed-loop evaluation metrics and recent safety events."""
    return {
        "control_mode": global_simulation_manager.control_mode,
        "control_path": "SIMULATOR_ONLY",
        "metrics": global_simulation_manager._closed_loop_metrics,
        "active_scenario_id": global_simulation_manager.scenario_id,
        "events": global_simulation_manager._scenario_events.get(global_simulation_manager.scenario_id, [])[-30:],
        "is_synthetic": True,
    }


@router.websocket("/ws/simulation")
async def websocket_simulation_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint streaming live simulation updates (~10 Hz).
    Allows client to receive real-time thermal evolution and send play/pause/mode commands.
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
                elif cmd == "control_mode":
                    mode = msg.get("control_mode", "BASELINE")
                    await global_simulation_manager.set_control_mode(mode)
            except Exception:
                pass
    except WebSocketDisconnect:
        global_simulation_manager.unregister_client(websocket)
    except Exception:
        global_simulation_manager.unregister_client(websocket)
