"""
HVEAC Brain V1 — Shadow Mode API Router

Provides REST endpoints for the HVEAC Brain shadow-mode integration.

All endpoints are read-only / observation-only.
There is NO path from any endpoint to HVAC control.

Endpoints:
- GET  /api/brain/status          → Model health and metadata
- GET  /api/brain/shadow-predict  → Run shadow prediction on current sim state
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter

from ml.inference import get_model_status, predict_room_setpoint
from ml.feature_adapter import FeatureAdapter

logger = logging.getLogger("hveac.brain.api")

router = APIRouter(tags=["brain"])

# Singleton feature adapter (maintains rolling history per scenario session)
_adapter = FeatureAdapter()


@router.get("/api/brain/status")
async def brain_status():
    """Returns model health, version, and configuration metadata."""
    status = get_model_status()
    return {
        "brain": status,
        "integration_mode": "SHADOW_ONLY",
        "control_path": False,  # EXPLICIT: no HVAC control path
    }


@router.get("/api/brain/shadow-predict")
async def shadow_predict():
    """
    Run the Brain model against the current simulation state.

    This is SHADOW MODE only:
    - The prediction is returned for UI display and comparison.
    - There is NO automated action path.
    - The simulation continues using its own rule-based optimizer.
    """
    from backend.simulation.manager import global_simulation_manager

    try:
        # 1. Get current simulation state
        sim_state = global_simulation_manager.get_state()

        if not sim_state:
            return {
                "status": "NO_STATE",
                "error": "No simulation state available",
                "prediction": None,
                "mode": "SHADOW_ONLY",
            }

        # 2. Adapt simulation state → 80-feature vector
        features, adapter_warnings = _adapter.adapt(sim_state)

        # 3. Run model inference
        result = predict_room_setpoint(features)

        # 4. Get the simulator's own recommendation for comparison
        targets = sim_state.get("targets", {})
        simulator_setpoint = targets.get("optimal_temperature_c", None)
        simulator_action = targets.get("optimal_hvac_action", None)

        # 5. Compute agreement
        ai_setpoint = result.get("predicted_class")
        agreement = None
        deviation = None
        if ai_setpoint is not None and simulator_setpoint is not None:
            deviation = round(abs(ai_setpoint - simulator_setpoint), 2)
            agreement = deviation <= 0.5  # Within 0.5°C = agreement

        return {
            "status": result.get("status", "ERROR"),
            "mode": "SHADOW_ONLY",
            "control_path": False,
            "prediction": {
                "ai_setpoint_c": ai_setpoint,
                "confidence": result.get("confidence"),
                "class_probabilities": result.get("class_probabilities"),
                "all_classes": result.get("all_classes"),
            },
            "simulator": {
                "setpoint_c": simulator_setpoint,
                "action": simulator_action,
            },
            "comparison": {
                "agreement": agreement,
                "deviation_c": deviation,
            },
            "diagnostics": {
                "feature_warnings": result.get("feature_warnings", []) + adapter_warnings,
                "feature_count": len(features),
            },
            "simulation_context": {
                "scenario_id": sim_state.get("scenario_id"),
                "scenario_name": sim_state.get("scenario_name"),
                "simulation_time_seconds": sim_state.get("simulation_time_seconds"),
                "status": sim_state.get("status"),
            },
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"[BRAIN API] Shadow predict error: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "mode": "SHADOW_ONLY",
            "control_path": False,
            "prediction": None,
            "error": str(e),
        }


@router.post("/api/brain/adapter-reset")
async def reset_adapter():
    """Reset the feature adapter's rolling history (e.g. after scenario change)."""
    _adapter.reset()
    try:
        from backend.simulation.manager import _get_shadow_predictor
        shadow = _get_shadow_predictor()
        if shadow is not None:
            shadow.reset()
    except Exception:
        pass
    return {"status": "ok", "message": "Feature adapter history cleared"}
