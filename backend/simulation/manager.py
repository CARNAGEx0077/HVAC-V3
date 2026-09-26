"""
Simulation Manager for HVEAC V3 Control Center.

Coordinates scenario playback, variable speed control (1x, 5x, 10x, 30x),
real-time state transitions, and WebSocket broadcast streaming.

Includes HVEAC Brain V1 shadow prediction integration:
- AI predictions are generated on each simulation step change
- Results are included in the WebSocket broadcast for frontend display
- Shadow mode is OBSERVATION ONLY — no HVAC control path exists
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fastapi import WebSocket

from simulator.config import SimulationEngineConfig
from simulator.engine import ScenarioEngine
from simulator.room import FIXED_COMPUTERS, FIXED_AC_UNITS
from simulator.scenarios import ALL_SCENARIOS

logger = logging.getLogger("hveac.simulation")

# Lazy-load shadow predictor to avoid blocking startup if ML deps are missing
_shadow_predictor = None

def _get_shadow_predictor():
    """Lazy-initialize the shadow predictor service."""
    global _shadow_predictor
    if _shadow_predictor is None:
        try:
            from ml.shadow_predictor import ShadowPredictor
            _shadow_predictor = ShadowPredictor()
            logger.info("[SIM] HVEAC Brain shadow predictor loaded")
        except Exception as e:
            logger.warning(f"[SIM] Shadow predictor unavailable: {e}")
    return _shadow_predictor

DATASET_DIR = Path("simulation_dataset")

SCENARIO_METADATA: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "Localized Heavy Compute",
        "description": "Four physically clustered computers (C1-C4) in Zone 1 run sustained heavy compute workloads while C5-C10 run light tasks with uniform room occupancy.",
        "dominant_heat_source": "Compute Cluster (Zone 1)",
        "expected_behavior": "Localized computational thermal hotspot forms in Zone 1 (NW quadrant).",
        "duration_seconds": 7200,
        "timestep_seconds": 10,
    },
    2: {
        "id": 2,
        "name": "Occupancy Concentration",
        "description": "All 10 computers maintain uniform light workloads, while 16 occupants gather for an intensive meeting in Zone 3 (SE quadrant).",
        "dominant_heat_source": "Human Occupancy (Zone 3)",
        "expected_behavior": "Occupancy metabolic heat drives a localized thermal hotspot in Zone 3.",
        "duration_seconds": 7200,
        "timestep_seconds": 10,
    },
    3: {
        "id": 3,
        "name": "Distributed Heavy Compute",
        "description": "All 10 computers spatially distributed across all 4 zones execute heavy concurrent compute tasks under balanced occupancy.",
        "dominant_heat_source": "Distributed Compute (10 Nodes)",
        "expected_behavior": "Thermal load rises broadly across the entire room, requiring uniform multi-unit cooling.",
        "duration_seconds": 7200,
        "timestep_seconds": 10,
    },
    4: {
        "id": 4,
        "name": "High Occupancy / Low Computer Load",
        "description": "Classroom/seminar setting with 34 occupants distributed room-wide while all 10 workstations operate in low-power idle.",
        "dominant_heat_source": "Human Occupancy (Room-Wide)",
        "expected_behavior": "Metabolic sensible heat dominates room heat budget by >4.5:1 over computers.",
        "duration_seconds": 7200,
        "timestep_seconds": 10,
    },
    5: {
        "id": 5,
        "name": "Opposing Thermal Zones",
        "description": "West side (Zones 1 & 4) is loaded with heavy compute nodes, while East side (Zones 2 & 3) hosts 20+ occupants under low compute.",
        "dominant_heat_source": "Dual Source (West Compute / East People)",
        "expected_behavior": "Two distinct opposing thermal regions develop from fundamentally different heat sources.",
        "duration_seconds": 7200,
        "timestep_seconds": 10,
    },
}


class SimulationManager:
    """Singleton simulation orchestrator driving real-time playback and WebSocket broadcasting."""

    def __init__(self):
        self.scenario_id: int = 1
        self.status: str = "READY"  # READY, RUNNING, PAUSED, COMPLETED
        self.speed: int = 10        # 1x, 5x, 10x, 30x
        self.sim_time_seconds: float = 0.0
        self.current_step_idx: int = 0
        self.timestep_seconds: float = 10.0
        self.duration_seconds: float = 7200.0

        # Control Mode: BASELINE (default), SHADOW (allowed), AI_CONTROL (temporarily disabled pending audit)
        self.control_mode: str = "BASELINE"
        self.allowed_control_modes: List[str] = ["BASELINE", "SHADOW"]
        self.disabled_control_modes: Dict[str, str] = {
            "AI_CONTROL": "AI_CONTROL is temporarily DISABLED pending thermal control audit"
        }

        # In-memory cached scenario datasets: {scenario_id: [record, ...]}
        self._scenario_data: Dict[int, List[Dict[str, Any]]] = {}
        # In-memory cached AI-controlled scenario datasets
        self._ai_scenario_data: Dict[int, List[Dict[str, Any]]] = {}
        # Closed-loop evaluation metrics and recent events
        self._closed_loop_metrics: Dict[str, Any] = {}
        self._scenario_events: Dict[int, List[Dict[str, Any]]] = {1: [], 2: [], 3: [], 4: [], 5: []}

        self._connected_sockets: Set[WebSocket] = set()
        self._playback_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Load existing scenario datasets or generate if missing
        self._load_or_generate_datasets()

    def _load_or_generate_datasets(self):
        """Loads pre-generated scenario records into memory for zero-latency replay."""
        import csv
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        config = SimulationEngineConfig()
        engine = ScenarioEngine(config)

        for sid in (1, 2, 3, 4, 5):
            jsonl_file = DATASET_DIR / f"scenario_{sid}.jsonl"
            records: List[Dict[str, Any]] = []

            if jsonl_file.exists():
                try:
                    with open(jsonl_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                records.append(json.loads(line))
                    logger.info(f"[SIM LOAD] Loaded Scenario {sid}: {len(records)} records from {jsonl_file.name}")
                except Exception as err:
                    logger.warning(f"[SIM LOAD] Failed to parse {jsonl_file}: {err}. Generating fresh...")
                    records = []

            if not records:
                scen_cls = ALL_SCENARIOS[sid]
                scen = scen_cls(duration_seconds=self.duration_seconds, timestep_seconds=self.timestep_seconds)
                records = engine.run_scenario(scen, seed=42)
                # Cache to disk
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                logger.info(f"[SIM GEN] Generated and cached Scenario {sid}: {len(records)} records")

            self._scenario_data[sid] = records

            # Also load AI-controlled scenario dataset if available
            ai_file = DATASET_DIR / f"scenario_{sid}_ai.jsonl"
            ai_records: List[Dict[str, Any]] = []
            if ai_file.exists():
                try:
                    with open(ai_file, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                ai_records.append(json.loads(line))
                    logger.info(f"[SIM LOAD] Loaded AI Scenario {sid}: {len(ai_records)} records from {ai_file.name}")
                except Exception as err:
                    logger.warning(f"[SIM LOAD] Failed to parse {ai_file}: {err}")
            self._ai_scenario_data[sid] = ai_records

        # Load closed-loop metrics
        metrics_file = Path("ml/reports/closed_loop_metrics.json")
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    self._closed_loop_metrics = json.load(f)
                logger.info("[SIM LOAD] Loaded closed_loop_metrics.json")
            except Exception as e:
                logger.warning(f"[SIM LOAD] Failed to load closed_loop_metrics.json: {e}")

        # Load control events
        events_file = Path("ml/reports/control_events.csv")
        if events_file.exists():
            try:
                with open(events_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            sid = int(row.get("scenario_id", 1))
                            if sid in self._scenario_events:
                                self._scenario_events[sid].append({
                                    "sim_time_s": float(row.get("sim_time_s", 0.0)),
                                    "time": row.get("time", "00:00:00"),
                                    "ai_requested_c": float(row.get("ai_requested_c", 25.5)),
                                    "applied_c": float(row.get("applied_c", 25.5)),
                                    "baseline_c": float(row.get("baseline_c", 22.5)),
                                    "status": row.get("status", "APPLIED"),
                                    "reason": row.get("reason", ""),
                                })
                        except (ValueError, TypeError):
                            continue
                logger.info(f"[SIM LOAD] Loaded control events: {sum(len(v) for v in self._scenario_events.values())} events")
            except Exception as e:
                logger.warning(f"[SIM LOAD] Failed to parse control_events.csv: {e}")

    # --------------------------------------------------------------------------
    # WebSocket Client Subscription
    # --------------------------------------------------------------------------

    async def register_client(self, websocket: WebSocket):
        await websocket.accept()
        self._connected_sockets.add(websocket)
        logger.info(f"[SIM WS] Client connected. Total clients: {len(self._connected_sockets)}")
        # Send immediate current state update
        await self._send_to_client(websocket, self.build_update_payload())

    def unregister_client(self, websocket: WebSocket):
        self._connected_sockets.discard(websocket)
        logger.info(f"[SIM WS] Client disconnected. Total clients: {len(self._connected_sockets)}")

    async def _send_to_client(self, websocket: WebSocket, payload: Dict[str, Any]):
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            self._connected_sockets.discard(websocket)

    async def broadcast_update(self):
        if not self._connected_sockets:
            return
        payload = self.build_update_payload()
        msg_text = json.dumps(payload)
        stale = set()
        for ws in self._connected_sockets:
            try:
                await ws.send_text(msg_text)
            except Exception:
                stale.add(ws)
        self._connected_sockets.difference_update(stale)

    # --------------------------------------------------------------------------
    # Telemetry Payload Assembly (Strictly adhering to Section 23)
    # --------------------------------------------------------------------------

    def build_update_payload(self) -> Dict[str, Any]:
        """Constructs machine-readable simulation snapshot for frontend visualization."""
        if self.control_mode == "AI_CONTROL" and self.scenario_id in self._ai_scenario_data and self._ai_scenario_data[self.scenario_id]:
            records = self._ai_scenario_data[self.scenario_id]
        else:
            records = self._scenario_data.get(self.scenario_id, [])

        base_records = self._scenario_data.get(self.scenario_id, [])
        max_steps = len(records)
        step = min(max(self.current_step_idx, 0), max(0, max_steps - 1))
        rec = records[step] if records else {}
        base_step = min(max(self.current_step_idx, 0), max(0, len(base_records) - 1))
        base_rec = base_records[base_step] if base_records else {}

        # 1. Computers array with trend tracking and thermal contribution
        computers = []
        prev_step = max(0, step - 3)
        prev_rec = records[prev_step] if records else {}

        for loc in FIXED_COMPUTERS:
            cid = loc.computer_id
            cpu = float(rec.get(f"cpu_util_comp_{cid}", 15.0))
            gpu = float(rec.get(f"gpu_util_comp_{cid}", 0.0))
            workload = str(rec.get(f"workload_cat_comp_{cid}", "IDLE"))
            heat = float(rec.get(f"heat_w_comp_{cid}", 45.0))

            # Thermal contribution indicator
            contrib = rec.get(f"thermal_contrib_comp_{cid}")
            if not contrib:
                if heat >= 240.0:
                    contrib = "VERY HIGH"
                elif heat >= 170.0:
                    contrib = "HIGH"
                elif heat >= 100.0:
                    contrib = "MODERATE"
                else:
                    contrib = "LOW"

            # Simulated status indicator
            st = rec.get(f"status_comp_{cid}")
            if not st:
                if workload in ("HEAVY", "CPU_INTENSIVE", "GPU_INTENSIVE") or cpu >= 75.0:
                    st = "HEAVY LOAD"
                elif workload == "IDLE" or cpu < 10.0:
                    st = "IDLE"
                else:
                    st = "RUNNING"

            # Live trend calculation from recent timesteps (last 30s)
            prev_cpu = float(prev_rec.get(f"cpu_util_comp_{cid}", cpu))
            prev_heat = float(prev_rec.get(f"heat_w_comp_{cid}", heat))
            cpu_delta = cpu - prev_cpu
            heat_delta = heat - prev_heat

            trend_cpu = "UP" if cpu_delta >= 1.5 else ("DOWN" if cpu_delta <= -1.5 else "STABLE")
            trend_heat = "RISING" if heat_delta >= 3.0 else ("FALLING" if heat_delta <= -3.0 else "STABLE")

            computers.append({
                "id": cid,
                "name": loc.name,
                "zone": loc.zone,
                "zone_display": loc.zone.replace("_", " ").title(),
                "x": loc.x,
                "y": loc.y,
                "cpu_util_percent": round(cpu, 1),
                "gpu_util_percent": round(gpu, 1),
                "workload_category": workload,
                "heat_watts": round(heat, 1),
                "thermal_contribution": contrib,
                "status": st,
                "trend_cpu": trend_cpu,
                "trend_heat": trend_heat,
                "cpu_thermal_index": round(min(max(cpu, 0.0), 100.0), 1),
                "workload_score": round(0.6 * cpu + 0.4 * gpu, 1),
            })

        # Summary statistics across 10 computers
        heavy_count = sum(1 for c in computers if c["status"] == "HEAVY LOAD" or c["workload_category"] in ("HEAVY", "CPU_INTENSIVE"))
        total_comp_heat = round(sum(c["heat_watts"] for c in computers), 1)
        avg_cpu = round(sum(c["cpu_util_percent"] for c in computers) / max(1, len(computers)), 1)
        avg_gpu = round(sum(c["gpu_util_percent"] for c in computers) / max(1, len(computers)), 1)

        comp_summary = {
            "total_count": len(computers),
            "active_count": len(computers),
            "heavy_count": heavy_count,
            "total_compute_heat_watts": total_comp_heat,
            "average_cpu_percent": avg_cpu,
            "average_gpu_percent": avg_gpu,
        }

        # 2. AC units array
        acs = []
        for loc in FIXED_AC_UNITS:
            ac_id = loc.ac_id
            suffix = ac_id.lower().replace("-", "")
            lvl = rec.get(f"cooling_level_{suffix}", 0.25)
            sp = rec.get(f"setpoint_{suffix}", 22.0)
            st = rec.get(f"state_{suffix}", "COOLING" if lvl > 0 else "OFF")
            delivered = round(lvl * 3500.0, 1)
            acs.append({
                "id": ac_id,
                "wall": loc.wall,
                "state": st,
                "cooling_level": lvl,
                "cooling_level_percent": round(lvl * 100, 1),
                "setpoint_c": sp,
                "delivered_cooling_watts": delivered,
            })

        # 3. Zone cooling map
        zone_cooling = {
            "zone_1": rec.get("hvac_cooling_zone_1_w", 0.0),
            "zone_2": rec.get("hvac_cooling_zone_2_w", 0.0),
            "zone_3": rec.get("hvac_cooling_zone_3_w", 0.0),
            "zone_4": rec.get("hvac_cooling_zone_4_w", 0.0),
        }

        # 4. Zone temperatures
        zone_temps = {
            "zone_1": rec.get("zone_1_temperature_c", 22.5),
            "zone_2": rec.get("zone_2_temperature_c", 22.5),
            "zone_3": rec.get("zone_3_temperature_c", 22.5),
            "zone_4": rec.get("zone_4_temperature_c", 22.5),
        }

        # 5. Occupancy
        occupancy = {
            "total": rec.get("total_occupancy", 0),
            "zones": {
                "zone_1": rec.get("occupancy_zone_1", 0),
                "zone_2": rec.get("occupancy_zone_2", 0),
                "zone_3": rec.get("occupancy_zone_3", 0),
                "zone_4": rec.get("occupancy_zone_4", 0),
            },
        }

        # 6. Thermal metrics
        thermal = {
            "zone_temperatures_c": zone_temps,
            "room_average_temperature_c": rec.get("room_average_temperature_c", 22.5),
            "minimum_temperature_c": rec.get("min_zone_temperature_c", 22.5),
            "maximum_temperature_c": rec.get("max_zone_temperature_c", 22.5),
            "temperature_difference_c": rec.get("temperature_gradient_c", 0.0),
            "computer_heat_watts": rec.get("total_computational_heat_w", 0.0),
            "occupancy_heat_watts": rec.get("total_occupancy_heat_w", 0.0),
            "hvac_cooling_watts": rec.get("total_hvac_cooling_w", 0.0),
            "total_net_heat_watts": rec.get("total_net_heat_w", 0.0),
        }

        # 7. Optimal ML Targets
        targets = {
            "optimal_cooling_ac1": rec.get("optimal_cooling_ac1", 0.0),
            "optimal_cooling_ac2": rec.get("optimal_cooling_ac2", 0.0),
            "optimal_cooling_ac3": rec.get("optimal_cooling_ac3", 0.0),
            "optimal_cooling_ac4": rec.get("optimal_cooling_ac4", 0.0),
            "optimal_temperature_c": rec.get("optimal_temperature_c", 22.5),
            "optimal_hvac_action": rec.get("optimal_hvac_action", "ECO_MAINTAIN"),
        }

        meta = SCENARIO_METADATA.get(self.scenario_id, {})

        payload = {
            "type": "simulation_update",
            "is_synthetic": True,
            "data_source": "HVEAC_SIMULATOR_V3",
            "scenario_id": self.scenario_id,
            "scenario_name": meta.get("name", "Unknown Scenario"),
            "status": self.status,
            "speed": self.speed,
            "simulation_time_seconds": round(self.sim_time_seconds, 1),
            "total_duration_seconds": self.duration_seconds,
            "step_index": step,
            "total_steps": max_steps,
            "progress_percent": round((step / max(1, max_steps - 1)) * 100, 1) if max_steps > 0 else 0.0,
            "occupancy": occupancy,
            "computers": computers,
            "computer_summary": comp_summary,
            "environment": {
                "outdoor_temperature_c": rec.get("outdoor_temperature_c", 28.0),
                "humidity_percent": rec.get("relative_humidity_percent", 50.0),
                "solar_irradiance_w_m2": rec.get("solar_irradiance_w_m2", 200.0),
            },
            "hvac": {
                "acs": acs,
                "zone_cooling_w": zone_cooling,
            },
            "thermal": thermal,
            "targets": targets,
        }

        # ── HVEAC Brain Shadow Prediction ──
        shadow = _get_shadow_predictor()
        if shadow is not None:
            try:
                shadow.predict(payload)
                payload["ai_shadow"] = shadow.get_full_shadow_state()
            except Exception as e:
                logger.debug(f"[SIM] Shadow prediction skipped: {e}")
                payload["ai_shadow"] = {
                    "enabled": True, "mode": "SHADOW",
                    "status": "ERROR", "model_version": "hveac_brain_v1",
                    "control_path": False, "error": str(e),
                }
        else:
            payload["ai_shadow"] = {
                "enabled": False, "mode": "SHADOW",
                "status": "UNAVAILABLE", "model_version": "hveac_brain_v1",
                "control_path": False,
            }

        # ── Closed-Loop AI Control Fields (Sections 13, 17, 18, 26) ──
        baseline_setpoint_c = float(base_rec.get("optimal_temperature_c", 22.5))

        if self.control_mode == "AI_CONTROL":
            ai_req = float(rec.get("ai_requested_setpoint_c", rec.get("optimal_temperature_c", 25.5)))
            ai_safe = float(rec.get("ai_safe_setpoint_c", rec.get("optimal_temperature_c", 25.5)))
            ai_applied = float(rec.get("ai_applied_setpoint_c", rec.get("optimal_temperature_c", 25.5)))
            safety_status = str(rec.get("safety_status", "APPLIED"))
            safety_reason = str(rec.get("safety_reason", "Safety Governor validated"))
            control_authority = "HVEAC BRAIN v1"
        elif self.control_mode == "SHADOW":
            shadow_state = payload.get("ai_shadow", {})
            pred_sp = shadow_state.get("predicted_setpoint_c")
            ai_req = float(pred_sp) if pred_sp is not None else 25.5
            ai_safe = ai_req
            ai_applied = baseline_setpoint_c
            safety_status = "OBSERVATION_ONLY"
            safety_reason = "Shadow observation mode — baseline authoritative"
            control_authority = "SIMULATOR OPTIMIZER"
        else:  # BASELINE
            ai_req = None
            ai_safe = None
            ai_applied = baseline_setpoint_c
            safety_status = "BASELINE_OPTIMIZER"
            safety_reason = "Simulator optimizer authoritative"
            control_authority = "SIMULATOR OPTIMIZER"

        sp_diff = round(ai_applied - baseline_setpoint_c, 2) if ai_applied is not None else 0.0

        # Section 18 WebSocket & REST telemetry fields
        payload["control_mode"] = self.control_mode
        payload["control_path"] = "SIMULATOR_ONLY"
        payload["control_authority"] = control_authority
        payload["ai_requested_setpoint_c"] = ai_req
        payload["ai_safe_setpoint_c"] = ai_safe
        payload["ai_applied_setpoint_c"] = ai_applied
        payload["baseline_setpoint_c"] = baseline_setpoint_c
        payload["setpoint_difference_c"] = sp_diff
        payload["safety_status"] = safety_status
        payload["safety_reason"] = safety_reason
        payload["model_status"] = "READY"
        payload["model_version"] = "hveac_brain_v1"

        # Bounded recent control events up to current simulation time
        events = self._scenario_events.get(self.scenario_id, [])
        recent_events = [
            e for e in events if e.get("sim_time_s", 0) <= self.sim_time_seconds
        ][-15:]

        scenario_metrics = self._closed_loop_metrics.get("scenarios", {}).get(str(self.scenario_id), {})

        payload["ai_control"] = {
            "mode": self.control_mode,
            "control_path": "SIMULATOR_ONLY",
            "control_authority": control_authority,
            "requested_setpoint_c": ai_req,
            "safe_setpoint_c": ai_safe,
            "applied_setpoint_c": ai_applied,
            "baseline_setpoint_c": baseline_setpoint_c,
            "difference_c": sp_diff,
            "safety_status": safety_status,
            "safety_reason": safety_reason,
            "recent_events": recent_events,
            "metrics": scenario_metrics,
        }

        return payload

    async def set_control_mode(self, mode: str) -> Dict[str, Any]:
        """Switches simulation control mode (BASELINE, SHADOW, AI_CONTROL)."""
        mode_upper = mode.upper().strip()
        if mode_upper not in self.allowed_control_modes:
            raise ValueError(f"Invalid control mode '{mode}'. Allowed modes: {self.allowed_control_modes}")

        async with self._lock:
            old_mode = self.control_mode
            self.control_mode = mode_upper
            logger.info(f"[SIM CONTROL] Mode transition: {old_mode} -> {self.control_mode}")

            # Reset shadow predictor rolling history
            shadow = _get_shadow_predictor()
            if shadow is not None:
                shadow.reset()

        await self.broadcast_update()
        return {
            "status": "success",
            "control_mode": self.control_mode,
            "previous_mode": old_mode,
            "control_path": "SIMULATOR_ONLY",
        }

    # --------------------------------------------------------------------------
    # Playback Lifecycle Management
    # --------------------------------------------------------------------------

    async def select_scenario(self, scenario_id: int) -> Dict[str, Any]:
        """Switches active scenario, stopping current playback and resetting timeline immediately."""
        async with self._lock:
            if scenario_id not in SCENARIO_METADATA:
                raise ValueError(f"Invalid scenario ID {scenario_id}. Choose 1 to 5.")
            self.stop_playback_loop()
            self.scenario_id = scenario_id
            self.current_step_idx = 0
            self.sim_time_seconds = 0.0
            self.status = "READY"
            # Reset shadow predictor rolling history for new scenario
            shadow = _get_shadow_predictor()
            if shadow is not None:
                shadow.reset()
            logger.info(f"[SIM] Selected Scenario {scenario_id}: {SCENARIO_METADATA[scenario_id]['name']}")

        await self.broadcast_update()
        return self.get_state()

    async def start(self) -> Dict[str, Any]:
        """Starts or resumes simulation playback."""
        async with self._lock:
            if self.status == "COMPLETED":
                # Restart from step 0
                self.current_step_idx = 0
                self.sim_time_seconds = 0.0

            self.status = "RUNNING"
            if self._playback_task is None or self._playback_task.done():
                self._playback_task = asyncio.create_task(self._run_playback_loop())
            logger.info(f"[SIM] Started Scenario {self.scenario_id} at {self.speed}x speed.")

        await self.broadcast_update()
        return self.get_state()

    async def pause(self) -> Dict[str, Any]:
        """Pauses simulation playback."""
        async with self._lock:
            if self.status == "RUNNING":
                self.status = "PAUSED"
                self.stop_playback_loop()
                logger.info(f"[SIM] Paused Scenario {self.scenario_id} at {self.sim_time_seconds:.1f}s.")

        await self.broadcast_update()
        return self.get_state()

    async def reset(self) -> Dict[str, Any]:
        """Resets simulation time to 0 while keeping current scenario."""
        async with self._lock:
            self.stop_playback_loop()
            self.current_step_idx = 0
            self.sim_time_seconds = 0.0
            self.status = "READY"
            # Reset shadow predictor rolling history
            shadow = _get_shadow_predictor()
            if shadow is not None:
                shadow.reset()
            logger.info(f"[SIM] Reset Scenario {self.scenario_id} to t=0.")

        await self.broadcast_update()
        return self.get_state()

    async def set_speed(self, speed: int) -> Dict[str, Any]:
        """Sets playback speed multiplier (1, 5, 10, 30)."""
        valid_speeds = (1, 5, 10, 30)
        if speed not in valid_speeds:
            raise ValueError(f"Speed must be one of {valid_speeds}, got {speed}")
        async with self._lock:
            self.speed = speed
            logger.info(f"[SIM] Playback speed set to {speed}x.")

        await self.broadcast_update()
        return self.get_state()

    def stop_playback_loop(self):
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        self._playback_task = None

    async def _run_playback_loop(self):
        """Background coroutine broadcasting ticks at ~10 Hz."""
        tick_interval = 0.1  # 100ms real time -> 10 updates / sec
        records = self._scenario_data.get(self.scenario_id, [])
        total_steps = len(records)

        try:
            while self.status == "RUNNING":
                await asyncio.sleep(tick_interval)

                # Advance simulated time
                sim_step_advance = tick_interval * self.speed
                self.sim_time_seconds += sim_step_advance
                self.current_step_idx = int(self.sim_time_seconds / self.timestep_seconds)

                if self.current_step_idx >= total_steps - 1:
                    self.current_step_idx = total_steps - 1
                    self.sim_time_seconds = self.duration_seconds
                    self.status = "COMPLETED"
                    logger.info(f"[SIM] Scenario {self.scenario_id} completed.")
                    await self.broadcast_update()
                    break

                await self.broadcast_update()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[SIM] Playback loop error: {e}")

    # --------------------------------------------------------------------------
    # Inspection and Reporting
    # --------------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return self.build_update_payload()

    def get_scenarios(self) -> List[Dict[str, Any]]:
        return list(SCENARIO_METADATA.values())

    def get_comparison_summary(self) -> List[Dict[str, Any]]:
        """Calculates cross-scenario comparison from real simulated data."""
        comparisons = []
        for sid in (1, 2, 3, 4, 5):
            records = self._scenario_data.get(sid, [])
            if not records:
                continue
            meta = SCENARIO_METADATA[sid]
            first_rec = records[0]
            last_rec = records[-1]

            avg_temps = [r["room_average_temperature_c"] for r in records]
            all_comp_heats = [r["total_computational_heat_w"] for r in records]
            all_occ_heats = [r["total_occupancy_heat_w"] for r in records]
            all_total_heats = [c + o for c, o in zip(all_comp_heats, all_occ_heats)]

            max_zone_temps = [r["max_zone_temperature_c"] for r in records]
            min_zone_temps = [r["min_zone_temperature_c"] for r in records]

            comparisons.append({
                "scenario_id": sid,
                "name": meta["name"],
                "dominant_heat_source": meta["dominant_heat_source"],
                "initial_temperature_c": first_rec["room_average_temperature_c"],
                "final_temperature_c": last_rec["room_average_temperature_c"],
                "maximum_temperature_c": max(max_zone_temps),
                "minimum_temperature_c": min(min_zone_temps),
                "average_temperature_c": round(sum(avg_temps) / len(avg_temps), 2),
                "peak_computational_heat_w": max(all_comp_heats),
                "peak_occupancy_heat_w": max(all_occ_heats),
                "peak_total_heat_w": round(max(all_total_heats), 1),
                "average_total_heat_w": round(sum(all_total_heats) / len(all_total_heats), 1),
            })
        return comparisons


# Singleton instance shared by FastAPI routes
global_simulation_manager = SimulationManager()
