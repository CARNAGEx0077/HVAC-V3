"""
Simulation Manager for HVEAC V3 Control Center.

Coordinates scenario playback, variable speed control (1x, 5x, 10x, 30x),
real-time state transitions, and WebSocket broadcast streaming.

RUNTIME ARCHITECTURE:
    SIMULATION STATE
        ↓
    THERMAL CONTROL ALGORITHM
        ↓
    SAFETY GOVERNOR
        ↓
    HVAC ACTUATORS
        ↓
    THERMAL PHYSICS
        ↓
    NEXT SIMULATION STATE

There is NO runtime optimizer. The Thermal Control Algorithm is the
ONLY decision-maker. It is a prototype substitute for a future HVEAC
AI model.
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
from backend.control.thermal_control_algorithm import (
    ThermalControlAlgorithm,
    HVACDecision,
    COMFORT_TARGET_C,
)
from backend.control.safety_governor import SafetyGovernor

logger = logging.getLogger("hveac.simulation")

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

        # Control Mode: PROTOTYPE_CONTROL (default active mode)
        self.control_mode: str = "PROTOTYPE_CONTROL"
        self.allowed_control_modes: List[str] = ["PROTOTYPE_CONTROL", "BASELINE", "SHADOW"]
        self.disabled_control_modes: Dict[str, str] = {
            "AI_CONTROL": "AI/ML model integration is disabled in prototype; Thermal Control Algorithm is the active controller."
        }

        # In-memory cached scenario datasets: {scenario_id: [record, ...]}
        self._scenario_data: Dict[int, List[Dict[str, Any]]] = {}

        # Thermal Control Algorithm — the ONLY runtime decision-maker
        self._controller = ThermalControlAlgorithm()
        self._safety_governor = SafetyGovernor()
        self._last_decision: Optional[HVACDecision] = None

        self._connected_sockets: Set[WebSocket] = set()
        self._playback_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Load existing scenario datasets or generate if missing
        self._load_or_generate_datasets()

    def _load_or_generate_datasets(self):
        """Loads pre-generated scenario records into memory for zero-latency replay."""
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        engine = None

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
                if engine is None:
                    config = SimulationEngineConfig()
                    engine = ScenarioEngine(config)
                scen_cls = ALL_SCENARIOS[sid]
                scen = scen_cls(duration_seconds=self.duration_seconds, timestep_seconds=self.timestep_seconds)
                records = engine.run_scenario(scen, seed=42)
                # Cache to disk
                with open(jsonl_file, "w", encoding="utf-8") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                logger.info(f"[SIM GEN] Generated and cached Scenario {sid}: {len(records)} records")

            self._scenario_data[sid] = records

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
    # Telemetry Payload Assembly
    # --------------------------------------------------------------------------

    def build_update_payload(self) -> Dict[str, Any]:
        """Constructs machine-readable simulation snapshot for frontend visualization."""
        records = self._scenario_data.get(self.scenario_id, [])
        max_steps = len(records)
        step = min(max(self.current_step_idx, 0), max(0, max_steps - 1))
        rec = records[step] if records else {}

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

        # 2. AC units array — driven by Thermal Control Algorithm
        acs = []
        for loc in FIXED_AC_UNITS:
            ac_id = loc.ac_id
            suffix = ac_id.lower().replace("-", "")

            # Use live controller decision if available
            if self._last_decision and not self._last_decision.is_fallback:
                cooling_pct = self._last_decision.ac_cooling_percent.get(ac_id, 0.0)
                lvl = cooling_pct / 100.0
                sp = self._last_decision.ac_setpoint_c.get(ac_id, 24.0)
            else:
                lvl = rec.get(f"cooling_level_{suffix}", 0.25)
                sp = rec.get(f"setpoint_{suffix}", 22.0)

            st = "COOLING" if lvl > 0.01 else "IDLE"
            delivered = round(lvl * 3500.0, 1)
            acs.append({
                "id": ac_id,
                "wall": loc.wall,
                "state": st,
                "cooling_level": round(lvl, 4),
                "cooling_level_percent": round(lvl * 100, 1),
                "setpoint_c": round(sp, 2),
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

        # ── Compute Thermal Control Algorithm Decision ──
        # Build a simulation state dict for the controller
        controller_input = {
            "thermal": thermal,
            "computers": computers,
            "occupancy": occupancy,
            "timestep_seconds": self.timestep_seconds,
        }
        raw_decision = self._controller.compute(controller_input)

        # ── Validate & Constrain via Safety Governor ──
        decision = self._safety_governor.validate_decision(raw_decision, sim_time_s=self.sim_time_seconds)
        self._last_decision = decision

        # Zone-level heat loads from the controller
        zone_heat_loads = {}
        zone_trends = {}
        zone_demands = {}
        for z_id in ("ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"):
            zs = decision.zone_states.get(z_id)
            if zs:
                zone_heat_loads[z_id.lower()] = zs.heat_load_w
                zone_trends[z_id.lower()] = zs.temperature_trend_c_per_min
                zone_demands[z_id.lower()] = zs.cooling_demand

        # Update AC values in the acs array with the validated decision
        for ac_entry in acs:
            ac_id = ac_entry["id"]
            cooling_pct = decision.ac_cooling_percent.get(ac_id, 0.0)
            lvl = cooling_pct / 100.0
            sp = decision.ac_setpoint_c.get(ac_id, 24.0)
            ac_entry["cooling_level"] = round(lvl, 4)
            ac_entry["cooling_level_percent"] = round(cooling_pct, 1)
            ac_entry["setpoint_c"] = round(sp, 2)
            ac_entry["state"] = "COOLING" if lvl > 0.01 else "IDLE"
            ac_entry["delivered_cooling_watts"] = round(lvl * 3500.0, 1)

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
        }

        # ── Thermal Control Algorithm & Safety Governor telemetry ──
        payload["control_mode"] = "PROTOTYPE_CONTROL"
        payload["control_path"] = "THERMAL_CONTROL_ALGORITHM → SAFETY_GOVERNOR → HVAC"
        payload["control_authority"] = "THERMAL CONTROL ALGORITHM"
        payload["controller_name"] = decision.controller_name
        gov_rep = self._safety_governor.last_report
        payload["safety_status"] = gov_rep.status if gov_rep else "ACTIVE"
        payload["safety_reason"] = gov_rep.reason if gov_rep else "Safety Governor validates all commands"
        payload["model_status"] = "NOT_INTEGRATED"
        payload["model_version"] = "prototype_deterministic_v1"

        # Per-zone control telemetry
        payload["control"] = {
            "mode": "PROTOTYPE_CONTROL",
            "controller": "THERMAL CONTROL ALGORITHM",
            "comfort_target_c": COMFORT_TARGET_C,
            "room_temperature_c": decision.room_temperature_c,
            "room_cooling_demand": decision.room_cooling_demand,
            "room_setpoint_c": decision.room_setpoint_c,
            "is_fallback": decision.is_fallback,
            "zone_heat_loads_w": zone_heat_loads,
            "zone_trends_c_per_min": zone_trends,
            "zone_demands_percent": zone_demands,
            "zones": {},
            "acs": {},
        }
        for z_id in ("ZONE_1", "ZONE_2", "ZONE_3", "ZONE_4"):
            zs = decision.zone_states.get(z_id)
            if zs:
                payload["control"]["zones"][z_id.lower()] = {
                    "temperature_c": zs.temperature_c,
                    "heat_load_w": zs.heat_load_w,
                    "trend_c_per_min": zs.temperature_trend_c_per_min,
                    "temperature_demand": zs.temperature_demand,
                    "heat_demand": zs.heat_demand,
                    "trend_demand": zs.trend_demand,
                    "cooling_demand": zs.cooling_demand,
                }
        for ac_id in ("AC-1", "AC-2", "AC-3", "AC-4"):
            payload["control"]["acs"][ac_id.lower().replace("-", "")] = {
                "cooling_percent": decision.ac_cooling_percent.get(ac_id, 0.0),
                "setpoint_c": decision.ac_setpoint_c.get(ac_id, 24.0),
            }

        # Decision trace for debug panel
        payload["decision_trace"] = {
            "flow": [
                "THERMAL SENSORS",
                "THERMAL CONTROL ALGORITHM",
                "SAFETY GOVERNOR",
                "HVAC ACTUATION",
                "THERMAL RESPONSE",
            ],
            "controller": decision.controller_name,
            "is_fallback": decision.is_fallback,
        }

        return payload

    async def set_control_mode(self, mode: str) -> Dict[str, Any]:
        """Control mode is fixed to PROTOTYPE_CONTROL."""
        mode_upper = mode.upper().strip()
        if mode_upper not in self.allowed_control_modes:
            raise ValueError(f"Invalid control mode '{mode}'. Allowed modes: {self.allowed_control_modes}")

        async with self._lock:
            self.control_mode = mode_upper
            logger.info(f"[SIM CONTROL] Mode: {self.control_mode}")

        await self.broadcast_update()
        return {
            "status": "success",
            "control_mode": self.control_mode,
            "control_authority": "THERMAL CONTROL ALGORITHM",
            "control_path": "THERMAL_CONTROL_ALGORITHM → SAFETY_GOVERNOR → HVAC",
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
            # Reset controller state for new scenario
            self._controller.reset()
            self._safety_governor.reset()
            self._last_decision = None
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
                self._controller.reset()
                self._safety_governor.reset()
                self._last_decision = None

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
            # Reset controller state
            self._controller.reset()
            self._safety_governor.reset()
            self._last_decision = None
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
