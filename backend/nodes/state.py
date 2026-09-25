"""
HVEAC V3 - Node State & Liveness Manager
Thread-safe in-memory store tracking compute nodes, computational thermal load proxies,
and multi-node cluster aggregations.
"""

import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from backend.nodes.schemas import NodeTelemetryPayload


class NodeRecord:
    def __init__(self, payload: NodeTelemetryPayload):
        now = datetime.now(timezone.utc)
        self.node_id = payload.node_id
        self.first_seen = now
        self.last_seen = now
        self.sequence = payload.sequence
        self.payload = payload

    def update(self, payload: NodeTelemetryPayload):
        self.last_seen = datetime.now(timezone.utc)
        self.sequence = payload.sequence
        self.payload = payload

    def get_liveness(self, now: datetime) -> str:
        diff_seconds = (now - self.last_seen).total_seconds()
        if diff_seconds <= 6.0:
            return "ONLINE"
        elif diff_seconds <= 15.0:
            return "STALE"
        else:
            return "OFFLINE"

    def to_summary_dict(self, now: datetime) -> Dict[str, Any]:
        liveness = self.get_liveness(now)
        diff_seconds = round((now - self.last_seen).total_seconds(), 1)
        p = self.payload

        # CPU telemetry & proxy values
        cpu_util = p.cpu.utilization_percent
        cpu_str = f"{round(cpu_util)}%"
        cpu_temp_str = f"{round(p.cpu.temperature_c)}°C" if p.cpu.temperature_c is not None else "N/A"
        cpu_pwr_str = f"{round(p.cpu.power_watts, 1)}W" if p.cpu.power_watts is not None else "N/A"

        cpu_workload = p.cpu.workload_category or "IDLE"
        cpu_thermal_idx = p.cpu.thermal_load_index
        cpu_thermal_idx_str = f"{round(cpu_thermal_idx)} / 100" if cpu_thermal_idx is not None else "N/A"
        cpu_signal_state = p.cpu.thermal_signal_state or "UNAVAILABLE"

        # GPU telemetry (primary / active GPU)
        gpu_str = "N/A"
        gpu_temp_str = "N/A"
        gpu_pwr_str = "N/A"
        gpu_name = "None"
        gpu_util = None

        if p.gpu.available and len(p.gpu.gpus) > 0:
            best_gpu = max(p.gpu.gpus, key=lambda g: (g.power_watts or 0.0, g.utilization_percent))
            gpu_util = best_gpu.utilization_percent
            gpu_str = f"{round(best_gpu.utilization_percent)}%"
            gpu_temp_str = f"{round(best_gpu.temperature_c)}°C" if best_gpu.temperature_c is not None else "N/A"
            gpu_pwr_str = f"{round(best_gpu.power_watts, 1)}W" if best_gpu.power_watts is not None else "N/A"
            gpu_name = best_gpu.name

        ram_str = f"{round(p.memory.utilization_percent)}%"

        # Node-level thermal proxy vs measured power
        node_thermal_idx = p.thermal.node_thermal_load_index
        node_thermal_idx_str = f"{round(node_thermal_idx)} / 100" if node_thermal_idx is not None else "N/A"
        thermal_signal = p.thermal.signal_state or p.thermal.estimation_status or "UNAVAILABLE"

        # Measured electrical power (strictly None if unavailable; NEVER 0 unless physically measured 0)
        est_heat_val = p.thermal.estimated_heat_watts
        est_heat_str = f"{round(est_heat_val, 1)}W" if est_heat_val is not None else "N/A"

        return {
            "id": self.node_id,
            "hostname": p.system.hostname,
            "status": liveness,
            "last_seen_seconds_ago": diff_seconds,

            # CPU telemetry
            "cpu": cpu_str,
            "cpu_num": cpu_util,
            "cpu_temp": cpu_temp_str,
            "cpu_power": cpu_pwr_str,
            "cpu_workload": cpu_workload,
            "cpu_thermal_index": cpu_thermal_idx,
            "cpu_thermal_index_str": cpu_thermal_idx_str,
            "cpu_short_term_avg": p.cpu.short_term_average_percent,
            "cpu_sustained_sec": p.cpu.sustained_load_seconds,
            "cpu_is_sustained_high": p.cpu.is_sustained_high_load,
            "cpu_thermal_signal": cpu_signal_state,

            # GPU telemetry
            "gpu": gpu_str,
            "gpu_num": gpu_util,
            "gpu_temp": gpu_temp_str,
            "gpu_power": gpu_pwr_str,
            "gpu_name": gpu_name,

            # RAM & Workload
            "ram": ram_str,
            "ram_num": p.memory.utilization_percent,
            "workload": p.workload.category,

            # Node Thermal Signals
            "node_thermal_index": node_thermal_idx,
            "node_thermal_index_str": node_thermal_idx_str,
            "thermal_signal": thermal_signal,
            "est_heat": est_heat_str,
            "est_heat_watts": est_heat_val if liveness == "ONLINE" else None,

            # Metadata
            "sequence": self.sequence,
            "agent_version": p.system.agent_version,
            "timestamp": p.timestamp
        }


class NodeStateManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._nodes: Dict[str, NodeRecord] = {}

    def record_telemetry(self, payload: NodeTelemetryPayload) -> Dict[str, Any]:
        with self._lock:
            node_id = payload.node_id
            if node_id in self._nodes:
                self._nodes[node_id].update(payload)
            else:
                self._nodes[node_id] = NodeRecord(payload)

            return {
                "status": "ok",
                "node_id": node_id,
                "sequence": payload.sequence,
                "timestamp": payload.timestamp
            }

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._nodes.get(node_id)
            if not record:
                return None
            now = datetime.now(timezone.utc)
            summary = record.to_summary_dict(now)
            summary["latest_telemetry"] = record.payload.model_dump()
            return summary

    def get_all_nodes(self) -> Dict[str, Any]:
        with self._lock:
            now = datetime.now(timezone.utc)
            nodes_list = [record.to_summary_dict(now) for record in self._nodes.values()]

            # Sort nodes by node_id
            nodes_list.sort(key=lambda n: n["id"])

            total_nodes = len(nodes_list)
            online_nodes = [n for n in nodes_list if n["status"] == "ONLINE"]
            online_count = len(online_nodes)
            stale_count = sum(1 for n in nodes_list if n["status"] == "STALE")
            offline_count = sum(1 for n in nodes_list if n["status"] == "OFFLINE")

            # 1. Average CPU Utilization
            avg_cpu_val = (sum(n["cpu_num"] for n in online_nodes) / online_count) if online_count > 0 else None

            # 2. Average GPU Utilization
            gpu_readings = [n["gpu_num"] for n in online_nodes if n.get("gpu_num") is not None]
            avg_gpu_val = (sum(gpu_readings) / len(gpu_readings)) if len(gpu_readings) > 0 else None

            # 3. Node Thermal Load Index (0 - 100 proxy)
            thermal_indexes = [n["node_thermal_index"] for n in online_nodes if n.get("node_thermal_index") is not None]
            avg_thermal_index = (sum(thermal_indexes) / len(thermal_indexes)) if len(thermal_indexes) > 0 else None
            max_thermal_index = max(thermal_indexes) if len(thermal_indexes) > 0 else None

            # 4. Total Measured Electrical Power (in Watts)
            # CRITICAL: Do NOT treat missing power as zero. Only aggregate nodes with real power.
            measured_powers = [n["est_heat_watts"] for n in online_nodes if n.get("est_heat_watts") is not None]
            total_measured_power = sum(measured_powers) if len(measured_powers) > 0 else None

            # Format cluster thermal summary text
            if total_measured_power is not None:
                total_heat_display = f"{round(total_measured_power, 1)} W"
                thermal_mode = "MEASURED" if all(n.get("thermal_signal") == "MEASURED" for n in online_nodes) else "PARTIAL"
            elif avg_thermal_index is not None:
                total_heat_display = f"{round(avg_thermal_index)} / 100"
                thermal_mode = "PROXY"
            else:
                total_heat_display = "0 W"
                thermal_mode = "UNAVAILABLE"

            return {
                "total_count": total_nodes,
                "online_count": online_count,
                "stale_count": stale_count,
                "offline_count": offline_count,

                # Aggregated Averages & Totals
                "average_cpu_utilization": round(avg_cpu_val, 1) if avg_cpu_val is not None else None,
                "avg_cpu": f"{round(avg_cpu_val)}%" if avg_cpu_val is not None else "N/A",

                "average_gpu_utilization": round(avg_gpu_val, 1) if avg_gpu_val is not None else None,
                "avg_gpu": f"{round(avg_gpu_val)}%" if avg_gpu_val is not None else "N/A",

                "average_node_thermal_load_index": round(avg_thermal_index, 1) if avg_thermal_index is not None else None,
                "avg_node_thermal_load_index_str": f"{round(avg_thermal_index)} / 100" if avg_thermal_index is not None else "N/A",
                "maximum_node_thermal_load_index": round(max_thermal_index, 1) if max_thermal_index is not None else None,

                "total_measured_power_watts": round(total_measured_power, 1) if total_measured_power is not None else None,
                "total_heat_watts": total_heat_display,
                "total_heat_watts_num": round(total_measured_power, 1) if total_measured_power is not None else None,
                "cluster_thermal_mode": thermal_mode,

                # Detailed Node List
                "nodes": nodes_list
            }


# Global singleton instance
global_node_state = NodeStateManager()
