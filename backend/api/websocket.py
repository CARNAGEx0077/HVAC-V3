"""
HVEAC Control Center - WebSocket Broadcast Manager
Pushes consolidated, non-blocking real-time telemetry updates to connected dashboard clients.
"""

import asyncio
import logging
from typing import Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.occupancy.state import global_state

logger = logging.getLogger("hveac.websocket")

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"[WS CONNECT] Client connected. Total active: {len(self.active_connections)}")
        # Send immediate initial snapshot
        initial_payload = global_state.get_websocket_payload()
        await websocket.send_json(initial_payload)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"[WS DISCONNECT] Client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, data: dict):
        if not self.active_connections:
            return

        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.debug(f"[WS BROADCAST ERROR] {e}")
                dead_connections.append(connection)

        for dead in dead_connections:
            self.disconnect(dead)


ws_manager = ConnectionManager()


@router.websocket("/ws/occupancy")
async def websocket_occupancy_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection open; receive ping/pong or control messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.debug(f"[WS ENDPOINT EXCEPTION] {e}")
        ws_manager.disconnect(websocket)


async def broadcast_telemetry_loop():
    """
    Background asyncio broadcaster task.
    Pushes state updates to all connected clients at regular intervals (~5 Hz).
    """
    logger.info("[WS BROADCASTER] Telemetry broadcaster started.")
    while True:
        try:
            if ws_manager.active_connections:
                payload = global_state.get_websocket_payload()
                await ws_manager.broadcast(payload)
        except Exception as e:
            logger.error(f"[WS BROADCASTER ERROR] {e}")
        await asyncio.sleep(0.2)
