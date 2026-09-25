/**
 * HVEAC Control Center - Centralized State Architecture
 * 
 * Provides an isolated, predictable single source of truth for the dashboard.
 * Designed to seamlessly connect to WebSocket or REST backends in future phases.
 */

(function (window) {
  'use strict';

  // Default initial state directly matching initial engineering specs
  const initialState = {
    system: {
      status: 'OFFLINE', // 'OFFLINE' | 'ONLINE'
      location: 'Vandalur, Chennai',
      mode: 'DEMO CONTROL',
      backendConnected: false
    },
    pipeline: [
      { id: 'camera', title: 'CAMERA / OCCUPANCY', status: 'OFFLINE', planned: false },
      { id: 'nodes', title: 'COMPUTER NODES', status: 'OFFLINE', planned: false },
      { id: 'environment', title: 'ENVIRONMENT', status: 'NOT CONNECTED', planned: false },
      { id: 'dataLayer', title: 'HVEAC DATA LAYER', status: 'ONLINE', planned: false },
      { id: 'thermal', title: 'THERMAL INTELLIGENCE', status: 'PLANNED', planned: true },
      { id: 'hvac', title: 'HVAC OPTIMIZATION', status: 'PLANNED', planned: true }
    ],
    overviewMetrics: {
      occupancy: '0',
      occupancySub: 'Current people count',
      nodes: '0 / 0',
      nodesSub: 'Nodes online',
      computerHeat: '0 W',
      computerHeatSub: 'Total estimated heat',
      outdoorTemp: 'N/A',
      outdoorTempSub: 'Regional outdoor temp'
    },
    occupancy: {
      currentCount: 0,
      confidence: 'N/A',
      cameraStatus: 'OFFLINE',
      fps: 0,
      feedStatus: 'NOT CONNECTED'
    },
    nodes: {
      onlineCount: 0,
      totalCount: 0,
      totalHeatWatts: '0 W',
      avgCpu: 'N/A',
      avgGpu: 'N/A',
      telemetryList: []
    },
    environment: {
      outdoorTemp: 'N/A',
      humidity: 'N/A',
      solar: 'N/A',
      status: 'NOT CONNECTED'
    },
    control: {
      systemBackend: 'OFFLINE',
      camera: 'OFFLINE',
      visionEngine: 'OFFLINE',
      occupancy: 'STANDBY'
    },
    commands: []
  };

  // State container
  let currentState = JSON.parse(JSON.stringify(initialState));
  const subscribers = new Set();

  /**
   * Returns a deep clone of the current state to guarantee immutability
   */
  function getState() {
    return JSON.parse(JSON.stringify(currentState));
  }

  /**
   * Updates state partially and notifies registered subscribers
   * @param {Object|Function} updater Partial state object or updater function
   */
  function setState(updater) {
    if (typeof updater === 'function') {
      currentState = updater(currentState);
    } else if (typeof updater === 'object' && updater !== null) {
      currentState = {
        ...currentState,
        ...updater
      };
    }
    notifySubscribers();
  }

  /**
   * Subscribe to state mutations
   * @param {Function} callback 
   * @returns {Function} Unsubscribe function
   */
  function subscribe(callback) {
    if (typeof callback === 'function') {
      subscribers.add(callback);
      // Immediately pass current state
      callback(getState());
    }
    return () => subscribers.delete(callback);
  }

  function notifySubscribers() {
    const stateSnapshot = getState();
    subscribers.forEach((fn) => {
      try {
        fn(stateSnapshot);
      } catch (err) {
        console.error('[HVEAC State] Error in subscriber:', err);
      }
    });
  }

  /**
   * Format current time as HH:MM:SS
   */
  function getCurrentTimeString() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;
  }

  /**
   * Real Subsystem Command Dispatcher
   * Calls FastAPI backend control endpoints and logs responses in Command History.
   */
  async function dispatchCommand(target, action) {
    const time = getCurrentTimeString();
    let endpoint = '';
    let payload = { action };

    if (target === 'CAMERA') {
      endpoint = '/api/control/camera';
    } else if (target === 'VISION ENGINE') {
      endpoint = '/api/control/vision';
    } else if (target === 'OCCUPANCY') {
      endpoint = '/api/control/occupancy';
    }

    let status = 'SENT';
    let result = 'AWAITING RESPONSE';

    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      status = (resp.ok && data.success) ? 'SUCCESS' : 'FAILED';
      result = data.message || (resp.ok ? 'COMPLETED' : 'REQUEST FAILED');

      // Update control state if present
      const updatedControl = { ...currentState.control };
      if (data.camera_status) updatedControl.camera = data.camera_status;
      if (data.vision_status) updatedControl.visionEngine = data.vision_status;

      const commandEntry = {
        id: Date.now() + Math.random(),
        time: time,
        target: target,
        action: action,
        status: status,
        result: result
      };

      const updatedCommands = [commandEntry, ...currentState.commands].slice(0, 20);

      setState({
        control: updatedControl,
        commands: updatedCommands
      });

      console.info(`[HVEAC Control] Command completed: ${target} -> ${action} [${status}: ${result}]`);
      return data;
    } catch (err) {
      status = 'FAILED';
      result = `Network error: ${err.message}`;

      const commandEntry = {
        id: Date.now() + Math.random(),
        time: time,
        target: target,
        action: action,
        status: status,
        result: result
      };

      setState({
        commands: [commandEntry, ...currentState.commands].slice(0, 20)
      });

      console.error(`[HVEAC Control] Network error dispatching command to ${endpoint}:`, err);
    }
  }

  // Export State API to window
  window.HVEAC_STATE = {
    getState,
    setState,
    subscribe,
    dispatchCommand,
    resetState: () => setState(JSON.parse(JSON.stringify(initialState)))
  };

})(window);
