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
      computerHeatLabel: 'COMPUTER HEAT',
      computerHeatSub: 'Aggregated thermal load',
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
      totalHeatLabel: 'TOTAL ESTIMATED HEAT',
      avgCpu: 'N/A',
      avgGpu: 'N/A',
      avgNodeThermalLoad: 'N/A',
      clusterThermalMode: 'UNAVAILABLE',
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
    commands: [],
    simulation: {
      activeScenarioId: 1,
      status: 'READY',
      speed: 10,
      simTimeSeconds: 0,
      totalDurationSeconds: 7200,
      stepIndex: 0,
      totalSteps: 720,
      pitchMode: false,
      controlMode: 'PROTOTYPE_CONTROL',
      scenarios: [],
      comparisons: [],
      selectedComputerId: null,
      history: [],
      latestTelemetry: null
    }
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

  // --------------------------------------------------------------------------
  // Simulation Lab API Actions
  // --------------------------------------------------------------------------

  async function fetchSimulationScenarios() {
    try {
      const resp = await fetch('/api/simulation/scenarios');
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            scenarios: data.scenarios || [],
            activeScenarioId: data.active_scenario_id || s.simulation.activeScenarioId
          }
        }));
      }
    } catch (e) {
      console.warn('[HVEAC Sim] Could not fetch scenarios:', e);
    }
  }

  async function selectSimulationScenario(scenarioId) {
    try {
      const resp = await fetch('/api/simulation/select', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario_id: Number(scenarioId) })
      });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            activeScenarioId: Number(scenarioId),
            status: data.state.status,
            simTimeSeconds: data.state.simulation_time_seconds,
            history: [],
            latestTelemetry: data.state
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error selecting scenario:', e);
    }
  }

  async function startSimulation() {
    try {
      const resp = await fetch('/api/simulation/start', { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            status: data.state.status
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error starting simulation:', e);
    }
  }

  async function pauseSimulation() {
    try {
      const resp = await fetch('/api/simulation/pause', { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            status: data.state.status
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error pausing simulation:', e);
    }
  }

  async function resetSimulation() {
    try {
      const resp = await fetch('/api/simulation/reset', { method: 'POST' });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            status: data.state.status,
            simTimeSeconds: data.state.simulation_time_seconds,
            history: [],
            latestTelemetry: data.state
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error resetting simulation:', e);
    }
  }

  async function setSimulationSpeed(speed) {
    try {
      const resp = await fetch('/api/simulation/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: Number(speed) })
      });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            speed: data.speed
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error setting speed:', e);
    }
  }

  async function fetchSimulationComparison() {
    try {
      const resp = await fetch('/api/simulation/comparison');
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            comparisons: data.comparisons || []
          }
        }));
      }
    } catch (e) {
      console.warn('[HVEAC Sim] Could not fetch comparison summary:', e);
    }
  }

  async function setSimulationControlMode(mode) {
    try {
      const resp = await fetch('/api/simulation/control-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ control_mode: mode })
      });
      if (resp.ok) {
        const data = await resp.json();
        setState(s => ({
          ...s,
          simulation: {
            ...s.simulation,
            controlMode: data.control_mode
          }
        }));
      }
    } catch (e) {
      console.error('[HVEAC Sim] Error setting control mode:', e);
    }
  }

  function togglePitchMode() {
    setState(s => ({
      ...s,
      simulation: {
        ...s.simulation,
        pitchMode: !s.simulation.pitchMode
      }
    }));
  }

  function selectSimulationComputer(compId) {
    setState(s => ({
      ...s,
      simulation: {
        ...s.simulation,
        selectedComputerId: s.simulation.selectedComputerId === compId ? null : compId
      }
    }));
  }

  // Export State API to window
  window.HVEAC_STATE = {
    getState,
    setState,
    subscribe,
    dispatchCommand,
    fetchSimulationScenarios,
    selectSimulationScenario,
    startSimulation,
    pauseSimulation,
    resetSimulation,
    setSimulationSpeed,
    setSimulationControlMode,
    fetchSimulationComparison,
    togglePitchMode,
    selectSimulationComputer,
    resetState: () => setState(JSON.parse(JSON.stringify(initialState)))
  };

})(window);
