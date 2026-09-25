/**
 * HVEAC Control Center - Main Application Controller
 * 
 * Orchestrates Hash-based SPA Routing, Event Delegation,
 * and Targeted Live Telemetry Updates via WebSocket.
 */

(function (window) {
  'use strict';

  const {
    getState,
    setState,
    subscribe,
    dispatchCommand,
    selectSimulationScenario,
    startSimulation,
    pauseSimulation,
    resetSimulation,
    setSimulationSpeed,
    fetchSimulationScenarios,
    fetchSimulationComparison,
    togglePitchMode
  } = window.HVEAC_STATE;

  const {
    renderOverview,
    renderOccupancy,
    renderNodes,
    renderEnvironment,
    renderSimulation,
    updateSimulationDom,
    updateBrainShadowDom,
    updateBrainStatusDom,
    clearBrainTrendHistory,
    renderThermal,
    renderHvac,
    renderAnalytics,
    renderControl
  } = window.HVEAC_PAGES;

  // DOM Elements
  const elPageTitle = document.getElementById('page-title');
  const elStatusDot = document.getElementById('status-dot');
  const elStatusText = document.getElementById('status-text');
  const elMainContent = document.getElementById('main-content');
  const elSidebarNav = document.getElementById('sidebar-nav');

  // Route definitions
  const routes = {
    '#/': {
      title: 'OVERVIEW',
      render: renderOverview
    },
    '#/occupancy': {
      title: 'OCCUPANCY',
      render: renderOccupancy
    },
    '#/nodes': {
      title: 'COMPUTER TELEMETRY NETWORK',
      render: renderNodes
    },
    '#/environment': {
      title: 'OUTDOOR ENVIRONMENT',
      render: renderEnvironment
    },
    '#/simulation': {
      title: 'HVEAC V3 SIMULATION LAB',
      render: renderSimulation
    },
    '#/thermal': {
      title: 'THERMAL INTELLIGENCE',
      render: renderThermal
    },
    '#/hvac': {
      title: 'HVAC OPTIMIZATION',
      render: renderHvac
    },
    '#/analytics': {
      title: 'ANALYTICS',
      render: renderAnalytics
    },
    '#/control': {
      title: 'SYSTEM CONTROL CENTER',
      render: renderControl
    }
  };

  /**
   * Determine current route from location hash
   */
  function getCurrentRoute() {
    const hash = window.location.hash || '#/';
    return routes[hash] ? hash : '#/';
  }

  /**
   * Updates Top Header & System Online/Offline indicator
   */
  function updateHeader(routeKey, isBackendOnline) {
    const routeConfig = routes[routeKey] || routes['#/'];
    if (elPageTitle) {
      elPageTitle.textContent = routeConfig.title;
    }

    if (elStatusDot && elStatusText) {
      elStatusDot.className = isBackendOnline ? 'status-dot status-dot-online' : 'status-dot status-dot-offline';
      elStatusText.textContent = isBackendOnline ? 'SYSTEM ONLINE' : 'SYSTEM OFFLINE';
    }
  }

  /**
   * Updates Sidebar active navigation item
   */
  function updateSidebar(currentRoute) {
    if (!elSidebarNav) return;
    const navItems = elSidebarNav.querySelectorAll('.nav-item');
    navItems.forEach(item => {
      const routeAttr = item.getAttribute('data-route');
      if (routeAttr === currentRoute) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });
  }

  /**
   * Renders the current view based on active route and state
   */
  function renderView() {
    const currentRoute = getCurrentRoute();
    const routeConfig = routes[currentRoute] || routes['#/'];
    const state = getState();

    // Update Header and Navigation
    updateHeader(currentRoute, state.system.backendConnected);
    updateSidebar(currentRoute);

    // Render page body
    if (elMainContent && typeof routeConfig.render === 'function') {
      elMainContent.innerHTML = routeConfig.render(state);

      // Post-render lifecycle hooks for Simulation Lab
      if (currentRoute === '#/simulation') {
        fetchSimulationScenarios();
        if (latestSimSnapshot) {
          updateSimulationDom(latestSimSnapshot);
          const chartCanvas = document.getElementById('sim-chart-canvas');
          if (chartCanvas && window.HVEAC_SIM_VISUALS) {
            window.HVEAC_SIM_VISUALS.renderHistoryChart(chartCanvas, simHistory);
          }
        }
        // Fetch brain status and start shadow polling
        fetchBrainStatus();
        startBrainShadowPolling();
      } else {
        stopBrainShadowPolling();
      }
    }
  }

  /**
   * Targeted DOM Updates
   * Updates only affected metric spans without replacing the page shell.
   */
  function updateLiveTelemetryDom(data) {
    // 1. Overview Page Elements
    const elOverviewOcc = document.getElementById('card-occupancy');
    if (elOverviewOcc) {
      const valEl = elOverviewOcc.querySelector('.metric-card-value');
      if (valEl) valEl.textContent = data.occupancy;
    }

    const elCamNode = document.querySelector('[data-node-id="camera"]');
    if (elCamNode) {
      const statusEl = elCamNode.querySelector('.pipeline-node-status');
      if (statusEl) {
        const isRunning = data.vision_status === 'RUNNING';
        statusEl.textContent = isRunning ? 'RUNNING' : (data.camera_status === 'CONNECTED' ? 'CONNECTED' : 'OFFLINE');
        statusEl.className = `pipeline-node-status ${isRunning ? 'status-online' : 'status-offline'}`;
      }
    }

    // 2. Occupancy Page Elements
    const occCountCard = document.getElementById('card-occ-count');
    if (occCountCard) {
      const val = occCountCard.querySelector('.metric-card-value');
      if (val) val.textContent = data.occupancy;
    }

    const occConfCard = document.getElementById('card-occ-confidence');
    if (occConfCard) {
      const val = occConfCard.querySelector('.metric-card-value');
      if (val) val.textContent = data.confidence_display || (data.confidence !== null ? `${Math.round(data.confidence * 100)}%` : 'N/A');
    }

    const occStatusCard = document.getElementById('card-occ-status');
    if (occStatusCard) {
      const val = occStatusCard.querySelector('.metric-card-value');
      if (val) val.textContent = data.camera_status;
      const sub = occStatusCard.querySelector('.metric-card-subtext');
      if (sub) sub.textContent = `Processing FPS: ${data.processing_fps}`;
    }

    const elBarFps = document.getElementById('telemetry-bar-fps');
    if (elBarFps) elBarFps.textContent = `Processing FPS: ${data.processing_fps}`;

    const elBarLat = document.getElementById('telemetry-bar-latency');
    if (elBarLat) elBarLat.textContent = `Inference Latency: ${data.inference_latency_ms} ms`;

    const elBarTracked = document.getElementById('telemetry-bar-tracked');
    if (elBarTracked) elBarTracked.textContent = `Tracked Persons: ${data.tracked_persons}`;

    // 3. Control Center Elements
    const elCtrlCam = document.getElementById('ctrl-status-camera');
    if (elCtrlCam) {
      elCtrlCam.textContent = data.camera_status;
      elCtrlCam.className = `control-card-status ${data.camera_status === 'CONNECTED' ? 'status-online' : 'status-offline'}`;
    }

    const elCtrlVis = document.getElementById('ctrl-status-vision');
    if (elCtrlVis) {
      elCtrlVis.textContent = data.vision_status;
      elCtrlVis.className = `control-card-status ${data.vision_status === 'RUNNING' ? 'status-online' : 'status-offline'}`;
    }
  }

  /**
   * WebSocket Connection Manager
   * Connects to /ws/occupancy for real-time live telemetry stream
   */
  let socket = null;
  let reconnectTimeout = null;

  function connectTelemetryWebSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/occupancy`;

    try {
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        console.info('[HVEAC WS] Connected to live occupancy telemetry stream.');
        updateHeader(getCurrentRoute(), true);
        setState(s => ({
          ...s,
          system: { ...s.system, status: 'ONLINE', backendConnected: true }
        }));
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'occupancy_update') {
            // 1. Update state object quietly
            setState(s => ({
              ...s,
              occupancy: {
                ...s.occupancy,
                currentCount: msg.occupancy,
                rawCount: msg.raw_count,
                confidence: msg.confidence_display || (msg.confidence !== null ? `${Math.round(msg.confidence * 100)}%` : 'N/A'),
                cameraStatus: msg.camera_status,
                visionStatus: msg.vision_status,
                fps: msg.processing_fps,
                inferenceLatency: msg.inference_latency_ms,
                trackedPersons: msg.tracked_persons
              },
              overviewMetrics: {
                ...s.overviewMetrics,
                occupancy: String(msg.occupancy)
              },
              control: {
                ...s.control,
                camera: msg.camera_status,
                visionEngine: msg.vision_status
              }
            }));

            // 2. Perform targeted DOM update
            updateLiveTelemetryDom(msg);
          }
        } catch (parseErr) {
          console.error('[HVEAC WS] Error parsing telemetry message:', parseErr);
        }
      };

      socket.onclose = () => {
        console.warn('[HVEAC WS] Telemetry socket closed. Reconnecting in 3s...');
        updateHeader(getCurrentRoute(), false);
        setState(s => ({
          ...s,
          system: { ...s.system, status: 'OFFLINE', backendConnected: false }
        }));
        clearTimeout(reconnectTimeout);
        reconnectTimeout = setTimeout(connectTelemetryWebSocket, 3000);
      };

      socket.onerror = (err) => {
        console.warn('[HVEAC WS] Socket error:', err);
        socket.close();
      };
    } catch (e) {
      console.error('[HVEAC WS] Failed to initiate connection:', e);
      clearTimeout(reconnectTimeout);
      reconnectTimeout = setTimeout(connectTelemetryWebSocket, 3000);
    }
  }

  /**
   * Simulation WebSocket Manager
   * Connects to /ws/simulation for live synthetic scenario telemetry stream
   */
  let simSocket = null;
  let simReconnectTimeout = null;
  let latestSimSnapshot = null;
  const simHistory = [];

  function connectSimulationWebSocket() {
    if (simSocket && (simSocket.readyState === WebSocket.OPEN || simSocket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/simulation`;

    try {
      simSocket = new WebSocket(wsUrl);

      simSocket.onopen = () => {
        console.info('[HVEAC Sim WS] Connected to live simulation stream.');
      };

      simSocket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === 'simulation_update') {
            latestSimSnapshot = msg;

            // Maintain rolling history
            simHistory.push({
              time: msg.simulation_time_seconds,
              avg: msg.thermal?.room_average_temperature_c ?? 22.5,
              z1: msg.thermal?.zone_temperatures_c?.zone_1 ?? 22.5,
              z2: msg.thermal?.zone_temperatures_c?.zone_2 ?? 22.5,
              z3: msg.thermal?.zone_temperatures_c?.zone_3 ?? 22.5,
              z4: msg.thermal?.zone_temperatures_c?.zone_4 ?? 22.5
            });
            if (simHistory.length > 35) simHistory.shift();

            // Targeted DOM & Canvas update if user is on Simulation page
            if (getCurrentRoute() === '#/simulation') {
              updateSimulationDom(msg);
              const chartCanvas = document.getElementById('sim-chart-canvas');
              if (chartCanvas && window.HVEAC_SIM_VISUALS) {
                window.HVEAC_SIM_VISUALS.renderHistoryChart(chartCanvas, simHistory);
              }
            }
          }
        } catch (parseErr) {
          console.error('[HVEAC Sim WS] Error parsing simulation message:', parseErr);
        }
      };

      simSocket.onclose = () => {
        clearTimeout(simReconnectTimeout);
        simReconnectTimeout = setTimeout(connectSimulationWebSocket, 2500);
      };

      simSocket.onerror = () => {
        simSocket.close();
      };
    } catch (e) {
      clearTimeout(simReconnectTimeout);
      simReconnectTimeout = setTimeout(connectSimulationWebSocket, 2500);
    }
  }

  // ════════════════════════════════════════════════════════════════════════
  // HVEAC BRAIN — Shadow Mode Polling
  // ════════════════════════════════════════════════════════════════════════
  let _brainPollInterval = null;
  const BRAIN_POLL_INTERVAL_MS = 2000; // Poll every 2 seconds

  /**
   * Fetch model status/metadata once and populate the info strip.
   */
  async function fetchBrainStatus() {
    try {
      const resp = await fetch('/api/brain/status');
      if (resp.ok) {
        const data = await resp.json();
        updateBrainStatusDom(data);
      }
    } catch (err) {
      console.warn('[HVEAC Brain] Failed to fetch brain status:', err);
    }
  }

  /**
   * Fetch a single shadow prediction and update the UI.
   */
  async function fetchBrainShadowPrediction() {
    if (getCurrentRoute() !== '#/simulation') return;
    try {
      const resp = await fetch('/api/brain/shadow-predict');
      if (resp.ok) {
        const data = await resp.json();
        updateBrainShadowDom(data);
      }
    } catch (err) {
      // Silent fail — shadow mode is observation only
    }
  }

  /**
   * Start periodic brain shadow predictions.
   */
  function startBrainShadowPolling() {
    stopBrainShadowPolling();
    // Immediate first prediction
    fetchBrainShadowPrediction();
    _brainPollInterval = setInterval(fetchBrainShadowPrediction, BRAIN_POLL_INTERVAL_MS);
  }

  /**
   * Stop brain shadow polling.
   */
  function stopBrainShadowPolling() {
    if (_brainPollInterval) {
      clearInterval(_brainPollInterval);
      _brainPollInterval = null;
    }
  }

  /**
   * Event delegation for interactive buttons and simulation controls
   */
  function setupEventDelegation() {
    if (!elMainContent) return;

    elMainContent.addEventListener('click', async (e) => {
      // 1. Control Center actions
      const ctrlBtn = e.target.closest('[data-action]');
      if (ctrlBtn) {
        const action = ctrlBtn.getAttribute('data-action');
        const target = ctrlBtn.getAttribute('data-target');

        if (action && target) {
          e.preventDefault();
          ctrlBtn.disabled = true;
          ctrlBtn.style.opacity = '0.5';

          try {
            await dispatchCommand(target, action);
            if (getCurrentRoute() === '#/control') {
              renderView();
            }
          } finally {
            ctrlBtn.disabled = false;
            ctrlBtn.style.opacity = '1';
          }
          return;
        }
      }

      // 2. Simulation Play / Pause
      const btnPlayPause = e.target.closest('#btn-sim-play-pause');
      if (btnPlayPause) {
        e.preventDefault();
        if (latestSimSnapshot && latestSimSnapshot.status === 'RUNNING') {
          await pauseSimulation();
        } else {
          await startSimulation();
        }
        return;
      }

      // 3. Simulation Reset
      const btnReset = e.target.closest('#btn-sim-reset');
      if (btnReset) {
        e.preventDefault();
        simHistory.length = 0; // Clear rolling chart history on reset
        await resetSimulation();
        return;
      }

      // 4. Simulation Speed Selector
      const speedBtn = e.target.closest('[data-speed]');
      if (speedBtn) {
        e.preventDefault();
        const spd = parseInt(speedBtn.getAttribute('data-speed'), 10);
        if (spd) {
          await setSimulationSpeed(spd);
          document.querySelectorAll('.btn-speed').forEach(b => b.classList.remove('active'));
          speedBtn.classList.add('active');
        }
        return;
      }

      // 5. Select Scenario
      const selectBtn = e.target.closest('[data-select-id]');
      if (selectBtn) {
        e.preventDefault();
        const sid = parseInt(selectBtn.getAttribute('data-select-id'), 10);
        if (sid) {
          simHistory.length = 0;
          window._simSelectedCompId = null;
          clearBrainTrendHistory();
          // Reset feature adapter rolling history
          fetch('/api/brain/adapter-reset', { method: 'POST' }).catch(() => {});
          await selectSimulationScenario(sid);
          renderView();
        }
        return;
      }

      // 6. Pitch Mode Toggle
      const btnPitch = e.target.closest('#btn-sim-pitch');
      if (btnPitch) {
        e.preventDefault();
        togglePitchMode();
        renderView();
        return;
      }

      // 7. Comparison Modal Toggle
      const btnCompare = e.target.closest('#btn-sim-compare');
      if (btnCompare) {
        e.preventDefault();
        const modal = document.getElementById('sim-comparison-modal');
        if (modal) {
          modal.style.display = 'flex';
          loadComparisonTable();
        }
        return;
      }

      const btnCloseModal = e.target.closest('#btn-close-comparison') || e.target.closest('#sim-modal-overlay');
      if (btnCloseModal) {
        e.preventDefault();
        const modal = document.getElementById('sim-comparison-modal');
        if (modal) modal.style.display = 'none';
        return;
      }

      // 8. Deselect Computer Node Button
      const btnDeselect = e.target.closest('#btn-deselect-node');
      if (btnDeselect) {
        e.preventDefault();
        window._simSelectedCompId = null;
        highlightSelectedComputer(null);
        return;
      }

      // 9. Click on Table Row or Inspect Button
      const inspectBtn = e.target.closest('.btn-inspect-node');
      const nodeRow = e.target.closest('.sim-node-row');
      if (inspectBtn || nodeRow) {
        const el = inspectBtn || nodeRow;
        const cid = parseInt(el.getAttribute('data-comp-id'), 10);
        if (cid) {
          window._simSelectedCompId = window._simSelectedCompId === cid ? null : cid;
          highlightSelectedComputer(window._simSelectedCompId);
          // Highlight and scroll pin into view
          const mapPin = document.getElementById(`sim-comp-${cid}`);
          if (mapPin && window._simSelectedCompId) {
            mapPin.classList.add('comp-selected');
          }
        }
        return;
      }

      // 10. Click on Room Stage Workstation Pin
      const compPin = e.target.closest('.sim-workstation');
      if (compPin) {
        const cid = parseInt(compPin.getAttribute('data-comp-id'), 10);
        if (cid) {
          window._simSelectedCompId = window._simSelectedCompId === cid ? null : cid;
          highlightSelectedComputer(window._simSelectedCompId);
          const targetRow = document.getElementById(`sim-row-c${cid}`);
          if (targetRow && window._simSelectedCompId) {
            targetRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
        return;
      }
    });

    function highlightSelectedComputer(compId) {
      // Toggle row selection classes
      for (let i = 1; i <= 10; i++) {
        const r = document.getElementById(`sim-row-c${i}`);
        if (r) {
          if (i === compId) r.classList.add('row-selected');
          else r.classList.remove('row-selected');
        }
        const pin = document.getElementById(`sim-comp-${i}`);
        if (pin) {
          if (i === compId) pin.classList.add('comp-selected');
          else pin.classList.remove('comp-selected');
        }
      }

      const panel = document.getElementById('sim-selected-inspector');
      if (panel) {
        if (!compId) {
          panel.style.display = 'none';
        } else if (latestSimSnapshot) {
          const comp = latestSimSnapshot.computers?.find(c => c.id === compId);
          if (comp) {
            panel.style.display = 'flex';
            const nameEl = document.getElementById('sim-insp-name');
            if (nameEl) nameEl.textContent = comp.name;
            const zoneEl = document.getElementById('sim-insp-zone');
            if (zoneEl) zoneEl.textContent = comp.zone_display || comp.zone;
            const cpuEl = document.getElementById('sim-insp-cpu');
            if (cpuEl) cpuEl.textContent = `${comp.cpu_util_percent.toFixed(1)}%`;
            const trendEl = document.getElementById('sim-insp-cputrend');
            if (trendEl) {
              const t = comp.trend_cpu || 'STABLE';
              trendEl.textContent = t === 'UP' ? '↑' : (t === 'DOWN' ? '↓' : '→');
              trendEl.className = `trend-icon trend-${t.toLowerCase()}`;
            }
            const gpuEl = document.getElementById('sim-insp-gpu');
            if (gpuEl) gpuEl.textContent = `${comp.gpu_util_percent.toFixed(1)}%`;
            const catEl = document.getElementById('sim-insp-cat');
            if (catEl) {
              catEl.textContent = comp.workload_category;
              catEl.className = `status-badge status-${comp.workload_category.toLowerCase().replace(/_/g, '-')}`;
            }
            const heatEl = document.getElementById('sim-insp-heat');
            if (heatEl) heatEl.textContent = `${comp.heat_watts.toFixed(1)} W`;
            const hTrendEl = document.getElementById('sim-insp-heattrend');
            if (hTrendEl) {
              const ht = comp.trend_heat || 'STABLE';
              hTrendEl.textContent = ht;
              hTrendEl.className = `trend-badge trend-${ht.toLowerCase()}`;
            }
            const contribEl = document.getElementById('sim-insp-contrib');
            if (contribEl) {
              contribEl.textContent = comp.thermal_contribution;
              contribEl.className = `contrib-badge contrib-${comp.thermal_contribution.toLowerCase().replace(/\s+/g, '-')}`;
            }
            const stEl = document.getElementById('sim-insp-status');
            if (stEl) {
              stEl.textContent = comp.status;
              stEl.className = `status-badge status-${comp.status.toLowerCase().replace(/\s+/g, '-')}`;
            }
          }
        }
      }
    }

    // Workstation & Table hover synchronization
    elMainContent.addEventListener('mouseover', (e) => {
      // Hover map marker -> highlight table row
      const compEl = e.target.closest('.sim-workstation');
      if (compEl) {
        const cid = parseInt(compEl.getAttribute('data-comp-id'), 10);
        const row = document.getElementById(`sim-row-c${cid}`);
        if (row) row.classList.add('row-hover');

        const tooltip = document.getElementById('sim-comp-tooltip');
        if (tooltip && latestSimSnapshot) {
          const compData = latestSimSnapshot.computers?.find(c => c.id === cid);
          if (compData) {
            document.getElementById('sim-tt-title').textContent = compData.name;
            document.getElementById('sim-tt-zone').textContent = compData.zone_display || compData.zone;
            document.getElementById('sim-tt-cpu').textContent = `${compData.cpu_util_percent.toFixed(1)}%`;
            document.getElementById('sim-tt-gpu').textContent = `${compData.gpu_util_percent.toFixed(1)}%`;
            document.getElementById('sim-tt-workload').textContent = compData.workload_category;
            document.getElementById('sim-tt-heat').textContent = `${compData.heat_watts.toFixed(1)} W (Simulated)`;

            const rect = compEl.getBoundingClientRect();
            const stageRect = document.getElementById('sim-room-stage').getBoundingClientRect();
            tooltip.style.left = `${rect.left - stageRect.left + 35}px`;
            tooltip.style.top = `${rect.top - stageRect.top - 10}px`;
            tooltip.style.display = 'block';
          }
        }
      }

      // Hover table row -> highlight map marker
      const rowEl = e.target.closest('.sim-node-row');
      if (rowEl) {
        const cid = parseInt(rowEl.getAttribute('data-comp-id'), 10);
        const mapPin = document.getElementById(`sim-comp-${cid}`);
        if (mapPin) mapPin.classList.add('comp-hover');
      }
    });

    elMainContent.addEventListener('mouseout', (e) => {
      const compEl = e.target.closest('.sim-workstation');
      if (compEl) {
        const cid = parseInt(compEl.getAttribute('data-comp-id'), 10);
        const row = document.getElementById(`sim-row-c${cid}`);
        if (row) row.classList.remove('row-hover');

        const tooltip = document.getElementById('sim-comp-tooltip');
        if (tooltip) tooltip.style.display = 'none';
      }

      const rowEl = e.target.closest('.sim-node-row');
      if (rowEl) {
        const cid = parseInt(rowEl.getAttribute('data-comp-id'), 10);
        const mapPin = document.getElementById(`sim-comp-${cid}`);
        if (mapPin) mapPin.classList.remove('comp-hover');
      }
    });
  }

  async function loadComparisonTable() {
    const tbody = document.getElementById('sim-comparison-tbody');
    if (!tbody) return;
    try {
      const resp = await fetch('/api/simulation/comparison');
      if (resp.ok) {
        const data = await resp.json();
        const rows = (data.comparisons || []).map(c => `
          <tr>
            <td><strong>SCENARIO ${c.scenario_id}:</strong> ${c.name}</td>
            <td><span class="meta-tag">${c.dominant_heat_source}</span></td>
            <td>${c.initial_temperature_c.toFixed(1)}°C</td>
            <td><strong>${c.final_temperature_c.toFixed(1)}°C</strong></td>
            <td style="color: #ff4757; font-weight: bold;">${c.maximum_temperature_c.toFixed(1)}°C</td>
            <td style="color: #2ed573;">${c.minimum_temperature_c.toFixed(1)}°C</td>
            <td><strong>${Math.round(c.peak_total_heat_w)} W</strong></td>
            <td>${Math.round(c.average_total_heat_w)} W</td>
          </tr>
        `).join('');
        tbody.innerHTML = rows || '<tr><td colspan="8">No data available</td></tr>';
      }
    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="8" style="color: red;">Failed to load comparison data.</td></tr>';
    }
  }

  /**
   * Node Telemetry Polling Manager
   * Polls GET /api/nodes every 2 seconds
   */
  let nodePollingInterval = null;

  async function pollNodesTelemetry() {
    try {
      const resp = await fetch('/api/nodes');
      if (resp.ok) {
        const data = await resp.json();

        const hasMeasuredPower = data.total_measured_power_watts !== null && data.total_measured_power_watts !== undefined;
        const heatLabel = hasMeasuredPower ? 'TOTAL MEASURED HEAT' : 'COMPUTER THERMAL LOAD';
        const heatVal = data.total_heat_watts;
        const heatSub = hasMeasuredPower ? 'Aggregated physical power dissipation' : 'Cluster computational proxy index';

        setState(s => {
          const isNodesOnline = data.online_count > 0;
          const pipeline = s.pipeline.map(p => {
            if (p.id === 'nodes') {
              return { ...p, status: isNodesOnline ? 'ONLINE' : (data.total_count > 0 ? 'OFFLINE' : 'OFFLINE') };
            }
            return p;
          });

          return {
            ...s,
            system: { ...s.system, status: 'ONLINE', backendConnected: true },
            pipeline,
            overviewMetrics: {
              ...s.overviewMetrics,
              nodes: `${data.online_count} / ${data.total_count}`,
              computerHeat: heatVal,
              computerHeatLabel: heatLabel,
              computerHeatSub: heatSub
            },
            nodes: {
              onlineCount: data.online_count,
              totalCount: data.total_count,
              totalHeatWatts: heatVal,
              totalHeatLabel: heatLabel,
              avgCpu: data.avg_cpu,
              avgGpu: data.avg_gpu,
              avgNodeThermalLoad: data.avg_node_thermal_load_index_str || 'N/A',
              clusterThermalMode: data.cluster_thermal_mode || 'UNAVAILABLE',
              telemetryList: data.nodes || []
            }
          };
        });

        // Targeted DOM updates
        const curRoute = getCurrentRoute();
        if (curRoute === '#/nodes') {
          renderView();
        } else if (curRoute === '#/') {
          const elNodesCard = document.getElementById('card-nodes');
          if (elNodesCard) {
            const val = elNodesCard.querySelector('.metric-card-value');
            if (val) val.textContent = `${data.online_count} / ${data.total_count}`;
          }
          const elHeatCard = document.getElementById('card-heat');
          if (elHeatCard) {
            const labelEl = elHeatCard.querySelector('.metric-card-label');
            if (labelEl) labelEl.textContent = heatLabel;
            const val = elHeatCard.querySelector('.metric-card-value');
            if (val) val.textContent = heatVal;
            const sub = elHeatCard.querySelector('.metric-card-subtext');
            if (sub) sub.textContent = heatSub;
          }
          const elNodePipeline = document.querySelector('[data-node-id="nodes"]');
          if (elNodePipeline) {
            const statusEl = elNodePipeline.querySelector('.pipeline-node-status');
            if (statusEl) {
              const isOnline = data.online_count > 0;
              statusEl.textContent = isOnline ? 'ONLINE' : 'OFFLINE';
              statusEl.className = `pipeline-node-status ${isOnline ? 'status-online' : 'status-offline'}`;
            }
          }
        }
      }
    } catch (err) {
      // Backend temporarily unreachable during polling
    }
  }

  function startNodePolling() {
    if (nodePollingInterval) clearInterval(nodePollingInterval);
    pollNodesTelemetry();
    nodePollingInterval = setInterval(pollNodesTelemetry, 2000);
  }

  /**
   * Initialize Application
   */
  function init() {
    // Listen for hash changes
    window.addEventListener('hashchange', () => {
      renderView();
    });

    // Handle initial route if missing or invalid
    if (!window.location.hash) {
      window.location.hash = '#/';
    }

    // Setup interactive handlers
    setupEventDelegation();

    // Initial render
    renderView();

    // Connect real-time telemetry WebSocket
    connectTelemetryWebSocket();

    // Connect real-time simulation lab WebSocket
    connectSimulationWebSocket();

    // Start polling compute node cluster telemetry
    startNodePolling();

    console.info('[HVEAC] Control Center initialized in connected operations mode.');
  }

  // Bootstrap when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window);
