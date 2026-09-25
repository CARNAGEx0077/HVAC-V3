/**
 * HVEAC Control Center - Main Application Controller
 * 
 * Orchestrates Hash-based SPA Routing, Event Delegation,
 * and Targeted Live Telemetry Updates via WebSocket.
 */

(function (window) {
  'use strict';

  const { getState, setState, subscribe, dispatchCommand } = window.HVEAC_STATE;
  const {
    renderOverview,
    renderOccupancy,
    renderNodes,
    renderEnvironment,
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
   * Event delegation for Control Center action buttons
   */
  function setupEventDelegation() {
    if (!elMainContent) return;

    elMainContent.addEventListener('click', async (e) => {
      const btn = e.target.closest('[data-action]');
      if (!btn) return;

      const action = btn.getAttribute('data-action');
      const target = btn.getAttribute('data-target');

      if (action && target) {
        e.preventDefault();
        btn.disabled = true;
        btn.style.opacity = '0.5';

        try {
          await dispatchCommand(target, action);
          // Re-render Control Center view to refresh Command History
          if (getCurrentRoute() === '#/control') {
            renderView();
          }
        } finally {
          btn.disabled = false;
          btn.style.opacity = '1';
        }
      }
    });
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

    console.info('[HVEAC] Control Center initialized in connected operations mode.');
  }

  // Bootstrap when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})(window);
