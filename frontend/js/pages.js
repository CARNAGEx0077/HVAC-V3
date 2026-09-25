/**
 * HVEAC Control Center - Page Views
 * 
 * Implements view renderers for each application route.
 * Strictly avoids synthetic or fabricated telemetry.
 */

(function (window) {
  'use strict';

  const { MetricCard, Panel, Pipeline, DataTable, Button, PageShell, escapeHtml } = window.HVEAC_COMPONENTS;

  /**
   * OVERVIEW PAGE (#/)
   * Primary engineering operations dashboard view
   */
  function renderOverview(state) {
    const pipelineHtml = Pipeline({ nodes: state.pipeline });

    const metricsHtml = `
      <div class="metrics-grid grid-cols-4">
        ${MetricCard({
          label: 'OCCUPANCY',
          value: state.overviewMetrics.occupancy,
          subtext: state.overviewMetrics.occupancySub,
          id: 'card-occupancy'
        })}
        ${MetricCard({
          label: 'COMPUTER NODES',
          value: state.overviewMetrics.nodes,
          subtext: state.overviewMetrics.nodesSub,
          id: 'card-nodes'
        })}
        ${MetricCard({
          label: state.overviewMetrics.computerHeatLabel || 'COMPUTER HEAT',
          value: state.overviewMetrics.computerHeat,
          subtext: state.overviewMetrics.computerHeatSub,
          id: 'card-heat'
        })}
        ${MetricCard({
          label: 'OUTDOOR TEMP',
          value: state.overviewMetrics.outdoorTemp,
          subtext: state.overviewMetrics.outdoorTempSub,
          id: 'card-temp'
        })}
      </div>
    `;

    return PageShell({
      content: pipelineHtml + metricsHtml,
      extraClass: 'page-overview'
    });
  }

  /**
   * OCCUPANCY PAGE (#/occupancy)
   * Camera vision stream & people occupancy telemetry
   */
  function renderOccupancy(state) {
    const metricsHtml = `
      <div class="metrics-grid grid-cols-3">
        ${MetricCard({
          label: 'CURRENT OCCUPANCY',
          value: state.occupancy.currentCount,
          subtext: 'People',
          id: 'card-occ-count'
        })}
        ${MetricCard({
          label: 'CONFIDENCE',
          value: state.occupancy.confidence,
          subtext: 'Detection confidence',
          id: 'card-occ-confidence'
        })}
        ${MetricCard({
          label: 'CAMERA STATUS',
          value: state.occupancy.cameraStatus,
          subtext: `Processing FPS: ${state.occupancy.fps}`,
          id: 'card-occ-status'
        })}
      </div>
    `;

    const feedHtml = `
      <div class="camera-feed-container">
        <div class="camera-viewport" id="camera-viewport">
          <img 
            src="/video_feed" 
            alt="HVEAC Camera Feed" 
            class="camera-stream-img" 
            id="camera-stream-img"
            onerror="this.style.display='none'; var p=document.getElementById('camera-placeholder'); if(p) p.style.display='flex';"
            onload="this.style.display='block'; var p=document.getElementById('camera-placeholder'); if(p) p.style.display='none';"
          />
          <div class="viewport-brackets">
            <div class="bracket bracket-tl"></div>
            <div class="bracket bracket-tr"></div>
            <div class="bracket bracket-bl"></div>
            <div class="bracket bracket-br"></div>
          </div>
          <div class="camera-placeholder-text" id="camera-placeholder" style="display: none;">
            <div class="camera-title">CAMERA FEED</div>
            <div class="camera-subtitle" id="camera-placeholder-subtitle">FEED UNAVAILABLE</div>
          </div>
        </div>
        <div class="camera-telemetry-bar">
          <span id="telemetry-bar-feed">Feed: Live Stream</span>
          <span id="telemetry-bar-fps">Processing FPS: ${state.occupancy.fps}</span>
          <span id="telemetry-bar-latency">Inference Latency: ${state.occupancy.inferenceLatency || 0} ms</span>
          <span id="telemetry-bar-tracked">Tracked Persons: ${state.occupancy.trackedPersons || 0}</span>
        </div>
      </div>
    `;

    const feedPanel = Panel({
      title: 'LIVE CAMERA FEED',
      content: feedHtml
    });

    return PageShell({
      content: metricsHtml + feedPanel,
      extraClass: 'page-occupancy'
    });
  }

  /**
   * COMPUTER NODES PAGE (#/nodes)
   * Cluster compute heat dissipation & hardware telemetry
   */
  function renderNodes(state) {
    const isMeasured = state.nodes.clusterThermalMode === 'MEASURED' || state.nodes.clusterThermalMode === 'PARTIAL';
    const heatLabel = isMeasured ? 'TOTAL MEASURED HEAT' : 'CLUSTER THERMAL LOAD';
    const heatSubtext = isMeasured ? 'Physical power dissipation' : 'Utilization-based proxy index';

    const metricsHtml = `
      <div class="metrics-grid grid-cols-5">
        ${MetricCard({
          label: 'NODES ONLINE',
          value: state.nodes.onlineCount,
          subtext: 'Active compute nodes'
        })}
        ${MetricCard({
          label: 'TOTAL NODES',
          value: state.nodes.totalCount,
          subtext: 'Cluster inventory'
        })}
        ${MetricCard({
          label: heatLabel,
          value: state.nodes.totalHeatWatts,
          subtext: heatSubtext
        })}
        ${MetricCard({
          label: 'AVG CPU LOAD',
          value: state.nodes.avgCpu,
          subtext: 'Cluster average utilization'
        })}
        ${MetricCard({
          label: 'AVG NODE THERMAL',
          value: state.nodes.avgNodeThermalLoad || 'N/A',
          subtext: 'Aggregate thermal load index'
        })}
      </div>
    `;

    const tableColumns = [
      'Node ID',
      'Hostname',
      'Status',
      'CPU Load',
      'CPU Workload',
      'CPU Thermal Load',
      'CPU Temp',
      'CPU Power',
      'GPU Load',
      'GPU Temp',
      'GPU Power',
      'Node Thermal Load',
      'Thermal Signal',
      'Est Heat'
    ];

    const tableRows = (state.nodes.telemetryList || []).map(node => {
      const signalClass = (node.thermal_signal || 'unavailable').toLowerCase();
      const signalBadge = `<span class="status-badge status-${signalClass}">${node.thermal_signal || 'UNAVAILABLE'}</span>`;
      const statusBadge = `<span class="status-badge status-${(node.status || 'offline').toLowerCase()}">${node.status}</span>`;

      return [
        node.id,
        node.hostname || 'N/A',
        statusBadge,
        node.cpu,
        node.cpu_workload || node.workload || 'IDLE',
        node.cpu_thermal_index_str || (node.cpu_thermal_index !== undefined && node.cpu_thermal_index !== null ? `${Math.round(node.cpu_thermal_index)} / 100` : 'N/A'),
        node.cpu_temp || 'N/A',
        node.cpu_power || 'N/A',
        node.gpu,
        node.gpu_temp || 'N/A',
        node.gpu_power || 'N/A',
        node.node_thermal_index_str || (node.node_thermal_index !== undefined && node.node_thermal_index !== null ? `${Math.round(node.node_thermal_index)} / 100` : 'N/A'),
        signalBadge,
        node.est_heat || 'N/A'
      ];
    });

    const tablePanel = Panel({
      title: 'COMPUTER NODE CLUSTER TELEMETRY & THERMAL LOAD PROXIES',
      content: DataTable({
        columns: tableColumns,
        rows: tableRows,
        emptyMessage: 'NO NODES CONNECTED'
      })
    });

    return PageShell({
      content: metricsHtml + tablePanel,
      extraClass: 'page-nodes'
    });
  }

  /**
   * ENVIRONMENT PAGE (#/environment)
   * Meteorological outdoor telemetry
   */
  function renderEnvironment(state) {
    const metricsHtml = `
      <div class="metrics-grid grid-cols-3">
        ${MetricCard({
          label: 'OUTDOOR TEMP',
          value: state.environment.outdoorTemp,
          subtext: 'Regional ambient temperature'
        })}
        ${MetricCard({
          label: 'HUMIDITY',
          value: state.environment.humidity,
          subtext: 'Relative humidity'
        })}
        ${MetricCard({
          label: 'SOLAR',
          value: state.environment.solar,
          subtext: 'Solar irradiance index'
        })}
      </div>
    `;

    const statusHtml = `
      <div class="phase-notice-box">
        <div class="phase-notice-title">ENVIRONMENT DATA</div>
        <div class="phase-notice-badge">STATUS: NOT CONNECTED</div>
        <div class="phase-notice-desc">
          Outdoor meteorological telemetry source is currently disconnected. Regional weather station integration is pending. No synthetic values are displayed.
        </div>
      </div>
    `;

    const envPanel = Panel({
      title: 'ENVIRONMENT DATA',
      content: statusHtml
    });

    return PageShell({
      content: metricsHtml + envPanel,
      extraClass: 'page-environment'
    });
  }

  /**
   * THERMAL INTELLIGENCE PAGE (#/thermal)
   * Future Phase Module
   */
  function renderThermal() {
    const noticeHtml = `
      <div class="phase-notice-box">
        <div class="phase-notice-title">THERMAL INTELLIGENCE</div>
        <div class="phase-notice-badge">COMING NEXT PHASE</div>
        <div class="phase-notice-desc">
          <strong>NOT IMPLEMENTED</strong><br><br>
          Thermodynamic modeling and predictive room temperature field mapping will be integrated in Phase 2. No synthetic predictions are simulated.
        </div>
      </div>
    `;

    return PageShell({
      content: Panel({
        title: 'THERMAL DYNAMICS ENGINE',
        content: noticeHtml
      }),
      extraClass: 'page-thermal'
    });
  }

  /**
   * HVAC OPTIMIZATION PAGE (#/hvac)
   * Future Phase Module
   */
  function renderHvac() {
    const noticeHtml = `
      <div class="phase-notice-box">
        <div class="phase-notice-title">HVAC OPTIMIZATION</div>
        <div class="phase-notice-badge">COMING NEXT PHASE</div>
        <div class="phase-notice-desc">
          <strong>NOT IMPLEMENTED</strong><br><br>
          Automated chiller and variable refrigerant flow optimization loops will be deployed in Phase 2. No synthetic actuator controls are activated.
        </div>
      </div>
    `;

    return PageShell({
      content: Panel({
        title: 'HVAC CONTROL OPTIMIZATION',
        content: noticeHtml
      }),
      extraClass: 'page-hvac'
    });
  }

  /**
   * ANALYTICS PAGE (#/analytics)
   * Historical telemetry and metrics charts
   */
  function renderAnalytics() {
    const emptyNotice = `
      <div class="phase-notice-box">
        <div class="phase-notice-title">HISTORICAL TELEMETRY &amp; SYSTEM ANALYTICS</div>
        <div class="phase-notice-badge">STATUS: IDLE</div>
        <div class="phase-notice-desc">
          <strong>NO DATA AVAILABLE</strong><br><br>
          Time-series telemetry database is waiting for live ingestion streams. Historical data aggregation will populate here once telemetry sources are online.
        </div>
      </div>
    `;

    return PageShell({
      content: Panel({
        title: 'ANALYTICS',
        content: emptyNotice
      }),
      extraClass: 'page-analytics'
    });
  }

  /**
   * CONTROL CENTER PAGE (#/control)
   * Subsystem Controls with Command History
   */
  function renderControl(state) {
    const isOnline = state.system.backendConnected;
    const noticeBanner = `
      <div class="control-notice-banner">
        <span class="control-notice-title">SYSTEM CONTROL CENTER</span>
        <span class="control-notice-badge ${isOnline ? 'status-online' : ''}" style="${isOnline ? 'color: var(--status-green); border-color: #1a4231; background: #0e241b;' : ''}">${isOnline ? 'LIVE BACKEND CONNECTED' : 'STANDALONE / DISCONNECTED'}</span>
      </div>
    `;

    const getStatusClass = (status) => {
      if (status.includes('ACTIVE')) return 'status-online';
      if (status.includes('RESTART')) return 'status-standby';
      return 'status-offline';
    };

    const controlsHtml = `
      <div class="control-sections-grid">
        <!-- Camera Subsystem -->
        <div class="control-card">
          <div class="control-card-header">
            <span class="control-card-title">CAMERA</span>
            <span class="control-card-status ${getStatusClass(state.control.camera)}" id="ctrl-status-camera">${escapeHtml(state.control.camera)}</span>
          </div>
          <div class="control-button-group">
            ${Button({ label: 'START', variant: 'start', action: 'START', target: 'CAMERA' })}
            ${Button({ label: 'STOP', variant: 'stop', action: 'STOP', target: 'CAMERA' })}
            ${Button({ label: 'RESTART', variant: 'restart', action: 'RESTART', target: 'CAMERA' })}
          </div>
        </div>

        <!-- Vision Engine Subsystem -->
        <div class="control-card">
          <div class="control-card-header">
            <span class="control-card-title">VISION ENGINE</span>
            <span class="control-card-status ${getStatusClass(state.control.visionEngine)}" id="ctrl-status-vision">${escapeHtml(state.control.visionEngine)}</span>
          </div>
          <div class="control-button-group">
            ${Button({ label: 'START', variant: 'start', action: 'START', target: 'VISION ENGINE' })}
            ${Button({ label: 'STOP', variant: 'stop', action: 'STOP', target: 'VISION ENGINE' })}
            ${Button({ label: 'RESTART', variant: 'restart', action: 'RESTART', target: 'VISION ENGINE' })}
          </div>
        </div>

        <!-- Occupancy Tracking -->
        <div class="control-card">
          <div class="control-card-header">
            <span class="control-card-title">OCCUPANCY</span>
            <span class="control-card-status status-active" id="ctrl-status-occupancy">${escapeHtml(state.control.occupancy)}</span>
          </div>
          <div class="control-button-group">
            ${Button({ label: 'RESET STATE', variant: 'reset', action: 'RESET STATE', target: 'OCCUPANCY' })}
          </div>
        </div>
      </div>
    `;

    // Command History Table
    const commandCols = ['Time', 'Target', 'Action', 'Status', 'Result'];
    const commandRows = state.commands.map(cmd => [
      cmd.time,
      cmd.target,
      cmd.action,
      cmd.status,
      cmd.result
    ]);

    const historyPanel = Panel({
      title: 'COMMAND HISTORY',
      content: DataTable({
        columns: commandCols,
        rows: commandRows,
        emptyMessage: 'NO COMMANDS ISSUED'
      })
    });

    return PageShell({
      content: noticeBanner + controlsHtml + historyPanel,
      extraClass: 'page-control'
    });
  }

  // --------------------------------------------------------------------------
  // Simulation Lab Page (Route: #/simulation)
  // Engineering Thermal Workstation & Building Automation UI
  // --------------------------------------------------------------------------

  function renderSimulation(state) {
    const sim = state.simulation || {};
    const activeId = sim.activeScenarioId || 1;
    const status = sim.status || 'READY';
    const speed = sim.speed || 10;
    const isPitch = sim.pitchMode || false;

    // 1. Engineering Header
    const header = `
      <div class="sim-workstation-header">
        <div class="sw-brand-group">
          <span class="sw-brand-title">HVEAC</span>
          <span class="sw-brand-sep">|</span>
          <span class="sw-brand-sub">Simulation Lab</span>
          <span class="sw-tag">THERMAL WORKSTATION</span>
        </div>
        <div class="sw-header-meta">
          <span class="sw-meta-item"><span class="sw-meta-lbl">ENVIRONMENT:</span> 12.0m × 10.0m Chamber</span>
          <span class="sw-meta-item"><span class="sw-meta-lbl">MODEL:</span> 2D Multi-Zone Energy Balance</span>
          <span class="sw-meta-item"><span class="sw-meta-lbl">CALIBRATION:</span> Synthetic Baseline</span>
        </div>
      </div>
    `;

    // Scenario meta catalog
    const scenarios = [
      {
        id: 1,
        title: 'SCENARIO 01 — Localized Heavy Compute',
        desc: 'Cluster of 4 computers in Zone 1 (NW) executes heavy batch compute while others idle with baseline occupancy.',
        source: 'Compute cluster (Zone 1)',
        result: 'Localized thermal hotspot'
      },
      {
        id: 2,
        title: 'SCENARIO 02 — Occupancy Concentration',
        desc: 'All 10 computers maintain uniform light workloads while 16 occupants gather in Zone 3 (SE).',
        source: 'Human occupancy (Zone 3)',
        result: 'Localized metabolic heat'
      },
      {
        id: 3,
        title: 'SCENARIO 03 — Distributed Heavy Compute',
        desc: 'All 10 computers spatially distributed across all 4 zones run heavy batch workloads simultaneously.',
        source: 'Distributed compute (10 nodes)',
        result: 'Uniform room-wide heat influx'
      },
      {
        id: 4,
        title: 'SCENARIO 04 — High Occupancy / Low Compute',
        desc: 'Seminar setting with 34 occupants distributed room-wide with minimal computer workload.',
        source: 'Human occupancy (Room-wide)',
        result: 'Metabolic heat dominance'
      },
      {
        id: 5,
        title: 'SCENARIO 05 — Opposing Thermal Zones',
        desc: 'West side loaded with heavy compute nodes; East side loaded with 20+ occupants under light compute.',
        source: 'Dual source (West Compute / East Occupancy)',
        result: 'Opposing thermal zones'
      }
    ];

    const currentScenario = scenarios.find(s => s.id === activeId) || scenarios[0];

    // 2. Compact Engineering Toolbar
    const toolbar = `
      <div class="sim-toolbar ${isPitch ? 'sim-toolbar-pitch' : ''}">
        <div class="sim-toolbar-left">
          <div class="sim-scenario-field">
            <span class="field-label">Scenario:</span>
            <span class="field-val" id="sim-active-title">${currentScenario.title}</span>
          </div>
          <div class="sim-status-badge status-${status.toLowerCase()}" id="sim-status-badge">
            ${status === 'RUNNING' ? '● RUNNING' : status === 'PAUSED' ? '❚❚ PAUSED' : status === 'COMPLETED' ? '✔ COMPLETED' : '○ READY'}
          </div>
        </div>

        <div class="sim-toolbar-center">
          <button class="btn-eng ${status === 'RUNNING' ? 'btn-eng-amber' : 'btn-eng-primary'}" id="btn-sim-play-pause">
            ${status === 'RUNNING' ? 'Pause' : 'Start'}
          </button>
          <button class="btn-eng btn-eng-subtle" id="btn-sim-reset">
            Reset
          </button>

          <div class="sim-speed-selector">
            <span class="speed-label">Speed:</span>
            ${[1, 5, 10, 30].map(s => `
              <button class="btn-speed ${speed === s ? 'active' : ''}" data-speed="${s}">${s}x</button>
            `).join('')}
          </div>
        </div>

        <div class="sim-toolbar-right">
          <div class="sim-time-field">
            <span class="time-label">Simulation:</span>
            <span class="time-val" id="sim-clock-display">00:00:00 / 02:00:00</span>
          </div>
          <button class="btn-eng btn-eng-subtle ${isPitch ? 'active' : ''}" id="btn-sim-pitch">
            ${isPitch ? 'Exit Pitch' : 'Pitch Mode'}
          </button>
          <button class="btn-eng btn-eng-subtle" id="btn-sim-compare">
            Compare
          </button>
        </div>
      </div>
      <div class="sim-progress-bar-container">
        <div class="sim-progress-bar" id="sim-progress-bar" style="width: 0%;"></div>
      </div>
    `;

    // Fixed computer coordinate definitions (relative percentages inside 12m x 10m room)
    const fixedCompDefs = [
      { id: 1, name: 'C1', left: '16.7%', top: '20%', zone: 'ZONE_1' },
      { id: 2, name: 'C2', left: '33.3%', top: '20%', zone: 'ZONE_1' },
      { id: 3, name: 'C3', left: '16.7%', top: '35%', zone: 'ZONE_1' },
      { id: 4, name: 'C4', left: '33.3%', top: '35%', zone: 'ZONE_1' },
      { id: 5, name: 'C5', left: '66.7%', top: '20%', zone: 'ZONE_2' },
      { id: 6, name: 'C6', left: '83.3%', top: '35%', zone: 'ZONE_2' },
      { id: 7, name: 'C7', left: '83.3%', top: '65%', zone: 'ZONE_3' },
      { id: 8, name: 'C8', left: '66.7%', top: '80%', zone: 'ZONE_3' },
      { id: 9, name: 'C9', left: '33.3%', top: '80%', zone: 'ZONE_4' },
      { id: 10, name: 'C10', left: '16.7%', top: '65%', zone: 'ZONE_4' }
    ];

    // 3. Technical Floorplan Stage
    const roomViewport = `
      <div class="sim-room-container ${isPitch ? 'sim-room-pitch' : ''}">
        <div class="sim-room-header">
          <div class="room-title-group">
            <span class="room-title">THERMAL MAP</span>
            <span class="room-meta">12.0m × 10.0m Architectural Floorplan</span>
          </div>
          <span class="room-legend-inline">4 Thermal Zones • 10 Workstations • 4 Perimeter ACs</span>
        </div>

        <div class="sim-room-stage" id="sim-room-stage">
          <!-- Dynamic Canvas Heatmap Underlay -->
          <canvas id="sim-heatmap-canvas" class="sim-heatmap-canvas" width="640" height="533"></canvas>

          <!-- 4 Wall-mounted AC Units (Technical Boundary Blocks) -->
          <div class="sim-ac-unit ac-north" id="sim-ac-north" title="AC-1 (North Wall)">
            <span class="ac-tag">AC1 NORTH</span>
            <span class="ac-val" id="sim-ac-north-val">22.0°C | 25% | IDLE</span>
          </div>
          <div class="sim-ac-unit ac-east" id="sim-ac-east" title="AC-2 (East Wall)">
            <span class="ac-tag">AC2 EAST</span>
            <span class="ac-val" id="sim-ac-east-val">22.0°C | 25% | IDLE</span>
          </div>
          <div class="sim-ac-unit ac-south" id="sim-ac-south" title="AC-3 (South Wall)">
            <span class="ac-tag">AC3 SOUTH</span>
            <span class="ac-val" id="sim-ac-south-val">22.0°C | 25% | IDLE</span>
          </div>
          <div class="sim-ac-unit ac-west" id="sim-ac-west" title="AC-4 (West Wall)">
            <span class="ac-tag">AC4 WEST</span>
            <span class="ac-val" id="sim-ac-west-val">22.0°C | 25% | IDLE</span>
          </div>

          <!-- Zone Spatial Overlay Badges -->
          <div class="sim-zone-overlay zone-nw" id="sim-zone-nw">
            <div class="zone-hdr"><span class="zone-name">Z1 NW</span><span class="zone-occ-tag" id="sim-occ-z1">1 OCC</span></div>
            <div class="zone-temp-telemetry" id="sim-tag-z1">22.5 °C</div>
          </div>
          <div class="sim-zone-overlay zone-ne" id="sim-zone-ne">
            <div class="zone-hdr"><span class="zone-name">Z2 NE</span><span class="zone-occ-tag" id="sim-occ-z2">10 OCC</span></div>
            <div class="zone-temp-telemetry" id="sim-tag-z2">22.5 °C</div>
          </div>
          <div class="sim-zone-overlay zone-se" id="sim-zone-se">
            <div class="zone-hdr"><span class="zone-name">Z3 SE</span><span class="zone-occ-tag" id="sim-occ-z3">10 OCC</span></div>
            <div class="zone-temp-telemetry" id="sim-tag-z3">22.5 °C</div>
          </div>
          <div class="sim-zone-overlay zone-sw" id="sim-zone-sw">
            <div class="zone-hdr"><span class="zone-name">Z4 SW</span><span class="zone-occ-tag" id="sim-occ-z4">0 OCC</span></div>
            <div class="zone-temp-telemetry" id="sim-tag-z4">22.5 °C</div>
          </div>

          <!-- 10 Fixed Workstation Markers (Technical CAD Blocks) -->
          ${fixedCompDefs.map(c => `
            <div class="sim-workstation" id="sim-comp-${c.id}" style="left: ${c.left}; top: ${c.top};" data-comp-id="${c.id}" title="Click to inspect node ${c.name}">
              <div class="comp-aura" id="sim-aura-${c.id}"></div>
              <div class="comp-body">
                <span class="comp-id">${c.name}</span>
                <div class="comp-load-bar"><div class="comp-load-fill" id="sim-load-${c.id}" style="width: 20%;"></div></div>
              </div>
            </div>
          `).join('')}

          <!-- Workstation Hover Tooltip -->
          <div class="sim-comp-tooltip" id="sim-comp-tooltip" style="display: none;">
            <div class="tooltip-header" id="sim-tt-title">Computer 1</div>
            <div class="tooltip-row"><span>Zone:</span> <strong id="sim-tt-zone">ZONE_1</strong></div>
            <div class="tooltip-row"><span>CPU:</span> <strong id="sim-tt-cpu">15%</strong></div>
            <div class="tooltip-row"><span>GPU:</span> <strong id="sim-tt-gpu">0%</strong></div>
            <div class="tooltip-row"><span>Workload:</span> <strong id="sim-tt-workload">IDLE</strong></div>
            <div class="tooltip-row"><span>Heat:</span> <strong id="sim-tt-heat">55 W</strong></div>
          </div>
        </div>

        <!-- Integrated Scientific Thermal Scale -->
        <div class="sim-heatmap-legend">
          <div class="legend-scale-group">
            <span class="legend-val">20.0°C</span>
            <div class="legend-bar-wrapper">
              <div class="legend-gradient-bar"></div>
              <div class="legend-ref-marker" style="left: 31.25%;">
                <span class="ref-line"></span>
                <span class="ref-tag">22.5°C Comfort</span>
              </div>
            </div>
            <span class="legend-val">28.0°C</span>
          </div>
          <span class="legend-desc">Scientific Thermal Scale (Cool → Normal → Warm → Hot)</span>
        </div>
      </div>
    `;

    // 4. Live Telemetry Monitor (Right Workstation Panel)
    const monitorPanel = `
      <div class="sim-monitor-container">
        <!-- 4 Telemetry KPI Blocks -->
        <div class="sim-kpi-grid">
          <div class="sim-kpi-block">
            <span class="kpi-label">ROOM AVG</span>
            <span class="kpi-val" id="sim-kpi-avg">22.5 °C</span>
          </div>
          <div class="sim-kpi-block">
            <span class="kpi-label">MAX ZONE</span>
            <span class="kpi-val" id="sim-kpi-max">22.5 °C</span>
          </div>
          <div class="sim-kpi-block">
            <span class="kpi-label">MIN ZONE</span>
            <span class="kpi-val" id="sim-kpi-min">22.5 °C</span>
          </div>
          <div class="sim-kpi-block">
            <span class="kpi-label">THERMAL GRADIENT</span>
            <span class="kpi-val" id="sim-kpi-gradient">0.0 °C</span>
          </div>
        </div>

        <!-- Zone Telemetry Table (Section 10) -->
        <div class="sim-card-panel">
          <div class="card-title-eng">ZONE TELEMETRY</div>
          <table class="sim-zone-table">
            <thead>
              <tr>
                <th>ZONE</th>
                <th style="text-align: right;">TEMP</th>
                <th style="text-align: right;">COMPUTE</th>
                <th style="text-align: right;">OCCUPANCY</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="cell-zone-name"><span class="zone-dot dot-z1"></span>Z1 NW</td>
                <td class="cell-num" id="sim-zm-t1">22.5 °C</td>
                <td class="cell-num" id="sim-zt-comp1">340 W</td>
                <td class="cell-num" id="sim-zt-occ1">1</td>
              </tr>
              <tr>
                <td class="cell-zone-name"><span class="zone-dot dot-z2"></span>Z2 NE</td>
                <td class="cell-num" id="sim-zm-t2">22.5 °C</td>
                <td class="cell-num" id="sim-zt-comp2">170 W</td>
                <td class="cell-num" id="sim-zt-occ2">10</td>
              </tr>
              <tr>
                <td class="cell-zone-name"><span class="zone-dot dot-z3"></span>Z3 SE</td>
                <td class="cell-num" id="sim-zm-t3">22.5 °C</td>
                <td class="cell-num" id="sim-zt-comp3">170 W</td>
                <td class="cell-num" id="sim-zt-occ3">10</td>
              </tr>
              <tr>
                <td class="cell-zone-name"><span class="zone-dot dot-z4"></span>Z4 SW</td>
                <td class="cell-num" id="sim-zm-t4">22.5 °C</td>
                <td class="cell-num" id="sim-zt-comp4">170 W</td>
                <td class="cell-num" id="sim-zt-occ4">0</td>
              </tr>
            </tbody>
          </table>
        </div>

        <!-- Thermal Load Balance Ledger (Section 11) -->
        <div class="sim-card-panel">
          <div class="card-title-eng">THERMAL LOAD BALANCE</div>
          <div class="sim-budget-ledger">
            <div class="ledger-row">
              <span class="ledger-lbl">Computer heat</span>
              <span class="ledger-val" id="sim-bud-comp">850 W</span>
            </div>
            <div class="ledger-row">
              <span class="ledger-lbl">Occupancy heat</span>
              <span class="ledger-val" id="sim-bud-occ">1190 W</span>
            </div>
            <div class="ledger-row">
              <span class="ledger-lbl">Envelope gain</span>
              <span class="ledger-val" id="sim-bud-env">450 W</span>
            </div>
            <div class="ledger-row ledger-cooling">
              <span class="ledger-lbl">HVAC cooling</span>
              <span class="ledger-val text-cooling" id="sim-bud-cool">-3500 W</span>
            </div>
            <div class="ledger-divider"></div>
            <div class="ledger-row ledger-net">
              <span class="ledger-lbl">NET HEAT FLUX</span>
              <span class="ledger-val-net" id="sim-bud-net">-1010 W</span>
            </div>
          </div>
        </div>

        <!-- Control Recommendation (Section 12 & 24 - Baseline Thermodynamic Recommendation) -->
        <div class="sim-card-panel sim-rec-panel">
          <div class="sim-rec-header">
            <span class="card-title-eng">CONTROL RECOMMENDATION</span>
            <span class="rec-action-badge" id="sim-target-action">ECO_MAINTAIN</span>
          </div>
          <div class="rec-content">
            <div class="rec-row-primary">
              <span class="rec-lbl">Recommended room setpoint:</span>
              <span class="rec-val" id="sim-target-temp">22.5 °C</span>
            </div>
            <div class="rec-row-secondary">
              <span class="rec-lbl-sub">Perimeter allocation:</span>
              <span class="rec-val-acs" id="sim-target-acs">AC1: 25% | AC2: 25% | AC3: 25% | AC4: 25%</span>
            </div>
            <div class="rec-source-note">
              <span class="source-lbl">Source:</span> Rule-based simulation baseline
            </div>
          </div>
        </div>

        <!-- 5-Minute Temperature History Chart (Section 13) -->
        <div class="sim-card-panel">
          <div class="card-title-eng">5-MINUTE TEMPERATURE HISTORY</div>
          <div class="sim-canvas-wrapper">
            <canvas id="sim-chart-canvas" width="520" height="160"></canvas>
          </div>
        </div>
      </div>
    `;

    // 5. Computer Node Telemetry Table & Summary Section (Section 14 & 15)
    const computerTelemetrySection = `
      <div class="sim-nodes-section" id="sim-nodes-telemetry-section">
        <div class="section-header-bar">
          <div class="sh-title-group">
            <h3 class="sh-title">COMPUTER TELEMETRY</h3>
            <span class="sh-subtitle">Live simulated computational workloads and heat output across 10 cluster nodes</span>
          </div>
        </div>

        <!-- Compact Summary Strip -->
        <div class="sim-comp-summary-strip">
          <div class="summary-metric">
            <span class="sm-lbl">TOTAL NODES</span>
            <span class="sm-val" id="sim-sum-total">10 / 10</span>
          </div>
          <div class="summary-metric">
            <span class="sm-lbl">HEAVY LOAD</span>
            <span class="sm-val" id="sim-sum-heavy">0</span>
          </div>
          <div class="summary-metric">
            <span class="sm-lbl">ACTIVE NODES</span>
            <span class="sm-val" id="sim-sum-active">10</span>
          </div>
          <div class="summary-metric">
            <span class="sm-lbl">TOTAL COMPUTE HEAT</span>
            <span class="sm-val" id="sim-sum-heat">850.0 W</span>
          </div>
          <div class="summary-metric">
            <span class="sm-lbl">AVERAGE CPU</span>
            <span class="sm-val" id="sim-sum-avg-cpu">20.0%</span>
          </div>
          <div class="summary-metric">
            <span class="sm-lbl">AVERAGE GPU</span>
            <span class="sm-val" id="sim-sum-avg-gpu">10.0%</span>
          </div>
        </div>

        <!-- Selected Node Telemetry Inspector -->
        <div class="sim-selected-inspector" id="sim-selected-inspector" style="display: none;">
          <div class="si-left">
            <span class="si-tag">SELECTED NODE:</span>
            <strong class="si-name" id="sim-insp-name">Computer 1</strong>
            <span class="si-zone" id="sim-insp-zone">Zone 1 (NW)</span>
          </div>
          <div class="si-metrics">
            <div class="si-metric"><span class="si-lbl">CPU:</span> <strong id="sim-insp-cpu">20.0%</strong> <span class="trend-icon" id="sim-insp-cputrend">→</span></div>
            <div class="si-metric"><span class="si-lbl">GPU:</span> <strong id="sim-insp-gpu">10.0%</strong></div>
            <div class="si-metric"><span class="si-lbl">WORKLOAD:</span> <span class="status-tag" id="sim-insp-cat">LIGHT</span></div>
            <div class="si-metric"><span class="si-lbl">HEAT:</span> <strong id="sim-insp-heat">86.7 W</strong> <span class="trend-tag" id="sim-insp-heattrend">STABLE</span></div>
            <div class="si-metric"><span class="si-lbl">CONTRIBUTION:</span> <span class="contrib-tag" id="sim-insp-contrib">LOW</span></div>
            <div class="si-metric"><span class="si-lbl">STATUS:</span> <span class="status-tag" id="sim-insp-status">RUNNING</span></div>
          </div>
          <button class="btn-eng btn-eng-subtle btn-xs" id="btn-deselect-node">Deselect</button>
        </div>

        <!-- Telemetry Table (Section 14 & 15) -->
        <div class="sim-table-wrapper">
          <table class="sim-nodes-table" id="sim-nodes-table">
            <thead>
              <tr>
                <th style="width: 70px;">ID</th>
                <th style="width: 110px;">ZONE</th>
                <th style="width: 140px;">CPU</th>
                <th style="width: 120px;">WORKLOAD</th>
                <th style="width: 130px;">GPU</th>
                <th style="width: 140px; text-align: right;">HEAT</th>
                <th style="width: 140px;">THERMAL LOAD</th>
                <th style="width: 110px;">STATUS</th>
                <th style="width: 80px; text-align: center;">ACTION</th>
              </tr>
            </thead>
            <tbody id="sim-nodes-tbody">
              ${[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(id => {
                const zoneName = id <= 4 ? 'Z1 NW' : (id <= 6 ? 'Z2 NE' : (id <= 8 ? 'Z3 SE' : 'Z4 SW'));
                return `
                  <tr class="sim-node-row" id="sim-row-c${id}" data-comp-id="${id}">
                    <td class="cell-node-id"><strong>C${id}</strong></td>
                    <td class="cell-zone" id="sim-row-zone-${id}">${zoneName}</td>
                    <td class="cell-cpu">
                      <div class="val-bar-group">
                        <span class="cell-num-fixed" id="sim-row-cpu-${id}">20.0%</span>
                        <span class="trend-icon" id="sim-row-cputrend-${id}">→</span>
                        <div class="eng-inline-bar"><div class="bar-fill" id="sim-row-cpubar-${id}" style="width: 20%;"></div></div>
                      </div>
                    </td>
                    <td class="cell-workload">
                      <span class="status-tag status-light" id="sim-row-cat-${id}">LIGHT</span>
                    </td>
                    <td class="cell-gpu">
                      <div class="val-bar-group">
                        <span class="cell-num-fixed" id="sim-row-gpu-${id}">10.0%</span>
                        <div class="eng-inline-bar"><div class="bar-fill" id="sim-row-gpubar-${id}" style="width: 10%;"></div></div>
                      </div>
                    </td>
                    <td class="cell-heat" style="text-align: right;">
                      <span class="cell-num-fixed" id="sim-row-heat-${id}">86.7 W</span>
                      <span class="trend-tag trend-stable" id="sim-row-heattrend-${id}">STABLE</span>
                    </td>
                    <td class="cell-contrib">
                      <span class="contrib-tag contrib-low" id="sim-row-contrib-${id}">LOW</span>
                    </td>
                    <td class="cell-status">
                      <span class="status-tag status-running" id="sim-row-status-${id}">RUNNING</span>
                    </td>
                    <td class="cell-action" style="text-align: center;">
                      <button class="btn-eng btn-eng-subtle btn-xs btn-inspect-node" data-comp-id="${id}">Select</button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        </div>
      </div>
    `;

    // 6. Technical Scenario Selector (Section 16)
    const scenarioCards = `
      <div class="sim-scenarios-section">
        <div class="section-header-bar">
          <div class="sh-title-group">
            <h3 class="sh-title">SCENARIO CATALOG</h3>
            <span class="sh-subtitle">Calibrated thermodynamic scenarios for thermal profiling and model validation</span>
          </div>
        </div>

        <div class="sim-scenarios-grid">
          ${scenarios.map(s => {
            const isAct = s.id === activeId;
            return `
              <div class="sim-scenario-card ${isAct ? 'card-active' : ''}" data-scenario-id="${s.id}">
                <div class="sc-header">
                  <span class="sc-id">SCENARIO 0${s.id}</span>
                  ${isAct ? '<span class="sc-status-tag">ACTIVE</span>' : ''}
                </div>
                <h4 class="sc-title">${s.title.replace(`SCENARIO 0${s.id} — `, '').replace(`SCENARIO ${s.id}: `, '')}</h4>
                <p class="sc-desc">${s.desc}</p>
                <div class="sc-spec-list">
                  <div class="spec-item"><span class="spec-k">Primary source:</span> <span class="spec-v">${s.source}</span></div>
                  <div class="spec-item"><span class="spec-k">Expected effect:</span> <span class="spec-v">${s.result}</span></div>
                </div>
                <button class="btn-eng ${isAct ? 'btn-eng-primary' : 'btn-eng-subtle'} btn-select-scen" data-select-id="${s.id}">
                  ${isAct ? 'Active' : 'Load'}
                </button>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    // 7. Scenario Comparison Modal
    const comparisonDrawer = `
      <div class="sim-comparison-modal" id="sim-comparison-modal" style="display: none;">
        <div class="modal-overlay" id="sim-modal-overlay"></div>
        <div class="modal-content">
          <div class="modal-header">
            <span class="modal-title">SCENARIO COMPARISON MATRIX</span>
            <button class="btn-close" id="btn-close-comparison">✕</button>
          </div>
          <div class="modal-body">
            <p class="modal-note">Summary thermodynamic statistics calculated from 720-timestep simulation runs across all 5 benchmark scenarios.</p>
            <div class="table-responsive">
              <table class="sim-nodes-table" id="sim-comparison-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Dominant Heat Source</th>
                    <th style="text-align: right;">Initial Temp</th>
                    <th style="text-align: right;">Final Temp</th>
                    <th style="text-align: right;">Peak Temp</th>
                    <th style="text-align: right;">Min Temp</th>
                    <th style="text-align: right;">Peak Heat</th>
                    <th style="text-align: right;">Avg Heat</th>
                  </tr>
                </thead>
                <tbody id="sim-comparison-tbody">
                  <tr><td colspan="8" style="text-align: center; color: #8b949e; padding: 20px;">Loading scenario comparisons...</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    `;

    // 8. HVEAC BRAIN — SHADOW MODE SECTION
    const brainShadowSection = `
      <div class="sim-brain-shadow-section" id="sim-brain-shadow-section">
        <div class="section-header-bar">
          <div class="sh-title-group">
            <h3 class="sh-title">HVEAC BRAIN — SHADOW MODE</h3>
            <span class="sh-subtitle">AI model predictions for observation only — no HVAC control path</span>
          </div>
          <div class="sh-badge-group">
            <span class="brain-mode-badge">SHADOW ONLY</span>
            <span class="brain-status-badge" id="brain-status-badge">LOADING…</span>
          </div>
        </div>

        <!-- Model Info Strip -->
        <div class="brain-info-strip" id="brain-info-strip">
          <div class="brain-info-item">
            <span class="bi-lbl">MODEL:</span>
            <span class="bi-val" id="brain-model-ver">—</span>
          </div>
          <div class="brain-info-item">
            <span class="bi-lbl">ARCHITECTURE:</span>
            <span class="bi-val" id="brain-architecture">—</span>
          </div>
          <div class="brain-info-item">
            <span class="bi-lbl">FEATURES:</span>
            <span class="bi-val" id="brain-feature-count">—</span>
          </div>
          <div class="brain-info-item">
            <span class="bi-lbl">CLASSES:</span>
            <span class="bi-val" id="brain-class-count">—</span>
          </div>
          <div class="brain-info-item">
            <span class="bi-lbl">TEST ACC:</span>
            <span class="bi-val" id="brain-test-acc">—</span>
          </div>
        </div>

        <!-- Prediction vs Simulator Comparison -->
        <div class="brain-comparison-grid">
          <div class="brain-compare-card">
            <div class="bcc-header">
              <span class="bcc-tag bcc-tag-ai">AI PREDICTION</span>
              <span class="bcc-confidence" id="brain-confidence">— %</span>
            </div>
            <div class="bcc-value" id="brain-ai-setpoint">—</div>
            <div class="bcc-sub">Predicted optimal room setpoint</div>
          </div>
          <div class="brain-compare-card">
            <div class="bcc-header">
              <span class="bcc-tag bcc-tag-sim">SIMULATOR</span>
              <span class="bcc-action" id="brain-sim-action">—</span>
            </div>
            <div class="bcc-value" id="brain-sim-setpoint">—</div>
            <div class="bcc-sub">Rule-based optimizer setpoint</div>
          </div>
          <div class="brain-compare-card brain-agreement-card">
            <div class="bcc-header">
              <span class="bcc-tag bcc-tag-compare">COMPARISON</span>
            </div>
            <div class="bcc-agreement" id="brain-agreement">—</div>
            <div class="bcc-deviation" id="brain-deviation">Deviation: —</div>
          </div>
        </div>

        <!-- Class Probability Distribution -->
        <div class="brain-proba-section">
          <div class="card-title-eng">CLASS PROBABILITY DISTRIBUTION</div>
          <div class="brain-proba-bars" id="brain-proba-bars">
            <div class="proba-placeholder">Waiting for prediction…</div>
          </div>
        </div>

        <!-- Shadow Trend History -->
        <div class="brain-trend-section">
          <div class="card-title-eng">AI vs SIMULATOR — SETPOINT TREND</div>
          <div class="sim-canvas-wrapper">
            <canvas id="brain-trend-canvas" width="520" height="160"></canvas>
          </div>
        </div>

        <!-- Model Inputs Inspector (collapsible) -->
        <details class="brain-inputs-inspector">
          <summary class="card-title-eng clickable-summary">MODEL INPUTS INSPECTOR ▸</summary>
          <div class="brain-inputs-grid" id="brain-inputs-grid">
            <div class="proba-placeholder">Run a prediction to inspect model inputs</div>
          </div>
          <div class="brain-input-warnings" id="brain-input-warnings"></div>
        </details>
      </div>
    `;

    const fullContent = `
      ${header}
      ${toolbar}
      <div class="sim-main-stage ${isPitch ? 'stage-pitch' : ''}">
        ${roomViewport}
        ${monitorPanel}
      </div>
      ${computerTelemetrySection}
      ${brainShadowSection}
      ${scenarioCards}
      ${comparisonDrawer}
    `;


    return PageShell({
      content: fullContent,
      extraClass: `page-simulation ${isPitch ? 'mode-pitch' : ''}`
    });
  }

  /**
   * Targeted DOM updater for live simulation ticks.
   * Updates only dynamic text, progress bar, workstations, and canvas heatmap without re-rendering page shell.
   */
  function updateSimulationDom(data) {
    if (!data) return;

    // 1. Status badge & Play/Pause button
    const statusBadge = document.getElementById('sim-status-badge');
    if (statusBadge) {
      statusBadge.className = `sim-status-badge status-${(data.status || 'ready').toLowerCase()}`;
      statusBadge.textContent = data.status === 'RUNNING' ? '● RUNNING'
        : data.status === 'PAUSED' ? '❚❚ PAUSED'
        : data.status === 'COMPLETED' ? '✔ COMPLETED' : '○ READY';
    }

    const btnPlay = document.getElementById('btn-sim-play-pause');
    if (btnPlay) {
      btnPlay.className = `btn-eng ${data.status === 'RUNNING' ? 'btn-eng-amber' : 'btn-eng-primary'}`;
      btnPlay.textContent = data.status === 'RUNNING' ? 'Pause' : 'Start';
    }

    // 2. Timeline Clock & Progress Bar
    const clockEl = document.getElementById('sim-clock-display');
    if (clockEl) {
      const curSec = Math.floor(data.simulation_time_seconds || 0);
      const totSec = Math.floor(data.total_duration_seconds || 7200);
      const pad = (n) => String(n).padStart(2, '0');
      const fmtTime = (s) => `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
      const step = data.step_index ?? 0;
      const totalSteps = data.total_steps ?? 720;
      clockEl.textContent = `${fmtTime(curSec)} / ${fmtTime(totSec)} (Step ${step}/${totalSteps})`;
    }

    const progBar = document.getElementById('sim-progress-bar');
    if (progBar) {
      progBar.style.width = `${data.progress_percent || 0}%`;
    }

    // 3. Thermal metrics
    const thermal = data.thermal || {};
    const zoneTemps = thermal.zone_temperatures_c || {};

    const elAvg = document.getElementById('sim-kpi-avg');
    if (elAvg) elAvg.textContent = `${(thermal.room_average_temperature_c ?? 22.5).toFixed(1)} °C`;

    const elMax = document.getElementById('sim-kpi-max');
    if (elMax) elMax.textContent = `${(thermal.maximum_temperature_c ?? 22.5).toFixed(1)} °C`;

    const elMin = document.getElementById('sim-kpi-min');
    if (elMin) elMin.textContent = `${(thermal.minimum_temperature_c ?? 22.5).toFixed(1)} °C`;

    const elGrad = document.getElementById('sim-kpi-gradient');
    if (elGrad) elGrad.textContent = `${(thermal.temperature_difference_c ?? 0.0).toFixed(1)} °C`;

    // Zone tags on top-down room
    const tag1 = document.getElementById('sim-tag-z1');
    if (tag1) tag1.textContent = `${(zoneTemps.zone_1 ?? 22.5).toFixed(1)} °C`;
    const tag2 = document.getElementById('sim-tag-z2');
    if (tag2) tag2.textContent = `${(zoneTemps.zone_2 ?? 22.5).toFixed(1)} °C`;
    const tag3 = document.getElementById('sim-tag-z3');
    if (tag3) tag3.textContent = `${(zoneTemps.zone_3 ?? 22.5).toFixed(1)} °C`;
    const tag4 = document.getElementById('sim-tag-z4');
    if (tag4) tag4.textContent = `${(zoneTemps.zone_4 ?? 22.5).toFixed(1)} °C`;

    // Zone occupancy tags on map
    const occZones = data.occupancy?.zones || {};
    const occ1 = document.getElementById('sim-occ-z1');
    if (occ1) occ1.textContent = `${occZones.zone_1 ?? 0} OCC`;
    const occ2 = document.getElementById('sim-occ-z2');
    if (occ2) occ2.textContent = `${occZones.zone_2 ?? 0} OCC`;
    const occ3 = document.getElementById('sim-occ-z3');
    if (occ3) occ3.textContent = `${occZones.zone_3 ?? 0} OCC`;
    const occ4 = document.getElementById('sim-occ-z4');
    if (occ4) occ4.textContent = `${occZones.zone_4 ?? 0} OCC`;

    // Zone telemetry table cells (Section 10)
    const zmT1 = document.getElementById('sim-zm-t1');
    if (zmT1) zmT1.textContent = `${(zoneTemps.zone_1 ?? 22.5).toFixed(1)} °C`;
    const zmT2 = document.getElementById('sim-zm-t2');
    if (zmT2) zmT2.textContent = `${(zoneTemps.zone_2 ?? 22.5).toFixed(1)} °C`;
    const zmT3 = document.getElementById('sim-zm-t3');
    if (zmT3) zmT3.textContent = `${(zoneTemps.zone_3 ?? 22.5).toFixed(1)} °C`;
    const zmT4 = document.getElementById('sim-zm-t4');
    if (zmT4) zmT4.textContent = `${(zoneTemps.zone_4 ?? 22.5).toFixed(1)} °C`;

    // Zone compute heat calculation
    const computers = data.computers || [];
    let compZ1 = 0, compZ2 = 0, compZ3 = 0, compZ4 = 0;
    computers.forEach(c => {
      const h = c.heat_watts || 0;
      if (c.id <= 4) compZ1 += h;
      else if (c.id <= 6) compZ2 += h;
      else if (c.id <= 8) compZ3 += h;
      else compZ4 += h;
    });

    const elZtC1 = document.getElementById('sim-zt-comp1');
    if (elZtC1) elZtC1.textContent = `${Math.round(compZ1)} W`;
    const elZtC2 = document.getElementById('sim-zt-comp2');
    if (elZtC2) elZtC2.textContent = `${Math.round(compZ2)} W`;
    const elZtC3 = document.getElementById('sim-zt-comp3');
    if (elZtC3) elZtC3.textContent = `${Math.round(compZ3)} W`;
    const elZtC4 = document.getElementById('sim-zt-comp4');
    if (elZtC4) elZtC4.textContent = `${Math.round(compZ4)} W`;

    const elZtO1 = document.getElementById('sim-zt-occ1');
    if (elZtO1) elZtO1.textContent = `${occZones.zone_1 ?? 0}`;
    const elZtO2 = document.getElementById('sim-zt-occ2');
    if (elZtO2) elZtO2.textContent = `${occZones.zone_2 ?? 0}`;
    const elZtO3 = document.getElementById('sim-zt-occ3');
    if (elZtO3) elZtO3.textContent = `${occZones.zone_3 ?? 0}`;
    const elZtO4 = document.getElementById('sim-zt-occ4');
    if (elZtO4) elZtO4.textContent = `${occZones.zone_4 ?? 0}`;

    // 4. AC Units on walls
    const acs = data.hvac?.acs || [];
    acs.forEach(ac => {
      const elVal = document.getElementById(`sim-ac-${ac.wall}-val`);
      if (elVal) {
        elVal.textContent = `${ac.setpoint_c}°C | ${ac.cooling_level_percent}% | ${ac.state}`;
      }
    });

    // 5. Workstations markers & Computer Telemetry Table (Section 14 & 15)
    computers.forEach(c => {
      // Map marker load bar & aura
      const fill = document.getElementById(`sim-load-${c.id}`);
      if (fill) fill.style.width = `${Math.min(Math.max(c.cpu_util_percent, 0), 100)}%`;

      const aura = document.getElementById(`sim-aura-${c.id}`);
      if (aura) {
        if (c.cpu_util_percent >= 75 || c.workload_category === 'HEAVY') {
          aura.className = 'comp-aura aura-heavy';
        } else if (c.cpu_util_percent >= 30) {
          aura.className = 'comp-aura aura-moderate';
        } else {
          aura.className = 'comp-aura aura-idle';
        }
      }

      // Telemetry table row cells
      const elCpu = document.getElementById(`sim-row-cpu-${c.id}`);
      if (elCpu) elCpu.textContent = `${c.cpu_util_percent.toFixed(1)}%`;

      const elCpuBar = document.getElementById(`sim-row-cpubar-${c.id}`);
      if (elCpuBar) elCpuBar.style.width = `${Math.min(Math.max(c.cpu_util_percent, 0), 100)}%`;

      const elCpuTrend = document.getElementById(`sim-row-cputrend-${c.id}`);
      if (elCpuTrend) {
        const trend = c.trend_cpu || 'STABLE';
        elCpuTrend.textContent = trend === 'UP' ? '↑' : (trend === 'DOWN' ? '↓' : '→');
        elCpuTrend.className = `trend-icon trend-${trend.toLowerCase()}`;
      }

      const elCat = document.getElementById(`sim-row-cat-${c.id}`);
      if (elCat) {
        const cat = c.workload_category || 'IDLE';
        elCat.textContent = cat;
        elCat.className = `status-tag status-${cat.toLowerCase().replace(/_/g, '-')}`;
      }

      const elGpu = document.getElementById(`sim-row-gpu-${c.id}`);
      if (elGpu) elGpu.textContent = `${c.gpu_util_percent.toFixed(1)}%`;

      const elGpuBar = document.getElementById(`sim-row-gpubar-${c.id}`);
      if (elGpuBar) elGpuBar.style.width = `${Math.min(Math.max(c.gpu_util_percent, 0), 100)}%`;

      const elHeat = document.getElementById(`sim-row-heat-${c.id}`);
      if (elHeat) elHeat.textContent = `${c.heat_watts.toFixed(1)} W`;

      const elHeatTrend = document.getElementById(`sim-row-heattrend-${c.id}`);
      if (elHeatTrend) {
        const hTrend = c.trend_heat || 'STABLE';
        elHeatTrend.textContent = hTrend;
        elHeatTrend.className = `trend-tag trend-${hTrend.toLowerCase()}`;
      }

      const elContrib = document.getElementById(`sim-row-contrib-${c.id}`);
      if (elContrib) {
        const contrib = c.thermal_contribution || 'LOW';
        elContrib.textContent = contrib;
        elContrib.className = `contrib-tag contrib-${contrib.toLowerCase().replace(/\s+/g, '-')}`;
      }

      const elStatus = document.getElementById(`sim-row-status-${c.id}`);
      if (elStatus) {
        const st = c.status || 'RUNNING';
        elStatus.textContent = st;
        elStatus.className = `status-tag status-${st.toLowerCase().replace(/\s+/g, '-')}`;
      }
    });

    // Summary statistics bar
    const sum = data.computer_summary;
    if (sum) {
      const elHeavy = document.getElementById('sim-sum-heavy');
      if (elHeavy) elHeavy.textContent = sum.heavy_count;
      const elHeat = document.getElementById('sim-sum-heat');
      if (elHeat) elHeat.textContent = `${sum.total_compute_heat_watts.toFixed(1)} W`;
      const elAvgCpu = document.getElementById('sim-sum-avg-cpu');
      if (elAvgCpu) elAvgCpu.textContent = `${sum.average_cpu_percent.toFixed(1)}%`;
      const elAvgGpu = document.getElementById('sim-sum-avg-gpu');
      if (elAvgGpu) elAvgGpu.textContent = `${sum.average_gpu_percent.toFixed(1)}%`;
    }

    // Update selected inspector if active
    if (window._simSelectedCompId) {
      const selComp = computers.find(c => c.id === window._simSelectedCompId);
      if (selComp) {
        const panel = document.getElementById('sim-selected-inspector');
        if (panel) {
          panel.style.display = 'flex';
          const nameEl = document.getElementById('sim-insp-name');
          if (nameEl) nameEl.textContent = selComp.name;
          const zoneEl = document.getElementById('sim-insp-zone');
          if (zoneEl) zoneEl.textContent = selComp.zone_display || selComp.zone;
          const cpuEl = document.getElementById('sim-insp-cpu');
          if (cpuEl) cpuEl.textContent = `${selComp.cpu_util_percent.toFixed(1)}%`;
          const trendEl = document.getElementById('sim-insp-cputrend');
          if (trendEl) {
            const t = selComp.trend_cpu || 'STABLE';
            trendEl.textContent = t === 'UP' ? '↑' : (t === 'DOWN' ? '↓' : '→');
            trendEl.className = `trend-icon trend-${t.toLowerCase()}`;
          }
          const gpuEl = document.getElementById('sim-insp-gpu');
          if (gpuEl) gpuEl.textContent = `${selComp.gpu_util_percent.toFixed(1)}%`;
          const catEl = document.getElementById('sim-insp-cat');
          if (catEl) {
            catEl.textContent = selComp.workload_category;
            catEl.className = `status-tag status-${selComp.workload_category.toLowerCase().replace(/_/g, '-')}`;
          }
          const heatEl = document.getElementById('sim-insp-heat');
          if (heatEl) heatEl.textContent = `${selComp.heat_watts.toFixed(1)} W`;
          const hTrendEl = document.getElementById('sim-insp-heattrend');
          if (hTrendEl) {
            const ht = selComp.trend_heat || 'STABLE';
            hTrendEl.textContent = ht;
            hTrendEl.className = `trend-tag trend-${ht.toLowerCase()}`;
          }
          const contribEl = document.getElementById('sim-insp-contrib');
          if (contribEl) {
            contribEl.textContent = selComp.thermal_contribution;
            contribEl.className = `contrib-tag contrib-${selComp.thermal_contribution.toLowerCase().replace(/\s+/g, '-')}`;
          }
          const stEl = document.getElementById('sim-insp-status');
          if (stEl) {
            stEl.textContent = selComp.status;
            stEl.className = `status-tag status-${selComp.status.toLowerCase().replace(/\s+/g, '-')}`;
          }
        }
      }
    }

    // 6. Heat Budget
    const bComp = document.getElementById('sim-bud-comp');
    if (bComp) bComp.textContent = `${Math.round(thermal.computer_heat_watts ?? 0)} W`;
    const bOcc = document.getElementById('sim-bud-occ');
    if (bOcc) bOcc.textContent = `${Math.round(thermal.occupancy_heat_watts ?? 0)} W`;
    const bEnv = document.getElementById('sim-bud-env');
    if (bEnv) bEnv.textContent = `${Math.round(thermal.envelope_gain_watts ?? 450)} W`;
    const bCool = document.getElementById('sim-bud-cool');
    if (bCool) bCool.textContent = `-${Math.abs(Math.round(thermal.hvac_cooling_watts ?? 0))} W`;
    const bNet = document.getElementById('sim-bud-net');
    if (bNet) {
      const net = Math.round(thermal.total_net_heat_watts ?? 0);
      bNet.textContent = `${net > 0 ? '+' : ''}${net} W`;
      bNet.style.color = net > 200 ? '#f85149' : net < -200 ? '#58a6ff' : '#e6edf3';
    }

    // 7. Control Recommendation (Rule-based simulation baseline)
    const targets = data.targets || {};
    const tAction = document.getElementById('sim-target-action');
    if (tAction) tAction.textContent = targets.optimal_hvac_action || 'ECO_MAINTAIN';
    const tTemp = document.getElementById('sim-target-temp');
    if (tTemp) tTemp.textContent = `${(targets.optimal_temperature_c ?? 22.5).toFixed(1)} °C`;
    const tAcs = document.getElementById('sim-target-acs');
    if (tAcs) {
      const ac1 = Math.round((targets.optimal_cooling_ac1 ?? 0) * 100);
      const ac2 = Math.round((targets.optimal_cooling_ac2 ?? 0) * 100);
      const ac3 = Math.round((targets.optimal_cooling_ac3 ?? 0) * 100);
      const ac4 = Math.round((targets.optimal_cooling_ac4 ?? 0) * 100);
      tAcs.textContent = `AC1: ${ac1}% | AC2: ${ac2}% | AC3: ${ac3}% | AC4: ${ac4}%`;
    }

    // 8. Render Canvas Heatmap
    const canvasHeat = document.getElementById('sim-heatmap-canvas');
    if (canvasHeat && window.HVEAC_SIM_VISUALS) {
      window.HVEAC_SIM_VISUALS.renderThermalHeatmap(canvasHeat, data);
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // HVEAC BRAIN — SHADOW MODE DOM UPDATER
  // ══════════════════════════════════════════════════════════════════════════

  // Trend history buffer for the AI vs Simulator chart
  const _brainTrendHistory = [];
  const _brainTrendMaxPoints = 60;

  /**
   * Updates the Brain Shadow Mode section from the /api/brain/shadow-predict response.
   * Called periodically (~every 2s) when the simulation is running.
   */
  function updateBrainShadowDom(brainData) {
    if (!brainData) return;

    const section = document.getElementById('sim-brain-shadow-section');
    if (!section) return;

    // 1. Model status badge
    const statusBadge = document.getElementById('brain-status-badge');
    if (statusBadge) {
      const st = brainData.status || 'UNAVAILABLE';
      statusBadge.textContent = st;
      statusBadge.className = `brain-status-badge brain-st-${st.toLowerCase()}`;
    }

    // 2. Prediction vs Simulator comparison
    const pred = brainData.prediction || {};
    const sim = brainData.simulator || {};
    const comp = brainData.comparison || {};

    const aiSetEl = document.getElementById('brain-ai-setpoint');
    if (aiSetEl) {
      aiSetEl.textContent = pred.ai_setpoint_c != null
        ? `${pred.ai_setpoint_c.toFixed(1)} °C`
        : '—';
    }

    const confEl = document.getElementById('brain-confidence');
    if (confEl) {
      confEl.textContent = pred.confidence != null
        ? `${(pred.confidence * 100).toFixed(1)}%`
        : '— %';
    }

    const simSetEl = document.getElementById('brain-sim-setpoint');
    if (simSetEl) {
      simSetEl.textContent = sim.setpoint_c != null
        ? `${sim.setpoint_c.toFixed(1)} °C`
        : '—';
    }

    const simActEl = document.getElementById('brain-sim-action');
    if (simActEl) simActEl.textContent = sim.action || '—';

    const agreeEl = document.getElementById('brain-agreement');
    if (agreeEl) {
      if (comp.agreement === true) {
        agreeEl.textContent = '✓ AGREE';
        agreeEl.className = 'bcc-agreement agree-yes';
      } else if (comp.agreement === false) {
        agreeEl.textContent = '✗ DISAGREE';
        agreeEl.className = 'bcc-agreement agree-no';
      } else {
        agreeEl.textContent = '—';
        agreeEl.className = 'bcc-agreement';
      }
    }

    const devEl = document.getElementById('brain-deviation');
    if (devEl) {
      devEl.textContent = comp.deviation_c != null
        ? `Deviation: ${comp.deviation_c.toFixed(1)} °C`
        : 'Deviation: —';
    }

    // 3. Class Probability Distribution bars
    const probaContainer = document.getElementById('brain-proba-bars');
    if (probaContainer && pred.class_probabilities) {
      const probs = pred.class_probabilities;
      const classes = Object.keys(probs).sort((a, b) => parseFloat(a) - parseFloat(b));
      const maxP = Math.max(...Object.values(probs), 0.01);

      let html = '';
      classes.forEach(cls => {
        const p = probs[cls];
        const pct = (p * 100).toFixed(1);
        const barW = Math.max(2, (p / maxP) * 100);
        const isPred = pred.ai_setpoint_c != null && parseFloat(cls) === pred.ai_setpoint_c;
        html += `
          <div class="proba-bar-row ${isPred ? 'proba-predicted' : ''}">
            <span class="proba-label">${cls} °C</span>
            <div class="proba-bar-track">
              <div class="proba-bar-fill" style="width: ${barW}%;"></div>
            </div>
            <span class="proba-pct">${pct}%</span>
          </div>
        `;
      });
      probaContainer.innerHTML = html;
    }

    // 4. Trend history chart
    if (pred.ai_setpoint_c != null && sim.setpoint_c != null) {
      const ctx = brainData.simulation_context || {};
      _brainTrendHistory.push({
        time: ctx.simulation_time_seconds || 0,
        ai: pred.ai_setpoint_c,
        sim: sim.setpoint_c,
      });
      if (_brainTrendHistory.length > _brainTrendMaxPoints) {
        _brainTrendHistory.splice(0, _brainTrendHistory.length - _brainTrendMaxPoints);
      }
      _renderBrainTrendChart();
    }

    // 5. Diagnostics / Feature warnings
    const warningsEl = document.getElementById('brain-input-warnings');
    if (warningsEl) {
      const diag = brainData.diagnostics || {};
      const warns = diag.feature_warnings || [];
      if (warns.length > 0) {
        warningsEl.innerHTML = `<div class="brain-warn-tag">⚠ ${warns.length} feature warnings</div>
          <div class="brain-warn-list">${warns.slice(0, 10).map(w => `<div class="warn-item">${w}</div>`).join('')}</div>`;
      } else {
        warningsEl.innerHTML = '<div class="brain-ok-tag">✓ All 80 features mapped</div>';
      }
    }
  }

  /**
   * Update model info strip from /api/brain/status response.
   */
  function updateBrainStatusDom(statusData) {
    if (!statusData || !statusData.brain) return;
    const brain = statusData.brain;

    const verEl = document.getElementById('brain-model-ver');
    if (verEl) verEl.textContent = brain.model_version || '—';

    const archEl = document.getElementById('brain-architecture');
    if (archEl) archEl.textContent = brain.architecture || '—';

    const featEl = document.getElementById('brain-feature-count');
    if (featEl) featEl.textContent = brain.feature_count || '—';

    const clsEl = document.getElementById('brain-class-count');
    if (clsEl) clsEl.textContent = brain.class_count || '—';

    const accEl = document.getElementById('brain-test-acc');
    if (accEl) {
      accEl.textContent = brain.test_accuracy != null
        ? `${(brain.test_accuracy * 100).toFixed(1)}%`
        : '—';
    }

    const badgeEl = document.getElementById('brain-status-badge');
    if (badgeEl) {
      badgeEl.textContent = brain.status || 'UNAVAILABLE';
      badgeEl.className = `brain-status-badge brain-st-${(brain.status || 'unavailable').toLowerCase()}`;
    }
  }

  /**
   * Render the AI vs Simulator setpoint trend line chart on canvas.
   */
  function _renderBrainTrendChart() {
    const canvas = document.getElementById('brain-trend-canvas');
    if (!canvas || _brainTrendHistory.length < 2) return;

    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const pad = { t: 18, r: 12, b: 24, l: 42 };
    const plotW = W - pad.l - pad.r;
    const plotH = H - pad.t - pad.b;

    ctx.clearRect(0, 0, W, H);

    // Determine Y range
    const allVals = _brainTrendHistory.flatMap(p => [p.ai, p.sim]);
    const yMin = Math.floor(Math.min(...allVals) * 2) / 2 - 0.5;
    const yMax = Math.ceil(Math.max(...allVals) * 2) / 2 + 0.5;
    const yRange = Math.max(yMax - yMin, 1);

    // Grid
    ctx.strokeStyle = 'rgba(139,148,158,0.15)';
    ctx.lineWidth = 1;
    for (let y = yMin; y <= yMax; y += 0.5) {
      const py = pad.t + plotH - ((y - yMin) / yRange) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.l, py);
      ctx.lineTo(pad.l + plotW, py);
      ctx.stroke();

      ctx.fillStyle = '#8b949e';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(`${y.toFixed(1)}`, pad.l - 4, py + 3);
    }

    const n = _brainTrendHistory.length;
    const xStep = plotW / Math.max(n - 1, 1);

    // AI line (blue)
    ctx.strokeStyle = '#58a6ff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    _brainTrendHistory.forEach((p, i) => {
      const px = pad.l + i * xStep;
      const py = pad.t + plotH - ((p.ai - yMin) / yRange) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Simulator line (amber)
    ctx.strokeStyle = '#d29922';
    ctx.lineWidth = 2;
    ctx.beginPath();
    _brainTrendHistory.forEach((p, i) => {
      const px = pad.l + i * xStep;
      const py = pad.t + plotH - ((p.sim - yMin) / yRange) * plotH;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Legend
    ctx.font = '10px monospace';
    ctx.fillStyle = '#58a6ff';
    ctx.fillText('● AI Prediction', pad.l + 4, pad.t - 4);
    ctx.fillStyle = '#d29922';
    ctx.fillText('● Simulator', pad.l + 120, pad.t - 4);
  }

  /**
   * Clear brain trend history (call on scenario change).
   */
  function clearBrainTrendHistory() {
    _brainTrendHistory.length = 0;
  }

  window.HVEAC_PAGES = {
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
  };

})(window);
