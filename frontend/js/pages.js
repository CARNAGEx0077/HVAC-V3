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

    // 1. Engineering Workstation Header & Metadata Strip
    const header = `
      <div class="sim-workstation-header">
        <div class="sw-brand-group">
          <span class="sw-brand-title">HVEAC V3</span>
          <span class="sw-brand-sep">|</span>
          <span class="sw-brand-sub">SIMULATION LAB</span>
          <span class="sw-tag">THERMAL WORKSTATION</span>
        </div>
        <div class="sw-header-meta">
          <span class="sw-meta-item"><span class="sw-meta-lbl">ENVIRONMENT:</span> 12.0m × 10.0m Chamber</span>
          <span class="sw-meta-item"><span class="sw-meta-lbl">MODEL:</span> 2D Multi-Zone Energy Balance</span>
          <span class="sw-meta-item"><span class="sw-meta-lbl">CALIBRATION:</span> Synthetic Baseline</span>
        </div>
      </div>
    `;

    // 5 Calibrated Scenarios Metadata
    const scenarios = [
      {
        id: 1,
        code: 'SCENARIO 01',
        title: 'Localized Heavy Compute',
        desc: 'Cluster of 4 computers in Zone 1 (NW) executes heavy batch compute while others idle with baseline occupancy.',
        source: 'Computer cluster (Zone 1)',
        result: 'Localized cooling demand'
      },
      {
        id: 2,
        code: 'SCENARIO 02',
        title: 'Occupancy Concentration',
        desc: 'All 10 computers maintain uniform light workloads while 16 occupants gather in Zone 3 (SE).',
        source: 'Human occupancy (Zone 3)',
        result: 'Localized metabolic heat'
      },
      {
        id: 3,
        code: 'SCENARIO 03',
        title: 'Distributed Heavy Compute',
        desc: 'All 10 computers spatially distributed across all 4 zones run heavy batch workloads simultaneously.',
        source: 'Distributed compute (10 nodes)',
        result: 'Uniform room-wide heat influx'
      },
      {
        id: 4,
        code: 'SCENARIO 04',
        title: 'High Occupancy / Low Compute',
        desc: 'Seminar setting with 34 occupants distributed room-wide with minimal computer workload.',
        source: 'Human occupancy (Room-wide)',
        result: 'Metabolic heat dominance'
      },
      {
        id: 5,
        code: 'SCENARIO 05',
        title: 'Opposing Thermal Zones',
        desc: 'West side loaded with heavy compute nodes; East side loaded with 20+ occupants under light compute.',
        source: 'Dual source (West Compute / East Occupancy)',
        result: 'Opposing thermal zones'
      }
    ];

    const currentScenario = scenarios.find(s => s.id === activeId) || scenarios[0];

    // 2. Control Mode Bar & Simulation Playback Toolbar
    const controlModeBar = `
      <div class="sim-control-mode-bar ${isPitch ? 'mode-pitch' : ''}">
        <div class="control-mode-left">
          <span class="ctrl-mode-lbl">CONTROL MODE</span>
          <span class="mode-badge-prototype" id="btn-mode-prototype">PROTOTYPE CONTROL</span>
          <div class="status-chip-group">
            <span class="status-chip status-chip-active">STATUS: ACTIVE</span>
            <span class="status-chip status-chip-ctrl">CONTROLLER: THERMAL CONTROL ALGORITHM</span>
            <span class="status-chip status-chip-ai">AI MODEL: NOT INTEGRATED</span>
          </div>
        </div>

        <div class="control-mode-right">
          <div class="sim-scenario-field">
            <span class="field-label">Scenario:</span>
            <span class="field-val" id="sim-active-title">${currentScenario.code} – ${currentScenario.title}</span>
          </div>
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

          <div class="sim-time-field">
            <span class="time-val" id="sim-clock-display">00:00:00 / 02:00:00</span>
          </div>

          <div class="sim-status-badge status-${status.toLowerCase()}" id="sim-status-badge">
            ${status === 'RUNNING' ? '● RUNNING' : status === 'PAUSED' ? '❚❚ PAUSED' : status === 'COMPLETED' ? '✔ COMPLETED' : '○ READY'}
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

    // 3. System Status Strip (Section 10)
    const statusStrip = `
      <div class="sim-status-strip">
        <div class="status-strip-item">
          <span class="status-strip-lbl">CONTROL AUTHORITY</span>
          <span class="status-strip-val" id="ctrl-authority" style="color: #58a6ff;">THERMAL CONTROL ALGORITHM</span>
        </div>
        <div class="status-strip-item">
          <span class="status-strip-lbl">SAFETY GOVERNOR</span>
          <span class="status-strip-val" id="ctrl-safety-status" style="color: #3fb950;">ACTIVE</span>
        </div>
        <div class="status-strip-item">
          <span class="status-strip-lbl">AI MODEL</span>
          <span class="status-strip-val" style="color: #d29922;">NOT INTEGRATED</span>
        </div>
        <div class="status-strip-item">
          <span class="status-strip-lbl">COMFORT TARGET</span>
          <span class="status-strip-val" id="ctrl-comfort-target" style="color: #e6edf3;">23.5°C</span>
        </div>
      </div>
    `;

    // 4. KPI Cards Row (Section 11)
    const kpiRow = `
      <div class="sim-kpi-row">
        <div class="sim-kpi-card">
          <span class="kpi-label">ROOM TEMPERATURE</span>
          <div class="kpi-value" id="ctrl-room-temp">22.5°C</div>
          <span class="kpi-sub">Average across 4 zones</span>
        </div>
        <div class="sim-kpi-card">
          <span class="kpi-label">ROOM COOLING DEMAND</span>
          <div class="kpi-value" id="ctrl-room-demand" style="color: #58a6ff;">0%</div>
          <span class="kpi-sub">Average zone demand</span>
        </div>
        <div class="sim-kpi-card">
          <span class="kpi-label">ROOM SETPOINT</span>
          <div class="kpi-value" id="ctrl-room-setpoint" style="color: #3fb950;">24.0°C</div>
          <span class="kpi-sub">Deterministic mapping from demand</span>
        </div>
        <div class="sim-kpi-card">
          <span class="kpi-label">CONTROLLER</span>
          <div class="kpi-value" style="font-size: 16px; color: #58a6ff; line-height: 1.3;">THERMAL CONTROL ALGORITHM</div>
          <span class="kpi-sub" id="ctrl-fallback-status" style="color: #3fb950;">NORMAL • SINGLE DECISION-MAKER</span>
        </div>
      </div>
    `;

    // 5. Technical CAD Floorplan Stage (Section 29)
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
          <canvas id="sim-heatmap-canvas" class="sim-heatmap-canvas" width="640" height="533"></canvas>

          <!-- 4 Wall-mounted AC Units -->
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

          <!-- 10 Fixed Workstation Markers -->
          ${fixedCompDefs.map(c => `
            <div class="sim-workstation" id="sim-comp-${c.id}" style="left: ${c.left}; top: ${c.top};" data-comp-id="${c.id}" title="Click to inspect node ${c.name}">
              <div class="comp-aura" id="sim-aura-${c.id}"></div>
              <div class="comp-body">
                <span class="comp-id">${c.name}</span>
                <div class="comp-load-bar"><div class="comp-load-fill" id="sim-load-${c.id}" style="width: 20%;"></div></div>
              </div>
            </div>
          `).join('')}

          <div class="sim-comp-tooltip" id="sim-comp-tooltip" style="display: none;">
            <div class="tooltip-header" id="sim-tt-title">Computer 1</div>
            <div class="tooltip-meta"><span id="sim-tt-zone">Zone 1 (NW)</span></div>
            <div class="tooltip-row"><span>CPU:</span> <strong id="sim-tt-cpu">20%</strong></div>
            <div class="tooltip-row"><span>GPU:</span> <strong id="sim-tt-gpu">10%</strong></div>
            <div class="tooltip-row"><span>Workload:</span> <strong id="sim-tt-workload">LIGHT</strong></div>
            <div class="tooltip-row"><span>Heat:</span> <strong id="sim-tt-heat">86.7 W</strong></div>
          </div>
        </div>

        <div class="sim-thermal-legend">
          <div class="legend-scale">
            <span class="legend-val">20.0°C</span>
            <div class="legend-bar">
              <div class="legend-gradient"></div>
              <div class="legend-target-marker" style="left: 31.25%;">
                <div class="marker-line"></div>
                <span class="ref-tag">22.5°C Comfort</span>
              </div>
            </div>
            <span class="legend-val">28.0°C</span>
          </div>
          <span class="legend-desc">Scientific Thermal Scale (Cool → Normal → Warm → Hot)</span>
        </div>
      </div>
    `;

    // 6. Live Telemetry Monitor & Heat Load Balance
    const monitorPanel = `
      <div class="sim-monitor-container">
        <!-- Telemetry Summary Grid -->
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

        <!-- Zone Telemetry Table -->
        <div class="sim-card-panel">
          <div class="card-title-eng">ZONE METRICS SUMMARY</div>
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

        <!-- Thermal Load Balance Ledger -->
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

        <!-- 5-Minute Temperature History Chart -->
        <div class="sim-card-panel">
          <div class="card-title-eng">5-MINUTE TEMPERATURE HISTORY</div>
          <div class="sim-canvas-wrapper">
            <canvas id="sim-chart-canvas" width="520" height="150"></canvas>
          </div>
        </div>
      </div>
    `;

    // 7. Zone-Level Control Telemetry (Section 13)
    const zoneTelemetrySection = `
      <div class="sim-card-section">
        <div class="card-section-header">
          <span class="card-title-eng">ZONE-LEVEL CONTROL TELEMETRY</span>
          <span class="card-subtitle-eng">Thermodynamic state and computed demand across 4 quadrants</span>
        </div>
        <div class="sim-zone-telemetry-grid">
          <div class="zone-telemetry-card">
            <div class="zone-card-header">
              <span class="zone-card-name">ZONE 1</span>
              <span class="zone-card-loc">NW</span>
            </div>
            <div class="zone-card-metrics">
              <div class="zone-metric-row"><span class="zone-metric-lbl">Temperature:</span> <span class="zone-metric-val" id="ctrl-z1-temp">22.5°C</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Heat:</span> <span class="zone-metric-val" id="ctrl-z1-heat">0 W</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Demand:</span> <span class="zone-metric-val" id="ctrl-z1-demand">0%</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Trend:</span> <span class="zone-metric-val" id="ctrl-z1-trend">+0.00°C/min</span></div>
            </div>
          </div>
          <div class="zone-telemetry-card">
            <div class="zone-card-header">
              <span class="zone-card-name">ZONE 2</span>
              <span class="zone-card-loc">NE</span>
            </div>
            <div class="zone-card-metrics">
              <div class="zone-metric-row"><span class="zone-metric-lbl">Temperature:</span> <span class="zone-metric-val" id="ctrl-z2-temp">22.5°C</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Heat:</span> <span class="zone-metric-val" id="ctrl-z2-heat">0 W</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Demand:</span> <span class="zone-metric-val" id="ctrl-z2-demand">0%</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Trend:</span> <span class="zone-metric-val" id="ctrl-z2-trend">+0.00°C/min</span></div>
            </div>
          </div>
          <div class="zone-telemetry-card">
            <div class="zone-card-header">
              <span class="zone-card-name">ZONE 3</span>
              <span class="zone-card-loc">SE</span>
            </div>
            <div class="zone-card-metrics">
              <div class="zone-metric-row"><span class="zone-metric-lbl">Temperature:</span> <span class="zone-metric-val" id="ctrl-z3-temp">22.5°C</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Heat:</span> <span class="zone-metric-val" id="ctrl-z3-heat">0 W</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Demand:</span> <span class="zone-metric-val" id="ctrl-z3-demand">0%</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Trend:</span> <span class="zone-metric-val" id="ctrl-z3-trend">+0.00°C/min</span></div>
            </div>
          </div>
          <div class="zone-telemetry-card">
            <div class="zone-card-header">
              <span class="zone-card-name">ZONE 4</span>
              <span class="zone-card-loc">SW</span>
            </div>
            <div class="zone-card-metrics">
              <div class="zone-metric-row"><span class="zone-metric-lbl">Temperature:</span> <span class="zone-metric-val" id="ctrl-z4-temp">22.5°C</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Heat:</span> <span class="zone-metric-val" id="ctrl-z4-heat">0 W</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Demand:</span> <span class="zone-metric-val" id="ctrl-z4-demand">0%</span></div>
              <div class="zone-metric-row"><span class="zone-metric-lbl">Trend:</span> <span class="zone-metric-val" id="ctrl-z4-trend">+0.00°C/min</span></div>
            </div>
          </div>
        </div>
      </div>
    `;

    // 8. AC-Level Control Output (Section 14)
    const acTelemetrySection = `
      <div class="sim-card-section">
        <div class="card-section-header">
          <span class="card-title-eng">AC-LEVEL CONTROL OUTPUT</span>
          <span class="card-subtitle-eng">Perimeter unit actuation levels mapped through spatial influence matrix</span>
        </div>
        <div class="sim-ac-telemetry-grid">
          <div class="ac-telemetry-card">
            <div class="ac-card-header">
              <span class="ac-card-name">AC1</span>
              <span class="ac-card-wall">NORTH</span>
            </div>
            <div class="ac-card-primary">
              <div class="ac-cooling-lbl">Cooling</div>
              <div class="ac-cooling-val" id="ctrl-ac1-cooling">0%</div>
            </div>
            <div class="ac-setpoint-row">
              <span>Setpoint</span>
              <span class="ac-sp-val" id="ctrl-ac1-sp">24.0°C</span>
            </div>
          </div>
          <div class="ac-telemetry-card">
            <div class="ac-card-header">
              <span class="ac-card-name">AC2</span>
              <span class="ac-card-wall">EAST</span>
            </div>
            <div class="ac-card-primary">
              <div class="ac-cooling-lbl">Cooling</div>
              <div class="ac-cooling-val" id="ctrl-ac2-cooling">0%</div>
            </div>
            <div class="ac-setpoint-row">
              <span>Setpoint</span>
              <span class="ac-sp-val" id="ctrl-ac2-sp">24.0°C</span>
            </div>
          </div>
          <div class="ac-telemetry-card">
            <div class="ac-card-header">
              <span class="ac-card-name">AC3</span>
              <span class="ac-card-wall">SOUTH</span>
            </div>
            <div class="ac-card-primary">
              <div class="ac-cooling-lbl">Cooling</div>
              <div class="ac-cooling-val" id="ctrl-ac3-cooling">0%</div>
            </div>
            <div class="ac-setpoint-row">
              <span>Setpoint</span>
              <span class="ac-sp-val" id="ctrl-ac3-sp">24.0°C</span>
            </div>
          </div>
          <div class="ac-telemetry-card">
            <div class="ac-card-header">
              <span class="ac-card-name">AC4</span>
              <span class="ac-card-wall">WEST</span>
            </div>
            <div class="ac-card-primary">
              <div class="ac-cooling-lbl">Cooling</div>
              <div class="ac-cooling-val" id="ctrl-ac4-cooling">0%</div>
            </div>
            <div class="ac-setpoint-row">
              <span>Setpoint</span>
              <span class="ac-sp-val" id="ctrl-ac4-sp">24.0°C</span>
            </div>
          </div>
        </div>
      </div>
    `;

    // 9. Runtime Control Pipeline (Sections 15, 16, 17, 18 — Decision Trace Replacement)
    const pipelineSection = `
      <div class="sim-pipeline-section">
        <div class="card-section-header">
          <span class="card-title-eng">RUNTIME CONTROL PIPELINE</span>
          <span class="card-subtitle-eng">Deterministic single-authority flow • No secondary decision-maker</span>
        </div>
        <div class="sim-pipeline-grid">
          <div class="pipeline-step-card">
            <div class="pipeline-step-header">
              <span class="pipeline-step-num">01</span>
              <span class="pipeline-step-title">THERMAL SENSORS</span>
            </div>
            <span class="pipeline-step-desc">Live zone temperatures and thermal loads</span>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step-card">
            <div class="pipeline-step-header">
              <span class="pipeline-step-num">02</span>
              <span class="pipeline-step-title">THERMAL CONTROL ALGORITHM</span>
            </div>
            <span class="pipeline-step-desc">Calculated zone cooling demand</span>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step-card">
            <div class="pipeline-step-header">
              <span class="pipeline-step-num">03</span>
              <span class="pipeline-step-title">SAFETY GOVERNOR</span>
            </div>
            <span class="pipeline-step-desc">Commands validated & limits enforced</span>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step-card">
            <div class="pipeline-step-header">
              <span class="pipeline-step-num">04</span>
              <span class="pipeline-step-title">HVAC ACTUATION</span>
            </div>
            <span class="pipeline-step-desc">AC1–AC4 perimeter commands applied</span>
          </div>
          <div class="pipeline-arrow">→</div>
          <div class="pipeline-step-card">
            <div class="pipeline-step-header">
              <span class="pipeline-step-num">05</span>
              <span class="pipeline-step-title">THERMAL RESPONSE</span>
            </div>
            <span class="pipeline-step-desc">Zone temperatures & heat balance updated</span>
          </div>
        </div>

        <!-- Live Decision Details (Section 18) -->
        <div class="sim-decision-summary-card">
          <span class="decision-summary-label">CURRENT DECISION</span>
          <span class="decision-summary-value" id="ctrl-current-decision">Room: 22.5°C  |  Average demand: 0%  |  AC1: 0%  |  AC2: 0%  |  AC3: 0%  |  AC4: 0%</span>
        </div>
      </div>
    `;

    // 10. Scenario Catalog (Sections 19, 20, 21)
    const scenarioCards = `
      <div class="sim-scenarios-section">
        <div class="section-header-bar">
          <div class="sh-title-group">
            <h3 class="sh-title">SCENARIO CATALOG</h3>
            <span class="sh-subtitle">Calibrated thermodynamic scenarios for thermal profiling and validation</span>
          </div>
        </div>

        <div class="sim-scenarios-grid">
          ${scenarios.map(s => {
            const isAct = s.id === activeId;
            return `
              <div class="sim-scenario-card ${isAct ? 'card-active' : ''}" data-scenario-id="${s.id}">
                <div class="sc-header">
                  <span class="sc-id">${s.code}</span>
                  ${isAct ? '<span class="sc-status-tag sc-status-active">ACTIVE SCENARIO</span>' : ''}
                </div>
                <h4 class="sc-title">${s.title}</h4>
                <p class="sc-desc">${s.desc}</p>
                <div class="sc-spec-list">
                  <div class="spec-item"><span class="spec-k">PRIMARY SOURCE:</span> <span class="spec-v">${s.source}</span></div>
                  <div class="spec-item"><span class="spec-k">EXPECTED RESPONSE:</span> <span class="spec-v">${s.result}</span></div>
                </div>
                <button class="btn-eng ${isAct ? 'btn-eng-primary' : 'btn-eng-subtle'} btn-select-scen" data-select-id="${s.id}">
                  ${isAct ? 'Active' : 'Load Scenario'}
                </button>
              </div>
            `;
          }).join('')}
        </div>
      </div>
    `;

    // 11. Collapsible Computer Workstations Telemetry Drawer
    const computerTelemetrySection = `
      <details class="sim-collapsible-nodes">
        <summary class="sim-nodes-summary">
          <span class="nodes-summary-title">WORKSTATION TELEMETRY</span>
          <span class="nodes-summary-sub">10 Cluster Nodes • Workloads, CPU, GPU & Heat Output (Click to Expand)</span>
        </summary>
        <div class="sim-nodes-content">
          <!-- Compact Summary Strip -->
          <div class="sim-comp-summary-strip">
            <div class="summary-metric"><span class="sm-lbl">TOTAL NODES</span><span class="sm-val" id="sim-sum-total">10 / 10</span></div>
            <div class="summary-metric"><span class="sm-lbl">HEAVY LOAD</span><span class="sm-val" id="sim-sum-heavy">0</span></div>
            <div class="summary-metric"><span class="sm-lbl">ACTIVE NODES</span><span class="sm-val" id="sim-sum-active">10</span></div>
            <div class="summary-metric"><span class="sm-lbl">TOTAL COMPUTE HEAT</span><span class="sm-val" id="sim-sum-heat">850.0 W</span></div>
            <div class="summary-metric"><span class="sm-lbl">AVERAGE CPU</span><span class="sm-val" id="sim-sum-avg-cpu">20.0%</span></div>
            <div class="summary-metric"><span class="sm-lbl">AVERAGE GPU</span><span class="sm-val" id="sim-sum-avg-gpu">10.0%</span></div>
          </div>

          <!-- Selected Node Inspector -->
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

          <!-- Telemetry Table -->
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
      </details>
    `;

    // 12. Scenario Comparison Matrix Modal
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
                    <th>Name</th>
                    <th>Type</th>
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

    // Assemble final clean coherent page structure matching Section 35
    const fullContent = `
      ${header}
      ${controlModeBar}
      ${statusStrip}
      ${kpiRow}
      <div class="sim-main-stage ${isPitch ? 'stage-pitch' : ''}">
        ${roomViewport}
        ${monitorPanel}
      </div>
      ${zoneTelemetrySection}
      ${acTelemetrySection}
      ${pipelineSection}
      ${scenarioCards}
      ${computerTelemetrySection}
      ${comparisonDrawer}
    `;

    return PageShell({
      content: fullContent,
      extraClass: `page-simulation ${isPitch ? 'mode-pitch' : ''}`
    });
  }

  // --------------------------------------------------------------------------
  // Targeted DOM Telemetry Updates (WebSocket 10 Hz)
  // --------------------------------------------------------------------------

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

    // Zone tags on top-down CAD floorplan
    const tag1 = document.getElementById('sim-tag-z1');
    if (tag1) tag1.textContent = `${(zoneTemps.zone_1 ?? 22.5).toFixed(1)} °C`;
    const tag2 = document.getElementById('sim-tag-z2');
    if (tag2) tag2.textContent = `${(zoneTemps.zone_2 ?? 22.5).toFixed(1)} °C`;
    const tag3 = document.getElementById('sim-tag-z3');
    if (tag3) tag3.textContent = `${(zoneTemps.zone_3 ?? 22.5).toFixed(1)} °C`;
    const tag4 = document.getElementById('sim-tag-z4');
    if (tag4) tag4.textContent = `${(zoneTemps.zone_4 ?? 22.5).toFixed(1)} °C`;

    // Occupancy counts on zone overlays
    const occ = data.occupancy || {};
    const occZones = occ.zones || {};
    const occ1 = document.getElementById('sim-occ-z1');
    if (occ1) occ1.textContent = `${occZones.zone_1 ?? 0} OCC`;
    const occ2 = document.getElementById('sim-occ-z2');
    if (occ2) occ2.textContent = `${occZones.zone_2 ?? 0} OCC`;
    const occ3 = document.getElementById('sim-occ-z3');
    if (occ3) occ3.textContent = `${occZones.zone_3 ?? 0} OCC`;
    const occ4 = document.getElementById('sim-occ-z4');
    if (occ4) occ4.textContent = `${occZones.zone_4 ?? 0} OCC`;

    // Zone summary table cells
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

    // 4. Perimeter AC Units on walls
    const acs = data.hvac?.acs || [];
    acs.forEach(ac => {
      const elVal = document.getElementById(`sim-ac-${ac.wall}-val`);
      if (elVal) {
        elVal.textContent = `${ac.setpoint_c}°C | ${ac.cooling_level_percent}% | ${ac.state}`;
      }
    });

    // 5. Workstations markers & Computer Telemetry Table
    computers.forEach(c => {
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

    // 7. Render Canvas Heatmap
    const canvasHeat = document.getElementById('sim-heatmap-canvas');
    if (canvasHeat && window.HVEAC_SIM_VISUALS) {
      window.HVEAC_SIM_VISUALS.renderThermalHeatmap(canvasHeat, data);
    }

    // 8. Thermal Control Algorithm Telemetry & Dynamic Pipeline Data
    const ctrl = data.control || {};
    if (ctrl) {
      // Room-level summary (Section 11)
      const ctrlRoomTemp = document.getElementById('ctrl-room-temp');
      if (ctrlRoomTemp) ctrlRoomTemp.textContent = `${(ctrl.room_temperature_c ?? 22.5).toFixed(1)}°C`;
      const ctrlRoomDemand = document.getElementById('ctrl-room-demand');
      if (ctrlRoomDemand) ctrlRoomDemand.textContent = `${(ctrl.room_cooling_demand ?? 0).toFixed(0)}%`;
      const ctrlRoomSp = document.getElementById('ctrl-room-setpoint');
      if (ctrlRoomSp) ctrlRoomSp.textContent = `${(ctrl.room_setpoint_c ?? 24.0).toFixed(1)}°C`;
      const ctrlFallback = document.getElementById('ctrl-fallback-status');
      if (ctrlFallback) {
        ctrlFallback.textContent = ctrl.is_fallback ? 'FALLBACK ACTIVATED' : 'NORMAL • SINGLE DECISION-MAKER';
        ctrlFallback.style.color = ctrl.is_fallback ? '#f85149' : '#3fb950';
      }

      // Zone-level control telemetry (Section 13)
      const zones = ctrl.zones || {};
      ['zone_1', 'zone_2', 'zone_3', 'zone_4'].forEach((zk, i) => {
        const n = i + 1;
        const z = zones[zk] || {};
        const elTmp = document.getElementById(`ctrl-z${n}-temp`);
        if (elTmp) elTmp.textContent = `${(z.temperature_c ?? 22.5).toFixed(1)}°C`;
        const elHt = document.getElementById(`ctrl-z${n}-heat`);
        if (elHt) elHt.textContent = `${Math.round(z.heat_load_w ?? 0)} W`;
        const elDm = document.getElementById(`ctrl-z${n}-demand`);
        if (elDm) {
          const d = z.cooling_demand ?? 0;
          elDm.textContent = `${d.toFixed(0)}%`;
          elDm.style.color = d > 50 ? '#f85149' : d > 20 ? '#d29922' : '#3fb950';
        }
        const elTr = document.getElementById(`ctrl-z${n}-trend`);
        if (elTr) {
          const tv = z.trend_c_per_min ?? 0;
          elTr.textContent = `${tv >= 0 ? '+' : ''}${tv.toFixed(2)}°C/min`;
          elTr.style.color = tv > 0.05 ? '#f85149' : tv < -0.05 ? '#58a6ff' : '#8b949e';
        }
      });

      // AC-level control output (Section 14)
      const ctrlAcs = ctrl.acs || {};
      ['ac1', 'ac2', 'ac3', 'ac4'].forEach(ak => {
        const a = ctrlAcs[ak] || {};
        const elCl = document.getElementById(`ctrl-${ak}-cooling`);
        if (elCl) {
          const cv = a.cooling_percent ?? 0;
          elCl.textContent = `${cv.toFixed(0)}%`;
          elCl.style.color = cv > 50 ? '#f85149' : cv > 20 ? '#d29922' : '#3fb950';
        }
        const elSp = document.getElementById(`ctrl-${ak}-sp`);
        if (elSp) elSp.textContent = `${(a.setpoint_c ?? 24.0).toFixed(1)}°C`;
      });

      // Live Decision Summary (Section 18)
      const elCurDec = document.getElementById('ctrl-current-decision');
      if (elCurDec && ctrl.acs) {
        const rT = (ctrl.room_temperature_c ?? 22.5).toFixed(1);
        const rD = Math.round(ctrl.room_cooling_demand ?? 0);
        const a1 = Math.round(ctrl.acs.ac1?.cooling_percent ?? 0);
        const a2 = Math.round(ctrl.acs.ac2?.cooling_percent ?? 0);
        const a3 = Math.round(ctrl.acs.ac3?.cooling_percent ?? 0);
        const a4 = Math.round(ctrl.acs.ac4?.cooling_percent ?? 0);
        elCurDec.textContent = `Room: ${rT}°C  |  Average demand: ${rD}%  |  AC1: ${a1}%  |  AC2: ${a2}%  |  AC3: ${a3}%  |  AC4: ${a4}%`;
      }
    }

    // Safety Governor status in status strip
    const govStatus = document.getElementById('ctrl-safety-status');
    if (govStatus && data.safety_status) {
      govStatus.textContent = data.safety_status;
      govStatus.style.color = data.safety_status === 'ACTIVE' ? '#3fb950' : data.safety_status === 'CLAMPED' ? '#d29922' : '#f85149';
    }
  }

  // ==========================================================================
  // EXPORTED FUNCTIONS
  // ==========================================================================

  window.HVEAC_PAGES = {
    renderOverview,
    renderOccupancy,
    renderNodes,
    renderEnvironment,
    renderSimulation,
    updateSimulationDom,
    renderThermal,
    renderHvac,
    renderAnalytics,
    renderControl
  };

})(window);
