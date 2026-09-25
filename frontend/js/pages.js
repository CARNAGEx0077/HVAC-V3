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

  window.HVEAC_PAGES = {
    renderOverview,
    renderOccupancy,
    renderNodes,
    renderEnvironment,
    renderThermal,
    renderHvac,
    renderAnalytics,
    renderControl
  };

})(window);
