/**
 * HVEAC V3 Simulation Lab - High-Performance Engineering Visualizations
 * 
 * Technical Simulation Workstation Components:
 * 1. 2D Thermal Heatmap with scientific spatial gradient interpolation,
 *    subtle computational dissipation, occupancy metabolic warmth,
 *    and perimeter AC cooling vents.
 * 2. Real-Time 5-Minute Temperature History Multi-Trace Trend Chart.
 */

(function (window) {
  'use strict';

  /**
   * Calibrated scientific thermal colormap (20.0°C to 28.0°C)
   * COOL (slate blue) -> NORMAL (sage teal/green) -> WARM (amber) -> HOT (brick red)
   */
  function temperatureToRgba(tempC, alpha = 0.5) {
    const t = Math.min(Math.max(tempC, 20.0), 28.0);
    // Normalized 0.0 (20°C) to 1.0 (28°C)
    const factor = (t - 20.0) / 8.0;

    let r, g, b;
    if (factor < 0.25) {
      // 20.0 to 22.0: Muted Slate Blue to Muted Teal
      const local = factor / 0.25;
      r = Math.round(44 + local * (38 - 44));
      g = Math.round(82 + local * (120 - 82));
      b = Math.round(130 + local * (115 - 130));
    } else if (factor < 0.40) {
      // 22.0 to 23.2: Muted Teal to Sage Green (Comfort Window)
      const local = (factor - 0.25) / 0.15;
      r = Math.round(38 + local * (52 - 38));
      g = Math.round(120 + local * (136 - 120));
      b = Math.round(115 + local * (78 - 115));
    } else if (factor < 0.65) {
      // 23.2 to 25.2: Sage Green to Technical Amber
      const local = (factor - 0.40) / 0.25;
      r = Math.round(52 + local * (180 - 52));
      g = Math.round(136 + local * (125 - 136));
      b = Math.round(78 + local * (35 - 78));
    } else if (factor < 0.85) {
      // 25.2 to 26.8: Technical Amber to Terracotta Ochre
      const local = (factor - 0.65) / 0.20;
      r = Math.round(180 + local * (195 - 180));
      g = Math.round(125 + local * (85 - 125));
      b = Math.round(35 + local * (40 - 35));
    } else {
      // 26.8 to 28.0+: Terracotta to Brick Red
      const local = (factor - 0.85) / 0.15;
      r = Math.round(195 + local * (185 - 195));
      g = Math.round(85 - local * (85 - 55));
      b = Math.round(40 + local * (45 - 40));
    }

    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  /**
   * Renders the 2D technical spatial thermal heatmap
   */
  function renderThermalHeatmap(canvas, state) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    const thermal = state?.thermal || {};
    const zoneTemps = thermal.zone_temperatures_c || {
      zone_1: 22.5,
      zone_2: 22.5,
      zone_3: 22.5,
      zone_4: 22.5
    };

    const t1 = zoneTemps.zone_1 ?? 22.5;
    const t2 = zoneTemps.zone_2 ?? 22.5;
    const t3 = zoneTemps.zone_3 ?? 22.5;
    const t4 = zoneTemps.zone_4 ?? 22.5;

    // Base background: dark charcoal engineering slate
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    // 1. Zone Quadrant Thermal Radial Gradients (Calibrated scientific dissipation)
    const zoneDefs = [
      { cx: width * 0.25, cy: height * 0.25, temp: t1, radius: width * 0.44 },
      { cx: width * 0.75, cy: height * 0.25, temp: t2, radius: width * 0.44 },
      { cx: width * 0.75, cy: height * 0.75, temp: t3, radius: width * 0.44 },
      { cx: width * 0.25, cy: height * 0.75, temp: t4, radius: width * 0.44 }
    ];

    ctx.save();
    ctx.globalCompositeOperation = 'screen';

    zoneDefs.forEach(z => {
      const grad = ctx.createRadialGradient(z.cx, z.cy, 15, z.cx, z.cy, z.radius);
      grad.addColorStop(0, temperatureToRgba(z.temp, 0.50));
      grad.addColorStop(0.55, temperatureToRgba(z.temp, 0.28));
      grad.addColorStop(1, 'rgba(13, 17, 23, 0.0)');

      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(z.cx, z.cy, z.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // 2. Micro Computational Heat Flux from Active Computers
    const computers = state?.computers || [];
    computers.forEach(c => {
      if (c.heat_watts && c.heat_watts > 110.0) {
        // Map 12m x 10m coordinates to canvas pixel space
        const px = (c.x / 12.0) * width;
        const py = ((10.0 - c.y) / 10.0) * height; // Invert Y (North is top)
        const intensity = Math.min((c.heat_watts - 90.0) / 210.0, 1.0);
        const radius = 20 + intensity * 35;

        const compGrad = ctx.createRadialGradient(px, py, 4, px, py, radius);
        compGrad.addColorStop(0, `rgba(185, 65, 45, ${0.30 + intensity * 0.25})`);
        compGrad.addColorStop(0.65, `rgba(180, 100, 35, ${0.15 + intensity * 0.15})`);
        compGrad.addColorStop(1, 'rgba(185, 65, 45, 0.0)');

        ctx.fillStyle = compGrad;
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // 3. Occupancy Metabolic Heat Flux
    const occ = state?.occupancy?.zones || {};
    const occPositions = {
      zone_1: { x: width * 0.25, y: height * 0.25 },
      zone_2: { x: width * 0.75, y: height * 0.25 },
      zone_3: { x: width * 0.75, y: height * 0.75 },
      zone_4: { x: width * 0.25, y: height * 0.75 }
    };

    Object.entries(occ).forEach(([zid, count]) => {
      if (count > 4) {
        const pos = occPositions[zid];
        const occRadius = Math.min(25 + count * 4.5, width * 0.32);
        const occGrad = ctx.createRadialGradient(pos.x, pos.y, 8, pos.x, pos.y, occRadius);
        occGrad.addColorStop(0, 'rgba(180, 115, 40, 0.25)');
        occGrad.addColorStop(0.55, 'rgba(165, 95, 35, 0.12)');
        occGrad.addColorStop(1, 'rgba(180, 115, 40, 0.0)');

        ctx.fillStyle = occGrad;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, occRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // 4. AC Cooling Airflow Inward Dispersion
    const acs = state?.hvac?.acs || [];
    acs.forEach(ac => {
      const coolingLvl = ac.cooling_level || 0.0;
      if (coolingLvl > 0.05) {
        let ax, ay, vx, vy, wR;
        if (ac.wall === 'north') {
          ax = width * 0.5; ay = 0; vx = 0; vy = 1; wR = width * 0.40;
        } else if (ac.wall === 'east') {
          ax = width; ay = height * 0.5; vx = -1; vy = 0; wR = width * 0.40;
        } else if (ac.wall === 'south') {
          ax = width * 0.5; ay = height; vx = 0; vy = -1; wR = width * 0.40;
        } else {
          // West
          ax = 0; ay = height * 0.5; vx = 1; vy = 0; wR = width * 0.40;
        }

        const coolGrad = ctx.createRadialGradient(ax, ay, 8, ax + vx * 45, ay + vy * 45, wR * coolingLvl);
        coolGrad.addColorStop(0, `rgba(56, 139, 253, ${0.30 * coolingLvl})`);
        coolGrad.addColorStop(0.60, `rgba(45, 115, 185, ${0.12 * coolingLvl})`);
        coolGrad.addColorStop(1, 'rgba(56, 139, 253, 0.0)');

        ctx.fillStyle = coolGrad;
        ctx.beginPath();
        ctx.arc(ax, ay, wR * coolingLvl, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    ctx.restore();

    // 5. Technical CAD Floorplan Grid & Axis Markings
    ctx.save();

    // Subdued quadrant dividers (dashed thin lines)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);

    ctx.beginPath();
    ctx.moveTo(width * 0.5, 0);
    ctx.lineTo(width * 0.5, height);
    ctx.moveTo(0, height * 0.5);
    ctx.lineTo(width, height * 0.5);
    ctx.stroke();

    // Center crosshair
    ctx.setLineDash([]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.beginPath();
    ctx.moveTo(width * 0.5 - 6, height * 0.5);
    ctx.lineTo(width * 0.5 + 6, height * 0.5);
    ctx.moveTo(width * 0.5, height * 0.5 - 6);
    ctx.lineTo(width * 0.5, height * 0.5 + 6);
    ctx.stroke();

    // Perimeter coordinate ticks (12m x 10m scale)
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.28)';
    ctx.font = '8px monospace';
    ctx.textAlign = 'center';

    // Top/Bottom ticks (every 2m)
    for (let m = 2; m < 12; m += 2) {
      const tx = (m / 12.0) * width;
      ctx.beginPath();
      ctx.moveTo(tx, 0); ctx.lineTo(tx, 4);
      ctx.moveTo(tx, height); ctx.lineTo(tx, height - 4);
      ctx.stroke();
    }

    // Left/Right ticks (every 2m)
    for (let m = 2; m < 10; m += 2) {
      const ty = (m / 10.0) * height;
      ctx.beginPath();
      ctx.moveTo(0, ty); ctx.lineTo(4, ty);
      ctx.moveTo(width, ty); ctx.lineTo(width - 4, ty);
      ctx.stroke();
    }

    // Outer perimeter boundary (clean neutral engineering border)
    ctx.strokeStyle = '#30363d';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

    ctx.restore();
  }

  /**
   * Renders the rolling 5-minute temperature history line chart (Technical Trend Monitor)
   */
  function renderHistoryChart(canvas, history) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Engineering dark background
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, width, height);

    const padLeft = 40;
    const padRight = 14;
    const padTop = 24;
    const padBottom = 22;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Y-axis range: 20.0°C to 28.0°C
    const yMin = 20.0;
    const yMax = 28.0;

    const getY = (val) => {
      const norm = (val - yMin) / (yMax - yMin);
      return padTop + plotH * (1.0 - Math.min(Math.max(norm, 0.0), 1.0));
    };

    // Horizontal grid lines & axis labels
    ctx.save();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#8b949e';
    ctx.font = '9px monospace';
    ctx.textAlign = 'right';

    for (let t = yMin; t <= yMax; t += 2.0) {
      const y = getY(t);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();
      ctx.fillText(`${t.toFixed(0)}°C`, padLeft - 6, y + 3);
    }

    // Target comfort reference line: 22.5°C (subtle dashed blue reference)
    const yComfort = getY(22.5);
    ctx.strokeStyle = 'rgba(56, 139, 253, 0.35)';
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, yComfort);
    ctx.lineTo(width - padRight, yComfort);
    ctx.stroke();
    ctx.fillText('22.5°', padLeft - 6, yComfort + 3);
    ctx.restore();

    if (!history || history.length < 2) {
      ctx.save();
      ctx.fillStyle = '#484f58';
      ctx.font = '11px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('AWAITING SIMULATION TELEMETRY...', width * 0.5, height * 0.5 + 4);
      ctx.restore();
      return;
    }

    // Keep last 30 data points (5 minutes at 10s intervals)
    const points = history.slice(-30);
    const n = points.length;

    const getX = (idx) => padLeft + (idx / (n - 1)) * plotW;

    // Helper to draw a single series
    function drawSeries(key, color, lineWidth = 1.2, dash = []) {
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.setLineDash(dash);
      ctx.beginPath();

      points.forEach((p, idx) => {
        const x = getX(idx);
        const y = getY(p[key] ?? 22.5);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.restore();
    }

    // Draw individual zones with restrained engineering colors
    drawSeries('z1', '#e06c75', 1.2);       // Zone 1: Muted Coral/Brick
    drawSeries('z2', '#d19a66', 1.2);       // Zone 2: Muted Amber
    drawSeries('z3', '#98c379', 1.2);       // Zone 3: Muted Sage
    drawSeries('z4', '#61afef', 1.2);       // Zone 4: Muted Slate Blue

    // Draw Room Average (Crisp off-white solid line)
    drawSeries('avg', '#f0f6fc', 1.8);

    // Compact engineering legend at top
    ctx.save();
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';

    const items = [
      { label: 'Room Avg', color: '#f0f6fc' },
      { label: 'Z1 NW', color: '#e06c75' },
      { label: 'Z2 NE', color: '#d19a66' },
      { label: 'Z3 SE', color: '#98c379' },
      { label: 'Z4 SW', color: '#61afef' }
    ];

    let legX = padLeft + 6;
    items.forEach(it => {
      ctx.fillStyle = it.color;
      ctx.fillRect(legX, 8, 8, 3);
      ctx.fillStyle = '#8b949e';
      ctx.fillText(it.label, legX + 12, 12);
      legX += 72;
    });

    ctx.restore();
  }

  window.HVEAC_SIM_VISUALS = {
    temperatureToRgba,
    renderThermalHeatmap,
    renderHistoryChart
  };

})(window);
