/**
 * HVEAC Control Center - Component Architecture
 * 
 * Reusable, clean functional HTML components for dashboard views.
 */

(function (window) {
  'use strict';

  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  /**
   * MetricCard Component
   * Displays high-level KPIs with large numbers and micro-labels
   */
  function MetricCard({ label, value, subtext, id = '' }) {
    return `
      <div class="metric-card" ${id ? `id="${id}"` : ''}>
        <div class="metric-card-label">${escapeHtml(label)}</div>
        <div class="metric-card-value">${escapeHtml(value)}</div>
        <div class="metric-card-subtext">${escapeHtml(subtext)}</div>
      </div>
    `;
  }

  /**
   * StatusBadge Component
   */
  function StatusBadge({ label, status = 'default' }) {
    const statusClass = `status-${status.toLowerCase().replace(/\s+/g, '-')}`;
    return `<span class="status-badge ${statusClass}">${escapeHtml(label)}</span>`;
  }

  /**
   * Panel Component
   * Standard dark container for widgets, feeds, and tables
   */
  function Panel({ title, content, subtext = '', extraHeader = '', className = '' }) {
    return `
      <section class="panel ${escapeHtml(className)}">
        <div class="panel-header">
          <div class="panel-title">${escapeHtml(title)}</div>
          ${subtext ? `<div class="panel-subtext">${escapeHtml(subtext)}</div>` : ''}
          ${extraHeader}
        </div>
        <div class="panel-body">
          ${content}
        </div>
      </section>
    `;
  }

  /**
   * Pipeline Component
   * Renders the 6-stage live system pipeline flow with nodes and connecting arrows
   */
  function Pipeline({ nodes = [] }) {
    const nodesHtml = nodes.map((node, index) => {
      const isPlanned = node.planned || node.status === 'PLANNED';
      const plannedClass = isPlanned ? 'planned' : '';
      const statusClass = `status-${node.status.toLowerCase().replace(/\s+/g, '-')}`;

      const nodeMarkup = `
        <div class="pipeline-node ${plannedClass}" data-node-id="${escapeHtml(node.id || '')}">
          <div class="pipeline-node-title">${escapeHtml(node.title)}</div>
          <div class="pipeline-node-status ${statusClass}">${escapeHtml(node.status)}</div>
        </div>
      `;

      const arrowMarkup = index < nodes.length - 1 ? '<div class="pipeline-arrow">→</div>' : '';

      return nodeMarkup + arrowMarkup;
    }).join('');

    return `
      <div class="pipeline-card">
        <div class="pipeline-header">LIVE SYSTEM PIPELINE</div>
        <div class="pipeline-flow">
          ${nodesHtml}
        </div>
      </div>
    `;
  }

  /**
   * DataTable Component
   * Renders columnar tabular telemetry data with graceful empty states
   */
  function DataTable({ columns = [], rows = [], emptyMessage = 'NO DATA AVAILABLE' }) {
    const ths = columns.map(col => `<th>${escapeHtml(col)}</th>`).join('');

    let tbodyContent = '';
    if (!rows || rows.length === 0) {
      tbodyContent = `
        <tr>
          <td colspan="${columns.length}" class="table-empty-cell">${escapeHtml(emptyMessage)}</td>
        </tr>
      `;
    } else {
      tbodyContent = rows.map(row => {
        const tds = row.map(cell => {
          if (typeof cell === 'string' && (cell.startsWith('<span') || cell.startsWith('<div'))) {
            return `<td>${cell}</td>`;
          }
          return `<td>${escapeHtml(cell)}</td>`;
        }).join('');
        return `<tr>${tds}</tr>`;
      }).join('');
    }

    return `
      <div class="data-table-container">
        <table class="data-table">
          <thead>
            <tr>${ths}</tr>
          </thead>
          <tbody>
            ${tbodyContent}
          </tbody>
        </table>
      </div>
    `;
  }

  /**
   * Button Component
   */
  function Button({ id = '', label, variant = 'default', action = '', target = '', extraClass = '' }) {
    return `
      <button 
        type="button" 
        ${id ? `id="${id}"` : ''} 
        class="btn btn-${escapeHtml(variant)} ${escapeHtml(extraClass)}"
        data-action="${escapeHtml(action)}"
        data-target="${escapeHtml(target)}"
      >
        ${escapeHtml(label)}
      </button>
    `;
  }

  /**
   * PageShell Component
   */
  function PageShell({ content, extraClass = '' }) {
    return `<div class="page-shell ${escapeHtml(extraClass)}">${content}</div>`;
  }

  window.HVEAC_COMPONENTS = {
    MetricCard,
    StatusBadge,
    Panel,
    Pipeline,
    DataTable,
    Button,
    PageShell,
    escapeHtml
  };

})(window);
