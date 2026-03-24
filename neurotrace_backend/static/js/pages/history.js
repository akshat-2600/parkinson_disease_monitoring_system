/* ============================================================
   static/js/pages/history.js
   ============================================================ */
const PageHistory = (() => {
  const ROOT = 'page-history';

  function _html() {
    return `
      <div class="page-header">
        <div>
          <div class="page-title">Historical Tracking</div>
          <div class="page-subtitle">Longitudinal progression analysis & intervention timeline</div>
        </div>
        <button class="btn btn-ghost" id="histRefreshBtn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>
          Refresh
        </button>
      </div>
      <div id="historyError"></div>
      <div id="historyLoading" style="display:none">${Skeletons.historySkeleton()}</div>

      <!-- Empty state -->
      <div id="historyEmpty">
        <div class="card" style="text-align:center;padding:60px 24px">
          <div style="font-size:56px;margin-bottom:16px;opacity:0.4">📊</div>
          <div style="font-family:var(--font-display);font-size:18px;font-weight:700;margin-bottom:8px">No history available</div>
          <div style="font-size:13px;color:var(--text-2);max-width:380px;margin:auto;margin-bottom:20px">
            History charts will appear here as predictions accumulate over time.
            Run your first prediction to start tracking.
          </div>
          <button class="btn btn-primary" onclick="Router.navigate('realtime')" style="margin:auto">
            ⚡ Run First Prediction
          </button>
        </div>
      </div>

      <!-- Data -->
      <div id="historyData" style="display:none">
        <div class="card card-glow-cyan section-gap">
          <div class="card-header">
            <span class="card-title">Severity Timeline</span>
            <span class="text-muted" style="font-size:11px;font-family:var(--font-mono)" id="histPeriodLabel">—</span>
          </div>
          <div class="chart-wrap" style="height:240px"><canvas id="historySeverityChart"></canvas></div>
        </div>
        <div class="grid-2 section-gap">
          <div class="card card-glow-violet">
            <div class="card-header"><span class="card-title">UPDRS Scores</span></div>
            <div class="chart-wrap" style="height:200px"><canvas id="historyUpdrsChart"></canvas></div>
          </div>
          <div class="card card-glow-green">
            <div class="card-header"><span class="card-title">Multi-Modality Trends</span></div>
            <div class="chart-wrap" style="height:200px"><canvas id="historyModalityChart"></canvas></div>
          </div>
        </div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">Intervention Timeline</span>
            <div class="card-icon" style="background:var(--cyan-dim)">📋</div>
          </div>
          <div class="history-timeline" id="interventionTimeline"></div>
        </div>
      </div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('histRefreshBtn')?.addEventListener('click', load);
  }

  async function load() {
    Helpers.clearError('historyError');
    Helpers.hide('historyData');
    Helpers.hide('historyLoading');

    let data = null;
    try {
      Helpers.show('historyLoading');
      Helpers.hide('historyEmpty');
      const res = await API.getHistory(App.patientId);
      data = res?.data || null;
    } catch (e) {
      Helpers.hide('historyLoading');
      if (!e.message.includes('401') && !e.message.includes('404')) {
        Helpers.showError('historyError', `Error: ${e.message}`);
      }
    }
    Helpers.hide('historyLoading');

    // Check if there is actual history (non-empty labels array)
    const hasHistory = data &&
      Array.isArray(data.labels) &&
      data.labels.length > 0 &&
      Array.isArray(data.severity) &&
      data.severity.some(v => v != null);

    if (hasHistory) {
      _render(data);
    } else {
      Helpers.show('historyEmpty');
    }
  }

  function _render(d) {
    Helpers.hide('historyEmpty');
    Helpers.show('historyData');

    const labels = d.labels || [];
    Helpers.setText('histPeriodLabel', `${labels.length} data point${labels.length !== 1 ? 's' : ''}`);

    _line('historySeverity', 'historySeverityChart', labels, [{
      label: 'Severity', data: d.severity || [],
      borderColor: '#00d4ff', fill: true, tension: 0.4, borderWidth: 2,
      pointRadius: 3, pointHoverRadius: 6,
    }], { minY: 0, maxY: 100 });

    // UPDRS only if data exists
    if (d.updrs && d.updrs.some(v => v != null)) {
      _line('historyUpdrs', 'historyUpdrsChart', labels, [{
        label: 'UPDRS', data: d.updrs,
        borderColor: '#7b5cff', backgroundColor: 'rgba(123,92,255,0.1)',
        fill: true, tension: 0.4, borderWidth: 2, pointRadius: 2,
      }], { maxX: 6 });
    } else {
      Helpers.setHTML('historyUpdrsChart', '');
      const wrap = document.getElementById('historyUpdrsChart')?.parentElement;
      if (wrap) wrap.innerHTML = `<div class="empty-state" style="padding:40px"><div class="empty-text">No UPDRS data yet</div></div>`;
    }

    // Multi-modality
    const modalityDatasets = [
      d.voice  && { label: 'Voice', data: d.voice,  borderColor: '#00e5a0' },
      d.mri    && { label: 'MRI',   data: d.mri,    borderColor: '#ffb547' },
      d.motor  && { label: 'Motor', data: d.motor,  borderColor: '#ff6b9d' },
    ].filter(Boolean).map(ds => ({ ...ds, tension: 0.4, borderWidth: 1.5, pointRadius: 0, fill: false }));

    if (modalityDatasets.length > 0) {
      _line('historyModality', 'historyModalityChart', labels, modalityDatasets, { minY: 0, maxY: 1, maxX: 6, legend: true });
    }

    // Intervention timeline
    const interventions = d.interventions || [];
    Helpers.setHTML('interventionTimeline', interventions.length
      ? interventions.map(i => `
          <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-date">${i.date}</div>
            <div class="timeline-event">${i.event}</div>
          </div>`).join('')
      : `<div class="empty-state" style="padding:20px"><div class="empty-text">No interventions recorded yet</div></div>`
    );
  }

  function _line(key, canvasId, labels, datasets, opts = {}) {
    const ctx = document.getElementById(canvasId)?.getContext('2d'); if (!ctx) return;
    datasets.forEach(ds => {
      if (ds.fill && !ds.backgroundColor && ds.borderColor) {
        ds.backgroundColor = Charts.gradientFill(ctx, ds.borderColor.replace('#', ''), 240);
      }
    });
    Charts.create(key, ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: opts.legend ? { position: 'bottom', labels: { boxWidth: 10, padding: 12, color: '#8892b0' } } : { display: false },
          tooltip: Charts.TOOLTIP,
        },
        scales: {
          x: { ...Charts.SCALE.x, ticks: { maxTicksLimit: opts.maxX || 8 } },
          y: { ...Charts.SCALE.y, min: opts.minY, max: opts.maxY },
        },
      },
    });
  }

  return { init, load };
})();