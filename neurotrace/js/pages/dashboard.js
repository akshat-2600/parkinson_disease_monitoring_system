/* ============================================================
   js/pages/dashboard.js — Dashboard page render logic
   ============================================================ */

const PageDashboard = (() => {

  const ROOT = 'page-dashboard';

  /* ── Template ── */
  function _html() {
    return `
      <div class="page-header">
        <div>
          <div class="page-title">Patient Overview</div>
          <div class="page-subtitle">Real-time Parkinson's monitoring — Fusion AI engine active</div>
        </div>
        <div class="page-actions">
          <button class="btn btn-ghost" id="dashRefreshBtn">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>
            Refresh
          </button>
          <button class="btn btn-primary" id="dashPredictBtn">⚡ Run Prediction</button>
        </div>
      </div>

      <div id="dashboardError"></div>
      <div id="dashboardLoading">${Skeletons.dashboardSkeleton()}</div>

      <div id="dashboardData" style="display:none">

        <!-- Patient summary -->
        <div class="patient-card section-gap">
          <div class="patient-avatar" id="patientAvatar">—</div>
          <div class="patient-info">
            <div class="patient-name" id="patientName">—</div>
            <div class="patient-id"   id="patientIdLabel">—</div>
            <div class="patient-meta">
              <span class="meta-tag" id="patientAge">Age: —</span>
              <span class="meta-tag" id="patientGender">Gender: —</span>
              <span class="meta-tag" id="patientDiagnosis">—</span>
              <span class="meta-tag" id="patientOnset">Onset: —</span>
            </div>
          </div>
          <div class="patient-status">
            <span class="status-badge" id="patientStatusBadge">—</span>
            <span style="font-size:11px;font-family:var(--font-mono);color:var(--text-3)" id="lastUpdated">—</span>
          </div>
        </div>

        <!-- Stat cards -->
        <div class="grid-4 section-gap">
          <div class="stat-card card-glow-cyan">
            <div class="stat-label" data-tooltip="Overall Parkinson's severity (0–100)">Severity Score</div>
            <div class="stat-value text-cyan" id="severityScore">—</div>
            <div class="stat-sub">
              <span id="severityStage">Stage: —</span>
              <span class="stat-change neutral" id="severityChange">—</span>
            </div>
          </div>
          <div class="stat-card card-glow-violet">
            <div class="stat-label" data-tooltip="Unified Parkinson's Disease Rating Scale">UPDRS Score</div>
            <div class="stat-value text-violet" id="updrsScore">—</div>
            <div class="stat-sub">
              <span>Total motor</span>
              <span class="stat-change neutral" id="updrsChange">—</span>
            </div>
          </div>
          <div class="stat-card card-glow-green">
            <div class="stat-label" data-tooltip="Fusion model confidence level">Fusion Confidence</div>
            <div class="stat-value text-green" id="fusionConf">—</div>
            <div class="stat-sub"><span>Model consensus</span></div>
          </div>
          <div class="stat-card">
            <div class="stat-label" data-tooltip="Active clinical risk flags">Active Alerts</div>
            <div class="stat-value text-amber" id="alertCount">—</div>
            <div class="stat-sub"><span>Open flags</span></div>
          </div>
        </div>

        <!-- Gauge + Modality -->
        <div class="grid-2 section-gap">
          <div class="card card-glow-cyan">
            <div class="card-header">
              <span class="card-title">Severity Gauge</span>
              <div class="card-icon" style="background:var(--cyan-dim)">📊</div>
            </div>
            <div id="gaugeContainer">${Gauge.template()}</div>
          </div>
          <div class="card card-glow-violet">
            <div class="card-header">
              <span class="card-title">Model Contributions</span>
              <div class="card-icon" style="background:var(--violet-dim)">🔬</div>
            </div>
            <div class="modality-grid" id="modalityGrid"></div>
          </div>
        </div>

        <!-- Progression chart -->
        <div class="card card-glow-cyan section-gap">
          <div class="card-header">
            <span class="card-title">Progression Trend</span>
            <span class="stat-change neutral" style="font-size:11px">30 days</span>
          </div>
          <div class="chart-wrap" style="height:220px">
            <canvas id="progressionChart"></canvas>
          </div>
        </div>

        <!-- Risk + Alerts -->
        <div class="grid-2">
          <div class="card">
            <div class="card-header">
              <span class="card-title">Risk Indicators</span>
              <div class="card-icon" style="background:var(--red-dim)">⚠️</div>
            </div>
            <div class="risk-list" id="riskList"></div>
          </div>
          <div class="card">
            <div class="card-header">
              <span class="card-title">Active Alerts</span>
              <div class="card-icon" style="background:var(--amber-dim)">🔔</div>
            </div>
            <div class="alert-list" id="alertList"></div>
          </div>
        </div>

      </div>`;
  }

  /* ── Init ── */
  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('dashRefreshBtn')?.addEventListener('click', load);
    document.getElementById('dashPredictBtn')?.addEventListener('click', () => Router.navigate('realtime'));
  }

  /* ── Load data ── */
  async function load() {
    Helpers.clearError('dashboardError');
    Helpers.show('dashboardLoading');
    Helpers.hide('dashboardData');

    let data;
    try {
      data = await API.getDashboard(App.patientId);
    } catch (e) {
      Helpers.showError('dashboardError', `API unreachable: ${e.message}`);
      data = Fallback.getDashboard(App.patientId);
    }

    _render(data);
  }

  /* ── Render ── */
  function _render(d) {
    Helpers.hide('dashboardLoading');
    Helpers.show('dashboardData');

    // Patient card
    Helpers.setText('patientAvatar',   d.initials || Helpers.initials(d.name || ''));
    Helpers.setText('patientName',     d.name || '—');
    Helpers.setText('patientIdLabel',  `${App.patientId} · Neurological Unit A`);
    Helpers.setText('patientAge',      `Age: ${d.age || '—'}`);
    Helpers.setText('patientGender',   `Gender: ${d.gender || '—'}`);
    Helpers.setText('patientDiagnosis', d.diagnosis || '—');
    Helpers.setText('patientOnset',    `Onset: ${d.onset || '—'}`);
    Helpers.setText('lastUpdated',     `Last updated: ${Helpers.timeNow()}`);

    // Status badge
    const sb = document.getElementById('patientStatusBadge');
    if (sb) {
      const map = { stable: ['🟢 Stable', 'status-stable'], warning: ['🟡 Monitoring', 'status-warning'], critical: ['🔴 Critical', 'status-critical'] };
      const [label, cls] = map[d.status] || map.stable;
      sb.textContent = label;
      sb.className   = `status-badge ${cls}`;
    }

    // Stats
    Helpers.setText('severityScore',  d.severity ?? '—');
    Helpers.setText('severityStage',  d.stage || '—');
    Helpers.setText('severityChange', `+${d.severity_change ?? '—'}% MoM`);
    Helpers.setText('updrsScore',     d.updrs ?? '—');
    Helpers.setText('updrsChange',    d.updrs_change ?? '—');
    Helpers.setText('fusionConf',     d.fusion_confidence ? `${Helpers.pct(d.fusion_confidence)}%` : '—');

    const alertCount = (d.alerts || []).length;
    Helpers.setText('alertCount', alertCount);
    Sidebar.setAlertBadge(alertCount);
    Topbar.setNotification(alertCount > 0);

    // Gauge
    Gauge.update(d.severity || 0, d.stage || '—');

    // Modality grid
    _renderModalities(d.modalities || {});

    // Progression chart
    _renderProgressionChart(d.severity || 60);

    // Risk list
    _renderRisks(d.risks || []);

    // Alerts
    _renderAlerts(d.alerts || []);
  }

  function _renderModalities(modalities) {
    const icons  = { voice:'🎤', clinical:'📋', timeseries:'📈', mri:'🧠', spiral:'🌀', motor:'🤚' };
    const labels = { voice:'Voice', clinical:'Clinical', timeseries:'Time-Series', mri:'MRI', spiral:'Spiral', motor:'Motor' };
    const grid   = document.getElementById('modalityGrid');
    if (!grid) return;

    grid.innerHTML = Object.entries(modalities).slice(0, 6).map(([key, val], i) => {
      const pct   = Helpers.pct(val);
      const color = Helpers.PALETTE[i % Helpers.PALETTE.length];
      return `
        <div class="modality-item" data-tooltip="${labels[key] || key} confidence">
          <div class="modality-icon">${icons[key] || '🔬'}</div>
          <div class="modality-name">${labels[key] || key}</div>
          <div class="modality-score" style="color:${color}">${pct}%</div>
          <div class="modality-bar-wrap">
            <div class="modality-bar" style="width:${pct}%;background:${color}"></div>
          </div>
        </div>`;
    }).join('');
  }

  function _renderProgressionChart(baseVal) {
    const labels = Array.from({ length: 30 }, (_, i) => {
      const d = new Date(); d.setDate(d.getDate() - (29 - i));
      return `${d.getDate()}/${d.getMonth() + 1}`;
    });
    const vals = labels.map((_, i) =>
      Math.min(100, baseVal - 8 + i * 0.3 + Math.sin(i / 3) * 3 + (Math.random() - 0.5) * 3)
    );

    const ctx = document.getElementById('progressionChart')?.getContext('2d');
    if (!ctx) return;

    Charts.create('progression', ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Severity Score',
          data: vals,
          borderColor: '#00d4ff',
          backgroundColor: Charts.gradientFill(ctx, '00d4ff', 220),
          fill: true, tension: 0.4, pointRadius: 0, pointHoverRadius: 5, borderWidth: 2,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false }, tooltip: Charts.TOOLTIP },
        scales: {
          x: { ...Charts.SCALE.x, ticks: { maxTicksLimit: 8 } },
          y: { ...Charts.SCALE.y, min: Math.max(0, baseVal - 20), max: Math.min(100, baseVal + 20) },
        },
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
      },
    });
  }

  function _renderRisks(risks) {
    const colorMap = { high: 'var(--red)', medium: 'var(--amber)', low: 'var(--green)', moderate: 'var(--amber)' };
    Helpers.setHTML('riskList', risks.length
      ? risks.map(r => `
          <div class="risk-item">
            <div class="risk-dot" style="background:${colorMap[r.level] || 'var(--cyan)'}"></div>
            <div class="risk-name">${r.name}</div>
            <div class="risk-level risk-${r.level || 'low'}">${(r.level || 'low').toUpperCase()}</div>
          </div>`).join('')
      : `<div class="empty-state"><div class="empty-icon">✅</div><div class="empty-text">No active risks</div></div>`
    );
  }

  function _renderAlerts(alerts) {
    Helpers.setHTML('alertList', alerts.length
      ? alerts.map(a => `
          <div class="alert-item alert-${a.type || 'info'}">
            <div class="alert-header">
              <span class="alert-type">${a.type || 'info'}</span>
              <span class="alert-time">${a.time || ''}</span>
            </div>
            <div class="alert-msg">${a.msg || a.message || ''}</div>
          </div>`).join('')
      : `<div class="empty-state"><div class="empty-icon">🔔</div><div class="empty-text">No active alerts</div></div>`
    );
  }

  return { init, load };
})();
