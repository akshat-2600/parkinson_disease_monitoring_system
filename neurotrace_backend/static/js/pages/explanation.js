/* ============================================================
   static/js/pages/explanation.js
   ============================================================ */
const PageExplanation = (() => {
  const ROOT = 'page-explanation';

  function _html() {
    return `
      <div class="page-header">
        <div>
          <div class="page-title">Explainability</div>
          <div class="page-subtitle">AI decision transparency — feature importance & attention maps</div>
        </div>
        <button class="btn btn-ghost" id="explRefreshBtn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>
          Refresh
        </button>
      </div>
      <div id="explanationError"></div>
      <div id="explanationLoading" style="display:none">${Skeletons.explanationSkeleton()}</div>

      <!-- Empty state -->
      <div id="explanationEmpty">
        <div class="card" style="text-align:center;padding:60px 24px">
          <div style="font-size:56px;margin-bottom:16px;opacity:0.4">🔍</div>
          <div style="font-family:var(--font-display);font-size:18px;font-weight:700;margin-bottom:8px">No explanation data yet</div>
          <div style="font-size:13px;color:var(--text-2);max-width:380px;margin:auto;margin-bottom:20px">
            Run a prediction first — explainability analysis will appear here after the AI processes patient data.
          </div>
          <button class="btn btn-primary" onclick="Router.navigate('realtime')" style="margin:auto">
            ⚡ Run Prediction First
          </button>
        </div>
      </div>

      <!-- Data -->
      <div id="explanationData" style="display:none">
        <div class="grid-2 section-gap">
          <div class="card card-glow-cyan">
            <div class="card-header">
              <span class="card-title">Feature Importance</span>
              <div class="card-icon" style="background:var(--cyan-dim)">📈</div>
            </div>
            <div class="feature-bars" id="featureBars"></div>
          </div>
          <div class="card card-glow-violet">
            <div class="card-header">
              <span class="card-title">MRI Attention Heatmap</span>
              <div class="card-icon" style="background:var(--violet-dim)">🧠</div>
            </div>
            <div class="heatmap-wrap" id="mriHeatmap">
              <div class="heatmap-placeholder">
                <div style="font-size:32px;margin-bottom:8px">🧠</div>
                <div>MRI heatmap available after MRI prediction</div>
              </div>
            </div>
          </div>
        </div>
        <div class="grid-2 section-gap">
          <div class="card">
            <div class="card-header">
              <span class="card-title">Spiral Drawing Heatmap</span>
              <div class="card-icon" style="background:var(--green-dim)">🌀</div>
            </div>
            <div class="heatmap-wrap" id="spiralHeatmap">
              <div class="heatmap-placeholder">
                <div style="font-size:32px;margin-bottom:8px">🌀</div>
                <div>Spiral heatmap available after spiral prediction</div>
              </div>
            </div>
          </div>
          <div class="card card-glow-green">
            <div class="card-header">
              <span class="card-title">Attention Weights</span>
              <div class="card-icon" style="background:var(--green-dim)">🎯</div>
            </div>
            <div class="chart-wrap" style="height:200px"><canvas id="attentionChart"></canvas></div>
          </div>
        </div>
        <div class="card card-glow-cyan">
          <div class="card-header">
            <span class="card-title">AI Explanation Summary</span>
            <div class="card-icon" style="background:var(--cyan-dim)">💡</div>
          </div>
          <div class="explanation-summary" id="explanationSummary"></div>
        </div>
      </div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('explRefreshBtn')?.addEventListener('click', load);
  }

  async function load() {
    Helpers.clearError('explanationError');
    Helpers.hide('explanationData');
    Helpers.hide('explanationLoading');

    let data = null;
    try {
      Helpers.show('explanationLoading');
      Helpers.hide('explanationEmpty');
      const res = await API.getExplanation(App.patientId);
      data = res?.data || null;
    } catch (e) {
      Helpers.hide('explanationLoading');
      if (!e.message.includes('401') && !e.message.includes('404')) {
        Helpers.showError('explanationError', `Error: ${e.message}`);
      }
    }
    Helpers.hide('explanationLoading');

    // Only render if we have real feature importance data
    const hasData = data && data.features && data.features.length > 0;
    if (hasData) {
      _render(data);
    } else {
      Helpers.show('explanationEmpty');
    }
  }

  /* Call this directly after a realtime prediction with explanation data */
  function renderDirect(explData) {
    if (!explData) return;
    _render(explData);
  }

  function _render(d) {
    Helpers.hide('explanationEmpty');
    Helpers.show('explanationData');

    // Feature bars
    Helpers.setHTML('featureBars', (d.features || []).map((f, i) => {
      const pct   = Helpers.pct(f.importance ?? f.value ?? 0);
      const color = Helpers.PALETTE[i % Helpers.PALETTE.length];
      return `
        <div class="feature-row">
          <div class="feature-name">${f.name || f.feature || '—'}</div>
          <div class="feature-bar-wrap">
            <div class="feature-bar" style="width:${pct}%;background:${color}"></div>
          </div>
          <div class="feature-pct">${pct}%</div>
        </div>`;
    }).join(''));

    // Heatmaps
    _setHeatmap('mriHeatmap',    d.mri_heatmap_url,    d.mri_heatmap_base64,    '🧠', 'MRI heatmap not available');
    _setHeatmap('spiralHeatmap', d.spiral_heatmap_url, d.spiral_heatmap_base64, '🌀', 'Spiral heatmap not available');

    // Attention radar
    if (d.attention && d.attention.length > 0) {
      const modLabels = ['Voice','Clinical','Time-Series','MRI','Spiral','Motor','Gait','Tremor'];
      Charts.radar('attention', 'attentionChart', modLabels.slice(0, d.attention.length), d.attention, '#00e5a0');
    }

    Helpers.setHTML('explanationSummary', d.summary || d.explanation_text || 'Explanation generated from model analysis.');
  }

  function _setHeatmap(id, url, b64, emoji, placeholder) {
    const el = document.getElementById(id); if (!el) return;
    if (url)      el.innerHTML = `<img src="${url}" alt="Heatmap" style="max-width:100%;border-radius:var(--radius-sm)"/>`;
    else if (b64) el.innerHTML = `<img src="data:image/png;base64,${b64}" alt="Heatmap" style="max-width:100%;border-radius:var(--radius-sm)"/>`;
    else          el.innerHTML = `<div class="heatmap-placeholder"><div style="font-size:32px;margin-bottom:8px">${emoji}</div><div>${placeholder}</div></div>`;
  }

  return { init, load, renderDirect };
})();