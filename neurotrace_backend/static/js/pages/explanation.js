/* static/js/pages/explanation.js */
const PageExplanation = (() => {
  const ROOT = 'page-explanation';

  function _html() {
    return `
      <div class="page-header">
        <div><div class="page-title">Explainability</div><div class="page-subtitle">AI decision transparency — feature importance & attention maps</div></div>
        <button class="btn btn-ghost" id="explRefreshBtn"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg> Refresh</button>
      </div>
      <div id="explanationError"></div>
      <div id="explanationLoading">${Skeletons.explanationSkeleton()}</div>
      <div id="explanationData" style="display:none">
        <div class="grid-2 section-gap">
          <div class="card card-glow-cyan"><div class="card-header"><span class="card-title">Feature Importance</span><div class="card-icon" style="background:var(--cyan-dim)">📈</div></div><div class="feature-bars" id="featureBars"></div></div>
          <div class="card card-glow-violet"><div class="card-header"><span class="card-title">MRI Attention Heatmap</span><div class="card-icon" style="background:var(--violet-dim)">🧠</div></div><div class="heatmap-wrap" id="mriHeatmap"><div class="heatmap-placeholder"><div style="font-size:32px;margin-bottom:8px">🧠</div><div>MRI heatmap from /api/fusion/explanation</div></div></div></div>
        </div>
        <div class="grid-2 section-gap">
          <div class="card"><div class="card-header"><span class="card-title">Spiral Drawing Heatmap</span><div class="card-icon" style="background:var(--green-dim)">🌀</div></div><div class="heatmap-wrap" id="spiralHeatmap"><div class="heatmap-placeholder"><div style="font-size:32px;margin-bottom:8px">🌀</div><div>Spiral analysis heatmap</div></div></div></div>
          <div class="card card-glow-green"><div class="card-header"><span class="card-title">Attention Weights</span><div class="card-icon" style="background:var(--green-dim)">🎯</div></div><div class="chart-wrap" style="height:200px"><canvas id="attentionChart"></canvas></div></div>
        </div>
        <div class="card card-glow-cyan"><div class="card-header"><span class="card-title">AI Explanation Summary</span><div class="card-icon" style="background:var(--cyan-dim)">💡</div></div><div class="explanation-summary" id="explanationSummary"></div></div>
      </div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('explRefreshBtn')?.addEventListener('click', load);
  }

  async function load() {
    Helpers.clearError('explanationError');
    Helpers.show('explanationLoading'); Helpers.hide('explanationData');
    let data;
    try {
      const res = await API.getExplanation(App.patientId, Auth.getToken());
      data = res.data;
    } catch (e) {
      Helpers.showError('explanationError', `API error: ${e.message}`);
      data = Fallback.getExplanation();
    }
    _render(data);
  }

  function _render(d) {
    Helpers.hide('explanationLoading'); Helpers.show('explanationData');
    Helpers.setHTML('featureBars', (d.features||[]).map((f,i) => {
      const pct = Helpers.pct(f.importance ?? f.value ?? 0);
      const color = Helpers.PALETTE[i % Helpers.PALETTE.length];
      return `<div class="feature-row"><div class="feature-name">${f.name||f.feature||'—'}</div><div class="feature-bar-wrap"><div class="feature-bar" style="width:${pct}%;background:${color}"></div></div><div class="feature-pct">${pct}%</div></div>`;
    }).join(''));
    _heatmap('mriHeatmap',    d.mri_heatmap_url,    d.mri_heatmap_base64,    '🧠', 'MRI heatmap');
    _heatmap('spiralHeatmap', d.spiral_heatmap_url, d.spiral_heatmap_base64, '🌀', 'Spiral heatmap');
    const labels = ['Voice','Clinical','Time-Series','MRI','Spiral','Motor','Gait','Tremor'];
    const weights = (d.attention||[]).length ? d.attention : labels.map(() => +(Math.random()*0.6+0.3).toFixed(2));
    Charts.radar('attention','attentionChart', labels.slice(0,weights.length), weights, '#00e5a0');
    Helpers.setHTML('explanationSummary', d.summary || d.explanation_text || 'No explanation available.');
  }

  function _heatmap(id, url, b64, emoji, placeholder) {
    const el = document.getElementById(id); if (!el) return;
    if (url)       el.innerHTML = `<img src="${url}" alt="Heatmap"/>`;
    else if (b64)  el.innerHTML = `<img src="data:image/png;base64,${b64}" alt="Heatmap"/>`;
    else           el.innerHTML = `<div class="heatmap-placeholder"><div style="font-size:32px;margin-bottom:8px">${emoji}</div><div>${placeholder}</div></div>`;
  }

  return { init, load };
})();
