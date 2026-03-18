/* ============================================================
   js/pages/recommendations.js — Recommendations page
   ============================================================ */

const PageRecommendations = (() => {

  const ROOT = 'page-recommendations';
  let _allRecs = [];

  function _html() {
    return `
      <div class="page-header">
        <div>
          <div class="page-title">Recommendations</div>
          <div class="page-subtitle">Personalised AI-driven clinical action plan</div>
        </div>
        <button class="btn btn-ghost" id="recRefreshBtn">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/>
          </svg>
          Refresh
        </button>
      </div>

      <div id="recommendationsError"></div>

      <div class="section-tabs" id="recTabs">
        <button class="tab-btn active" data-filter="all">All</button>
        <button class="tab-btn" data-filter="high">🔴 High Priority</button>
        <button class="tab-btn" data-filter="moderate">🟡 Moderate</button>
        <button class="tab-btn" data-filter="preventive">🟢 Preventive</button>
      </div>

      <div id="recLoading">${Skeletons.recommendationsSkeleton()}</div>
      <div class="rec-cards" id="recCards" style="display:none"></div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('recRefreshBtn')?.addEventListener('click', load);
    document.getElementById('recTabs')?.addEventListener('click', e => {
      const btn = e.target.closest('.tab-btn');
      if (!btn) return;
      document.querySelectorAll('#recTabs .tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      _renderFiltered(btn.dataset.filter);
    });
  }

  async function load() {
    Helpers.clearError('recommendationsError');
    Helpers.show('recLoading');
    Helpers.hide('recCards');

    try {
      const raw  = await API.getRecommendations(App.patientId);
      _allRecs   = Array.isArray(raw) ? raw : (raw.recommendations || []);
    } catch (e) {
      Helpers.showError('recommendationsError', `API unreachable: ${e.message}`);
      _allRecs = Fallback.getRecommendations();
    }

    Helpers.hide('recLoading');
    Helpers.show('recCards');

    // Reset filter tab to "all"
    document.querySelectorAll('#recTabs .tab-btn').forEach(b => b.classList.remove('active'));
    const allBtn = document.querySelector('#recTabs .tab-btn[data-filter="all"]');
    if (allBtn) allBtn.classList.add('active');

    _renderFiltered('all');
  }

  function _renderFiltered(filter) {
    const list = filter === 'all'
      ? _allRecs
      : _allRecs.filter(r => r.priority === filter);
    _renderCards(list);
  }

  function _renderCards(recs) {
    const container = document.getElementById('recCards');
    if (!container) return;

    if (!recs.length) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">✅</div>
          <div class="empty-text">No recommendations in this category</div>
        </div>`;
      return;
    }

    container.innerHTML = recs.map(r => {
      const pct = Helpers.pct(r.confidence ?? 0);
      return `
        <div class="rec-card">
          <div class="rec-card-header">
            <div class="rec-title">${r.title}</div>
            <span class="rec-priority priority-${r.priority}">
              ${{ high: '🔴', moderate: '🟡', preventive: '🟢' }[r.priority] || '⚪'} ${r.priority}
            </span>
          </div>
          <div class="rec-reasoning">${r.reasoning}</div>
          <div class="rec-footer">
            <div class="rec-confidence">
              Confidence: ${pct}%
              <span class="confidence-bar-wrap">
                <span class="confidence-bar" style="width:${pct}%"></span>
              </span>
            </div>
            <span class="rec-category-tag">${r.category || 'General'}</span>
          </div>
        </div>`;
    }).join('');
  }

  return { init, load };
})();
