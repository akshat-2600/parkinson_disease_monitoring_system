/* ============================================================
   js/app.js — Application bootstrap, Router, App state
   ============================================================ */

/* ── App State ── */
const App = (() => {
  let _patientId   = 'PT-001';
  let _currentPage = 'dashboard';

  function setPatient(pid) {
    _patientId = pid;
    Topbar.setPatient(pid);
    Router.refresh();
  }

  return {
    get patientId()   { return _patientId; },
    get currentPage() { return _currentPage; },
    set currentPage(v){ _currentPage = v; },
    setPatient,
  };
})();

/* ── Router ── */
const Router = (() => {

  const PAGES = {
    dashboard:       PageDashboard,
    explanation:     PageExplanation,
    recommendations: PageRecommendations,
    history:         PageHistory,
    realtime:        PageRealtime,
  };

  let _initialised = false;

  function navigate(page) {
    if (!PAGES[page]) return;

    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    // Show target
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.add('active');

    // Init page HTML once
    if (!target.dataset.inited) {
      PAGES[page].init();
      target.dataset.inited = '1';
    }

    // Update UI chrome
    Sidebar.setActive(page);
    Topbar.setPage(page);
    App.currentPage = page;

    // Load data
    PAGES[page].load();
  }

  function refresh() {
    // Re-navigate current page so data reloads
    const page = App.currentPage;
    if (PAGES[page]) PAGES[page].load();
  }

  return { navigate, refresh };
})();

/* ── Bootstrap ── */
document.addEventListener('DOMContentLoaded', () => {
  // Apply Chart.js global defaults
  Charts.applyDefaults();

  // Render shell components
  Sidebar.render();
  Topbar.render();

  // Navigate to default page
  Router.navigate('dashboard');

  // Auto-refresh dashboard every 5 minutes
  setInterval(() => {
    if (App.currentPage === 'dashboard') PageDashboard.load();
  }, 5 * 60 * 1000);
});
