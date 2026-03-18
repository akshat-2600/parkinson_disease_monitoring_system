/* ============================================================
   static/js/app.js — Application bootstrap, Router, App state
   ============================================================ */

/* ── App State ── */
const App = (() => {
  let _patientId   = 'PT-001';
  let _currentPage = 'dashboard';

  function setPatient(pid) {
    _patientId = pid;
    Topbar.setPatient(pid);
    const sel = document.getElementById('patientSelector');
    if (sel && sel.value !== pid) {
      let opt = sel.querySelector(`option[value="${pid}"]`);
      if (!opt) { opt = new Option(pid, pid); sel.add(opt); }
      sel.value = pid;
    }
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

  function navigate(page) {
    if (!PAGES[page]) return;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const target = document.getElementById(`page-${page}`);
    if (target) target.classList.add('active');
    if (!target.dataset.inited) { PAGES[page].init(); target.dataset.inited = '1'; }
    Sidebar.setActive(page);
    Topbar.setPage(page);
    App.currentPage = page;
    PAGES[page].load();
  }

  function refresh() {
    const page = App.currentPage;
    if (PAGES[page]) PAGES[page].load();
  }

  return { navigate, refresh };
})();

/* ── Bootstrap ── */
document.addEventListener('DOMContentLoaded', () => {
  Charts.applyDefaults();

  function showMainShell() {
    const main    = document.getElementById('main-shell');
    const sidebar = document.getElementById('sidebar-root');
    if (main)    main.style.display    = '';
    if (sidebar) sidebar.style.display = '';
  }

  // Patch AuthUI.hideAuth to also reveal the main shell
  const _origHide = AuthUI.hideAuth.bind(AuthUI);
  AuthUI.hideAuth = function () {
    _origHide();
    showMainShell();
  };

  if (Auth.isLoggedIn()) {
    showMainShell();
    Sidebar.render();
    Topbar.render();
    if (Auth.getRole() === 'doctor') {
      DoctorDashboard.init();
    } else {
      PatientDashboard.init();
    }
  } else {
    AuthUI.showLogin();
  }

  // Auto-refresh dashboard every 5 minutes
  setInterval(() => {
    if (Auth.isLoggedIn() && App.currentPage === 'dashboard') {
      PageDashboard.load();
    }
  }, 5 * 60 * 1000);
});