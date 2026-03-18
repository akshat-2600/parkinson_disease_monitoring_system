/* ============================================================
   static/js/api/client.js
   API client — uses same-origin relative /api/* paths by default.
   Flask injects the actual base via window.NEUROTRACE_CONFIG.apiBase.
   ============================================================ */

const API = (() => {
  // Flask template injects: window.NEUROTRACE_CONFIG = { apiBase: "" }
  // Empty string = same-origin, so /api/auth/login works from any host/port.
  const BASE = (window.NEUROTRACE_CONFIG && window.NEUROTRACE_CONFIG.apiBase)
    ? window.NEUROTRACE_CONFIG.apiBase.replace(/\/$/, '')
    : '';
  const API_ROOT = BASE + '/api';

  async function _get(path, token) {
    const res = await fetch(API_ROOT + path, {
      headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  }

  async function _post(path, body, token) {
    const isForm = body instanceof FormData;
    const res = await fetch(API_ROOT + path, {
      method: 'POST',
      headers: {
        'Authorization': token ? `Bearer ${token}` : undefined,
        ...(isForm ? {} : { 'Content-Type': 'application/json' }),
      },
      body: isForm ? body : JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  }

  // ── Auth ──────────────────────────────────────────────────
  const login   = (email, password) => _post('/auth/login',  { email, password });
  const signup  = (payload)         => _post('/auth/signup', payload);
  const logout  = (token)           => fetch(API_ROOT + '/auth/logout', { method: 'DELETE', headers: { Authorization: `Bearer ${token}` } });
  const refreshToken = (rt)         => fetch(API_ROOT + '/auth/refresh', { method: 'POST', headers: { Authorization: `Bearer ${rt}` } }).then(r => r.json());

  // ── Fusion dashboard endpoints ────────────────────────────
  const getDashboard       = (pid, tok) => _get(`/fusion/dashboard/${pid}`, tok);
  const getExplanation     = (pid, tok) => _get(`/fusion/explanation/${pid}`, tok);
  const getRecommendations = (pid, tok) => _get(`/fusion/recommendations/${pid}`, tok);
  const getHistory         = (pid, tok) => _get(`/fusion/history/${pid}`, tok);
  const realtimePredict    = (fd,  tok) => _post('/fusion/realtime_predict', fd, tok);

  // ── Patient management ────────────────────────────────────
  const listPatients          = (tok, page=1, pp=200) => _get(`/patients/?page=${page}&per_page=${pp}`, tok);
  const getPatient            = (pid, tok)  => _get(`/patients/${pid}`, tok);
  const getPatientPredictions = (pid, tok)  => _get(`/patients/${pid}/predictions`, tok);
  const getPatientReports     = (pid, tok)  => _get(`/patients/${pid}/reports`, tok);

  return {
    API_ROOT,
    login, signup, logout, refreshToken,
    getDashboard, getExplanation, getRecommendations, getHistory, realtimePredict,
    listPatients, getPatient, getPatientPredictions, getPatientReports,
  };
})();