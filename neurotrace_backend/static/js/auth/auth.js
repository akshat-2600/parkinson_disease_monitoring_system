/* ============================================================
   static/js/auth/auth.js
   JWT session management — uses API module (no hardcoded URLs).
   ============================================================ */

const Auth = (() => {
  const TOKEN_KEY   = 'nt_access_token';
  const REFRESH_KEY = 'nt_refresh_token';
  const USER_KEY    = 'nt_user';

  /* ── Token storage ─────────────────────────────────────── */
  function saveSession(data) {
    localStorage.setItem(TOKEN_KEY,   data.access_token);
    localStorage.setItem(REFRESH_KEY, data.refresh_token);
    localStorage.setItem(USER_KEY,    JSON.stringify(data.user));
    if (data.patient_uid) localStorage.setItem('nt_patient_uid', data.patient_uid);
    if (data.doctor_id)   localStorage.setItem('nt_doctor_id',   data.doctor_id);
  }

  function clearSession() {
    ['nt_access_token', 'nt_refresh_token', 'nt_user', 'nt_patient_uid', 'nt_doctor_id']
      .forEach(k => localStorage.removeItem(k));
  }

  const getToken      = () => localStorage.getItem(TOKEN_KEY);
  const getRefresh    = () => localStorage.getItem(REFRESH_KEY);
  const getUser       = () => { try { return JSON.parse(localStorage.getItem(USER_KEY)); } catch { return null; } };
  const getRole       = () => (getUser() || {}).role || null;
  const getPatientUid = () => localStorage.getItem('nt_patient_uid');
  const isLoggedIn    = () => !!(getToken() && getUser());

  /* ── Auth-aware fetch (auto-refresh on 401) ──────────────  */
  async function authFetch(url, options = {}) {
    const headers = { ...(options.headers || {}), 'Authorization': `Bearer ${getToken()}` };
    if (!(options.body instanceof FormData)) headers['Content-Type'] = 'application/json';

    let res = await fetch(url, { ...options, headers });

    if (res.status === 401) {
      const ok = await _tryRefresh();
      if (ok) {
        headers['Authorization'] = `Bearer ${getToken()}`;
        res = await fetch(url, { ...options, headers });
      } else {
        clearSession();
        AuthUI.showLogin();
        throw new Error('Session expired. Please log in again.');
      }
    }
    return res;
  }

  async function _tryRefresh() {
    const rt = getRefresh();
    if (!rt) return false;
    try {
      const data = await API.refreshToken(rt);
      if (!data.data?.access_token) return false;
      localStorage.setItem(TOKEN_KEY, data.data.access_token);
      return true;
    } catch { return false; }
  }

  /* ── API calls (delegate to API module) ──────────────────  */
  async function login(email, password) {
    const data = await API.login(email, password);
    if (data.status !== 'success') throw new Error(data.message || 'Login failed');
    saveSession(data.data);
    return data.data;
  }

  async function signup(payload) {
    const data = await API.signup(payload);
    if (data.status !== 'success') throw new Error(data.message || 'Signup failed');
    saveSession(data.data);
    return data.data;
  }

  async function logout() {
    const tok = getToken();
    clearSession();
    try { if (tok) await API.logout(tok); } catch { /* ignore */ }
    AuthUI.showLogin();
  }

  return {
    login, signup, logout, authFetch,
    saveSession, clearSession,
    getToken, getRefresh, getUser, getRole, getPatientUid, isLoggedIn,
    API_ROOT: () => API.API_ROOT,   // expose for other modules
  };
})();