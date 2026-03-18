/* ============================================================
   js/api/client.js — Flask backend API client
   Base: http://localhost:5000
   ============================================================ */

const API = (() => {
  const BASE = 'http://localhost:5000';

  /* ── Core fetch wrapper ── */
  async function request(url, options = {}) {
    const res = await fetch(url, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  }

  /* ── Endpoint methods ── */

  /** GET /fusion/dashboard/:patientId */
  async function getDashboard(patientId) {
    return request(`${BASE}/fusion/dashboard/${patientId}`);
  }

  /** GET /fusion/explanation/:patientId */
  async function getExplanation(patientId) {
    return request(`${BASE}/fusion/explanation/${patientId}`);
  }

  /** GET /fusion/recommendations/:patientId */
  async function getRecommendations(patientId) {
    return request(`${BASE}/fusion/recommendations/${patientId}`);
  }

  /** GET /fusion/history/:patientId */
  async function getHistory(patientId) {
    return request(`${BASE}/fusion/history/${patientId}`);
  }

  /**
   * POST /fusion/realtime_predict
   * @param {FormData} formData — multipart payload with uploaded files
   */
  async function realtimePredict(formData) {
    const res = await fetch(`${BASE}/fusion/realtime_predict`, {
      method: 'POST',
      body: formData,     // Do NOT set Content-Type; browser sets multipart boundary
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    return res.json();
  }

  return { getDashboard, getExplanation, getRecommendations, getHistory, realtimePredict };
})();
