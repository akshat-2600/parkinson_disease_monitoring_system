/* static/js/auth/patient_dashboard.js */
const PatientDashboard = (() => {

  function init() {
    const user = Auth.getUser();
    const uid  = Auth.getPatientUid();
    if (uid) App.setPatient(uid);
    Sidebar.render();
    Topbar.render();
    _patchTopbar(user);
    Router.navigate('dashboard');
    _overrideView(user, uid || App.patientId);
  }

  function _patchTopbar(user) {
    const right = document.querySelector('.topbar-right'); if (!right) return;
    const badge = document.createElement('span');
    badge.className = 'role-badge patient'; badge.textContent = '🧑 Patient'; right.prepend(badge);
    const profileBtn = document.createElement('button');
    profileBtn.className='topbar-btn'; profileBtn.setAttribute('data-tooltip','My Profile');
    profileBtn.innerHTML=`<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
    profileBtn.addEventListener('click',()=>_showProfile(user));
    right.insertBefore(profileBtn, right.lastElementChild);
    const logoutBtn = document.createElement('button');
    logoutBtn.className='topbar-btn'; logoutBtn.setAttribute('data-tooltip','Sign out');
    logoutBtn.innerHTML=`<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>`;
    logoutBtn.addEventListener('click',()=>Auth.logout());
    right.appendChild(logoutBtn);
    // Hide patient selector
    const sel = document.querySelector('.patient-select-wrap'); if(sel) sel.style.display='none';
    Helpers.setText('topbarPatient', Auth.getPatientUid()||'My Dashboard');
  }

  function _overrideView(user, uid) {
    const root = document.getElementById('page-dashboard'); if (!root) return;
    root.dataset.inited = '1';
    root.innerHTML = `
      <div class="patient-welcome">
        <div class="welcome-avatar">${_ini(user)}</div>
        <div class="welcome-text">
          <div class="greeting">Welcome back, ${user?.first_name||'Patient'} 👋</div>
          <div class="sub">Here's your personal health monitoring summary.</div>
          <div style="font-size:11px;font-family:var(--font-mono);color:var(--cyan);margin-top:6px">Patient ID: ${uid||'—'}</div>
        </div>
        <div class="welcome-actions">
          <button class="btn btn-primary" onclick="Router.navigate('realtime')">⚡ New Prediction</button>
          <button class="btn btn-ghost"   onclick="PatientDashboard.showReports()">📋 My Reports</button>
        </div>
      </div>
      <div style="background:var(--cyan-dim);border:1px solid rgba(0,212,255,0.2);border-radius:var(--radius-sm);padding:12px 16px;margin-bottom:20px;font-size:12px;color:var(--cyan);display:flex;align-items:center;gap:8px">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>
        You are viewing only your own health data. Your information is private and secure.
      </div>
      <div id="patientStatsError"></div>
      <div id="patientStatsLoading"><div class="grid-4 section-gap">${[1,2,3,4].map(()=>`<div class="skeleton skeleton-card" style="height:100px"></div>`).join('')}</div></div>
      <div id="patientStatsData" style="display:none">
        <div class="grid-4 section-gap">
          <div class="stat-card card-glow-cyan"><div class="stat-label">My Severity</div><div class="stat-value text-cyan" id="mySeverity">—</div><div class="stat-sub"><span id="myStage">—</span></div></div>
          <div class="stat-card card-glow-violet"><div class="stat-label">UPDRS Score</div><div class="stat-value text-violet" id="myUpdrs">—</div></div>
          <div class="stat-card card-glow-green"><div class="stat-label">AI Confidence</div><div class="stat-value text-green" id="myConfidence">—</div></div>
          <div class="stat-card"><div class="stat-label">My Predictions</div><div class="stat-value text-amber" id="myPredCount">—</div></div>
        </div>
        <div class="patient-card section-gap">
          <div class="patient-avatar">${_ini(user)}</div>
          <div class="patient-info">
            <div class="patient-name">${user?.first_name||''} ${user?.last_name||''}</div>
            <div class="patient-id">${uid||'—'}</div>
            <div class="patient-meta"><span class="meta-tag" id="myAge">Age: —</span><span class="meta-tag" id="myGender">—</span><span class="meta-tag" id="myDiagnosis">—</span><span class="meta-tag" id="myOnset">Onset: —</span></div>
          </div>
          <div class="patient-status"><span class="status-badge" id="myStatus">Loading…</span><span style="font-size:11px;font-family:var(--font-mono);color:var(--text-3)" id="myLastUpdated">—</span></div>
        </div>
        <div class="card card-glow-cyan section-gap"><div class="card-header"><span class="card-title">My Progression Trend</span></div><div class="chart-wrap" style="height:200px"><canvas id="myProgressionChart"></canvas></div></div>
        <div class="card section-gap"><div class="card-header"><span class="card-title">My Reports</span><div class="card-icon" style="background:var(--violet-dim)">📋</div></div><div id="myReportsList"><div class="loading-overlay"><div class="spinner"></div> Loading…</div></div></div>
        <div class="card"><div class="card-header"><span class="card-title">My Alerts</span></div><div class="alert-list" id="myAlertList"><div class="loading-overlay"><div class="spinner"></div></div></div></div>
      </div>
      <!-- Modals -->
      <div class="modal-backdrop" id="profileModal">
        <div class="modal"><div class="modal-header"><span class="modal-title">My Profile</span><button class="modal-close" onclick="document.getElementById('profileModal').classList.remove('open')">✕</button></div><div id="profileModalContent"></div></div>
      </div>
      <div class="modal-backdrop" id="reportsModal">
        <div class="modal" style="max-width:600px"><div class="modal-header"><span class="modal-title">My Reports</span><button class="modal-close" onclick="document.getElementById('reportsModal').classList.remove('open')">✕</button></div><div id="reportsModalContent" style="max-height:60vh;overflow-y:auto"></div></div>
      </div>`;

    _loadData(uid);
  }

  async function _loadData(uid) {
    if (!uid) return;
    try {
      const res = await API.getDashboard(uid, Auth.getToken());
      const d   = res.data;
      Helpers.hide('patientStatsLoading'); Helpers.show('patientStatsData');
      Helpers.setText('mySeverity',   d.severity??'—'); Helpers.setText('myStage',d.stage||'—');
      Helpers.setText('myUpdrs',      d.updrs??'—');
      Helpers.setText('myConfidence', d.fusion_confidence?`${Helpers.pct(d.fusion_confidence)}%`:'—');
      Helpers.setText('myAge',        `Age: ${d.age||'—'}`);
      Helpers.setText('myGender',     d.gender||'—');
      Helpers.setText('myDiagnosis',  d.diagnosis||'—');
      Helpers.setText('myOnset',      `Onset: ${d.onset||'—'}`);
      Helpers.setText('myLastUpdated',`Last updated: ${Helpers.timeNow()}`);
      const sb=document.getElementById('myStatus');
      if(sb){const m={stable:['🟢 Stable','status-stable'],warning:['🟡 Monitoring','status-warning'],critical:['🔴 Critical','status-critical']};const[l,c]=m[d.status]||m.stable;sb.textContent=l;sb.className=`status-badge ${c}`;}
      _renderAlerts(d.alerts||[]);
    } catch(e) {
      Helpers.showError('patientStatsError', `Failed to load: ${e.message}`);
      Helpers.hide('patientStatsLoading'); Helpers.show('patientStatsData');
    }

    // Prediction count
    try { const r=await API.getPatientPredictions(uid,Auth.getToken()); Helpers.setText('myPredCount',r.data?.total??'—'); } catch {}

    // History chart
    try {
      const r=await API.getHistory(uid,Auth.getToken());
      const h=r.data; const ctx=document.getElementById('myProgressionChart')?.getContext('2d');
      if(ctx&&h.labels?.length) Charts.create('myProgression',ctx,{type:'line',data:{labels:h.labels,datasets:[{label:'Severity',data:h.severity,borderColor:'#00d4ff',backgroundColor:Charts.gradientFill(ctx,'00d4ff',200),fill:true,tension:0.4,borderWidth:2,pointRadius:3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},tooltip:Charts.TOOLTIP},scales:{x:{...Charts.SCALE.x,ticks:{maxTicksLimit:6}},y:{...Charts.SCALE.y,min:0,max:100}}}});
    } catch {}

    // Reports
    try {
      const r=await API.getPatientReports(uid,Auth.getToken());
      const reports=r.data?.reports||[];
      Helpers.setHTML('myReportsList', reports.length
        ? reports.map(rp=>`<div class="report-card section-gap"><div class="report-title">${rp.title||'Report'}</div><div class="report-meta"><span>📅 ${new Date(rp.created_at).toLocaleDateString()}</span></div>${rp.content?.notes?`<div class="report-body">${rp.content.notes}</div>`:''}</div>`).join('')
        : `<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-text">No reports yet</div><div class="empty-sub">Your doctor will add reports here</div></div>`);
    } catch { Helpers.setHTML('myReportsList',`<div class="empty-state"><div class="empty-text">Could not load reports</div></div>`); }
  }

  function _renderAlerts(alerts) {
    Helpers.setHTML('myAlertList', alerts.length
      ? alerts.map(a=>`<div class="alert-item alert-${a.type||'info'}"><div class="alert-header"><span class="alert-type">${a.type||'info'}</span><span class="alert-time">${a.time||''}</span></div><div class="alert-msg">${a.msg||''}</div></div>`).join('')
      : `<div class="empty-state"><div class="empty-icon">✅</div><div class="empty-text">No active alerts</div></div>`);
  }

  function showReports() {
    const modal=document.getElementById('reportsModal'); if(!modal) return;
    modal.classList.add('open');
    const uid=Auth.getPatientUid()||App.patientId;
    Helpers.setHTML('reportsModalContent','<div class="loading-overlay"><div class="spinner"></div> Loading…</div>');
    API.getPatientReports(uid,Auth.getToken())
      .then(r=>{ const reports=r.data?.reports||[]; Helpers.setHTML('reportsModalContent', reports.length?reports.map(rp=>`<div class="report-card section-gap"><div class="report-title">${rp.title||'Report'}</div><div class="report-meta">📅 ${new Date(rp.created_at).toLocaleDateString()}</div>${rp.content?.notes?`<div class="report-body">${rp.content.notes}</div>`:''}</div>`).join(''):
        `<div class="empty-state"><div class="empty-icon">📋</div><div class="empty-text">No reports yet</div></div>`); })
      .catch(()=>Helpers.setHTML('reportsModalContent','<div class="empty-state"><div class="empty-text">Could not load reports</div></div>'));
  }

  function _showProfile(user) {
    const modal=document.getElementById('profileModal'); if(!modal) return;
    modal.classList.add('open');
    const uid=Auth.getPatientUid();
    Helpers.setHTML('profileModalContent',`
      <div style="display:flex;flex-direction:column;gap:16px">
        <div style="display:flex;align-items:center;gap:14px">
          <div style="width:56px;height:56px;border-radius:50%;background:linear-gradient(135deg,var(--green),var(--cyan));display:flex;align-items:center;justify-content:center;font-family:var(--font-display);font-size:20px;font-weight:800;color:white">${_ini(user)}</div>
          <div><div style="font-family:var(--font-display);font-size:18px;font-weight:700">${user?.first_name||''} ${user?.last_name||''}</div><div style="font-size:11px;font-family:var(--font-mono);color:var(--cyan)">${uid||'—'}</div></div>
        </div>
        ${_row('Email',user?.email||'—')}${_row('Role','Patient')}${_row('Patient ID',uid||'—')}
        <button class="auth-submit" onclick="Auth.logout()" style="margin-top:8px">Sign Out</button>
      </div>`);
  }

  function _row(label, value) {
    return `<div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius-sm)"><span style="font-size:11px;font-family:var(--font-mono);color:var(--text-3);text-transform:uppercase;letter-spacing:1px">${label}</span><span style="font-size:13px;font-weight:500">${value}</span></div>`;
  }

  function _ini(user) {
    if (!user) return '??';
    return ((user.first_name||'')[0]+(user.last_name||'')[0]).toUpperCase()||'??';
  }

  return { init, showReports };
})();
