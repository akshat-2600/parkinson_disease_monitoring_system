/* ============================================================
   static/js/pages/realtime.js
   ─────────────────────────────────────────────────────────
   ============================================================ */

const PageRealtime = (() => {
  const ROOT = 'page-realtime';

  /* Map each file slot to its modality endpoint */
  const MODALITIES = [
    { id: 'voiceFile',      field: 'voice',      label: 'Voice Sample',    icon: '🎤', accept: 'audio/*',              endpoint: '/voice/predict',    fileKey: 'audio'    },
    { id: 'mriFile',        field: 'mri',         label: 'MRI Scan',        icon: '🧠', accept: 'image/*,.nii,.nii.gz', endpoint: '/mri/predict',      fileKey: 'mri_scan' },
    { id: 'spiralFile',     field: 'spiral',      label: 'Spiral Drawing',  icon: '🌀', accept: 'image/*',              endpoint: '/spiral/predict',   fileKey: 'spiral_image' },
    { id: 'timeseriesFile', field: 'timeseries',  label: 'Time-Series',     icon: '📈', accept: '.csv,.json',           endpoint: '/timeseries/predict',fileKey: 'timeseries_data' },
    { id: 'clinicalFile',   field: 'clinical',    label: 'Clinical Data',   icon: '📋', accept: '.csv,.json',           endpoint: '/clinical/predict', fileKey: 'clinical_data' },
    { id: 'motorFile',      field: 'motor',       label: 'Motor Scores',    icon: '🤚', accept: '.csv,.json',           endpoint: '/motor/predict',    fileKey: 'motor_data'    },
  ];

  /* ── HTML ─────────────────────────────────────────── */
  function _html() {
    const fileInputsHTML = MODALITIES.map(m => `
      <div class="file-input-item" id="${m.id}Wrap" onclick="document.getElementById('${m.id}').click()">
        <label for="${m.id}">${m.label}</label>
        <div class="file-slot">
          <span class="file-slot-icon">${m.icon}</span>
          <span id="${m.id}Name" style="font-size:12px;color:var(--text-2)">Click to select</span>
        </div>
        <input type="file" id="${m.id}" accept="${m.accept}" />
      </div>`).join('');

    return `
      <div class="page-header">
        <div>
          <div class="page-title">Prediction</div>
          <div class="page-subtitle">Upload data — system detects single model or fusion automatically</div>
        </div>
      </div>

      <div id="realtimeError"></div>

      <div class="grid-2">
        <!-- LEFT: File upload panel -->
        <div>
          <div class="card section-gap">
            <div class="card-header">
              <span class="card-title">Upload Data</span>
              <div class="card-icon" style="background:var(--cyan-dim)">📤</div>
            </div>

            <!-- Guidance -->
            <div style="background:var(--cyan-dim);border:1px solid rgba(0,212,255,0.2);border-radius:var(--radius-sm);padding:10px 14px;margin-bottom:16px;font-size:12px;color:var(--cyan)">
              📌 Upload <strong>one file</strong> for a single-model prediction, or <strong>multiple files</strong> for fusion.
            </div>

            <div class="file-inputs">${fileInputsHTML}</div>

            <!-- Dynamic predict button -->
            <div style="display:flex;gap:10px;margin-top:16px;align-items:center">
              <button class="btn btn-primary" id="predictBtn" style="flex:1" disabled>
                Select a file to predict
              </button>
              <button class="btn btn-ghost" id="clearBtn">Clear</button>
            </div>

            <!-- Selected files summary -->
            <div id="selectedSummary" style="margin-top:12px;display:none">
              <div style="font-size:11px;font-family:var(--font-mono);color:var(--text-3);text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">
                Will run:
              </div>
              <div id="selectedModels" style="display:flex;flex-wrap:wrap;gap:6px"></div>
            </div>
          </div>
        </div>

        <!-- RIGHT: Results panel -->
        <div>
          <!-- Placeholder (before prediction) -->
          <div id="rtPlaceholder" class="card" style="text-align:center;padding:60px 20px">
            <div style="font-size:48px;margin-bottom:14px;opacity:0.3">⚡</div>
            <div style="font-size:15px;font-weight:600;color:var(--text-2)">Awaiting prediction</div>
            <div style="font-size:12px;color:var(--text-3);margin-top:6px;font-family:var(--font-mono)">
              Select files and click predict
            </div>
          </div>

          <!-- Result (shown after prediction) -->
          <div id="rtResult" class="rt-result">

            <!-- Mode indicator -->
            <div id="rtModeBar" style="display:flex;align-items:center;gap:10px;margin-bottom:16px;padding:10px 14px;background:var(--glass);border:1px solid var(--glass-border);border-radius:var(--radius-sm)">
              <span id="rtModeIcon" style="font-size:20px">⚡</span>
              <div>
                <div id="rtModeName" style="font-size:13px;font-weight:600">—</div>
                <div id="rtModeDesc" style="font-size:11px;color:var(--text-3);font-family:var(--font-mono)">—</div>
              </div>
            </div>

            <!-- Fusion severity (shown for fusion) -->
            <div class="fusion-score section-gap" id="rtFusionSection">
              <div class="fusion-label">Fusion Severity Score</div>
              <div class="fusion-value" id="rtSeverityNum">—</div>
              <div class="fusion-confidence" id="rtConfidence">Confidence: —</div>
              <div class="rt-proc-time" id="rtProcTime">—</div>
            </div>

            <!-- Single model result (shown for single model) -->
            <div class="card section-gap" id="rtSingleSection" style="display:none">
              <div class="card-header">
                <span class="card-title" id="rtSingleTitle">Model Result</span>
                <span id="rtSingleIcon" style="font-size:20px"></span>
              </div>
              <div style="text-align:center;padding:20px 0">
                <div style="font-family:var(--font-display);font-size:48px;font-weight:800" id="rtSingleProb">—</div>
                <div style="font-size:14px;font-weight:600;margin-top:8px" id="rtSingleLabel">—</div>
                <div style="font-size:12px;color:var(--text-3);font-family:var(--font-mono);margin-top:4px" id="rtSingleConf">—</div>
                <div class="rt-proc-time" id="rtSingleTime">—</div>
              </div>
            </div>

            <!-- Individual model breakdown (fusion only) -->
            <div class="card section-gap" id="rtBreakdownSection" style="display:none">
              <div class="card-header">
                <span class="card-title">Individual Model Results</span>
              </div>
              <div id="rtBreakdownContent"></div>
            </div>

            <!-- Risk + Recommendation -->
            <div class="card section-gap">
              <div class="card-header"><span class="card-title">Risk Assessment</span></div>
              <div id="rtRiskContent"></div>
            </div>

            <div class="card card-glow-green section-gap">
              <div class="card-header"><span class="card-title">Quick Recommendation</span></div>
              <div id="rtRecContent" style="font-size:13px;color:var(--text-2);line-height:1.7"></div>
            </div>

            <div class="card">
              <div class="card-header"><span class="card-title">Explanation Summary</span></div>
              <div class="explanation-summary" id="rtExplanation"></div>
            </div>

            <!-- Navigate to dashboard button -->
            <button class="btn btn-primary" style="width:100%;margin-top:16px;justify-content:center" onclick="Router.navigate('dashboard')">
              View Full Dashboard →
            </button>
          </div>
        </div>
      </div>`;
  }

  /* ── Init ─────────────────────────────────────────── */
  function init() {
    document.getElementById(ROOT).innerHTML = _html();

    MODALITIES.forEach(m => {
      document.getElementById(m.id)?.addEventListener('change', function () {
        const name = this.files[0]?.name || 'Click to select';
        Helpers.setText(`${m.id}Name`, name);
        if (this.files[0]) {
          document.getElementById(`${m.id}Wrap`)?.classList.add('has-file');
        } else {
          document.getElementById(`${m.id}Wrap`)?.classList.remove('has-file');
        }
        _updatePredictButton();
      });
    });

    document.getElementById('predictBtn')?.addEventListener('click', _runPrediction);
    document.getElementById('clearBtn')?.addEventListener('click', _clear);
  }

  /* ── Update button label based on selected files ──── */
  function _updatePredictButton() {
    const selected = _getSelectedModalities();
    const btn = document.getElementById('predictBtn');
    const summary = document.getElementById('selectedSummary');
    const models  = document.getElementById('selectedModels');

    if (selected.length === 0) {
      btn.disabled   = true;
      btn.innerHTML  = 'Select a file to predict';
      summary.style.display = 'none';
      return;
    }

    btn.disabled = false;
    summary.style.display = '';

    if (selected.length === 1) {
      btn.innerHTML = `${selected[0].icon} Run ${selected[0].label} Prediction`;
    } else {
      btn.innerHTML = `⚡ Run Fusion Prediction (${selected.length} modalities)`;
    }

    models.innerHTML = selected.map(m =>
      `<span style="background:var(--cyan-dim);color:var(--cyan);border:1px solid rgba(0,212,255,0.2);padding:3px 10px;border-radius:99px;font-size:11px;font-family:var(--font-mono)">${m.icon} ${m.label}</span>`
    ).join('');
  }

  function _getSelectedModalities() {
    return MODALITIES.filter(m => {
      const el = document.getElementById(m.id);
      return el?.files?.length > 0;
    });
  }

  /* ── Run prediction — smart routing ──────────────── */
  async function _runPrediction() {
    Helpers.clearError('realtimeError');
    const selected = _getSelectedModalities();

    if (selected.length === 0) {
      Helpers.showError('realtimeError', 'Please select at least one file');
      return;
    }

    const btn = document.getElementById('predictBtn');
    btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div> Processing…';
    btn.disabled  = true;

    const t0 = Date.now();

    try {
      if (selected.length === 1) {
        await _runSinglePrediction(selected[0], t0);
      } else {
        await _runFusionPrediction(selected, t0);
      }
    } catch (e) {
      Helpers.showError('realtimeError', `Prediction failed: ${e.message}`);
      // Show a fallback result clearly marked as demo
      _renderFallbackResult(selected, Date.now() - t0);
    }

    btn.innerHTML = selected.length === 1
      ? `${selected[0].icon} Run ${selected[0].label} Prediction`
      : `⚡ Run Fusion Prediction (${selected.length} modalities)`;
    btn.disabled = false;
  }

  /* ── Single modality prediction ───────────────────── */
  async function _runSinglePrediction(mod, t0) {
    const fd = new FormData();
    fd.append(mod.fileKey, document.getElementById(mod.id).files[0]);
    fd.append('patient_id', App.patientId);

    const json   = await API._request('POST', mod.endpoint, fd);
    const result = json?.result || json?.data?.result || json?.data || json;
    const elapsed = Date.now() - t0;

    _renderSingleResult(mod, result, elapsed);

    // Update dashboard with single model result
    if (typeof PageDashboard !== 'undefined') {
      PageDashboard.renderFromPrediction({
        severity:   result.severity,
        stage:      result.severity >= 70 ? 'Stage IV' : result.severity >= 40 ? 'Stage II-III' : 'Stage I',
        confidence: result.confidence,
        alerts:     [],
        risks:      [],
        modality_contributions: { [mod.field]: result.confidence || 0 },
      });
    }
  }

  /* ── Fusion (multi-modality) prediction ───────────── */
  async function _runFusionPrediction(selected, t0) {
    const fd = new FormData();
    fd.append('patient_id', App.patientId);

    selected.forEach(m => {
      fd.append(m.field, document.getElementById(m.id).files[0]);
    });

    const json    = await API.realtimePredict(fd);
    const result  = json?.data || json;
    const elapsed = Date.now() - t0;

    _renderFusionResult(result, elapsed);

    // Update dashboard tab with fusion result
    if (typeof PageDashboard !== 'undefined') {
      PageDashboard.renderFromPrediction(result);
    }
  }

  /* ── Render single model result ───────────────────── */
  function _renderSingleResult(mod, d, elapsed) {
    Helpers.hide('rtPlaceholder');
    document.getElementById('rtResult')?.classList.add('visible');

    // Mode bar
    Helpers.setText('rtModeIcon', mod.icon);
    Helpers.setText('rtModeName', `${mod.label} Model`);
    Helpers.setText('rtModeDesc', `Single modality prediction`);

    // Show single section, hide fusion section
    document.getElementById('rtFusionSection').style.display  = 'none';
    document.getElementById('rtSingleSection').style.display  = '';
    document.getElementById('rtBreakdownSection').style.display = 'none';

    Helpers.setText('rtSingleTitle', `${mod.label} Result`);
    Helpers.setText('rtSingleIcon',  mod.icon);

    const prob    = d.probability ?? d.result ?? 0;
    const hasPd   = d.has_parkinson;
    const probPct = Math.round(prob * 100);

    const probEl = document.getElementById('rtSingleProb');
    if (probEl) {
      probEl.textContent = `${probPct}%`;
      probEl.style.color = probPct >= 50 ? 'var(--red)' : 'var(--green)';
    }

    Helpers.setText('rtSingleLabel', d.label || (hasPd ? "Parkinson's Detected" : "No Parkinson's Detected"));
    Helpers.setText('rtSingleConf',  `Confidence: ${Helpers.pct(d.confidence)}%`);
    Helpers.setText('rtSingleTime',  `Processed in ${elapsed} ms`);

    // Risk + Rec
    _renderRisk(d.severity || prob * 100, null, null);
    Helpers.setText('rtRecContent',   'Run fusion prediction with more modalities for treatment recommendations.');
    Helpers.setHTML('rtExplanation',  d.explanation || `${mod.label} model analysed the input and found ${hasPd ? 'indicators consistent with' : 'no strong indicators of'} Parkinson's disease (probability: ${probPct}%).`);
  }

  /* ── Render fusion result ─────────────────────────── */
  function _renderFusionResult(d, elapsed) {
    Helpers.hide('rtPlaceholder');
    document.getElementById('rtResult')?.classList.add('visible');

    // Mode bar
    const mods = d.modalities_used || [];
    Helpers.setText('rtModeIcon', '⚡');
    Helpers.setText('rtModeName', 'Fusion Model');
    Helpers.setText('rtModeDesc', `Combined ${mods.length} modality(ies): ${mods.join(', ')}`);

    // Show fusion section
    document.getElementById('rtFusionSection').style.display  = '';
    document.getElementById('rtSingleSection').style.display  = 'none';

    Helpers.setText('rtSeverityNum', d.severity ?? '—');
    Helpers.setText('rtConfidence',  `Confidence: ${Helpers.pct(d.confidence)}%`);
    Helpers.setText('rtProcTime',    `Processed in ${elapsed} ms · Method: ${d.fusion_method || 'ensemble'}`);

    // Individual model breakdown
    const individual = d.individual_results || {};
    if (Object.keys(individual).length > 0) {
      document.getElementById('rtBreakdownSection').style.display = '';
      const modIcons = { voice:'🎤', clinical:'📋', mri:'🧠', spiral:'🌀', motor:'🤚', timeseries:'📈' };
      Helpers.setHTML('rtBreakdownContent', Object.entries(individual).map(([mod, res]) => `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--glass-border)">
          <div style="display:flex;align-items:center;gap:8px">
            <span style="font-size:16px">${modIcons[mod] || '🔬'}</span>
            <span style="font-size:13px;font-weight:500;text-transform:capitalize">${mod}</span>
          </div>
          <div style="display:flex;align-items:center;gap:10px">
            <span style="font-size:13px;font-family:var(--font-mono)">${Math.round((res.probability || 0)*100)}%</span>
            <span style="font-size:11px;padding:2px 8px;border-radius:99px;${res.has_parkinson ? 'background:var(--red-dim);color:var(--red)' : 'background:var(--green-dim);color:var(--green)'}">${res.label || (res.has_parkinson ? 'PD' : 'No PD')}</span>
          </div>
        </div>`).join(''));
    }

    // Risk + Rec + Explanation
    _renderRisk(d.severity, d.risk_flag, d.risk_description);
    Helpers.setText('rtRecContent',  d.recommendation || '—');
    Helpers.setHTML('rtExplanation', d.explanation || '—');
  }

  function _renderRisk(severity, flag, desc) {
    const sev    = severity || 0;
    const rFlag  = flag || (sev >= 70 ? 'HIGH' : sev >= 40 ? 'MODERATE' : 'LOW');
    const rDesc  = desc || (sev >= 70
      ? 'High severity detected. Urgent clinical review recommended.'
      : sev >= 40
      ? 'Moderate indicators. Monitor closely and consider follow-up.'
      : 'Low severity. Continue regular monitoring.'
    );
    const rColor = { HIGH: 'var(--red)', MODERATE: 'var(--amber)', LOW: 'var(--green)' };
    Helpers.setHTML('rtRiskContent', `
      <div style="display:flex;align-items:center;gap:12px">
        <div style="font-size:26px;font-family:var(--font-display);font-weight:800;color:${rColor[rFlag] || 'var(--cyan)'}">${rFlag}</div>
        <div style="font-size:13px;color:var(--text-2);line-height:1.6">${rDesc}</div>
      </div>`);
  }

  /* ── Fallback demo result (clearly marked) ────────── */
  function _renderFallbackResult(selected, elapsed) {
    Helpers.hide('rtPlaceholder');
    document.getElementById('rtResult')?.classList.add('visible');
    Helpers.setText('rtModeIcon', '⚠️');
    Helpers.setText('rtModeName', 'Demo Result (API Unavailable)');
    Helpers.setText('rtModeDesc', 'These are placeholder values — not real predictions');
    document.getElementById('rtFusionSection').style.display  = '';
    document.getElementById('rtSingleSection').style.display  = 'none';
    document.getElementById('rtBreakdownSection').style.display = 'none';
    Helpers.setText('rtSeverityNum', '—');
    Helpers.setText('rtConfidence',  'API unavailable — check server logs');
    Helpers.setText('rtProcTime',    `Failed after ${elapsed} ms`);
    _renderRisk(null, null, null);
    Helpers.setText('rtRecContent',  'Please check that the Flask server is running and you are logged in.');
    Helpers.setHTML('rtExplanation', 'API call failed. Verify: (1) Flask is running on port 5000, (2) You are logged in, (3) The Authorization token is valid.');
  }

  /* ── Clear ────────────────────────────────────────── */
  function _clear() {
    MODALITIES.forEach(m => {
      const el = document.getElementById(m.id);
      if (el) el.value = '';
      Helpers.setText(`${m.id}Name`, 'Click to select');
      document.getElementById(`${m.id}Wrap`)?.classList.remove('has-file');
    });
    document.getElementById('rtResult')?.classList.remove('visible');
    Helpers.show('rtPlaceholder');
    document.getElementById('selectedSummary').style.display = 'none';
    Helpers.clearError('realtimeError');
    _updatePredictButton();
  }

  function load() {} // no auto-fetch

  return { init, load };
})();