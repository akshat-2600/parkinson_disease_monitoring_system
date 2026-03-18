/* ============================================================
   js/pages/realtime.js — Real-Time Prediction page
   ============================================================ */

const PageRealtime = (() => {

  const ROOT = 'page-realtime';

  const FILE_INPUTS = [
    { id: 'voiceFile',      label: 'Voice Sample',     icon: '🎤', accept: 'audio/*',           field: 'voice'      },
    { id: 'mriFile',        label: 'MRI Scan',          icon: '🧠', accept: 'image/*,.nii,.nii.gz', field: 'mri'     },
    { id: 'spiralFile',     label: 'Spiral Drawing',    icon: '🌀', accept: 'image/*',            field: 'spiral'    },
    { id: 'timeseriesFile', label: 'Time-Series Data',  icon: '📈', accept: '.csv,.json',          field: 'timeseries'},
    { id: 'clinicalFile',   label: 'Clinical Data',     icon: '📋', accept: '.csv,.json',          field: 'clinical'  },
    { id: 'motorFile',      label: 'Motor Score Data',  icon: '🤚', accept: '.csv,.json',          field: 'motor'     },
  ];

  function _fileInputsHTML() {
    return FILE_INPUTS.map(f => `
      <div class="file-input-item" onclick="document.getElementById('${f.id}').click()">
        <label for="${f.id}">${f.label}</label>
        <div class="file-slot">
          <span class="file-slot-icon">${f.icon}</span>
          <span id="${f.id}Name" style="font-size:12px;color:var(--text-2)">Click to select</span>
        </div>
        <input type="file" id="${f.id}" accept="${f.accept}" />
      </div>`).join('');
  }

  function _html() {
    return `
      <div class="page-header">
        <div>
          <div class="page-title">Real-Time Prediction</div>
          <div class="page-subtitle">Upload multimodal data for instant AI fusion analysis</div>
        </div>
      </div>

      <div id="realtimeError"></div>

      <div class="grid-2">
        <!-- Input panel -->
        <div>
          <div class="card section-gap">
            <div class="card-header">
              <span class="card-title">Upload Data Inputs</span>
              <div class="card-icon" style="background:var(--cyan-dim)">📤</div>
            </div>
            <div class="file-inputs">${_fileInputsHTML()}</div>
            <div style="display:flex;gap:10px;margin-top:16px">
              <button class="btn btn-primary" id="predictBtn" style="flex:1">⚡ Run Fusion Prediction</button>
              <button class="btn btn-ghost"   id="clearBtn">Clear</button>
            </div>
          </div>
        </div>

        <!-- Results panel -->
        <div>
          <div id="rtPlaceholder" class="card" style="text-align:center;padding:60px 20px">
            <div style="font-size:48px;margin-bottom:14px;opacity:0.3">⚡</div>
            <div style="font-size:15px;font-weight:600;color:var(--text-2)">Awaiting prediction</div>
            <div style="font-size:12px;color:var(--text-3);margin-top:6px;font-family:var(--font-mono)">Upload files and run fusion model</div>
          </div>

          <div id="rtResult" class="rt-result">
            <!-- Fusion score -->
            <div class="fusion-score section-gap">
              <div class="fusion-label">Fusion Severity Score</div>
              <div class="fusion-value" id="rtSeverityNum">—</div>
              <div class="fusion-confidence" id="rtConfidence">Confidence: —</div>
              <div class="rt-proc-time" id="rtProcTime">Processed in — ms</div>
            </div>

            <!-- Risk flag -->
            <div class="card section-gap">
              <div class="card-header">
                <span class="card-title">Risk Assessment</span>
              </div>
              <div id="rtRiskContent"></div>
            </div>

            <!-- Quick recommendation -->
            <div class="card card-glow-green section-gap">
              <div class="card-header">
                <span class="card-title">Quick Recommendation</span>
              </div>
              <div id="rtRecContent" style="font-size:13px;color:var(--text-2);line-height:1.7"></div>
            </div>

            <!-- Explanation -->
            <div class="card">
              <div class="card-header">
                <span class="card-title">Explanation Summary</span>
              </div>
              <div class="explanation-summary" id="rtExplanation"></div>
            </div>
          </div>
        </div>
      </div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    _bindEvents();
  }

  function _bindEvents() {
    // File name preview
    FILE_INPUTS.forEach(f => {
      document.getElementById(f.id)?.addEventListener('change', function () {
        const name = this.files[0]?.name || 'Click to select';
        Helpers.setText(`${f.id}Name`, name);
      });
    });

    document.getElementById('predictBtn')?.addEventListener('click', _runPrediction);
    document.getElementById('clearBtn')?.addEventListener('click', _clear);
  }

  async function _runPrediction() {
    Helpers.clearError('realtimeError');

    const btn = document.getElementById('predictBtn');
    btn.innerHTML = '<div class="spinner" style="width:16px;height:16px;border-width:2px"></div> Processing…';
    btn.disabled  = true;

    const formData = new FormData();
    formData.append('patient_id', App.patientId);
    FILE_INPUTS.forEach(f => {
      const el = document.getElementById(f.id);
      if (el?.files[0]) formData.append(f.field, el.files[0]);
    });

    const t0 = Date.now();
    let data;

    try {
      data = await API.realtimePredict(formData);
    } catch (e) {
      Helpers.showError('realtimeError', `Prediction API unavailable: ${e.message}`);
      // Fallback mock result
      data = {
        severity:         Math.round(45 + Math.random() * 30),
        confidence:       parseFloat((0.75 + Math.random() * 0.2).toFixed(2)),
        risk_flag:        'MODERATE',
        risk_description: 'Motor fluctuation detected. Wearing-off episodes likely between doses.',
        recommendation:   'Consider adjusting levodopa dosing schedule. Schedule physiotherapy assessment within 72 hours.',
        explanation:      'Primary indicators: elevated tremor frequency (6.2 Hz), reduced arm swing asymmetry ratio (0.74), voice jitter above threshold (2.8%).',
        processing_time_ms: Date.now() - t0,
      };
    }

    _renderResult(data, data.processing_time_ms ?? (Date.now() - t0));
    btn.innerHTML = '⚡ Run Fusion Prediction';
    btn.disabled  = false;
  }

  function _renderResult(d, elapsed) {
    Helpers.hide('rtPlaceholder');

    Helpers.setText('rtSeverityNum', d.severity ?? '—');
    Helpers.setText('rtConfidence',  `Confidence: ${Helpers.pct(d.confidence)}%`);
    Helpers.setText('rtProcTime',    `Processed in ${elapsed} ms`);

    const riskColors = { HIGH: 'var(--red)', MODERATE: 'var(--amber)', LOW: 'var(--green)' };
    const riskColor  = riskColors[d.risk_flag] || 'var(--cyan)';
    Helpers.setHTML('rtRiskContent', `
      <div style="display:flex;align-items:center;gap:12px">
        <div style="font-size:28px;font-family:var(--font-display);font-weight:800;color:${riskColor}">${d.risk_flag || '—'}</div>
        <div style="font-size:13px;color:var(--text-2);line-height:1.6">${d.risk_description || '—'}</div>
      </div>`);

    Helpers.setText('rtRecContent',   d.recommendation  || '—');
    Helpers.setHTML('rtExplanation',  d.explanation      || '—');

    document.getElementById('rtResult')?.classList.add('visible');
  }

  function _clear() {
    FILE_INPUTS.forEach(f => {
      const el = document.getElementById(f.id);
      if (el) el.value = '';
      Helpers.setText(`${f.id}Name`, 'Click to select');
    });
    document.getElementById('rtResult')?.classList.remove('visible');
    Helpers.show('rtPlaceholder');
    Helpers.clearError('realtimeError');
  }

  // load() is a no-op for this page (no auto-fetch on navigate)
  function load() {}

  return { init, load };
})();
