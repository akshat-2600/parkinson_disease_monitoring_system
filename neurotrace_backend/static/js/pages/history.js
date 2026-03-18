/* static/js/pages/history.js */
const PageHistory = (() => {
  const ROOT = 'page-history';

  function _html() {
    return `
      <div class="page-header">
        <div><div class="page-title">Historical Tracking</div><div class="page-subtitle">Longitudinal progression analysis & intervention timeline</div></div>
        <button class="btn btn-ghost" id="histRefreshBtn"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg> Refresh</button>
      </div>
      <div id="historyError"></div>
      <div id="historyLoading">${Skeletons.historySkeleton()}</div>
      <div id="historyData" style="display:none">
        <div class="card card-glow-cyan section-gap"><div class="card-header"><span class="card-title">Severity Timeline</span><span class="text-muted" style="font-size:11px;font-family:var(--font-mono)">12-month view</span></div><div class="chart-wrap" style="height:240px"><canvas id="historySeverityChart"></canvas></div></div>
        <div class="grid-2 section-gap">
          <div class="card card-glow-violet"><div class="card-header"><span class="card-title">UPDRS Scores</span></div><div class="chart-wrap" style="height:200px"><canvas id="historyUpdrsChart"></canvas></div></div>
          <div class="card card-glow-green"><div class="card-header"><span class="card-title">Multi-Modality Trends</span></div><div class="chart-wrap" style="height:200px"><canvas id="historyModalityChart"></canvas></div></div>
        </div>
        <div class="card"><div class="card-header"><span class="card-title">Intervention Timeline</span><div class="card-icon" style="background:var(--cyan-dim)">📋</div></div><div class="history-timeline" id="interventionTimeline"></div></div>
      </div>`;
  }

  function init() {
    document.getElementById(ROOT).innerHTML = _html();
    document.getElementById('histRefreshBtn')?.addEventListener('click', load);
  }

  async function load() {
    Helpers.clearError('historyError');
    Helpers.show('historyLoading'); Helpers.hide('historyData');
    let data;
    try {
      const res = await API.getHistory(App.patientId, Auth.getToken());
      data = res.data;
    } catch (e) {
      Helpers.showError('historyError', `API error: ${e.message}`);
      data = Fallback.getHistory(App.patientId);
    }
    _render(data);
  }

  function _render(d) {
    Helpers.hide('historyLoading'); Helpers.show('historyData');
    const labels = d.labels || [];

    function _line(key, canvasId, datasets, opts={}) {
      const ctx = document.getElementById(canvasId)?.getContext('2d'); if (!ctx) return;
      datasets.forEach(ds => { if (ds.fill && !ds.backgroundColor) ds.backgroundColor = Charts.gradientFill(ctx, ds.borderColor.replace('#',''), 240); });
      Charts.create(key, ctx, { type:'line', data:{labels,datasets}, options:{ responsive:true, maintainAspectRatio:false, plugins:{ legend: opts.legend ? {position:'bottom',labels:{boxWidth:10,padding:12,color:'#8892b0'}} : {display:false}, tooltip:Charts.TOOLTIP }, scales:{ x:{...Charts.SCALE.x, ticks:{maxTicksLimit:opts.maxX||8}}, y:{...Charts.SCALE.y, min:opts.minY, max:opts.maxY} } } });
    }

    _line('historySeverity','historySeverityChart',[{label:'Severity',data:d.severity||[],borderColor:'#00d4ff',fill:true,tension:0.4,borderWidth:2,pointRadius:3,pointHoverRadius:6}],{minY:0,maxY:100});
    _line('historyUpdrs','historyUpdrsChart',[{label:'UPDRS',data:d.updrs||[],borderColor:'#7b5cff',backgroundColor:'rgba(123,92,255,0.1)',fill:true,tension:0.4,borderWidth:2,pointRadius:2}],{maxX:6});
    _line('historyModality','historyModalityChart',[
      {label:'Voice',data:d.voice||[],borderColor:'#00e5a0',tension:0.4,borderWidth:1.5,pointRadius:0,fill:false},
      {label:'MRI',data:d.mri||[],borderColor:'#ffb547',tension:0.4,borderWidth:1.5,pointRadius:0,fill:false},
      {label:'Motor',data:d.motor||[],borderColor:'#ff6b9d',tension:0.4,borderWidth:1.5,pointRadius:0,fill:false},
    ],{minY:0,maxY:1,maxX:6,legend:true});

    Helpers.setHTML('interventionTimeline', (d.interventions||[]).length
      ? (d.interventions||[]).map(i=>`<div class="timeline-item"><div class="timeline-dot"></div><div class="timeline-date">${i.date}</div><div class="timeline-event">${i.event}</div></div>`).join('')
      : `<div class="empty-state"><div class="empty-text">No interventions recorded</div></div>`);
  }

  return { init, load };
})();
