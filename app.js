/**
 * app.js — AI Stock Signal Dashboard
 * Lightweight Charts integration + Position Sizer state + live updates
 */

/* ─── Signal data (replace with API fetch) ─────────────────────── */
const SIGNALS_DATA = [
  {
    id: 'tsla', ticker: 'TSLA', signal: 'buy',  model: 'LSTM',
    price: 245.67, confidence: 92, time: '2m ago', consensus: true,
  },
  {
    id: 'nvda', ticker: 'NVDA', signal: 'sell', model: 'Random Forest',
    price: 478.23, confidence: 88, time: '5m ago', consensus: false,
  },
  {
    id: 'spy',  ticker: 'SPY',  signal: 'hold', model: 'LSTM',
    price: 442.15, confidence: 76, time: '8m ago', consensus: false,
  },
  {
    id: 'msft', ticker: 'MSFT', signal: 'buy',  model: 'Random Forest',
    price: 378.90, confidence: 85, time: '12m ago', consensus: true,
  },
  {
    id: 'amzn', ticker: 'AMZN', signal: 'sell', model: 'LSTM',
    price: 167.45, confidence: 91, time: '15m ago', consensus: true,
  },
];

const METRICS_DATA = {
  sharpe:   { value: '1.84', delta: '+0.12', dir: 'up' },
  drawdown: { value: '-12.3%', delta: '+2.1%', dir: 'up' },
  annReturn:{ value: '+34.7%', delta: '+5.2%', dir: 'up' },
};

const SIGNAL_ICONS = { buy: '↗', sell: '↘', hold: '→' };

/* ─── Render signal cards ──────────────────────────────────────── */
function renderSignals() {
  const list = document.getElementById('signals-list');
  list.innerHTML = '';

  SIGNALS_DATA.forEach((s) => {
    const card = document.createElement('div');
    card.className = `signal-card ${s.signal}`;
    card.id = `signal-card-${s.id}`;

    const consensusHtml = s.consensus
      ? `<span class="consensus-badge">
           <svg width="9" height="9" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0a8 8 0 100 16A8 8 0 008 0zm3.97 5.03l-4.5 4.5a.75.75 0 01-1.06 0l-2-2a.75.75 0 011.06-1.06l1.47 1.47 3.97-3.97a.75.75 0 011.06 1.06z"/></svg>
           Consensus
         </span>` : '';

    card.innerHTML = `
      <div class="card-top">
        <div class="card-left">
          <span class="card-ticker" id="ticker-${s.id}">${s.ticker}</span>
          <span class="signal-badge ${s.signal}" id="signal-type-${s.id}">
            ${SIGNAL_ICONS[s.signal]} ${s.signal.toUpperCase()}
          </span>
          ${consensusHtml}
        </div>
        <span class="card-time" id="signal-time-${s.id}">${s.time}</span>
      </div>

      <div class="card-meta">
        <div class="meta-group">
          <span class="meta-label">Model</span>
          <span class="meta-value" id="signal-model-${s.id}">${s.model}</span>
        </div>
        <div class="meta-group" style="text-align:right">
          <span class="meta-label">Price</span>
          <span class="price-value" id="signal-price-${s.id}">$${s.price.toFixed(2)}</span>
        </div>
      </div>

      <div class="confidence-row">
        <span class="confidence-label">
          Confidence
          <span class="info-icon" title="Model confidence score">i</span>
        </span>
        <span class="confidence-pct ${s.signal}" id="signal-confidence-${s.id}">${s.confidence}%</span>
      </div>
      <div class="progress-track">
        <div class="progress-fill ${s.signal}" id="signal-bar-${s.id}" style="width:0%"></div>
      </div>
    `;

    list.appendChild(card);

    // Animate bar on next frame
    requestAnimationFrame(() => {
      setTimeout(() => {
        document.getElementById(`signal-bar-${s.id}`).style.width = s.confidence + '%';
      }, 80);
    });
  });
}

/* ─── Position Sizer ────────────────────────────────────────────── */
let sizer = {
  portfolio: 100000,
  risk: 2,
  confidence: 85,
};

function calcPosition() {
  const port = parseFloat(document.getElementById('input-portfolio').value) || 0;
  const risk = parseFloat(document.getElementById('input-risk').value) || 0;
  const conf = parseFloat(document.getElementById('input-confidence').value) || 0;

  sizer = { portfolio: port, risk, confidence: conf };

  const maxRisk   = port * (risk / 100);
  const adjSize   = maxRisk * (conf / 100);
  const pct       = port > 0 ? (adjSize / port) * 100 : 0;

  document.getElementById('recommended-size').textContent = '$' + adjSize.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  document.getElementById('result-bar-fill').style.width  = Math.min(pct * 10, 100) + '%';
  document.getElementById('result-pct-label').textContent = pct.toFixed(2) + '% of portfolio';
  document.getElementById('result-max-risk').textContent  = 'Max Risk: $' + maxRisk.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function bindSizerEvents() {
  ['input-portfolio','input-risk','input-confidence'].forEach(id => {
    document.getElementById(id).addEventListener('input', calcPosition);
  });
  document.getElementById('calc-btn').addEventListener('click', () => {
    calcPosition();
    const btn = document.getElementById('calc-btn');
    btn.textContent = '✓ Calculated';
    setTimeout(() => { btn.innerHTML = '<span>⚡</span> Calculate'; }, 1200);
  });
}

/* ─── Lightweight Chart ─────────────────────────────────────────── */
function initChart() {
  const container = document.getElementById('lightweight-chart');
  if (!window.LightweightCharts) { renderFallbackCanvas(container); return; }

  const chart = LightweightCharts.createChart(container, {
    layout: {
      background: { type: 'solid', color: '#161820' },
      textColor: '#8890b0',
    },
    grid: {
      vertLines: { color: '#2a2d3e', style: 1 },
      horzLines: { color: '#2a2d3e', style: 1 },
    },
    crosshair: {
      vertLine: { color: '#3a3e55', labelBackgroundColor: '#1c1e28' },
      horzLine: { color: '#3a3e55', labelBackgroundColor: '#1c1e28' },
    },
    rightPriceScale: { borderColor: '#2a2d3e' },
    timeScale: { borderColor: '#2a2d3e', timeVisible: true },
    width:  container.clientWidth,
    height: container.clientHeight,
  });

  window._dashChart = chart;

  const series = chart.addAreaSeries({
    lineColor:  '#00e676',
    topColor:   'rgba(0,230,118,0.2)',
    bottomColor:'rgba(0,230,118,0)',
    lineWidth: 2,
    priceLineColor: '#2a2d3e',
  });

  // Synthetic AAPL-like data
  const now   = Math.floor(Date.now() / 1000);
  const day   = 86400;
  const base  = 148;
  const data  = [];

  for (let i = 90; i >= 0; i--) {
    const t = now - i * day;
    const noise = (Math.random() - 0.48) * 2.5;
    const trend = (90 - i) * 0.055;
    data.push({ time: t, value: parseFloat((base + trend + noise).toFixed(2)) });
  }

  series.setData(data);

  // Buy/Sell markers
  const markers = [
    { time: data[20].time, position: 'belowBar', color: '#00e676', shape: 'arrowUp',   text: 'Buy' },
    { time: data[45].time, position: 'aboveBar', color: '#ff1744', shape: 'arrowDown', text: 'Sell' },
    { time: data[70].time, position: 'belowBar', color: '#00e676', shape: 'arrowUp',   text: 'Buy' },
  ];
  series.setMarkers(markers);

  chart.timeScale().fitContent();

  // Resize observer
  const ro = new ResizeObserver(() => {
    chart.applyOptions({ width: container.clientWidth, height: container.clientHeight });
  });
  ro.observe(container);

  // Expose update function for Python API
  window.updateChartData = (newData) => series.setData(newData);
  window.appendChartPoint = (point) => series.update(point);
}

/* ─── Fallback canvas chart (when CDN not available) ────────────── */
function renderFallbackCanvas(container) {
  const canvas = document.createElement('canvas');
  canvas.width  = container.clientWidth  || 700;
  canvas.height = container.clientHeight || 400;
  canvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;';
  container.appendChild(canvas);
  window._dashCanvas = canvas;
  drawCanvas(canvas);
}

function drawCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 30, right: 20, bottom: 50, left: 55 };
  const pts = 80;
  const prices = [];
  let v = 152;
  for (let i = 0; i < pts; i++) {
    v += (Math.random() - 0.46) * 2.2;
    prices.push(+v.toFixed(2));
  }
  const minP = Math.min(...prices) - 2;
  const maxP = Math.max(...prices) + 2;

  function xOf(i) { return PAD.left + (i / (pts - 1)) * (W - PAD.left - PAD.right); }
  function yOf(p) { return PAD.top + (1 - (p - minP) / (maxP - minP)) * (H - PAD.top - PAD.bottom); }

  // Background
  ctx.fillStyle = '#161820';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#2a2d3e'; ctx.lineWidth = 1;
  for (let r = 0; r <= 4; r++) {
    const y = PAD.top + (r / 4) * (H - PAD.top - PAD.bottom);
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(W - PAD.right, y); ctx.stroke();
    const label = (maxP - (r / 4) * (maxP - minP)).toFixed(0);
    ctx.fillStyle = '#8890b0'; ctx.font = '11px Inter,sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(label, PAD.left - 6, y + 4);
  }

  // Gradient area
  const grad = ctx.createLinearGradient(0, PAD.top, 0, H - PAD.bottom);
  grad.addColorStop(0, 'rgba(0,230,118,0.22)');
  grad.addColorStop(1, 'rgba(0,230,118,0)');
  ctx.beginPath();
  ctx.moveTo(xOf(0), yOf(prices[0]));
  for (let i = 1; i < pts; i++) ctx.lineTo(xOf(i), yOf(prices[i]));
  ctx.lineTo(xOf(pts-1), H - PAD.bottom);
  ctx.lineTo(xOf(0), H - PAD.bottom);
  ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#00e676'; ctx.lineWidth = 2;
  ctx.shadowColor = '#00e676'; ctx.shadowBlur = 6;
  ctx.moveTo(xOf(0), yOf(prices[0]));
  for (let i = 1; i < pts; i++) ctx.lineTo(xOf(i), yOf(prices[i]));
  ctx.stroke(); ctx.shadowBlur = 0;

  // Buy/Sell markers
  [[18,'buy'],[44,'sell'],[68,'buy']].forEach(([i, type]) => {
    const x = xOf(i), y = yOf(prices[i]);
    const isB = type === 'buy';
    ctx.beginPath(); ctx.arc(x, y + (isB ? 12 : -12), 5, 0, Math.PI * 2);
    ctx.fillStyle = isB ? '#00e676' : '#ff1744'; ctx.fill();
    ctx.fillStyle = isB ? '#00e676' : '#ff1744'; ctx.font = 'bold 10px Inter,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText(isB ? '▲' : '▼', x, y + (isB ? 26 : -18));
  });

  // X-axis labels
  const labels = ['1m','10m','20m','30m','40m','50m','60m'];
  ctx.fillStyle = '#8890b0'; ctx.font = '11px Inter,sans-serif'; ctx.textAlign = 'center';
  labels.forEach((l, i) => {
    const xi = Math.round((i / (labels.length - 1)) * (pts - 1));
    ctx.fillText(l, xOf(xi), H - PAD.bottom + 18);
  });
}

/* ─── Sidebar navigation ─────────────────────────────────────────── */
function bindNav() {
  document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      item.classList.add('active');
    });
  });
}

/* ─── Timeframe buttons ──────────────────────────────────────────── */
function bindTimeframes() {
  document.querySelectorAll('.tf-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

/* ─── Live clock / last-update ───────────────────────────────────── */
function startClock() {
  const el = document.getElementById('last-update-time');
  const tickerUpdated = document.getElementById('ticker-last-updated');
  let secs = 30;

  setInterval(() => {
    secs++;
    if (secs > 60) secs = 0;
    if (el) el.textContent = `Last update: ${secs}s ago`;
    if (tickerUpdated) tickerUpdated.textContent = `Last updated: ${secs}s ago`;
  }, 1000);
}

/* ─── Public API hooks (callable from Python backend) ──────────── */
window.DashboardAPI = {
  /**
   * Update ticker header
   * @param {string} ticker  e.g. "AAPL"
   * @param {string} name    e.g. "Apple Inc."
   * @param {number} price
   * @param {string} change  e.g. "+2.3%"
   */
  setTicker(ticker, name, price, change) {
    document.getElementById('ticker-name-text').textContent = `${ticker} - ${name}`;
    document.getElementById('ticker-price').textContent = `$${price.toFixed(2)}`;
    document.getElementById('ticker-change').textContent = `(${change})`;
  },

  /**
   * Inject a new signal card at the top of the feed
   * @param {Object} signal  { id, ticker, signal, model, price, confidence, time, consensus }
   */
  pushSignal(signal) {
    SIGNALS_DATA.unshift(signal);
    if (SIGNALS_DATA.length > 10) SIGNALS_DATA.pop();
    renderSignals();
    document.getElementById('active-count').textContent = SIGNALS_DATA.filter(s => s.signal !== 'hold').length + ' Active';
  },

  /** Update a specific signal card's confidence  */
  updateConfidence(id, confidence) {
    const pct = document.getElementById(`signal-confidence-${id}`);
    const bar = document.getElementById(`signal-bar-${id}`);
    if (pct) pct.textContent = confidence + '%';
    if (bar) bar.style.width = confidence + '%';
  },

  /** Update performance metrics */
  setMetrics({ sharpe, drawdown, annReturn }) {
    if (sharpe)    { document.getElementById('metric-sharpe').textContent     = sharpe.value;    document.getElementById('metric-sharpe-delta').textContent     = sharpe.delta; }
    if (drawdown)  { document.getElementById('metric-drawdown').textContent   = drawdown.value;  document.getElementById('metric-drawdown-delta').textContent   = drawdown.delta; }
    if (annReturn) { document.getElementById('metric-ann-return').textContent = annReturn.value; document.getElementById('metric-ann-return-delta').textContent = annReturn.delta; }
  },
};

/* ─── Init ───────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  renderSignals();
  calcPosition();
  bindSizerEvents();
  bindNav();
  bindTimeframes();
  startClock();

  // Defer chart after layout paint
  requestAnimationFrame(() => setTimeout(initChart, 50));
});
