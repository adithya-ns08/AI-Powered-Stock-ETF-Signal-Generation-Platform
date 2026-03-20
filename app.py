"""
app.py — AI-Powered Stock & ETF Signal Generation Platform
Streamlit Dashboard  |  Run: streamlit run app.py
Requirements: pip install streamlit plotly pandas numpy
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ── Page config — MUST be first Streamlit call ──────────────────
st.set_page_config(
    page_title="AI Signal Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS  — dark charcoal + neon accents
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-void:    #0a0b0d;
  --bg-base:    #0d0f14;
  --bg-surface: #13161f;
  --bg-card:    #191c28;
  --bg-hover:   #1f2235;
  --bg-input:   #11131a;
  --border:     #252840;
  --border-hi:  #353860;
  --emerald:    #00e676;
  --crimson:    #ff1744;
  --amber:      #ffab00;
  --blue:       #2979ff;
  --purple:     #d500f9;
  --text-1:     #e8eaf6;
  --text-2:     #8890b0;
  --text-3:     #555a72;
}

/* ── Reset ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  background-color: var(--bg-void) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--text-1) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }
.block-container {
  padding: 1rem 1.2rem 1rem 1.2rem !important;
  max-width: 100% !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-base) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
[data-testid="stSidebarNav"] { display: none; }

/* Sidebar buttons — make them icon-style */
[data-testid="stSidebar"] [data-testid="stButton"] button {
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  color: var(--text-3) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  padding: 8px 6px !important;
  margin: 1px 0 !important;
  transition: all 0.2s !important;
  box-shadow: none !important;
  width: 100% !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
  background: var(--bg-hover) !important;
  color: var(--text-1) !important;
  transform: none !important;
}

/* ── Number inputs ── */
[data-testid="stNumberInput"] input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-1) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 14px !important;
}
[data-testid="stNumberInput"] input:focus {
  border-color: var(--emerald) !important;
  box-shadow: 0 0 0 3px rgba(0,230,118,0.12) !important;
  outline: none !important;
}
[data-testid="stNumberInput"] label { color: var(--text-2) !important; font-size: 12px !important; }

/* ── Slider ── */
[data-testid="stSlider"] label { color: var(--text-2) !important; font-size: 12px !important; }
[data-baseweb="slider"] [role="slider"] { background: var(--emerald) !important; }
[data-baseweb="slider"] div[class*="track"] { background: var(--border) !important; }
[data-baseweb="slider"] div[class*="inner"] { background: var(--emerald) !important; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { color: var(--text-2) !important; font-size: 12px !important; }
[data-baseweb="select"] > div {
  background: var(--bg-input) !important;
  border-color: var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-1) !important;
}

/* ── Calculate button (main action) ── */
[data-testid="stButton"] button {
  background: linear-gradient(135deg, #00c853, #00e676) !important;
  color: #000 !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  box-shadow: 0 0 18px rgba(0,230,118,0.28) !important;
  transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
  box-shadow: 0 0 28px rgba(0,230,118,0.45) !important;
  transform: translateY(-1px) !important;
}

/* ── Radio (timeframe switch) ── */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] [role="radiogroup"] {
  display: flex !important; flex-direction: row !important;
  gap: 4px !important; flex-wrap: wrap !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 4px 12px !important;
  cursor: pointer !important;
  color: var(--text-2) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  transition: all 0.15s !important;
  margin: 0 !important;
}
[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
  background: var(--emerald) !important;
  border-color: var(--emerald) !important;
  color: #000 !important;
  font-weight: 700 !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] span:first-child { display: none !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 12px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
::-webkit-scrollbar-track { background: transparent; }

/* ── Plotly chart bg ── */
.js-plotly-plot { border-radius: 12px !important; }

/* ── Metric pulse animation ── */
@keyframes pulse-dot { 0%,100%{opacity:1; box-shadow: 0 0 6px var(--emerald);} 50%{opacity:.35; box-shadow: 0 0 2px var(--emerald);} }

/* ── Active sidebar nav ── */
.nav-active button {
  background: rgba(0,230,118,0.1) !important;
  color: var(--emerald) !important;
  border-color: rgba(0,230,118,0.25) !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  DATA (Engine Integration)
# ══════════════════════════════════════════════════════════════════
from signal_engine.engine import SignalEngine, EngineConfig
from pathlib import Path

@st.cache_resource
def load_engine():
    # Safely disable LSTM/XGBoost due to missing system libs on user's machine
    config = EngineConfig(use_lstm=False, use_xgboost=False)
    engine = SignalEngine(config=config)
    
    models_dir = Path("models")
    if not models_dir.exists() or not any(models_dir.iterdir()):
        engine._needs_training = True
    else:
        engine._needs_training = False
        engine.load_models()
    return engine

try:
    engine = load_engine()
    
    if getattr(engine, "_needs_training", False):
        with st.spinner("First run detected: Training AI models..."):
            engine.run(["AAPL", "TSLA", "NVDA", "SPY", "MSFT"], evaluate=True, train_mode=True)
            engine.save_models()
        engine._needs_training = False  # Ensure subsequent partial reruns don't retrain
        st.success("Models trained and cached! Refreshing stream...")
        st.rerun()

    # Execute zero-latency inference
    engine.run(["AAPL", "TSLA", "NVDA", "SPY", "MSFT"], evaluate=False, train_mode=False)
    SIGNALS = engine.get_signals_for_streamlit()
except Exception as e:
    st.error(f"Error loading Signal Engine: {e}")
    SIGNALS = []

TICKERS = {
    "AAPL": {"name": "Apple Inc.",   "price": 152.34, "change": 2.3,  "sentiment": 72},
    "TSLA": {"name": "Tesla Inc.",   "price": 245.67, "change": 1.8,  "sentiment": 65},
    "NVDA": {"name": "NVIDIA Corp.", "price": 478.23, "change": -1.2, "sentiment": 35},
    "SPY":  {"name": "S&P 500 ETF", "price": 442.15, "change": 0.4,  "sentiment": 55},
    "MSFT": {"name": "Microsoft",    "price": 378.90, "change": 3.1,  "sentiment": 80},
}

METRICS = {
    "sharpe":     {"value": "1.84",   "delta": "+0.12", "up": True},
    "drawdown":   {"value": "-12.3%", "delta": "+2.1%", "up": True},
    "ann_return": {"value": "+34.7%", "delta": "+5.2%", "up": True},
    "accuracy":   {"value": "82.4%",  "delta": "+1.5%", "up": True},
}

# ── Session state ────────────────────────────────────────────────
for k, v in [("nav", "Dashboard"), ("tf", "1D"), ("ticker", "AAPL")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════
#  PRICE SERIES (synthetic, cached)
# ══════════════════════════════════════════════════════════════════
@st.cache_data
def get_price_series(ticker: str, tf: str) -> pd.DataFrame:
    np.random.seed(abs(hash(ticker + tf)) % (2**31))
    n     = {"1H": 60, "4H": 120, "1D": 90, "1W": 52, "1M": 24}[tf]
    base  = TICKERS.get(ticker, TICKERS["AAPL"])["price"] * 0.97
    v     = float(base)
    prices = []
    for _ in range(n):
        v += float(np.random.randn()) * 1.8 + 0.06
        prices.append(round(max(v, 5.0), 2))
    delta = {"1H": timedelta(minutes=1), "4H": timedelta(hours=1),
             "1D": timedelta(days=1),   "1W": timedelta(weeks=1),
             "1M": timedelta(days=30)}[tf]
    end   = datetime.now()
    dates = [end - delta * (n - i) for i in range(n)]
    return pd.DataFrame({"date": dates, "price": prices})


# ══════════════════════════════════════════════════════════════════
#  PLOTLY CHART BUILDER
# ══════════════════════════════════════════════════════════════════
def build_chart(ticker: str, tf: str) -> go.Figure:
    df = get_price_series(ticker, tf)
    n  = len(df)
    # Place markers at ~25% and 65% of the series
    b_idx = [max(0, n // 4), max(0, n * 2 // 3)]
    s_idx = [max(0, n * 2 // 5), max(0, n * 4 // 5)]

    fig = go.Figure()

    # Main area line
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["price"],
        mode="lines",
        line=dict(color="#00e676", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,230,118,0.06)",
        name=ticker,
        hovertemplate="<b>%{x|%b %d %H:%M}</b><br>$%{y:.2f}<extra></extra>",
    ))

    # Buy markers
    valid_b = [i for i in b_idx if 0 <= i < n]
    if valid_b:
        fig.add_trace(go.Scatter(
            x=df["date"].iloc[valid_b], y=df["price"].iloc[valid_b],
            mode="markers+text",
            marker=dict(symbol="triangle-up", size=14, color="#00e676",
                        line=dict(color="#003a20", width=1.5)),
            text=["▲ BUY"] * len(valid_b),
            textposition="bottom center",
            textfont=dict(color="#00e676", size=10, family="Inter"),
            name="AI Buy Signal",
        ))

    # Sell markers
    valid_s = [i for i in s_idx if 0 <= i < n]
    if valid_s:
        fig.add_trace(go.Scatter(
            x=df["date"].iloc[valid_s], y=df["price"].iloc[valid_s],
            mode="markers+text",
            marker=dict(symbol="triangle-down", size=14, color="#ff1744",
                        line=dict(color="#5a0010", width=1.5)),
            text=["▼ SELL"] * len(valid_s),
            textposition="top center",
            textfont=dict(color="#ff1744", size=10, family="Inter"),
            name="AI Sell Signal",
        ))

    fig.update_layout(
        plot_bgcolor  = "#13161f",
        paper_bgcolor = "#13161f",
        margin=dict(l=60, r=20, t=24, b=44),
        font=dict(family="Inter,system-ui", color="#8890b0", size=11),
        xaxis=dict(
            gridcolor="#1f2235", showgrid=True,
            zeroline=False, showline=False,
            tickfont=dict(color="#555a72", size=10),
        ),
        yaxis=dict(
            gridcolor="#1f2235", showgrid=True,
            zeroline=False, showline=False,
            tickfont=dict(color="#555a72", size=10),
            tickprefix="$",
        ),
        legend=dict(
            bgcolor="rgba(19,22,31,0.9)", bordercolor="#252840",
            borderwidth=1, font=dict(color="#8890b0", size=10),
            orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.01,
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#191c28", bordercolor="#252840",
                        font=dict(color="#e8eaf6", family="Inter")),
    )
    return fig


# ══════════════════════════════════════════════════════════════════
#  HTML COMPONENT HELPERS
# ══════════════════════════════════════════════════════════════════
_SIG_COLOR  = {"buy": "#00e676", "sell": "#ff1744", "hold": "#ffab00"}
_SIG_BG     = {"buy": "rgba(0,230,118,0.12)", "sell": "rgba(255,23,68,0.12)", "hold": "rgba(255,171,0,0.12)"}
_SIG_BORDER = {"buy": "rgba(0,230,118,0.28)", "sell": "rgba(255,23,68,0.28)", "hold": "rgba(255,171,0,0.28)"}
_SIG_GSTART = {"buy": "#00897b", "sell": "#c62828", "hold": "#e65100"}
_SIG_ICON   = {"buy": "↗", "sell": "↘", "hold": "→"}


def signal_card_html(s: dict) -> str:
    sig   = s["signal"]
    sid   = s["id"]
    col   = _SIG_COLOR[sig]
    bg    = _SIG_BG[sig]
    bord  = _SIG_BORDER[sig]
    gs    = _SIG_GSTART[sig]
    icon  = _SIG_ICON[sig]
    pct   = s["confidence"]
    price = f"${s['price']:.2f}"
    con_html = (
        '<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 7px;'
        'background:rgba(255,255,255,0.05);border:1px solid #252840;border-radius:5px;'
        'font-size:10px;color:#555a72;">&#10003; Consensus</span>'
    ) if s["consensus"] else ""

    return (
        f'<div id="signal-card-{sid}" style="'
        f'background:#191c28;border:1px solid #252840;border-left:3px solid {col};'
        f'border-radius:10px;padding:12px 13px;margin-bottom:8px;'
        f'transition:border-color .2s;">'

        # Row 1: ticker + badge + time
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:7px;">'
        f'<span id="ticker-{sid}" style="font-size:15px;font-weight:700;'
        f'font-family:JetBrains Mono,monospace;color:#e8eaf6;">{s["ticker"]}</span>'
        f'<span id="signal-type-{sid}" style="display:inline-flex;align-items:center;gap:3px;'
        f'padding:2px 8px;border-radius:5px;font-size:10px;font-weight:700;letter-spacing:.05em;'
        f'background:{bg};color:{col};border:1px solid {bord};">{icon} {sig.upper()}</span>'
        f'{con_html}'
        f'</div>'
        f'<span id="signal-time-{sid}" style="font-size:10px;color:#555a72;">{s["time"]}</span>'
        f'</div>'

        # Row 2: model + price
        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
        f'<div><div style="font-size:10px;color:#555a72;">Model</div>'
        f'<div id="signal-model-{sid}" style="font-size:12px;font-weight:500;color:#8890b0;">{s["model"]}</div></div>'
        f'<div style="text-align:right"><div style="font-size:10px;color:#555a72;">Price</div>'
        f'<div id="signal-price-{sid}" style="font-size:13px;font-weight:600;'
        f'font-family:JetBrains Mono,monospace;color:#e8eaf6;">{price}</div></div>'
        f'</div>'

        # Row 3: confidence label + pct
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="font-size:10px;color:#555a72;">Confidence &#9432;</span>'
        f'<span id="signal-confidence-{sid}" style="font-size:12px;font-weight:700;'
        f'font-family:JetBrains Mono,monospace;color:{col};">{pct}%</span>'
        f'</div>'

        # Progress bar
        f'<div style="height:4px;background:#1f2235;border-radius:2px;overflow:hidden;">'
        f'<div id="signal-bar-{sid}" style="height:100%;width:{pct}%;border-radius:2px;'
        f'background:linear-gradient(90deg,{gs},{col});'
        f'box-shadow:0 0 8px {col}55;"></div>'
        f'</div>'
        f'</div>'
    )


def metric_card_html(emoji, label, value, delta, is_up, val_id, delta_id, glow_color):
    delta_color = "#00e676" if is_up else "#ff1744"
    arrow       = "▲" if is_up else "▼"
    return (
        f'<div style="background:#191c28;border:1px solid #252840;border-radius:10px;'
        f'padding:12px 16px;display:flex;align-items:center;gap:12px;">'
        f'<div style="width:36px;height:36px;border-radius:9px;background:{glow_color}18;'
        f'display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;">'
        f'{emoji}</div>'
        f'<div><div style="font-size:10px;color:#555a72;margin-bottom:2px;">{label}</div>'
        f'<div style="display:flex;align-items:baseline;gap:8px;">'
        f'<span id="{val_id}" style="font-size:20px;font-weight:700;'
        f'font-family:JetBrains Mono,monospace;color:#e8eaf6;">{value}</span>'
        f'<span id="{delta_id}" style="font-size:11px;font-weight:600;'
        f'font-family:JetBrains Mono,monospace;color:{delta_color};">{arrow} {delta}</span>'
        f'</div></div></div>'
    )


# ══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════
NAV = [
    ("Dashboard",   "📊"),
    ("Market",      "📈"),
    ("Backtesting", "🧪"),
    ("Alerts",      "🔔"),
]

with st.sidebar:
    # Logo
    st.markdown(
        '<div style="width:44px;height:44px;border-radius:11px;margin:12px auto 20px;'
        'background:linear-gradient(135deg,#00e676,#00897b);display:flex;align-items:center;'
        'justify-content:center;font-weight:800;font-size:15px;color:#000;'
        'box-shadow:0 0 20px rgba(0,230,118,.4);">AI</div>',
        unsafe_allow_html=True,
    )

    for label, emoji in NAV:
        is_active = st.session_state.nav == label
        bg  = "background:rgba(0,230,118,.1);border-color:rgba(0,230,118,.25);color:#00e676;" if is_active else "color:#555a72;"
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:center;'
            f'width:56px;height:54px;border-radius:10px;margin:2px auto;cursor:pointer;'
            f'border:1px solid transparent;{bg}justify-content:center;gap:2px;'
            f'font-size:11px;text-transform:uppercase;font-weight:500;letter-spacing:.03em;">'
            f'<span style="font-size:20px;">{emoji}</span>{label[:5]}</div>',
            unsafe_allow_html=True,
        )
        if st.button(label, key=f"nav_{label}", help=label):
            st.session_state.nav = label
            st.rerun()

    st.markdown("<br>" * 3, unsafe_allow_html=True)
    st.markdown(
        '<div style="width:38px;height:38px;border-radius:50%;margin:0 auto;'
        'background:linear-gradient(135deg,#00897b,#2979ff);display:flex;'
        'align-items:center;justify-content:center;font-weight:700;font-size:13px;color:#fff;'
        'border:2px solid #353860;">JD</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
h_left, h_mid, h_right = st.columns([2, 3, 2])

with h_left:
    ticker_select = st.selectbox(
        "Ticker",
        list(TICKERS.keys()),
        index=list(TICKERS.keys()).index(st.session_state.ticker),
        key="ticker_box",
        label_visibility="collapsed",
    )
    st.session_state.ticker = ticker_select

with h_mid:
    st.markdown(
        '<div style="background:#11131a;border:1px solid #252840;border-radius:8px;'
        'padding:7px 14px 7px 12px;display:flex;align-items:center;gap:8px;color:#555a72;">'
        '<span style="font-size:13px;">🔍</span>'
        '<span style="font-size:12px;">Search tickers (e.g., AAPL, SPY)…</span>'
        '</div>',
        unsafe_allow_html=True,
    )

with h_right:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:flex-end;gap:10px;padding-top:4px;">'
        '<div style="display:flex;align-items:center;gap:7px;background:rgba(0,230,118,.08);'
        'border:1px solid rgba(0,230,118,.25);border-radius:20px;padding:5px 14px;">'
        '<span style="width:7px;height:7px;border-radius:50%;background:#00e676;'
        'box-shadow:0 0 6px #00e676;display:inline-block;animation:pulse-dot 2s infinite;"></span>'
        '<span id="api-status-text" style="font-size:12px;font-weight:600;color:#00e676;">API Connected</span>'
        '</div>'
        '<span style="font-size:20px;color:#555a72;cursor:pointer;position:relative;" title="Notifications">'
        '🔔<span style="position:absolute;top:-2px;right:-2px;width:7px;height:7px;'
        'background:#ff1744;border-radius:50%;border:2px solid #0d0f14;display:block;"></span>'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  MAIN 3-column layout: [chart (7) | right panel (3)]
# ══════════════════════════════════════════════════════════════════
tick      = TICKERS[ticker_select]
sent      = tick["sentiment"]
chg       = tick["change"]
chg_sign  = "+" if chg >= 0 else ""
chg_color = "#00e676" if chg >= 0 else "#ff1744"
sent_label, sent_color = (
    ("Bullish", "#00e676") if sent > 60 else
    ("Bearish", "#ff1744") if sent < 40 else
    ("Neutral", "#ffab00")
)

col_main, col_right = st.columns([7, 3], gap="medium")

# ─────────────────────────────────────────────────────────────────
#  LEFT: ticker + chart + metrics
# ─────────────────────────────────────────────────────────────────
with col_main:

    # ── Ticker header ────────────────────────────────────────────
    t1, t2 = st.columns([1, 1])
    with t1:
        st.markdown(
            f'<div style="padding:4px 0 2px;">'
            f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
            f'<span id="ticker-name-text" style="font-size:20px;font-weight:700;color:#e8eaf6;">'
            f'{ticker_select} — {tick["name"]}</span>'
            f'<div style="display:inline-flex;align-items:center;gap:7px;background:#191c28;'
            f'border:1px solid #252840;border-radius:6px;padding:4px 10px;">'
            f'<div style="width:46px;height:4px;border-radius:2px;'
            f'background:linear-gradient(90deg,#ff1744 0%,#ffab00 50%,#00e676 100%);position:relative;">'
            f'<div style="position:absolute;top:-3px;left:{sent}%;transform:translateX(-50%);'
            f'width:10px;height:10px;border-radius:50%;background:{sent_color};'
            f'border:2px solid #191c28;box-shadow:0 0 6px {sent_color};"></div>'
            f'</div>'
            f'<span id="sentiment-label" style="font-size:11px;font-weight:600;color:{sent_color};">'
            f'{sent_label}</span>'
            f'</div></div>'
            f'<div style="display:flex;align-items:center;gap:10px;margin-top:5px;">'
            f'<span style="color:#00e676;font-size:14px;">⊕</span>'
            f'<span id="ticker-price" style="font-size:20px;font-weight:700;color:#00e676;'
            f'font-family:JetBrains Mono,monospace;">${tick["price"]:.2f}</span>'
            f'<span id="ticker-change" style="font-size:13px;font-weight:600;color:{chg_color};">'
            f'({chg_sign}{chg}%)</span>'
            f'<span id="ticker-last-updated" style="font-size:11px;color:#555a72;">Last updated: 2 min ago</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with t2:
        # Timeframe selector
        tf = st.radio(
            "Timeframe",
            ["1H", "4H", "1D", "1W", "1M"],
            index=["1H", "4H", "1D", "1W", "1M"].index(st.session_state.tf),
            horizontal=True,
            key="tf_radio",
            label_visibility="collapsed",
        )
        st.session_state.tf = tf

    # ── Chart ────────────────────────────────────────────────────
    fig = build_chart(ticker_select, st.session_state.tf)
    st.plotly_chart(fig, width="stretch", config={
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "toImageButtonOptions": {"format": "png", "filename": f"{ticker_select}_signal_chart"},
    })

    # ── Metrics bar ──────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    with m1:
        st.markdown(metric_card_html(
            "📊", "Sharpe Ratio",
            METRICS["sharpe"]["value"], METRICS["sharpe"]["delta"],
            METRICS["sharpe"]["up"], "metric-sharpe", "metric-sharpe-delta", "#00e676"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card_html(
            "📉", "Max Drawdown",
            METRICS["drawdown"]["value"], METRICS["drawdown"]["delta"],
            METRICS["drawdown"]["up"], "metric-drawdown", "metric-drawdown-delta", "#ff1744"
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card_html(
            "🚀", "Annualized Return",
            METRICS["ann_return"]["value"], METRICS["ann_return"]["delta"],
            METRICS["ann_return"]["up"], "metric-ann-return", "metric-ann-return-delta", "#2979ff"
        ), unsafe_allow_html=True)
    with m4:
        acc_str = METRICS["accuracy"]["value"]
        try:
            acc_val = float(acc_str.strip('%'))
        except:
            acc_val = 0.0
        
        if acc_val > 75:
            acc_color = "#00e676"  # Green
        elif acc_val >= 60:
            acc_color = "#ffab00"  # Yellow
        else:
            acc_color = "#ff1744"  # Red

        st.markdown(metric_card_html(
            "🎯", "Model Accuracy",
            acc_str, METRICS["accuracy"]["delta"],
            METRICS["accuracy"]["up"], "metric-accuracy", "metric-accuracy-delta", acc_color
        ), unsafe_allow_html=True)
    with m5:
        st.markdown(
            '<div style="background:#191c28;border:1px solid #252840;border-radius:10px;'
            'padding:12px 16px;height:100%;display:flex;align-items:center;justify-content:center;">'
            '<div style="display:flex;align-items:center;gap:8px;">'
            '<span style="width:8px;height:8px;border-radius:50%;background:#00e676;'
            'box-shadow:0 0 6px #00e676;display:inline-block;'
            'animation:pulse-dot 2s infinite;"></span>'
            '<span id="last-update-time" style="font-size:11px;color:#555a72;">Live · 30s ago</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────
#  RIGHT: Position Sizer + Live Signals
# ─────────────────────────────────────────────────────────────────
with col_right:

    # ╔══════════════════════════════════╗
    #  POSITION SIZER
    # ╚══════════════════════════════════╝
    st.markdown(
        '<div style="background:#13161f;border:1px solid #252840;border-radius:12px;padding:16px;margin-bottom:14px;">'
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px;">'
        '<span style="font-size:18px;">📐</span>'
        '<span style="font-size:14px;font-weight:600;color:#e8eaf6;">Position Sizer</span>'
        '</div>'
        '<div style="font-size:11px;color:#555a72;margin-bottom:14px;">Risk-based trade calculator</div>',
        unsafe_allow_html=True,
    )

    portfolio = st.number_input(
        "Portfolio Size ($)",
        min_value=1_000, max_value=100_000_000,
        value=100_000, step=5_000,
        key="portfolio_size",
    )
    risk_pct = st.number_input(
        "Risk per Trade (%)",
        min_value=0.1, max_value=25.0,
        value=2.0, step=0.1, format="%.1f",
        key="risk_pct",
    )
    confidence = st.slider(
        "Model Confidence (%)",
        min_value=1, max_value=100, value=85,
        key="model_conf",
    )

    # ── Live calculation ─────────────────────────────────────────
    max_risk      = portfolio * (risk_pct / 100)
    position_size = max_risk * (confidence / 100)
    pct_of_port   = (position_size / portfolio * 100) if portfolio else 0
    bar_w         = min(pct_of_port * 10, 100)

    if st.button("⚡  Calculate Position", key="calc_btn"):
        pass  # reactive — reruns automatically

    st.markdown(
        f'<div style="background:#0f111a;border:1px solid #252840;border-radius:10px;'
        f'padding:14px;margin-top:12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        f'<span style="font-size:11px;color:#8890b0;font-weight:500;">Recommended Size</span>'
        f'<span id="recommended-size" style="font-size:20px;font-weight:700;color:#00e676;'
        f'font-family:JetBrains Mono,monospace;">${position_size:,.2f}</span>'
        f'</div>'
        f'<div style="height:4px;background:#1f2235;border-radius:2px;margin-bottom:8px;">'
        f'<div id="result-bar-fill" style="height:100%;width:{bar_w:.1f}%;background:#00e676;'
        f'border-radius:2px;transition:width .4s;box-shadow:0 0 6px rgba(0,230,118,.4);"></div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#555a72;">'
        f'<span id="result-pct-label">{pct_of_port:.2f}% of portfolio</span>'
        f'<span id="result-max-risk">Max Risk: ${max_risk:,.2f}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)  # close sizer box

    # ╔══════════════════════════════════╗
    #  LIVE SIGNALS FEED
    # ╚══════════════════════════════════╝
    active_count = sum(1 for s in SIGNALS if s["signal"] != "hold")

    st.markdown(
        f'<div style="background:#13161f;border:1px solid #252840;border-radius:12px;padding:16px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:18px;">📡</span>'
        f'<span id="signals-title" style="font-size:14px;font-weight:600;color:#e8eaf6;">Live Signals</span>'
        f'</div>'
        f'<span id="active-count" style="background:rgba(0,230,118,.12);color:#00e676;'
        f'border:1px solid rgba(0,230,118,.25);border-radius:12px;padding:3px 10px;'
        f'font-size:11px;font-weight:600;">{active_count} Active</span>'
        f'</div>'
        f'<div style="font-size:11px;color:#555a72;margin-bottom:12px;">Real-time AI predictions</div>',
        unsafe_allow_html=True,
    )

    for s in SIGNALS:
        st.markdown(signal_card_html(s), unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close signals box

# ══════════════════════════════════════════════════════════════════
#  DIAGNOSTICS EXPANDER
# ══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Model Performance Diagnostics", expanded=False):
    st.markdown(
        '<div style="color:#e8eaf6; font-size:16px; font-weight:600; margin-bottom:10px;">'
        'Confusion Matrix (Last 100 Data Points)</div>', 
        unsafe_allow_html=True
    )
    
    # Synthetic Confusion Matrix Data matching prompt
    c1, c2, c3 = st.columns(3)
    c1.metric("True Positives (Correct Buys)", "28")
    c2.metric("False Positives (Bad Buys)", "4")
    c3.metric("True Negatives (Correct Holds/Sells)", "62")
    
    st.markdown("<hr style='margin: 15px 0; border-color: #252840;'>", unsafe_allow_html=True)
    st.markdown(
        '<div style="color:#e8eaf6; font-size:16px; font-weight:600; margin-bottom:10px;">'
        'Classification Report</div>', 
        unsafe_allow_html=True
    )
    
    # Synthetic Classification Report DataFrame
    report_data = {
        "Signal Type": ["BUY", "HOLD", "SELL", "Accuracy (Hit Ratio)"],
        "F1-Score": ["0.85", "0.84", "0.79", "0.82"],
        "Precision": ["0.88", "0.82", "0.81", "-"],
        "Recall": ["0.82", "0.86", "0.78", "-"],
        "Support": ["34", "45", "21", "100"]
    }
    # Create stylized dataframe using st.dataframe
    st.dataframe(pd.DataFrame(report_data), width="stretch", hide_index=True)

# ══════════════════════════════════════════════════════════════════
#  REAL-TIME ALERTING SYSTEM (Milestone 3)
# ══════════════════════════════════════════════════════════════════
from signal_engine.notifier import SlackNotifier

notifier = SlackNotifier()
notifier.process_new_signals(SIGNALS)
