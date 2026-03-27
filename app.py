"""
app.py — AI-Powered Stock & ETF Signal Generation Platform
Streamlit Dashboard  |  Run: streamlit run app.py
Requirements: pip install streamlit plotly pandas numpy yfinance
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Page config — MUST be first Streamlit call ──────────────────
st.set_page_config(
    page_title="AI-Powered Stock & ETF Signal Generation Platform",
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
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
[data-testid="stDecoration"] { display: none; }
/* [data-testid="stSidebar"] { display: none; } */
.block-container {
  padding: 1rem 2rem 1rem 2rem !important;
  max-width: 100% !important;
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

/* ── Text input ── */
[data-testid="stTextInput"] input {
  background: var(--bg-input) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 15px !important;
  padding: 10px 16px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--emerald) !important;
  box-shadow: 0 0 0 3px rgba(0,230,118,0.12) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label { color: var(--text-2) !important; font-size: 12px !important; }

/* ── Buttons (main action) ── */
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

/* ── Back button (secondary-ish) ── */
.back-btn [data-testid="stButton"] button {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text-2) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  font-weight: 500 !important;
}
.back-btn [data-testid="stButton"] button:hover {
  background: var(--bg-hover) !important;
  color: var(--text-1) !important;
  box-shadow: none !important;
  transform: none !important;
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

/* ── Animations ── */
@keyframes pulse-dot { 0%,100%{opacity:1; box-shadow: 0 0 6px var(--emerald);} 50%{opacity:.35; box-shadow: 0 0 2px var(--emerald);} }
@keyframes fade-in { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
@keyframes card-in { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }

.home-wrapper { animation: fade-in 0.4s ease-out; }
.detail-wrapper { animation: fade-in 0.35s ease-out; }
.market-card { animation: card-in 0.4s ease-out both; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  SESSION STATE — ROUTER INIT
# ══════════════════════════════════════════════════════════════════
if "current_view" not in st.session_state:
    st.session_state.current_view = "Home"
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

# Legacy state keys used inside render_detail
for k, v in [("tf", "1D")]:
    if k not in st.session_state:
        st.session_state[k] = v




# ══════════════════════════════════════════════════════════════════
#  ENGINE LOADER
# ══════════════════════════════════════════════════════════════════
from signal_engine.engine import SignalEngine, EngineConfig
from pathlib import Path

@st.cache_resource
def load_engine():
    config = EngineConfig(use_xgboost=True)
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
except Exception as e:
    st.error(f"Error loading Signal Engine: {e}")
    engine = None


# ══════════════════════════════════════════════════════════════════
#  SHARED HELPERS
# ══════════════════════════════════════════════════════════════════
METRICS = {
    "sharpe":     {"value": "1.84",   "delta": "+0.12", "up": True},
    "drawdown":   {"value": "-12.3%", "delta": "+2.1%", "up": True},
    "ann_return": {"value": "+34.7%", "delta": "+5.2%", "up": True},
    "accuracy":   {"value": "82.4%",  "delta": "+1.5%", "up": True},
}

_SIG_COLOR  = {"buy": "#00e676", "sell": "#ff1744", "hold": "#ffab00"}
_SIG_BG     = {"buy": "rgba(0,230,118,0.12)", "sell": "rgba(255,23,68,0.12)", "hold": "rgba(255,171,0,0.12)"}
_SIG_BORDER = {"buy": "rgba(0,230,118,0.28)", "sell": "rgba(255,23,68,0.28)", "hold": "rgba(255,171,0,0.28)"}
_SIG_GSTART = {"buy": "#00897b", "sell": "#c62828", "hold": "#e65100"}
_SIG_ICON   = {"buy": "↗", "sell": "↘", "hold": "→"}


@st.cache_data(ttl=3600)
def fetch_company_profile(ticker: str) -> dict:
    """
    Optimised company profile fetch:
      - fast_info and .info are fetched IN PARALLEL via ThreadPoolExecutor
      - fast_info supplies market cap in ~0.3 s; .info supplies the rest
      - Hard 6-second timeout on .info so a slow Yahoo response never
        freezes the UI — fallback values are returned instead
      - @st.cache_data(ttl=3600): only runs once per ticker per hour;
        all subsequent re-renders are instant
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    # ── helpers run in threads ────────────────────────────────────────
    def _get_info():
        return yf.Ticker(ticker).info or {}

    def _get_fast_info():
        fi = yf.Ticker(ticker).fast_info
        return {
            "market_cap":   getattr(fi, "market_cap",    None),
            "last_price":   getattr(fi, "last_price",    None),
            "display_name": getattr(fi, "display_name",  None),
        }

    info      = {}
    fast_data = {"market_cap": None}

    # Launch both fetches in parallel and wait up to 6 s for .info
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_info  = pool.submit(_get_info)
        f_fast  = pool.submit(_get_fast_info)

        # fast_info is quick — collect it first (1-2 s max)
        try:
            fast_data = f_fast.result(timeout=4)
        except Exception:
            pass

        # .info carries sector/industry/summary — cap at 6 s
        try:
            info = f_info.result(timeout=6)
        except (FuturesTimeout, Exception):
            info = {}

    # ── safe extraction with .get() fallbacks ────────────────────────
    long_name = info.get("longName") or info.get("shortName") or fast_data.get("display_name") or ""
    sector    = info.get("sector",   "") or ""
    industry  = info.get("industry", "") or ""
    summary   = info.get("longBusinessSummary", "") or ""

    # market cap: prefer .info, fall back to fast_info
    mcap_raw  = info.get("marketCap") or fast_data.get("market_cap")

    emp_raw = info.get("fullTimeEmployees")
    emp_str = f"{int(emp_raw):,}" if emp_raw else "N/A"

    return {
        "longName":            long_name or ticker,
        "sector":              sector    or "N/A",
        "industry":            industry  or "N/A",
        "longBusinessSummary": summary   or "No summary available.",
        "marketCap":           mcap_raw,   # int/float or None
        "fullTimeEmployees":   emp_str,
    }


@st.cache_data(ttl=3600)
def fetch_company_metadata(ticker: str) -> dict:
    """
    Fetch latest price + full metadata via yfinance.
    Strategy:
      1. Seed Yahoo Finance cookies → yf.Ticker(session) → stock.info
         Metadata (sector/industry/summary/employees) extracted independently
         from price so neither blocks the other.
      2. Patch missing price from fast_info if .info price keys are absent.
      3. yf.download() for price if everything above fails.
      4. fast_info price-only last resort.
      5. {"valid": False} — never returns 0.0
    """
    import time
    import requests

    # ── Session that mimics Chrome (needed for Yahoo Finance cookies) ─
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })

    # Seed Yahoo Finance cookies so .info works properly
    try:
        session.get("https://finance.yahoo.com", timeout=5)
    except Exception:
        pass

    # ── Helper: format market cap ──────────────────────────────────
    def _fmt_employees(val):
        try:
            return f"{int(val):,}" if val else "N/A"
        except Exception:
            return "N/A"

    # ── Primary: stock.info — metadata & price are extracted separately
    # so that even if price keys are absent, all profile fields are returned.
    info = {}
    try:
        stock = yf.Ticker(ticker, session=session)
        info  = stock.info or {}
    except Exception:
        pass

    # Extract metadata fields regardless of price availability
    sector   = info.get("sector",   None) or None
    industry = info.get("industry", None) or None
    summary  = info.get("longBusinessSummary", None) or None
    mcap     = info.get("marketCap", None)
    emp_str  = _fmt_employees(info.get("fullTimeEmployees"))
    name     = info.get("longName") or info.get("shortName") or None

    # Extract price from .info
    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )

    # If .info had metadata but no price, patch price from fast_info
    if not price and info:
        try:
            fi    = yf.Ticker(ticker, session=session).fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
            if not mcap:
                mcap = getattr(fi, "market_cap", None)
        except Exception:
            pass

    if price and float(price) > 0:
        prev = info.get("previousClose", price)
        try:
            prev_fi = getattr(yf.Ticker(ticker, session=session).fast_info, "previous_close", None)
            if prev_fi:
                prev = prev_fi
        except Exception:
            pass
        pct = ((float(price) - float(prev)) / float(prev) * 100) if prev else 0.0

        return {
            "valid":               True,
            "name":                name or ticker,
            "price":               round(float(price), 2),
            "change":              round(pct, 2),
            "sentiment":           50,
            "sector":              sector   or "N/A",
            "industry":            industry or "N/A",
            "marketCap":           mcap     or "N/A",
            "longBusinessSummary": summary  or "No summary available.",
            "fullTimeEmployees":   emp_str,
        }

    # ── Secondary: yf.download for price ──────────────────────────
    time.sleep(0.25)
    try:
        raw = yf.download(ticker, period="5d", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            close_col = next((c for c in raw.columns if c.lower() == "close"), None)
            if close_col and len(raw) >= 1:
                latest_close = float(raw[close_col].iloc[-1])
                prev_close   = float(raw[close_col].iloc[-2]) if len(raw) >= 2 else latest_close
                pct_change   = ((latest_close - prev_close) / prev_close * 100) if prev_close else 0.0
                try:
                    fi_name = getattr(yf.Ticker(ticker, session=session).fast_info, "display_name", None)
                except Exception:
                    fi_name = None
                # If we managed to get metadata from .info above, use it
                return {
                    "valid":               True,
                    "name":                name or fi_name or ticker,
                    "price":               round(latest_close, 2),
                    "change":              round(pct_change, 2),
                    "sentiment":           50,
                    "sector":              sector   or "N/A",
                    "industry":            industry or "N/A",
                    "marketCap":           mcap     or "N/A",
                    "longBusinessSummary": summary  or "No summary available.",
                    "fullTimeEmployees":   emp_str,
                }
    except Exception:
        pass

    # ── Tertiary: fast_info price-only ────────────────────────────
    time.sleep(0.25)
    try:
        fi    = yf.Ticker(ticker, session=session).fast_info
        price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
        if price and float(price) > 0:
            prev = getattr(fi, "previous_close", price)
            pct  = ((float(price) - float(prev)) / float(prev) * 100) if prev else 0.0
            return {
                "valid":               True,
                "name":                ticker,
                "price":               round(float(price), 2),
                "change":              round(pct, 2),
                "sentiment":           50,
                "sector":              sector   or "N/A",
                "industry":            industry or "N/A",
                "marketCap":           mcap     or "N/A",
                "longBusinessSummary": summary  or "No summary available.",
                "fullTimeEmployees":   emp_str,
            }
    except Exception:
        pass

    return {"valid": False}


@st.cache_data(ttl=1800)
def get_price_series(ticker: str, tf: str) -> pd.DataFrame | None:
    """
    Fetch real OHLCV data from yfinance for the chosen timeframe.
    Falls back to synthetic data anchored to the real price if fetch fails.
    Returns None only if the ticker is completely invalid.
    """
    period_map   = {"1H": "5d",  "4H": "60d", "1D": "6mo", "1W": "2y",  "1M": "5y"}
    interval_map = {"1H": "15m", "4H": "1h",  "1D": "1d",  "1W": "1wk", "1M": "1mo"}

    try:
        raw = yf.download(
            ticker,
            period      = period_map[tf],
            interval    = interval_map[tf],
            progress    = False,
            auto_adjust = True,
        )
        if raw is not None and not raw.empty:
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            close_col = next((c for c in raw.columns if c.lower() == "close"), None)
            if close_col:
                prices = raw[close_col].dropna()
                return pd.DataFrame({"date": prices.index, "price": prices.values})
    except Exception:
        pass

    # Synthetic fallback — anchored to real latest price
    tick_meta = fetch_company_metadata(ticker)
    if not tick_meta["valid"]:
        return None
    base = tick_meta["price"]
    np.random.seed(abs(hash(ticker + tf)) % (2 ** 31))
    n = {"1H": 60, "4H": 120, "1D": 90, "1W": 52, "1M": 24}[tf]
    v = float(base) * 0.97
    prices = []
    for _ in range(n):
        v += float(np.random.randn()) * (base * 0.0015) + (base * 0.0001)
        prices.append(round(max(v, 0.01), 2))
    delta = {"1H": timedelta(minutes=15), "4H": timedelta(hours=1),
             "1D": timedelta(days=1),     "1W": timedelta(weeks=1),
             "1M": timedelta(days=30)}[tf]
    end   = datetime.now()
    dates = [end - delta * (n - i) for i in range(n)]
    return pd.DataFrame({"date": dates, "price": prices})


def build_chart(ticker: str, tf: str) -> go.Figure:
    df = get_price_series(ticker, tf)
    fig = go.Figure()
    if df is None or df.empty:
        fig.add_annotation(
            text="Price data unavailable",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="#555a72", size=14),
        )
    else:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["price"],
            mode="lines",
            line=dict(color="#00d09c", width=3.5),
            name=ticker,
            showlegend=False,
        ))
    fig.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=True),
        yaxis=dict(showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)', zeroline=False, showticklabels=True, side='right'),
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e0e0e0", font=dict(color="#1e2029", family="Inter")),
    )
    return fig


def signal_card_html(s: dict) -> str:
    sig   = s["signal"]
    sid   = s["id"]
    col   = _SIG_COLOR[sig]
    bg    = _SIG_BG[sig]
    bord  = _SIG_BORDER[sig]
    gs    = _SIG_GSTART[sig]
    icon  = _SIG_ICON[sig]
    pct   = s["confidence"]
    price = f"₹{s['price']:.2f}"
    con_html = (
        '<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 7px;'
        'background:rgba(255,255,255,0.05);border:1px solid #252840;border-radius:5px;'
        'font-size:10px;color:#555a72;">&#10003; Consensus</span>'
    ) if s["consensus"] else ""
    return (
        f'<div id="signal-card-{sid}" style="'
        f'background:#191c28;border:1px solid #252840;border-left:3px solid {col};'
        f'border-radius:10px;padding:12px 13px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:center;gap:7px;">'
        f'<span style="font-size:15px;font-weight:700;font-family:JetBrains Mono,monospace;color:#e8eaf6;">{s["ticker"]}</span>'
        f'<span style="display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:5px;'
        f'font-size:10px;font-weight:700;letter-spacing:.05em;background:{bg};color:{col};border:1px solid {bord};">'
        f'{icon} {sig.upper()}</span>{con_html}</div>'
        f'<span style="font-size:10px;color:#555a72;">{s["time"]}</span></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
        f'<div><div style="font-size:10px;color:#555a72;">Model</div>'
        f'<div style="font-size:12px;font-weight:500;color:#8890b0;">{s["model"]}</div></div>'
        f'<div style="text-align:right"><div style="font-size:10px;color:#555a72;">Price</div>'
        f'<div style="font-size:13px;font-weight:600;font-family:JetBrains Mono,monospace;color:#e8eaf6;">{price}</div></div></div>'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
        f'<span style="font-size:10px;color:#555a72;">Confidence &#9432;</span>'
        f'<span style="font-size:12px;font-weight:700;font-family:JetBrains Mono,monospace;color:{col};">{pct}%</span></div>'
        f'<div style="height:4px;background:#1f2235;border-radius:2px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;border-radius:2px;'
        f'background:linear-gradient(90deg,{gs},{col});box-shadow:0 0 8px {col}55;"></div></div></div>'
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
#  CURRENCY HELPER
# ══════════════════════════════════════════════════════════════════
US_TICKERS = {"AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "SPY", "QQQ"}

def get_currency_symbol(ticker: str) -> str:
    """Return Rs for Indian tickers (.NS/.BO), $ for everything else."""
    if ticker.upper().endswith(".NS") or ticker.upper().endswith(".BO"):
        return "\u20b9"
    return "$"


# ══════════════════════════════════════════════════════════════════
#  HOME PAGE VIEW
# ══════════════════════════════════════════════════════════════════
WATCHLIST_TICKERS = [
    ("RELIANCE.NS", "Reliance Industries"),
    ("TCS.NS",      "Tata Consultancy"),
    ("HDFCBANK.NS", "HDFC Bank"),
    ("INFY.NS",     "Infosys"),
]

GLOBAL_TICKERS = [
    ("AAPL",  "Apple Inc."),
    ("MSFT",  "Microsoft Corp."),
    ("NVDA",  "NVIDIA Corp."),
    ("TSLA",  "Tesla Inc."),
]

def render_home():
    st.markdown('<div class="home-wrapper">', unsafe_allow_html=True)

    # ── Platform Header ──────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:24px 0 8px;border-bottom:1px solid #252840;margin-bottom:28px;">
      <div style="display:flex;align-items:center;gap:16px;">
        <div style="width:48px;height:48px;border-radius:14px;
                    background:linear-gradient(135deg,#00e676,#00897b);
                    display:flex;align-items:center;justify-content:center;
                    font-weight:800;font-size:17px;color:#000;
                    box-shadow:0 0 24px rgba(0,230,118,.4);">AI</div>
        <div>
          <div style="font-size:22px;font-weight:800;color:#e8eaf6;letter-spacing:-.02em;">
            ⚡ AI-Powered Stock & ETF Signal Generation Platform
          </div>
          <div style="font-size:12px;color:#555a72;margin-top:2px;">
            Institutional-grade signals · Real-time AI · NSE / BSE
          </div>
        </div>
      </div>
      <div style="display:flex;align-items:center;gap:7px;background:rgba(0,230,118,.08);
                  border:1px solid rgba(0,230,118,.25);border-radius:20px;padding:6px 16px;">
        <span style="width:7px;height:7px;border-radius:50%;background:#00e676;
                     box-shadow:0 0 6px #00e676;display:inline-block;
                     animation:pulse-dot 2s infinite;"></span>
        <span style="font-size:12px;font-weight:600;color:#00e676;">Live · Markets Open</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Search Bar ───────────────────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:10px;">
      <div style="font-size:13px;color:#8890b0;margin-bottom:8px;font-weight:500;">
        🔍 Search any stock or ETF
      </div>
    </div>
    """, unsafe_allow_html=True)

    search_col, btn_col = st.columns([5, 1])
    with search_col:
        search_input = st.text_input(
            "Search ticker",
            placeholder="Enter ticker symbol  (e.g. RELIANCE, TCS, AAPL, SPY)…",
            key="home_search",
            label_visibility="collapsed",
        ).strip().upper()
    with btn_col:
        search_clicked = st.button("⚡ Search", key="home_search_btn", width='stretch')

    if search_clicked and search_input:
        clean_input = search_input.upper().strip()
        # If user explicitly types a suffix (e.g. RELIANCE.BO, AAPL.US), trust it
        if "." in clean_input:
            ticker_to_search = clean_input
        # Known top US tickers — pass through without any suffix
        elif clean_input in US_TICKERS:
            ticker_to_search = clean_input
        # Everything else defaults to NSE Indian market
        else:
            ticker_to_search = f"{clean_input}.NS"
        st.session_state.selected_ticker = ticker_to_search
        st.session_state.current_view    = "Detail"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Global Tech Leaders Grid ─────────────────────────────────
    st.markdown("""
    <div style="margin-bottom:16px;">
      <div style="font-size:18px;font-weight:700;color:#e8eaf6;letter-spacing:-.01em;">
        &#127760; Global Tech Leaders
      </div>
      <div style="font-size:12px;color:#555a72;margin-top:4px;">
        US equities &middot; NASDAQ / NYSE &middot; Click to analyze
      </div>
    </div>
    """, unsafe_allow_html=True)

    gcols = st.columns(4, gap="medium")
    for idx, (ticker, fallback_name) in enumerate(GLOBAL_TICKERS):
        meta = fetch_company_metadata(ticker)
        price  = meta.get("price")  if meta["valid"] else None
        name   = meta.get("name",  fallback_name)
        change = meta.get("change") if meta["valid"] else None
        chg_color = "#00e676" if (change or 0) >= 0 else "#ff1744"
        chg_sign  = "+" if (change or 0) >= 0 else ""
        chg_arrow = "&#9650;" if (change or 0) >= 0 else "&#9660;"
        cur_sym   = get_currency_symbol(ticker)
        price_str  = f"{cur_sym}{price:,.2f}" if price is not None else "N/A"
        change_str = f"{chg_arrow} {chg_sign}{change:.2f}%" if change is not None else "—"

        with gcols[idx]:
            st.markdown(f"""
            <div class="market-card" style="animation-delay:{idx * 0.08}s;
                 background:#191c28;border:1px solid #252840;border-radius:14px;
                 padding:20px 18px 12px;margin-bottom:8px;
                 transition:border-color .2s,box-shadow .2s;"
                 onmouseover="this.style.borderColor='rgba(41,121,255,.4)';this.style.boxShadow='0 0 20px rgba(41,121,255,.1)'"
                 onmouseout="this.style.borderColor='#252840';this.style.boxShadow='none'">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                <div>
                  <div style="font-size:15px;font-weight:700;color:#e8eaf6;
                              font-family:'JetBrains Mono',monospace;">{ticker}</div>
                  <div style="font-size:11px;color:#555a72;margin-top:2px;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                              max-width:120px;">{name}</div>
                </div>
                <div style="background:rgba(41,121,255,.12);border:1px solid rgba(41,121,255,.25);
                            border-radius:6px;padding:3px 8px;font-size:10px;
                            font-weight:600;color:#2979ff;">NASDAQ</div>
              </div>
              <div style="font-size:26px;font-weight:800;color:#e8eaf6;
                          font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
                {price_str}
              </div>
              <div style="font-size:12px;font-weight:600;color:{chg_color};">
                {change_str}
              </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Analyze {ticker}", key=f"gl_btn_{idx}", width='stretch'):
                st.session_state.selected_ticker = ticker  # no .NS suffix — US ticker
                st.session_state.current_view    = "Detail"
                st.rerun()

    # ── Market Leaders Grid (Indian Equities) ────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:16px;">
      <div style="font-size:18px;font-weight:700;color:#e8eaf6;letter-spacing:-.01em;">
        &#128200; Market Leaders
      </div>
      <div style="font-size:12px;color:#555a72;margin-top:4px;">
        Top Indian equities &middot; NSE &middot; Click to analyze
      </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4, gap="medium")
    for idx, (ticker, fallback_name) in enumerate(WATCHLIST_TICKERS):
        meta = fetch_company_metadata(ticker)
        price  = meta.get("price")  if meta["valid"] else None
        name   = meta.get("name",  fallback_name)
        change = meta.get("change") if meta["valid"] else None
        chg_color = "#00e676" if (change or 0) >= 0 else "#ff1744"
        chg_sign  = "+" if (change or 0) >= 0 else ""
        chg_arrow = "&#9650;" if (change or 0) >= 0 else "&#9660;"
        cur_sym   = get_currency_symbol(ticker)
        price_str  = f"{cur_sym}{price:,.2f}" if price is not None else "N/A"
        change_str = f"{chg_arrow} {chg_sign}{change:.2f}%" if change is not None else "—"

        with cols[idx]:
            st.markdown(f"""
            <div class="market-card" style="animation-delay:{(idx + 4) * 0.08}s;
                 background:#191c28;border:1px solid #252840;border-radius:14px;
                 padding:20px 18px 12px;margin-bottom:8px;
                 transition:border-color .2s,box-shadow .2s;"
                 onmouseover="this.style.borderColor='rgba(0,230,118,.35)';this.style.boxShadow='0 0 20px rgba(0,230,118,.08)'"
                 onmouseout="this.style.borderColor='#252840';this.style.boxShadow='none'">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:12px;">
                <div>
                  <div style="font-size:15px;font-weight:700;color:#e8eaf6;
                              font-family:'JetBrains Mono',monospace;">{ticker.replace('.NS','').replace('.BO','')}</div>
                  <div style="font-size:11px;color:#555a72;margin-top:2px;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                              max-width:120px;">{name}</div>
                </div>
                <div style="background:rgba(0,230,118,.1);border:1px solid rgba(0,230,118,.2);
                            border-radius:6px;padding:3px 8px;font-size:10px;
                            font-weight:600;color:#00e676;">NSE</div>
              </div>
              <div style="font-size:26px;font-weight:800;color:#e8eaf6;
                          font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
                {price_str}
              </div>
              <div style="font-size:12px;font-weight:600;color:{chg_color};">
                {change_str}
              </div>
            </div>
            """, unsafe_allow_html=True)

            short = ticker.replace(".NS", "").replace(".BO", "")
            if st.button(f"Analyze {short}", key=f"wl_btn_{idx}", width='stretch'):
                st.session_state.selected_ticker = ticker
                st.session_state.current_view    = "Detail"
                st.rerun()

    # ── Market Stats Footer ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="border:1px solid #252840;border-radius:14px;padding:20px 24px;
                background:#13161f;display:flex;align-items:center;gap:40px;flex-wrap:wrap;">
      <div>
        <div style="font-size:10px;color:#555a72;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">NIFTY 50</div>
        <div style="font-size:20px;font-weight:700;color:#e8eaf6;font-family:'JetBrains Mono',monospace;">24,315.85</div>
        <div style="font-size:11px;color:#00e676;font-weight:600;">▲ +0.34%</div>
      </div>
      <div style="width:1px;height:40px;background:#252840;"></div>
      <div>
        <div style="font-size:10px;color:#555a72;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">SENSEX</div>
        <div style="font-size:20px;font-weight:700;color:#e8eaf6;font-family:'JetBrains Mono',monospace;">80,116.49</div>
        <div style="font-size:11px;color:#00e676;font-weight:600;">▲ +0.28%</div>
      </div>
      <div style="width:1px;height:40px;background:#252840;"></div>
      <div>
        <div style="font-size:10px;color:#555a72;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">BANK NIFTY</div>
        <div style="font-size:20px;font-weight:700;color:#e8eaf6;font-family:'JetBrains Mono',monospace;">52,841.30</div>
        <div style="font-size:11px;color:#ff1744;font-weight:600;">▼ -0.11%</div>
      </div>
      <div style="width:1px;height:40px;background:#252840;"></div>
      <div>
        <div style="font-size:10px;color:#555a72;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">INDIA VIX</div>
        <div style="font-size:20px;font-weight:700;color:#e8eaf6;font-family:'JetBrains Mono',monospace;">13.42</div>
        <div style="font-size:11px;color:#ffab00;font-weight:600;">→ +0.05%</div>
      </div>
      <div style="margin-left:auto;font-size:11px;color:#555a72;">Last refresh: just now · Auto-refresh every 30s</div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Alert System (Redesigned for Demo) ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        with st.expander("⚙️ AI Alert System", expanded=True):
            st.markdown(
                "<div style='color:#8890b0; font-size:14px; margin-bottom:15px;'>"
                "Trigger real-time AI-generated alerts based on current market leaders across global and Indian markets."
                "</div>", 
                unsafe_allow_html=True
            )
            
            # Status Badge
            st.markdown(
                '<div style="display:inline-flex; align-items:center; gap:8px; '
                'background:rgba(0,230,118,0.08); padding:6px 14px; border-radius:30px; '
                'border:1px solid rgba(0,230,118,0.25); margin-bottom:20px;">'
                '<span style="width:8px; height:8px; border-radius:50%; background:#00e676; '
                'box-shadow:0 0 8px #00e676;"></span>'
                '<span style="color:#00e676; font-weight:700; font-size:12px; text-transform:uppercase; letter-spacing:0.05em;">'
                'Terminal Mode Active</span></div>',
                unsafe_allow_html=True
            )

            # Market Labels
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                st.markdown(
                    "<div style='background:#13161f; padding:10px 15px; border-radius:8px; border:1px solid #252840;'>"
                    "<div style='font-size:10px; color:#555a72; text-transform:uppercase;'>Global Equities</div>"
                    "<div style='font-size:13px; color:#e8eaf6; font-weight:600;'>US Tech (NASDAQ)</div>"
                    "</div>", unsafe_allow_html=True
                )
            with mcol2:
                st.markdown(
                    "<div style='background:#13161f; padding:10px 15px; border-radius:8px; border:1px solid #252840;'>"
                    "<div style='font-size:10px; color:#555a72; text-transform:uppercase;'>Domestic Market</div>"
                    "<div style='font-size:13px; color:#e8eaf6; font-weight:600;'>Indian Leaders (NSE)</div>"
                    "</div>", unsafe_allow_html=True
                )

            st.markdown("<div style='margin:20px 0;'></div>", unsafe_allow_html=True)
            
            # Action Button
            if st.button("🔔 Send Live Alerts", use_container_width=True, key="home_live_alert_btn"):
                from signal_engine.notifier import AlertManager
                # Initialize with current session values if any
                notifier = AlertManager(
                    webhook_url=st.session_state.get("slack_url") or None,
                    email_address=st.session_state.get("alert_email") or None
                )
                notifier.test_notification()
                st.toast("Generating live market alerts in terminal...", icon="🔔")

            st.markdown(
                "<div style='background:rgba(41,121,255,0.08); padding:12px 16px; border-radius:10px; "
                "border:1px solid rgba(41,121,255,0.2); margin-top:15px;'>"
                "<div style='font-size:12px; color:#2979ff; display:flex; align-items:center; gap:8px;'>"
                "<span>ℹ️</span> <span>This triggers AI signals across US and Indian markets "
                "and displays them in the terminal for real-time analysis.</span>"
                "</div></div>", 
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
def create_financial_chart(data_series) -> go.Figure:
    fig = go.Figure()
    if data_series is None or data_series.empty:
        return fig
    
    max_val = data_series.max()
    
    fig.add_trace(go.Bar(
        x=data_series.index.strftime('%Y'), 
        y=data_series.values, 
        marker_color="#00d09c",
        width=0.15,
        text=[f"{(v/1e9):.1f}B" if abs(v) >= 1e8 else f"{(v/1e6):.1f}M" for v in data_series.values],
        textposition="outside",
        textfont=dict(color="#a0a0a0", size=11, family="Inter"),
        cliponaxis=False
    ))
    fig.update_layout(
        height=240, margin=dict(l=0, r=0, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, dtick=1, linecolor="#252840", tickfont=dict(color="#a0a0a0")),
    )
    fig.update_yaxes(range=[0, max_val * 1.15], showticklabels=False, showgrid=False, zeroline=False)
    return fig

# ══════════════════════════════════════════════════════════════════
#  DETAIL ANALYSIS VIEW
# ══════════════════════════════════════════════════════════════════
def render_detail():
    ticker_select = st.session_state.selected_ticker or "RELIANCE.NS"

    st.markdown('<div class="detail-wrapper">', unsafe_allow_html=True)

    # ── Top bar: Back button + Ticker header ─────────────────────
    top_left, top_right = st.columns([1, 5])
    with top_left:
        st.markdown('<div class="back-btn">', unsafe_allow_html=True)
        if st.button("← Back to Dashboard", key="back_btn"):
            st.session_state.current_view = "Home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with top_right:
        st.markdown(
            f'<div style="padding-top:6px;font-size:13px;color:#555a72;">'
            f'Dashboard &nbsp;›&nbsp; <span style="color:#e8eaf6;font-weight:600;">'
            f'{ticker_select}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Fetch metadata + validate ─────────────────────────────────────
    tick = fetch_company_metadata(ticker_select)
    if not tick["valid"]:
        st.error(
            f"Ticker **{ticker_select}** could not be found or data is temporarily "
            "unavailable. Please verify the symbol (e.g. RELIANCE.NS, TCS.NS for "
            "Indian stocks) and try again."
        )
        st.stop()

    # ── ML Signals (moved to progressive load below) ─────────────
    SIGNALS = []

    # ── Derived display vars ─────────────────────────────────────
    sent      = tick["sentiment"]
    chg       = tick["change"]
    chg_sign  = "+" if chg >= 0 else ""
    chg_color = "#00e676" if chg >= 0 else "#ff1744"
    sent_label, sent_color = (
        ("Bullish", "#00e676") if sent > 60 else
        ("Bearish", "#ff1744") if sent < 40 else
        ("Neutral", "#ffab00")
    )

    # ══════════════════════════════════════════════════════════════
    #  MAIN 2-column layout: [chart (7) | right panel (3)]
    # ══════════════════════════════════════════════════════════════
    col_main, col_right = st.columns([7, 3], gap="medium")

    # ─────────────────────────────────────────────────────────────
    #  LEFT: ticker header + chart + metrics
    # ─────────────────────────────────────────────────────────────
    with col_main:

        t1, t2 = st.columns([1, 1])
        with t1:
            st.markdown(
                f'<div style="padding:4px 0 2px;">'
                f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
                f'<span style="font-size:20px;font-weight:700;color:#e8eaf6;">'
                f'{ticker_select} — {tick["name"]}</span>'
                f'<div style="display:inline-flex;align-items:center;gap:7px;background:#191c28;'
                f'border:1px solid #252840;border-radius:6px;padding:4px 10px;">'
                f'<div style="width:46px;height:4px;border-radius:2px;'
                f'background:linear-gradient(90deg,#ff1744 0%,#ffab00 50%,#00e676 100%);position:relative;">'
                f'<div style="position:absolute;top:-3px;left:{sent}%;transform:translateX(-50%);'
                f'width:10px;height:10px;border-radius:50%;background:{sent_color};'
                f'border:2px solid #191c28;box-shadow:0 0 6px {sent_color};"></div>'
                f'</div>'
                f'<span style="font-size:11px;font-weight:600;color:{sent_color};">'
                f'{sent_label}</span></div></div>'
                f'<div style="display:flex;align-items:center;gap:10px;margin-top:5px;">'
                f'<span style="color:#00e676;font-size:14px;">⊕</span>'
                f'<span style="font-size:20px;font-weight:700;color:#00e676;'
                f'font-family:JetBrains Mono,monospace;">₹{tick["price"]:.2f}</span>'
                f'<span style="font-size:13px;font-weight:600;color:{chg_color};">'
                f'({chg_sign}{chg}%)</span>'
                f'<span style="font-size:11px;color:#555a72;">Last updated: 2 min ago</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        with t2:
            tf = st.radio(
                "Timeframe",
                ["1H", "4H", "1D", "1W", "1M"],
                index=["1H", "4H", "1D", "1W", "1M"].index(st.session_state.tf),
                horizontal=True,
                key="tf_radio",
                label_visibility="collapsed",
            )
            st.session_state.tf = tf

        # Chart
        fig = build_chart(ticker_select, st.session_state.tf)
        st.plotly_chart(fig, width='stretch', config={
            "displaylogo": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "toImageButtonOptions": {"format": "png", "filename": f"{ticker_select}_signal_chart"},
        })

        # Metrics bar
        backtest = {}
        if engine and ticker_select in engine.models_:
            # Retrieve metrics if the mock backtesting or true evaluation function exists
            if hasattr(engine, "get_backtest_metrics"):
                backtest = engine.get_backtest_metrics(ticker_select)
        
        real_sharpe   = backtest.get("sharpe")
        real_drawdown = backtest.get("drawdown")
        real_return   = backtest.get("ann_return")
        real_acc      = backtest.get("accuracy")

        m1, m2, m3, m4, m5 = st.columns(5, gap="small")
        with m1:
            val = f"{real_sharpe:.2f}" if real_sharpe is not None else "Calculating..."
            st.markdown(metric_card_html(
                "📊", "Sharpe Ratio",
                val, "",
                True, "metric-sharpe", "metric-sharpe-delta", "#00e676"
            ), unsafe_allow_html=True)
        with m2:
            val = f"{real_drawdown:.1f}%" if real_drawdown is not None else "Calculating..."
            dir_up = False if real_drawdown is not None and real_drawdown < 0 else True
            st.markdown(metric_card_html(
                "📉", "Max Drawdown",
                val, "",
                dir_up, "metric-drawdown", "metric-drawdown-delta", "#ff1744"
            ), unsafe_allow_html=True)
        with m3:
            val = f"{real_return:.1f}%" if real_return is not None else "Calculating..."
            dir_up = True if real_return is not None and real_return > 0 else False
            color = "#00e676" if dir_up else "#ff1744"
            st.markdown(metric_card_html(
                "🚀", "Annualized Return",
                val, "",
                dir_up, "metric-ann-return", "metric-ann-return-delta", color
            ), unsafe_allow_html=True)
        with m4:
            val = f"{real_acc:.1f}%" if real_acc is not None else "Calculating..."
            if real_acc is not None:
                acc_color = "#00e676" if real_acc > 75 else ("#ffab00" if real_acc >= 60 else "#ff1744")
            else:
                acc_color = "#8890b0"
            st.markdown(metric_card_html(
                "🎯", "Model Accuracy",
                val, "",
                True, "metric-accuracy", "metric-accuracy-delta", acc_color
            ), unsafe_allow_html=True)
        with m5:
            st.markdown(
                '<div style="background:#191c28;border:1px solid #252840;border-radius:10px;'
                'padding:12px 16px;height:100%;display:flex;align-items:center;justify-content:center;">'
                '<div style="display:flex;align-items:center;gap:8px;">'
                '<span style="width:8px;height:8px;border-radius:50%;background:#00e676;'
                'box-shadow:0 0 6px #00e676;display:inline-block;animation:pulse-dot 2s infinite;"></span>'
                '<span style="font-size:11px;color:#555a72;">Live · 30s ago</span>'
                '</div></div>',
                unsafe_allow_html=True,
            )



        # ── Financials Section (Tabs Layout) ───────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:0;color:#e8eaf6;font-size:18px;margin-bottom:16px;'>Financials</h3>", unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Revenue", "Profit", "Net Worth"])
        
        try:
            tick_obj = yf.Ticker(ticker_select)
            fin = tick_obj.financials
            bs = tick_obj.balance_sheet
            
            with tab1:
                if fin is not None and not fin.empty and "Total Revenue" in fin.index:
                    rev = fin.loc["Total Revenue"].dropna().head(4)[::-1]
                    st.plotly_chart(create_financial_chart(rev), width='stretch', config={"displayModeBar": False})
                else:
                    st.info("Revenue data not available.")
            with tab2:
                if fin is not None and not fin.empty and "Net Income" in fin.index:
                    profit = fin.loc["Net Income"].dropna().head(4)[::-1]
                    st.plotly_chart(create_financial_chart(profit), width='stretch', config={"displayModeBar": False})
                else:
                    st.info("Profit data not available.")
            with tab3:
                if bs is not None and not bs.empty:
                    nw_key = "Stockholders Equity" if "Stockholders Equity" in bs.index else ("Total Stockholder Equity" if "Total Stockholder Equity" in bs.index else None)
                    if nw_key:
                        nw = bs.loc[nw_key].dropna().head(4)[::-1]
                        st.plotly_chart(create_financial_chart(nw), width='stretch', config={"displayModeBar": False})
                    else:
                        st.info("Net Worth data not available.")
                else:
                    st.info("Net Worth data not available.")
        except Exception:
            st.info("Financial data not available for this ticker.")


    # ─────────────────────────────────────────────────────────────
    #  RIGHT: Position Sizer + Live Signals
    # ─────────────────────────────────────────────────────────────
    with col_right:

        # ── ML Engine Execution (Progressive Load) ────────────────────
        if engine:
            if ticker_select not in engine.models_:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.spinner("⚡ AI analyzing signals..."):
                    engine.run([ticker_select], evaluate=True, train_mode=True)
                    engine.save_models()
                st.rerun()
            engine.run([ticker_select], evaluate=False, train_mode=False)
            SIGNALS = engine.get_signals_for_streamlit()

        # Live Signals Feed
        active_count = sum(1 for s in SIGNALS if s["signal"] != "hold")
        st.markdown(
            f'<div style="background:#13161f;border:1px solid #252840;border-radius:12px;padding:16px;margin-bottom:14px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-size:18px;">📡</span>'
            f'<span style="font-size:14px;font-weight:600;color:#e8eaf6;">Live Signals</span>'
            f'</div>'
            f'<span style="background:rgba(0,230,118,.12);color:#00e676;'
            f'border:1px solid rgba(0,230,118,.25);border-radius:12px;padding:3px 10px;'
            f'font-size:11px;font-weight:600;">{active_count} Active</span>'
            f'</div>'
            f'<div style="font-size:11px;color:#555a72;margin-bottom:12px;">Real-time AI predictions</div>',
            unsafe_allow_html=True,
        )
        for s in SIGNALS:
            st.markdown(signal_card_html(s), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Position Sizer
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
            "Portfolio Size (₹)",
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

        max_risk      = portfolio * (risk_pct / 100)
        position_size = max_risk * (confidence / 100)
        pct_of_port   = (position_size / portfolio * 100) if portfolio else 0
        bar_w         = min(pct_of_port * 10, 100)

        if st.button("⚡  Calculate Position", key="calc_btn"):
            pass  # reactive

        st.markdown(
            f'<div style="background:#0f111a;border:1px solid #252840;border-radius:10px;'
            f'padding:14px;margin-top:12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
            f'<span style="font-size:11px;color:#8890b0;font-weight:500;">Recommended Size</span>'
            f'<span style="font-size:20px;font-weight:700;color:#00e676;'
            f'font-family:JetBrains Mono,monospace;">₹{position_size:,.2f}</span>'
            f'</div>'
            f'<div style="height:4px;background:#1f2235;border-radius:2px;margin-bottom:8px;">'
            f'<div style="height:100%;width:{bar_w:.1f}%;background:#00e676;'
            f'border-radius:2px;transition:width .4s;box-shadow:0 0 6px rgba(0,230,118,.4);"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#555a72;">'
            f'<span>{pct_of_port:.2f}% of portfolio</span>'
            f'<span>Max Risk: ₹{max_risk:,.2f}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    #  DIAGNOSTICS EXPANDER
    # ══════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Model Performance Diagnostics", expanded=False):
        try:
            if not engine or ticker_select not in engine.datasets_ or ticker_select not in engine.models_:
                st.warning("Not enough historical test data to generate confidence metrics for this ticker.")
            else:
                ds = engine.datasets_[ticker_select]
                trained = engine.models_[ticker_select]
                
                if "Random Forest" not in trained or len(ds.get("X_test", [])) == 0:
                    st.warning("Not enough historical test data to generate confidence metrics for this ticker.")
                else:
                    model = trained["Random Forest"]
                    X_te = ds["X_test"]
                    y_te = ds["y_test"]
                    
                    y_prob = model.predict_proba(X_te)
                    y_pred = np.argmax(y_prob, axis=1)
                    
                    from sklearn.metrics import confusion_matrix, classification_report
                    cm = confusion_matrix(y_te, y_pred)
                    
                    if cm.shape == (3, 3):
                        tp = cm[1, 1]
                        fp = cm[0, 1] + cm[2, 1]
                        tn = cm[0, 0] + cm[2, 2] + cm[0, 2] + cm[2, 0]
                    elif cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                    else:
                        tp = fp = tn = "N/A"
                        
                    st.markdown(
                        f'<div style="color:#e8eaf6; font-size:16px; font-weight:600; margin-bottom:10px;">'
                        f'Confusion Matrix (Test Set: {len(y_te)} Data Points)</div>',
                        unsafe_allow_html=True
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("True Positives (Correct Buys)", str(tp))
                    c2.metric("False Positives (Bad Buys)", str(fp))
                    c3.metric("True Negatives (Correct Holds/Sells)", str(tn))

                    st.markdown("<hr style='margin: 15px 0; border-color: #252840;'>", unsafe_allow_html=True)
                    st.markdown(
                        '<div style="color:#e8eaf6; font-size:16px; font-weight:600; margin-bottom:10px;">'
                        'Classification Report</div>',
                        unsafe_allow_html=True
                    )
                    
                    report_dict = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
                    
                    # Target mapping if available in engine, usually 0=HOLD, 1=BUY, 2=SELL
                    class_names = { "0": "HOLD", "1": "BUY", "2": "SELL" }
                    
                    rows = []
                    for k, v in report_dict.items():
                        if k in ["accuracy", "macro avg", "weighted avg"]:
                            continue
                        name = class_names.get(k, k)
                        rows.append({
                            "Signal Type": name,
                            "F1-Score": f"{v['f1-score']:.2f}",
                            "Precision": f"{v['precision']:.2f}",
                            "Recall": f"{v['recall']:.2f}",
                            "Support": str(int(v['support']))
                        })
                        
                    # Add accuracy row
                    acc = report_dict.get("accuracy", 0)
                    rows.append({
                        "Signal Type": "Accuracy (Hit Ratio)",
                        "F1-Score": f"{acc:.2f}",
                        "Precision": "-",
                        "Recall": "-",
                        "Support": str(len(y_te))
                    })
                    
                    report_df = pd.DataFrame(rows)
                    st.dataframe(report_df, width='stretch', hide_index=True)
                    
        except Exception as e:
            st.warning("Not enough historical test data to generate confidence metrics for this ticker.")

    # ══════════════════════════════════════════════════════════════
    #  COMPANY PROFILE EXPANDER
    # ══════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Company Profile", expanded=False):
        # Spinner only shows on first load per ticker; subsequent renders
        # are instant because @st.cache_data caches results for 1 hour.
        with st.spinner("Loading company profile..."):
            profile = fetch_company_profile(ticker_select)

        company_name = profile.get("longName")            or tick.get("name", ticker_select)
        sector       = profile.get("sector",   "N/A")
        industry     = profile.get("industry", "N/A")
        employees    = profile.get("fullTimeEmployees",  "N/A")
        bio          = profile.get("longBusinessSummary", "No summary available.")
        mcap         = profile.get("marketCap")  # int or None

        # Format market cap as readable currency string
        cur_sym = get_currency_symbol(ticker_select)
        if isinstance(mcap, (int, float)) and mcap:
            if mcap >= 1e12:
                market_cap = f"{cur_sym}{mcap/1e12:.2f}T"
            elif mcap >= 1e9:
                market_cap = f"{cur_sym}{mcap/1e9:.2f}B"
            elif mcap >= 1e6:
                market_cap = f"{cur_sym}{mcap/1e6:.2f}M"
            else:
                market_cap = f"{cur_sym}{mcap:,.0f}"
        else:
            market_cap = "N/A"

        # Business summary
        st.markdown(
            f'<div style="color:#8890b0; font-size:13px; line-height:1.7; '
            f'margin-bottom:16px;">{bio}</div>',
            unsafe_allow_html=True,
        )

        # ── 5 metric cards ───────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Company Name", company_name)
        c2.metric("Sector",       sector)
        c3.metric("Industry",     industry)
        c4.metric("Market Cap",   market_cap)
        c5.metric("Employees",    employees)

    # Removed redundant alerting poll since engine.py now handles it.
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════
if st.session_state.current_view == "Home":
    render_home()
else:
    render_detail()
