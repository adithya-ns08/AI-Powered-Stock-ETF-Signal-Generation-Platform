"""
signal_engine/data_pipeline.py
================================
OHLCV → Feature Engineering → Supervised Labels → Train/Val/Out-of-Time Split

Pipeline steps
--------------
1. Download or accept raw OHLCV data.
2. Compute technical indicators.
3. Create forward-looking labels.
4. Scale features: StandardScaler for tree models.
5. Split into three chronological segments:
       TRAIN  (60%) | VALIDATION (20%) | OUT-OF-TIME TEST (20%)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SIGNAL_BUY  = 2
SIGNAL_HOLD = 1
SIGNAL_SELL = 0
LABEL_MAP   = {SIGNAL_SELL: "sell", SIGNAL_HOLD: "hold", SIGNAL_BUY: "buy"}

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bollinger(series: pd.Series, period: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]

    df["ret_1d"]   = df["Close"].pct_change(1)
    df["ret_3d"]   = df["Close"].pct_change(3)
    df["ret_5d"]   = df["Close"].pct_change(5)
    df["ret_10d"]  = df["Close"].pct_change(10)
    df["log_ret"]  = np.log(df["Close"] / df["Close"].shift(1))

    for span in [10, 20, 50, 100, 200]:
        df[f"ma{span}"] = df["Close"].rolling(span).mean()
        df[f"dev_ma{span}"] = (df["Close"] - df[f"ma{span}"]) / df[f"ma{span}"]

    df["rsi_14"] = _rsi(df["Close"], 14)
    df["rsi_7"]  = _rsi(df["Close"],  7)

    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["Close"])
    df["macd_cross"] = np.sign(df["macd"] - df["macd_signal"])

    bb_up, bb_mid, bb_lo = _bollinger(df["Close"])
    df["bb_up"]    = bb_up
    df["bb_mid"]   = bb_mid
    df["bb_lo"]    = bb_lo
    df["bb_width"] = (bb_up - bb_lo) / bb_mid
    df["bb_pct"]   = (df["Close"] - bb_lo) / (bb_up - bb_lo + 1e-9)

    df["atr_14"] = _atr(df["High"], df["Low"], df["Close"], 14)
    df["atr_pct"] = df["atr_14"] / df["Close"]

    df["vol_ma20"]   = df["Volume"].rolling(20).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma20"] + 1)
    df["vol_change"] = df["Volume"].pct_change()

    df["body"]     = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    df["upper_sh"] = (df["High"]  - df[["Open","Close"]].max(axis=1)) / (df["High"] - df["Low"] + 1e-9)
    df["lower_sh"] = (df[["Open","Close"]].min(axis=1) - df["Low"])   / (df["High"] - df["Low"] + 1e-9)

    keep_cols = [c for c in df.columns if c not in
                 ["Open","High","Low","Volume","ma10","ma20","ma50","ma100","ma200",
                  "bb_up","bb_mid","bb_lo","vol_ma20","macd_signal","atr_14"]]
    result = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return result

def make_labels(close: pd.Series, horizon: int = 5,
                buy_threshold: float = 0.012, sell_threshold: float = -0.012) -> pd.Series:
    fwd_ret = close.shift(-horizon) / close - 1
    labels  = pd.Series(SIGNAL_HOLD, index=close.index, name="label")
    labels[fwd_ret >  buy_threshold]  = SIGNAL_BUY
    labels[fwd_ret <  sell_threshold] = SIGNAL_SELL
    return labels

def _synthetic_ohlcv(ticker: str, n: int = 500) -> pd.DataFrame:
    np.random.seed(abs(hash(ticker)) % (2**31))
    idx    = pd.bdate_range(end=pd.Timestamp.now().normalize(), periods=n)
    close  = np.cumprod(1 + np.random.randn(n) * 0.012 + 0.0002) * 150
    spread = np.abs(np.random.randn(n) * 0.01) * close
    return pd.DataFrame({
        "Open":   close - spread * 0.3,
        "High":   close + spread * 0.7,
        "Low":    close - spread * 0.7,
        "Close":  close,
        "Volume": np.random.randint(5_000_000, 30_000_000, n).astype(float),
    }, index=idx)

import streamlit as st

@st.cache_data(ttl=86400)
def fetch_ohlcv(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    if YFINANCE_AVAILABLE:
        try:
            import requests as _req
            sess = _req.Session()
            sess.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            })
            raw = yf.download(
                ticker, period=period, interval=interval,
                progress=False, auto_adjust=True,
                session=sess,
            )
            if raw is None or raw.empty:
                raise ValueError("Empty response")
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            return raw[["Open","High","Low","Close","Volume"]].dropna()
        except Exception as exc:
            print(f"[DataPipeline] yfinance failed for {ticker}: {exc} — using synthetic data.")
    return _synthetic_ohlcv(ticker)

def scale_features(X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)
    return X_tr_s, X_va_s, X_te_s, scaler

def build_dataset(ticker: str,
                  horizon: int = 5,
                  buy_threshold: float = 0.012,
                  sell_threshold: float = -0.012,
                  train_frac: float = 0.60,
                  val_frac: float = 0.20,
                  verbose: bool = True) -> dict:
    raw = fetch_ohlcv(ticker)
    if verbose:
        print(f"[{ticker}] Raw OHLCV: {len(raw)} rows")

    feat_df = build_features(raw)
    lbl = make_labels(raw["Close"], horizon, buy_threshold, sell_threshold)
    lbl = lbl.reindex(feat_df.index)
    valid_mask = lbl.notna()
    feat_df = feat_df[valid_mask]
    lbl     = lbl[valid_mask].astype(int)

    if verbose:
        counts = lbl.value_counts().to_dict()
        print(f"[{ticker}] Label distribution: {counts}")

    X_all = feat_df.values.astype(np.float32)
    y_all = lbl.values.astype(int)
    d_all = feat_df.index

    X_train_val, X_te, y_train_val, y_te, d_train_val, d_te = train_test_split(
        X_all, y_all, d_all, test_size=0.2, shuffle=False
    )
    X_tr, X_va, y_tr, y_va, d_tr, d_va = train_test_split(
        X_train_val, y_train_val, d_train_val, test_size=0.25, shuffle=False
    )

    X_tr_s, X_va_s, X_te_s, scaler_std = scale_features(X_tr, X_va, X_te)

    if verbose:
        print(f"[{ticker}] Splits  — train:{len(X_tr_s)} | val:{len(X_va_s)} | test:{len(X_te_s)}")

    return dict(
        ticker       = ticker,
        feature_names= list(feat_df.columns),
        close        = raw["Close"].reindex(feat_df.index),
        dates        = d_all,
        X_train=X_tr_s, X_val=X_va_s, X_test=X_te_s,
        y_train=y_tr,   y_val=y_va,   y_test=y_te,
        dates_train=d_tr, dates_val=d_va, dates_test=d_te,
        scaler_std   = scaler_std,
        label_counts = lbl.value_counts().to_dict(),
        n_features   = X_all.shape[1],
    )
