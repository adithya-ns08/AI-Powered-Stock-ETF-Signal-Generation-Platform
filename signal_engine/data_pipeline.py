"""
signal_engine/data_pipeline.py
================================
OHLCV → Feature Engineering → Supervised Labels → Train/Val/Out-of-Time Split

Pipeline steps
--------------
1. Download or accept raw OHLCV data (yfinance or DataFrame).
2. Compute technical indicators: RSI, MACD, Bollinger Bands, MAs, volume ratio.
3. Create forward-looking labels (Buy / Hold / Sell) from future N-day returns.
4. Scale features: StandardScaler for tree models, MinMaxScaler for LSTM sequences.
5. Split into three non-overlapping, chronological segments with NO lookahead bias:
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

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Label constants ──────────────────────────────────────────────
SIGNAL_BUY  = 2   # High-confidence bullish
SIGNAL_HOLD = 1   # Neutral / low confidence
SIGNAL_SELL = 0   # High-confidence bearish
LABEL_MAP   = {SIGNAL_SELL: "sell", SIGNAL_HOLD: "hold", SIGNAL_BUY: "buy"}


# ════════════════════════════════════════════════════════════════
#  Technical Indicator Helpers
# ════════════════════════════════════════════════════════════════

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — bounded 0‥100."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, Signal line, Histogram."""
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, period: int = 20, std: float = 2.0
               ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Upper band, middle (SMA), lower band."""
    mid   = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return mid + std * sigma, mid, mid - std * sigma


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = 14) -> pd.Series:
    """Average True Range — proxy for volatility."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ════════════════════════════════════════════════════════════════
#  Feature Engineering
# ════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a raw OHLCV DataFrame into a rich feature set.

    Expected columns: Open, High, Low, Close, Volume  (case-insensitive).
    Returns a new DataFrame with engineered features (rows with NaN dropped).
    """
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]

    # ── Returns & momentum ──────────────────────────────────────
    df["ret_1d"]   = df["Close"].pct_change(1)
    df["ret_3d"]   = df["Close"].pct_change(3)
    df["ret_5d"]   = df["Close"].pct_change(5)
    df["ret_10d"]  = df["Close"].pct_change(10)
    df["log_ret"]  = np.log(df["Close"] / df["Close"].shift(1))

    # ── Moving averages & deviation ──────────────────────────────
    for w in [10, 20, 50, 100, 200]:
        df[f"ma{w}"]    = df["Close"].rolling(w).mean()
        df[f"dev_ma{w}"] = (df["Close"] - df[f"ma{w}"]) / df[f"ma{w}"]

    # ── RSI ──────────────────────────────────────────────────────
    df["rsi_14"] = _rsi(df["Close"], 14)
    df["rsi_7"]  = _rsi(df["Close"],  7)

    # ── MACD ────────────────────────────────────────────────────
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["Close"])
    df["macd_cross"] = np.sign(df["macd"] - df["macd_signal"])

    # ── Bollinger Bands ──────────────────────────────────────────
    bb_up, bb_mid, bb_lo = _bollinger(df["Close"])
    df["bb_up"]    = bb_up
    df["bb_mid"]   = bb_mid
    df["bb_lo"]    = bb_lo
    df["bb_width"] = (bb_up - bb_lo) / bb_mid
    df["bb_pct"]   = (df["Close"] - bb_lo) / (bb_up - bb_lo + 1e-9)

    # ── ATR (volatility) ─────────────────────────────────────────
    df["atr_14"] = _atr(df["High"], df["Low"], df["Close"], 14)
    df["atr_pct"] = df["atr_14"] / df["Close"]

    # ── Volume features ──────────────────────────────────────────
    df["vol_ma20"]   = df["Volume"].rolling(20).mean()
    df["vol_ratio"]  = df["Volume"] / (df["vol_ma20"] + 1)
    df["vol_change"] = df["Volume"].pct_change()

    # ── Candlestick shape ────────────────────────────────────────
    df["body"]     = (df["Close"] - df["Open"]) / (df["High"] - df["Low"] + 1e-9)
    df["upper_sh"] = (df["High"]  - df[["Open","Close"]].max(axis=1)) / (df["High"] - df["Low"] + 1e-9)
    df["lower_sh"] = (df[["Open","Close"]].min(axis=1) - df["Low"])   / (df["High"] - df["Low"] + 1e-9)

    # Drop raw OHLCV + intermediate helpers before returning
    keep_cols = [c for c in df.columns if c not in
                 ["Open","High","Low","Volume","ma10","ma20","ma50","ma100","ma200",
                  "bb_up","bb_mid","bb_lo","vol_ma20","macd_signal","atr_14"]]
    result = df[keep_cols].replace([np.inf, -np.inf], np.nan).dropna()
    return result


# ════════════════════════════════════════════════════════════════
#  Label Generation
# ════════════════════════════════════════════════════════════════

def make_labels(close: pd.Series,
                horizon: int = 5,
                buy_threshold: float = 0.012,
                sell_threshold: float = -0.012) -> pd.Series:
    """
    Forward-looking labels over `horizon` trading days.

    BUY  (2) : future_return  > +buy_threshold
    SELL (0) : future_return  < sell_threshold
    HOLD (1) : otherwise

    ⚠ Labels are shift()-aligned and then NaN rows are dropped by the caller.
    """
    fwd_ret = close.shift(-horizon) / close - 1
    labels  = pd.Series(SIGNAL_HOLD, index=close.index, name="label")
    labels[fwd_ret >  buy_threshold]  = SIGNAL_BUY
    labels[fwd_ret <  sell_threshold] = SIGNAL_SELL
    return labels


# ════════════════════════════════════════════════════════════════
#  Data Acquisition
# ════════════════════════════════════════════════════════════════

def _synthetic_ohlcv(ticker: str, n: int = 500) -> pd.DataFrame:
    """Generate reproducible synthetic OHLCV when yfinance is unavailable."""
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


def fetch_ohlcv(ticker: str,
                period: str = "2y",
                interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV from yfinance; falls back to synthetic data.
    Returns a DataFrame with columns [Open, High, Low, Close, Volume].
    """
    if YFINANCE_AVAILABLE:
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw.empty:
                raise ValueError("Empty response")
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            return raw[["Open","High","Low","Close","Volume"]].dropna()
        except Exception as exc:
            print(f"[DataPipeline] yfinance failed for {ticker}: {exc} — using synthetic data.")
    return _synthetic_ohlcv(ticker)


# ════════════════════════════════════════════════════════════════
#  Scaling
# ════════════════════════════════════════════════════════════════

def scale_features(X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray,
                   scaler_type: str = "standard"
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Fit scaler on train only. Transform val+test — NO leakage.

    scaler_type : 'standard' (RF / XGBoost) | 'minmax' (LSTM sequences)
    Returns (X_train_s, X_val_s, X_test_s, fitted_scaler)
    """
    Scaler = MinMaxScaler if scaler_type == "minmax" else StandardScaler
    scaler = Scaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_test)
    return X_tr_s, X_va_s, X_te_s, scaler


def make_lstm_sequences(X: np.ndarray,
                        y: np.ndarray,
                        lookback: int = 20
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    Reshape flat (n_samples, n_features) arrays into LSTM-ready
    (n_samples, lookback, n_features) sequences with matching labels.
    """
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback: i, :])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ════════════════════════════════════════════════════════════════
#  Main pipeline entry point
# ════════════════════════════════════════════════════════════════

def build_dataset(ticker: str,
                  horizon: int = 5,
                  buy_threshold: float = 0.012,
                  sell_threshold: float = -0.012,
                  lstm_lookback: int = 20,
                  train_frac: float = 0.60,
                  val_frac: float = 0.20,
                  verbose: bool = True
                  ) -> dict:
    """
    Full pipeline: raw data → features → labels → scaled splits.

    Returns
    -------
    dict with keys:
        feature_names   : list[str]
        close           : pd.Series  — raw close prices (aligned with features)
        dates           : pd.DatetimeIndex

        # Flat arrays for RF / XGBoost
        X_train, X_val, X_test            : np.ndarray (scaled, StandardScaler)
        y_train, y_val, y_test            : np.ndarray (int labels 0/1/2)
        dates_train, dates_val, dates_test: pd.DatetimeIndex

        # Sequences for LSTM (MinMaxScaler)
        X_lstm_train, X_lstm_val, X_lstm_test : np.ndarray (batch, lookback, feats)
        y_lstm_train, y_lstm_val, y_lstm_test : np.ndarray

        scaler_std   : StandardScaler  — for RF / XGBoost
        scaler_mm    : MinMaxScaler    — for LSTM
        label_counts : dict            — class distribution
    """
    # 1. Fetch --------------------------------------------------
    raw = fetch_ohlcv(ticker)
    if verbose:
        print(f"[{ticker}] Raw OHLCV: {len(raw)} rows")

    # 2. Features -----------------------------------------------
    feat_df = build_features(raw)

    # 3. Labels (forward-looking, aligned by index) ─────────────
    lbl = make_labels(raw["Close"], horizon, buy_threshold, sell_threshold)
    lbl = lbl.reindex(feat_df.index)  # align to feature index
    valid_mask = lbl.notna()
    feat_df = feat_df[valid_mask]
    lbl     = lbl[valid_mask].astype(int)

    if verbose:
        counts = lbl.value_counts().to_dict()
        print(f"[{ticker}] Label distribution: {counts}")

    # 4. Chronological 60/20/20 split (NO shuffle) ──────────────
    n = len(feat_df)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)

    X_all = feat_df.values.astype(np.float32)
    y_all = lbl.values.astype(int)
    d_all = feat_df.index

    X_tr, X_va, X_te = X_all[:n_train], X_all[n_train:n_train+n_val], X_all[n_train+n_val:]
    y_tr, y_va, y_te = y_all[:n_train], y_all[n_train:n_train+n_val], y_all[n_train+n_val:]
    d_tr = d_all[:n_train]
    d_va = d_all[n_train:n_train+n_val]
    d_te = d_all[n_train+n_val:]

    # 5. Scale (fit on train only) ───────────────────────────────
    X_tr_s, X_va_s, X_te_s, scaler_std = scale_features(X_tr, X_va, X_te, "standard")
    X_tr_mm, X_va_mm, X_te_mm, scaler_mm = scale_features(X_tr, X_va, X_te, "minmax")

    # 6. LSTM sequences ──────────────────────────────────────────
    X_lstm_tr, y_lstm_tr = make_lstm_sequences(X_tr_mm, y_tr, lstm_lookback)
    X_lstm_va, y_lstm_va = make_lstm_sequences(X_va_mm, y_va, lstm_lookback)
    X_lstm_te, y_lstm_te = make_lstm_sequences(X_te_mm, y_te, lstm_lookback)

    if verbose:
        print(f"[{ticker}] Splits  — train:{len(X_tr_s)} | val:{len(X_va_s)} | test:{len(X_te_s)}")
        print(f"[{ticker}] LSTM seq — train:{X_lstm_tr.shape} | val:{X_lstm_va.shape} | test:{X_lstm_te.shape}")

    return dict(
        ticker       = ticker,
        feature_names= list(feat_df.columns),
        close        = raw["Close"].reindex(feat_df.index),
        dates        = d_all,

        X_train=X_tr_s, X_val=X_va_s, X_test=X_te_s,
        y_train=y_tr,   y_val=y_va,   y_test=y_te,
        dates_train=d_tr, dates_val=d_va, dates_test=d_te,

        X_lstm_train=X_lstm_tr, X_lstm_val=X_lstm_va, X_lstm_test=X_lstm_te,
        y_lstm_train=y_lstm_tr, y_lstm_val=y_lstm_va, y_lstm_test=y_lstm_te,

        scaler_std   = scaler_std,
        scaler_mm    = scaler_mm,
        label_counts = lbl.value_counts().to_dict(),
        n_features   = X_all.shape[1],
        lstm_lookback= lstm_lookback,
    )
