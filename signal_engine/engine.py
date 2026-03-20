"""
signal_engine/engine.py
========================
SignalEngine — main orchestrator.

Workflow per ticker:
    1. build_dataset()       — data_pipeline.py
    2. Fit RF, LSTM, XGBoost — models.py
    3. Ensemble probability   — average of all available models
    4. Convert to signal      — BUY / HOLD / SELL + confidence
    5. Evaluate on OOT test   — evaluator.py
    6. Return JSON-serialisable dict / pd.DataFrame for Streamlit

Public API
----------
engine = SignalEngine(config)
signals_df  = engine.run(tickers=["AAPL","TSLA"], mode="latest")
report_dict = engine.evaluation_reports   # keyed by ticker+model
"""

from __future__ import annotations

import json
import time
import warnings
import joblib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .data_pipeline import build_dataset, LABEL_MAP, SIGNAL_BUY, SIGNAL_SELL, SIGNAL_HOLD
from .models         import RandomForestSignalModel, XGBoostSignalModel
from .evaluator      import evaluate_model, print_report, calculate_historical_accuracy

try:
    from .models import LSTMSignalModel
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False


# ════════════════════════════════════════════════════════════════
#  Config dataclass
# ════════════════════════════════════════════════════════════════

@dataclass
class EngineConfig:
    # Data
    data_period: str    = "2y"        # yfinance period string
    label_horizon: int  = 5           # forward-looking days for labels
    buy_threshold: float  = 0.012     # +1.2% → BUY
    sell_threshold: float = -0.012    # -1.2% → SELL
    lstm_lookback: int  = 20          # LSTM sequence length

    # Training toggles
    use_rf:      bool = True
    use_lstm:    bool = True          # skipped if TF unavailable
    use_xgboost: bool = True

    # Signal fusion
    buy_confidence_threshold: float  = 0.55   # min P(BUY)  → BUY signal
    sell_confidence_threshold: float = 0.55   # min P(SELL) → SELL signal

    # Output
    output_dir: str = "signal_outputs"   # for saving reports/plots


# ════════════════════════════════════════════════════════════════
#  Signal converter
# ════════════════════════════════════════════════════════════════

def proba_to_signal(proba: np.ndarray,
                    buy_thresh: float  = 0.55,
                    sell_thresh: float = 0.55) -> tuple[str, float]:
    """
    Convert a 3-element probability vector [P(SELL), P(HOLD), P(BUY)]
    to a discrete signal and confidence percentage.

    Rules (in priority order):
    ─────────────────────────
    BUY  : P(BUY)  >= buy_thresh   AND P(BUY)  > P(SELL)
    SELL : P(SELL) >= sell_thresh  AND P(SELL) > P(BUY)
    HOLD : otherwise
    """
    p_sell, p_hold, p_buy = float(proba[0]), float(proba[1]), float(proba[2])
    confidence = max(p_sell, p_hold, p_buy)

    if p_buy >= buy_thresh and p_buy > p_sell:
        return "buy", round(p_buy * 100, 1)
    if p_sell >= sell_thresh and p_sell > p_buy:
        return "sell", round(p_sell * 100, 1)
    return "hold", round(p_hold * 100, 1)


def ensemble_proba(probas: list[np.ndarray],
                   weights: list[float] | None = None) -> np.ndarray:
    """
    Weighted average ensemble over multiple (n, 3) probability arrays.
    Default: equal weights.
    """
    if weights is None:
        weights = [1.0 / len(probas)] * len(probas)
    assert len(weights) == len(probas)
    total = sum(weights)
    out   = sum(w * p for w, p in zip(weights, probas)) / total
    return out


# ════════════════════════════════════════════════════════════════
#  Engine
# ════════════════════════════════════════════════════════════════

class SignalEngine:
    """
    Orchestrates data ingestion, model training, ensembling, and signal output.

    Usage
    -----
    >>> engine = SignalEngine()
    >>> df = engine.run(["AAPL", "TSLA", "NVDA", "SPY", "MSFT"])
    >>> print(df.to_json(orient="records"))   # → Streamlit live signals input
    """

    def __init__(self, config: EngineConfig | None = None):
        self.cfg      = config or EngineConfig()
        self.models_  : dict[str, dict] = {}   # {ticker: {model_name: model_obj}}
        self.datasets_: dict[str, dict] = {}   # {ticker: dataset dict}
        self.evaluation_reports: dict   = {}
        self._signals_df: pd.DataFrame | None = None

        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def save_models(self):
        """Save trained models and evaluation reports to disk for fast inference."""
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        for ticker, trained in self.models_.items():
            ticker_dir = models_dir / ticker
            ticker_dir.mkdir(exist_ok=True)
            for m_name, model_obj in trained.items():
                if m_name == "LSTM" and LSTM_AVAILABLE:
                    try:
                        from tensorflow.keras.models import save_model
                        save_model(model_obj.model, ticker_dir / f"{m_name}.keras")
                    except Exception as e:
                        print(f"[Engine] Failed to save LSTM for {ticker}: {e}")
                else:
                    joblib.dump(model_obj, ticker_dir / f"{m_name}.joblib")
                    
        # Retain validation reports for historical accuracy weighting
        joblib.dump(self.evaluation_reports, models_dir / "evaluation_reports.joblib")
        print(f"[Engine] Models saved to {models_dir.absolute()}")

    def load_models(self):
        """Load pre-trained models and evaluation reports from disk."""
        models_dir = Path("models")
        if not models_dir.exists():
            print(f"[Engine] No models directory found at {models_dir.absolute()}")
            return
            
        rep_path = models_dir / "evaluation_reports.joblib"
        if rep_path.exists():
            self.evaluation_reports = joblib.load(rep_path)
            
        for ticker_dir in models_dir.iterdir():
            if not ticker_dir.is_dir(): continue
            ticker = ticker_dir.name
            trained = {}
            for file in ticker_dir.iterdir():
                if file.suffix == ".joblib":
                    m_name = file.stem
                    trained[m_name] = joblib.load(file)
                elif file.suffix in [".h5", ".keras"]:
                    m_name = file.stem
                    if LSTM_AVAILABLE:
                        try:
                            from .models import LSTMSignalModel
                            from tensorflow.keras.models import load_model
                            lstm_wrapper = LSTMSignalModel()
                            lstm_wrapper.model = load_model(file)
                            lstm_wrapper.is_fitted = True
                            trained[m_name] = lstm_wrapper
                        except Exception as e:
                            print(f"[Engine] Failed to load LSTM for {ticker}: {e}")
            if trained:
                self.models_[ticker] = trained
        print(f"[Engine] Loaded models for {len(self.models_)} tickers.")

    # ── 1. Training ────────────────────────────────────────────

    def _train_ticker(self, ticker: str) -> dict[str, Any]:
        """Build dataset and train all configured models for one ticker."""
        print(f"\n{'─'*60}")
        print(f"  Processing {ticker} …")
        print(f"{'─'*60}")

        ds = build_dataset(
            ticker,
            horizon       = self.cfg.label_horizon,
            buy_threshold = self.cfg.buy_threshold,
            sell_threshold= self.cfg.sell_threshold,
            lstm_lookback  = self.cfg.lstm_lookback,
            verbose       = True,
        )
        self.datasets_[ticker] = ds

        trained: dict[str, Any] = {}

        # ── Random Forest ──────────────────────────────────
        if self.cfg.use_rf:
            rf = RandomForestSignalModel()
            rf.fit(ds["X_train"], ds["y_train"])
            trained["Random Forest"] = rf

        # ── XGBoost ───────────────────────────────────────
        if self.cfg.use_xgboost:
            xgb = XGBoostSignalModel()
            xgb.fit(ds["X_train"], ds["y_train"],
                    ds["X_val"],   ds["y_val"])
            trained["XGBoost"] = xgb

        # ── LSTM ──────────────────────────────────────────
        if self.cfg.use_lstm and LSTM_AVAILABLE:
            try:
                lstm = LSTMSignalModel(epochs=50, batch_size=32)
                lstm.fit(ds["X_lstm_train"], ds["y_lstm_train"],
                         ds["X_lstm_val"],   ds["y_lstm_val"])
                trained["LSTM"] = lstm
            except Exception as exc:
                print(f"[LSTM] Training failed for {ticker}: {exc}")

        self.models_[ticker] = trained
        return trained

    # ── 2. Evaluation on OOT set ─────────────────────────────

    def _evaluate_ticker(self, ticker: str):
        """Run evaluate_model on the Out-of-Time test set, store reports & print."""
        ds      = self.datasets_[ticker]
        trained = self.models_[ticker]

        for model_name, model in trained.items():
            # Use flat data for RF/XGBoost; LSTM sequences for LSTM
            if model_name == "LSTM":
                X_te = ds["X_lstm_test"]
                y_te = ds["y_lstm_test"]
            else:
                X_te = ds["X_test"]
                y_te = ds["y_test"]

            if len(X_te) == 0:
                print(f"[Eval] {ticker}/{model_name}: empty test set, skipping.")
                continue

            y_prob = model.predict_proba(X_te)
            y_pred = np.argmax(y_prob, axis=1)

            imp   = getattr(model, "feature_importances_", None)
            fnames = ds["feature_names"]

            report = evaluate_model(y_te, y_pred, y_prob,
                                    model_name=f"{ticker} — {model_name}",
                                    feature_names=fnames,
                                    importances=imp)
            
            # --- 100-day Hit Ratio for Milestone 2.1 ---
            hist_metrics = calculate_historical_accuracy(y_te, y_pred, n=100)
            report["historical_100d"] = hist_metrics

            key = f"{ticker}_{model_name}"
            self.evaluation_reports[key] = report
            print_report(report)

    # ── 3. Latest signal (last row) ───────────────────────────

    def _latest_signal(self, ticker: str) -> dict:
        """
        Generate the latest trading signal by running all trained models
        on the most recent available feature row (or sequence for LSTM)
        and ensembling their probability outputs.
        """
        ds      = self.datasets_[ticker]
        trained = self.models_[ticker]
        ts      = datetime.now().strftime("%H:%M:%S")

        indiv_probas: list[np.ndarray] = []
        indiv_signals: dict = {}

        for model_name, model in trained.items():
            try:
                if model_name == "LSTM":
                    # Last valid LSTM sequence from the full dataset
                    X_mm = ds.get("X_lstm_test")
                    if X_mm is None or len(X_mm) == 0:
                        continue
                    row = X_mm[[-1]]         # shape (1, lookback, feats)
                else:
                    row = ds["X_test"][[-1]] if len(ds["X_test"]) > 0 \
                          else ds["X_val"][[-1]]

                p = model.predict_proba(row)[0]  # shape (3,)
                sig, conf = proba_to_signal(p,
                                            self.cfg.buy_confidence_threshold,
                                            self.cfg.sell_confidence_threshold)
                indiv_signals[model_name] = {"signal": sig, "confidence": conf, "proba": p.tolist()}
                indiv_probas.append(p)
            except Exception as exc:
                print(f"[Signal] {ticker}/{model_name}: {exc}")

        if not indiv_probas:
            return {"ticker": ticker, "signal": "hold", "confidence": 50.0,
                    "model": "N/A", "timestamp": ts, "consensus": False,
                    "individual": {}}

        # Ensemble with dynamic weighting based on historical_100d Hit Ratio
        weights = []
        for model_name in indiv_signals:
            key = f"{ticker}_{model_name}"
            w = 0.5  # Base default
            if key in self.evaluation_reports:
                report = self.evaluation_reports[key]
                w = report.get("historical_100d", {}).get("hit_ratio", 0.5)
            weights.append(w)
            
        if sum(weights) == 0:
            weights = None

        ens_proba            = ensemble_proba(indiv_probas, weights=weights)
        ens_signal, ens_conf = proba_to_signal(ens_proba,
                                               self.cfg.buy_confidence_threshold,
                                               self.cfg.sell_confidence_threshold)

        # Consensus: majority of individual models agree with ensemble
        individual_labels = [v["signal"] for v in indiv_signals.values()]
        consensus = individual_labels.count(ens_signal) >= max(1, len(individual_labels) // 2 + 1)

        # Choose primary model label (highest confidence individual)
        best_model = max(indiv_signals, key=lambda m: indiv_signals[m]["confidence"])

        # Most recent close price from dataset
        close_series = ds.get("close")
        latest_price = float(close_series.iloc[-1]) if close_series is not None and len(close_series) > 0 else 0.0

        return {
            "id":          ticker.lower(),
            "ticker":      ticker,
            "signal":      ens_signal,
            "confidence":  ens_conf,
            "model":       best_model,
            "price":       round(latest_price, 2),
            "consensus":   consensus,
            "time":        f"{ts} Live",
            "ensemble_proba": {LABEL_MAP[i]: round(float(ens_proba[i]), 4) for i in range(3)},
            "individual_signals": indiv_signals,
        }

    # ── 4. Public run() ──────────────────────────────────────

    def run(self,
            tickers: list[str] | None = None,
            mode: str = "latest",
            evaluate: bool = True,
            train_mode: bool = True) -> pd.DataFrame:
        """
        Full pipeline: train models, evaluate, generate signals.

        Parameters
        ----------
        tickers    : list of ticker symbols  (default: AAPL, TSLA, NVDA, SPY, MSFT)
        mode       : 'latest' — signal for the most recent data point (default)
        evaluate   : if True, run evaluation on OOT test set and print reports
        train_mode : if True, trains models from scratch. If False, skips training
                     and relies exclusively on pre-loaded models for rapid inference.

        Returns
        -------
        pd.DataFrame with columns:
            id, ticker, signal, confidence, model, price, consensus,
            time, ensemble_proba, individual_signals
        """
        if tickers is None:
            tickers = ["AAPL", "TSLA", "NVDA", "SPY", "MSFT"]

        signals = []
        for tk in tickers:
            try:
                if train_mode:
                    self._train_ticker(tk)
                    if evaluate:
                        self._evaluate_ticker(tk)
                else:
                    if tk not in self.datasets_:
                        # Populate dataset for inference only 
                        self.datasets_[tk] = build_dataset(
                            tk,
                            horizon=self.cfg.label_horizon,
                            buy_threshold=self.cfg.buy_threshold,
                            sell_threshold=self.cfg.sell_threshold,
                            lstm_lookback=self.cfg.lstm_lookback,
                            verbose=False
                        )
                sig_row = self._latest_signal(tk)
                signals.append(sig_row)
                print(f"\n✅ {tk:5s} → Signal: {sig_row['signal'].upper():4s} | "
                      f"Confidence: {sig_row['confidence']:.1f}% | "
                      f"Model: {sig_row['model']}")
            except Exception as exc:
                print(f"❌ {tk}: engine error — {exc}")
                import traceback; traceback.print_exc()

        self._signals_df = pd.DataFrame(signals)
        return self._signals_df

    # ── 5. Streamlit integration helpers ─────────────────────

    def to_json(self) -> str:
        """
        Export latest signals as a JSON string for Streamlit / REST API.

        Format: list of signal dicts (one per ticker).
        The Streamlit app can consume this via:
            signals = json.loads(engine.to_json())
            SIGNALS = [{
                "id": s["id"], "ticker": s["ticker"],
                "signal": s["signal"],    "model": s["model"],
                "price": s["price"],      "confidence": s["confidence"],
                "time": s["time"],        "consensus": s["consensus"]
            } for s in signals]
        """
        if self._signals_df is None:
            raise RuntimeError("Run engine.run() first.")
        records = self._signals_df.to_dict(orient="records")
        # Coerce numpy types to native Python for JSON serialisation
        def _native(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        return json.dumps(records, default=_native, indent=2)

    def save_signals(self, path: str = "signal_outputs/signals.json"):
        """Write signals JSON to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out = self.to_json()
        Path(path).write_text(out)
        print(f"[Engine] Signals saved → {path}")

    def get_signals_for_streamlit(self) -> list[dict]:
        """
        Return list of clean dicts formatted for the Streamlit Live Signals feed.
        Matches the SIGNALS schema in app.py exactly.
        """
        if self._signals_df is None:
            raise RuntimeError("Run engine.run() first.")
        return [
            {
                "id":         row["id"],
                "ticker":     row["ticker"],
                "signal":     row["signal"],
                "model":      row["model"],
                "price":      row["price"],
                "confidence": int(round(float(row["confidence"]))),
                "time":       row["time"],
                "consensus":  bool(row["consensus"]),
            }
            for _, row in self._signals_df.iterrows()
        ]

    def get_evaluation_summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame with key metrics per ticker+model."""
        rows = []
        for key, r in self.evaluation_reports.items():
            rows.append({
                "key":              key,
                "model":            r["model_name"],
                "accuracy":         r["accuracy"],
                "precision_macro":  r["precision_macro"],
                "recall_macro":     r["recall_macro"],
                "f1_macro":         r["f1_macro"],
                "auc_roc":          r["auc_roc"],
                "precision_BUY":    r["precision_per_class"]["BUY"],
                "recall_BUY":       r["recall_per_class"]["BUY"],
                "precision_SELL":   r["precision_per_class"]["SELL"],
                "recall_SELL":      r["recall_per_class"]["SELL"],
            })
        return pd.DataFrame(rows)
