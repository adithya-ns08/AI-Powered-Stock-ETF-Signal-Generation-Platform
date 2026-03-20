"""
run_engine.py
=============
Standalone demo/test runner for the Signal Generation Engine.

Run:
    python run_engine.py

This will:
  1. Download (or synthesise) 2 years of OHLCV data for each ticker
  2. Train Random Forest, LSTM, and XGBoost models
  3. Print detailed evaluation reports (OOT test set)
  4. Print a final signal summary table
  5. Save signals to  signal_outputs/signals.json
  6. Save evaluation summary to signal_outputs/eval_summary.csv
  7. Export plots to signal_outputs/plots/

Integration with Streamlit (app.py):
--------------------------------------
    import json
    from signal_engine import SignalEngine

    @st.cache_resource
    def load_engine():
        engine = SignalEngine()
        engine.run(["AAPL","TSLA","NVDA","SPY","MSFT"], evaluate=False)
        return engine

    engine  = load_engine()
    SIGNALS = engine.get_signals_for_streamlit()
    # → Drop this list straight into app.py to power Live Signals cards

    # To refresh on a timer:
    # import time
    # if time.time() - st.session_state.get("last_refresh", 0) > 300:
    #     engine.run(["AAPL","TSLA"], evaluate=False)
    #     st.session_state["last_refresh"] = time.time()
"""

import sys
import json
import time
from pathlib import Path

# ── Make sure project root is on path ───────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from signal_engine import SignalEngine
from signal_engine.engine import EngineConfig
from signal_engine.evaluator import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
    plot_training_history,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TICKERS    = ["AAPL", "TSLA", "NVDA", "SPY", "MSFT"]
PLOT_DIR   = Path("signal_outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig: plt.Figure, name: str):
    path = PLOT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   📊  Saved → {path}")


def main():
    print("\n" + "=" * 65)
    print("  AI Stock & ETF Signal Generation Engine")
    print("  Tickers:", " | ".join(TICKERS))
    print("=" * 65)

    t0 = time.time()

    # ── 1. Configure ──────────────────────────────────────────
    cfg = EngineConfig(
        data_period        = "2y",
        label_horizon      = 5,          # predict 5-day direction
        buy_threshold      = 0.012,      # +1.2% → BUY label
        sell_threshold     = -0.012,     # -1.2% → SELL label
        lstm_lookback      = 20,
        use_rf             = True,
        use_lstm           = True,       # set False if TF not installed
        use_xgboost        = True,
        buy_confidence_threshold  = 0.55,
        sell_confidence_threshold = 0.55,
    )

    # ── 2. Run engine ─────────────────────────────────────────
    engine = SignalEngine(config=cfg)
    signals_df = engine.run(TICKERS, evaluate=True)

    # ── 3. Console signal summary ─────────────────────────────
    print("\n" + "═" * 65)
    print("  FINAL SIGNAL SUMMARY")
    print("═" * 65)
    print(f"  {'Ticker':<8} {'Signal':<6} {'Confidence':>11} {'Model':<16} {'Consensus'}")
    print("  " + "─" * 60)
    for _, row in signals_df.iterrows():
        con = "✔" if row.get("consensus") else " "
        print(f"  {row['ticker']:<8} {str(row['signal']).upper():<6} "
              f"{row['confidence']:>10.1f}%  {str(row['model']):<16} {con}")
    print("═" * 65)

    # ── 4. Save JSON signals ──────────────────────────────────
    engine.save_signals("signal_outputs/signals.json")

    # ── 5. Save evaluation CSV ────────────────────────────────
    eval_df = engine.get_evaluation_summary()
    eval_path = "signal_outputs/eval_summary.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"\n📋 Evaluation summary → {eval_path}")
    if not eval_df.empty:
        cols = ["model","accuracy","precision_macro","recall_macro","f1_macro","auc_roc"]
        print(eval_df[[c for c in cols if c in eval_df.columns]].to_string(index=False))

    # ── 6. Save plots ─────────────────────────────────────────
    print("\n🎨 Generating plots …")
    for key, report in engine.evaluation_reports.items():
        if report["confusion_matrix"] is None:
            continue
        safe_key = key.replace(" ", "_").replace("—", "-")

        # Confusion matrix
        fig = plot_confusion_matrix(report["confusion_matrix"],
                                    title=f"Confusion Matrix — {report['model_name']}")
        save_fig(fig, f"{safe_key}_confusion.png")

        # Feature importance (RF / XGBoost)
        if report.get("importances") is not None and report.get("feature_names"):
            fig = plot_feature_importance(
                report["feature_names"], report["importances"],
                top_n=15, title=f"Feature Importance — {report['model_name']}")
            save_fig(fig, f"{safe_key}_importance.png")

    # LSTM training history plots
    for ticker, trained in engine.models_.items():
        if "LSTM" in trained:
            lstm = trained["LSTM"]
            if lstm.history is not None:
                fig = plot_training_history(lstm.history, title=f"LSTM — {ticker}")
                save_fig(fig, f"{ticker}_lstm_history.png")

    elapsed = time.time() - t0
    print(f"\n⏱  Total runtime: {elapsed:.1f}s")
    print(f"📁 All outputs in: signal_outputs/")
    print("\n" + "=" * 65)
    print("  ENGINE COMPLETE — ready to feed Streamlit UI")
    print("=" * 65)

    # ── 7. Demo: show Streamlit-ready dict ────────────────────
    streamlit_signals = engine.get_signals_for_streamlit()
    print("\nStreamlit-compatible signals (drop into app.py SIGNALS list):")
    print(json.dumps(streamlit_signals, indent=2))
    return signals_df


if __name__ == "__main__":
    main()
