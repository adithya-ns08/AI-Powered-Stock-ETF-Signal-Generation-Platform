"""
signal_engine — AI Stock & ETF Signal Generation Engine
========================================================
Package exposing the SignalEngine orchestrator.

Usage:
    from signal_engine import SignalEngine
    engine = SignalEngine()
    signals_df = engine.run(tickers=["AAPL", "TSLA", "NVDA"])
"""
from .engine import SignalEngine

__all__ = ["SignalEngine"]
__version__ = "1.0.0"
