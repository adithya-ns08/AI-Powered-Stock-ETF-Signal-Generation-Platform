"""
signal_engine/evaluator.py
============================
Evaluation metrics for classification models in a trading context.

Functions
---------
evaluate_model(y_true, y_pred, y_proba, model_name, feature_names, importances)
    → dict with accuracy, precision, recall, F1, AUC-ROC, confusion matrix

plot_confusion_matrix(cm, title)   → matplotlib Figure
plot_roc_curve(y_true, y_proba)    → matplotlib Figure
plot_feature_importance(features)  → matplotlib Figure
generate_report(results_dict)      → prints a formatted summary
"""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
try:
    import seaborn as sns
    SNS_AVAILABLE = True
except ImportError:
    SNS_AVAILABLE = False

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import label_binarize

LABEL_NAMES = ["SELL", "HOLD", "BUY"]

# ── Plot style ───────────────────────────────────────────────────
_DARK  = "#0a0b0d"
_CARD  = "#191c28"
_BORD  = "#252840"
_TEXT1 = "#e8eaf6"
_TEXT2 = "#8890b0"
_EMRLD = "#00e676"
_CRIMN = "#ff1744"
_AMBER = "#ffab00"
_BLUE  = "#2979ff"

plt.rcParams.update({
    "figure.facecolor": _DARK,
    "axes.facecolor":   _CARD,
    "axes.edgecolor":   _BORD,
    "axes.labelcolor":  _TEXT1,
    "xtick.color":      _TEXT2,
    "ytick.color":      _TEXT2,
    "text.color":       _TEXT1,
    "grid.color":       _BORD,
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
})

SIGNAL_COLORS = [_CRIMN, _AMBER, _EMRLD]  # sell, hold, buy


# ════════════════════════════════════════════════════════════════
#  Core metrics
# ════════════════════════════════════════════════════════════════

def evaluate_model(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_proba: np.ndarray,
                   model_name: str = "Model",
                   feature_names: list[str] | None = None,
                   importances: np.ndarray | None = None,
                   ) -> dict:
    """
    Compute a full evaluation suite and return it as a dict.

    Parameters
    ----------
    y_true       : 1-D int array, true labels {0, 1, 2}
    y_pred       : 1-D int array, predicted labels
    y_proba      : (n, 3) float array, class probabilities
    model_name   : display name
    feature_names: list of feature strings (for importance plot)
    importances  : feature importance array from RF / XGBoost

    Returns dict with keys:
        accuracy, precision_macro, recall_macro, f1_macro,
        precision_per_class, recall_per_class, f1_per_class,
        auc_roc, confusion_matrix (np.ndarray),
        classification_report (str), model_name
    """
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro  = recall_score   (y_true, y_pred, average="macro", zero_division=0)
    f1_macro   = f1_score       (y_true, y_pred, average="macro", zero_division=0)

    # Per-class metrics
    prec_cls = precision_score(y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    rec_cls  = recall_score   (y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)
    f1_cls   = f1_score       (y_true, y_pred, average=None, labels=[0,1,2], zero_division=0)

    # AUC-ROC (OvR, macro)
    try:
        y_bin   = label_binarize(y_true, classes=[0, 1, 2])
        auc_roc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc_roc = float("nan")

    cm     = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(y_true, y_pred,
                                   target_names=LABEL_NAMES, zero_division=0)

    result = dict(
        model_name          = model_name,
        accuracy            = round(float(acc), 4),
        precision_macro     = round(float(prec_macro), 4),
        recall_macro        = round(float(rec_macro), 4),
        f1_macro            = round(float(f1_macro), 4),
        auc_roc             = round(float(auc_roc), 4),
        precision_per_class = {LABEL_NAMES[i]: round(float(prec_cls[i]), 4) for i in range(3)},
        recall_per_class    = {LABEL_NAMES[i]: round(float(rec_cls[i]),  4) for i in range(3)},
        f1_per_class        = {LABEL_NAMES[i]: round(float(f1_cls[i]),   4) for i in range(3)},
        confusion_matrix    = cm,
        classification_report = report,
        feature_names       = feature_names,
        importances         = importances,
    )
    return result


def calculate_historical_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n: int = 100) -> dict:
    """
    Calculate Hit Ratio (Accuracy), Precision, and Recall specifically over the last `n` data points.
    This serves as the rolling historical accuracy metric for the most recent signals.
    """
    if len(y_true) > n:
        y_t = y_true[-n:]
        y_p = y_pred[-n:]
    else:
        y_t = y_true
        y_p = y_pred

    acc = accuracy_score(y_t, y_p)

    # Precision and Recall per class
    prec_cls = precision_score(y_t, y_p, average=None, labels=[0, 1, 2], zero_division=0)
    rec_cls  = recall_score(y_t, y_p, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "period_points": len(y_t),
        "hit_ratio": round(float(acc), 4),
        "precision_BUY": round(float(prec_cls[2]), 4),
        "recall_BUY": round(float(rec_cls[2]), 4),
        "precision_SELL": round(float(prec_cls[0]), 4),
        "recall_SELL": round(float(rec_cls[0]), 4),
    }


# ════════════════════════════════════════════════════════════════
#  Plots
# ════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
    """Annotated heatmap of the 3×3 confusion matrix (Sell / Hold / Buy)."""
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_DARK)
    ax.set_facecolor(_CARD)

    # Normalise row-wise for color (raw counts as annotations)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    cmap = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        cm_norm, ax=ax,
        annot=cm,               # show raw counts
        fmt="d",
        cmap=cmap,
        linewidths=0.5,
        linecolor=_BORD,
        cbar=False,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        annot_kws={"size": 13, "weight": "bold"},
    )
    ax.set_xlabel("Predicted",  color=_TEXT1, fontsize=11, labelpad=8)
    ax.set_ylabel("Actual",     color=_TEXT1, fontsize=11, labelpad=8)
    ax.set_title(title,         color=_TEXT1, fontsize=13, pad=12, fontweight="bold")
    ax.tick_params(colors=_TEXT2, labelsize=10)
    fig.tight_layout()
    return fig


def plot_roc_curves(y_true: np.ndarray,
                    y_proba: np.ndarray,
                    title: str = "ROC Curves (OvR)") -> plt.Figure:
    """Per-class ROC curves on a dark background."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    colors = SIGNAL_COLORS

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_DARK)
    ax.set_facecolor(_CARD)
    ax.set_xlabel("False Positive Rate", color=_TEXT1, fontsize=10)
    ax.set_ylabel("True Positive Rate",  color=_TEXT1, fontsize=10)
    ax.set_title(title,                  color=_TEXT1, fontsize=12, fontweight="bold")
    ax.tick_params(colors=_TEXT2)

    for i, (cls_name, col) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2.0,
                label=f"{cls_name}  (AUC={roc_auc:.3f})")

    ax.plot([0,1],[0,1], ":", color=_BORD, lw=1.2, label="Random")
    ax.legend(loc="lower right", fontsize=9,
              facecolor=_CARD, edgecolor=_BORD, labelcolor=_TEXT1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: list[str],
                            importances: np.ndarray,
                            top_n: int = 15,
                            title: str = "Feature Importances") -> plt.Figure:
    """Horizontal bar chart of the top-N most important features."""
    pairs  = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 0.38 * top_n + 1.2), facecolor=_DARK)
    ax.set_facecolor(_CARD)

    bars = ax.barh(names, values, color=_EMRLD, edgecolor="none", height=0.6)
    # Colour bars by rank
    for i, bar in enumerate(bars):
        alpha = 0.4 + 0.6 * (i / max(len(bars) - 1, 1))
        bar.set_alpha(alpha)

    ax.set_xlabel("Importance", color=_TEXT1, fontsize=10)
    ax.set_title(title,         color=_TEXT1, fontsize=12, fontweight="bold")
    ax.tick_params(colors=_TEXT2, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_training_history(history, title: str = "LSTM Training History") -> plt.Figure:
    """Loss & accuracy curves from Keras history object."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor=_DARK)

    for ax, metric, val_metric, label, color in [
        (ax1, "loss",     "val_loss",     "Loss",     _CRIMN),
        (ax2, "accuracy", "val_accuracy", "Accuracy", _EMRLD),
    ]:
        ax.set_facecolor(_CARD)
        epochs = range(1, len(history.history[metric]) + 1)
        ax.plot(epochs, history.history[metric],     color=color, lw=1.8, label="Train")
        if val_metric in history.history:
            ax.plot(epochs, history.history[val_metric], color=color,
                    lw=1.8, linestyle="--", alpha=0.7, label="Val")
        ax.set_xlabel("Epoch",  color=_TEXT1, fontsize=10)
        ax.set_ylabel(label,    color=_TEXT1, fontsize=10)
        ax.set_title(f"{title} — {label}", color=_TEXT1, fontsize=11, fontweight="bold")
        ax.tick_params(colors=_TEXT2)
        ax.legend(fontsize=9, facecolor=_CARD, edgecolor=_BORD, labelcolor=_TEXT1)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════
#  Console report
# ════════════════════════════════════════════════════════════════

def print_report(results: dict):
    """Pretty-print the evaluation dict to stdout."""
    name = results["model_name"]
    sep  = "=" * 54
    print(f"\n{sep}")
    print(f"  {name} -- Evaluation Report")
    print(sep)
    print(f"  Accuracy         : {results['accuracy']:.4f}")
    print(f"  Precision (macro): {results['precision_macro']:.4f}")
    print(f"  Recall    (macro): {results['recall_macro']:.4f}")
    print(f"  F1        (macro): {results['f1_macro']:.4f}")
    print(f"  AUC-ROC   (macro): {results['auc_roc']:.4f}")
    print(f"\n  Per-class Precision:")
    for k, v in results["precision_per_class"].items():
        print(f"    {k:<6}: {v:.4f}")
    print(f"\n  Per-class Recall:")
    for k, v in results["recall_per_class"].items():
        print(f"    {k:<6}: {v:.4f}")
    print(f"\n  Confusion Matrix (rows=Actual, cols=Predicted):")
    print(f"  Labels: {LABEL_NAMES}")
    for row, lbl in zip(results["confusion_matrix"], LABEL_NAMES):
        print(f"  {lbl:<6}: {row.tolist()}")
    print(f"\n{results['classification_report']}")
    print(f"{sep}\n")
