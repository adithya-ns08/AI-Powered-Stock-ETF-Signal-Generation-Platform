"""
signal_engine/models.py
========================
Two model families for signal generation (TensorFlow/LSTM removed).

Classes
-------
RandomForestSignalModel   — sklearn RandomForestClassifier on flat features
XGBoostSignalModel        — XGBClassifier for gradient boosting on flat features

Each class implements:
    .fit(X_train, y_train, X_val, y_val)
    .predict_proba(X) → np.ndarray  shape (n, 3) probabilities [sell, hold, buy]
    .predict(X)       → np.ndarray  shape (n,)   discrete labels {0, 1, 2}
    .feature_importances_
"""

from __future__ import annotations

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ─────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

# ── XGBoost ──────────────────────────────────────────────────────
from xgboost import XGBClassifier
XGB_AVAILABLE = True


# ════════════════════════════════════════════════════════════════
#  Base helper
# ════════════════════════════════════════════════════════════════

class _BaseModel:
    """Shared interface contract."""
    def fit(self, X_train, y_train, X_val=None, y_val=None): raise NotImplementedError
    def predict_proba(self, X) -> np.ndarray: raise NotImplementedError
    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ════════════════════════════════════════════════════════════════
#  1. Random Forest
# ════════════════════════════════════════════════════════════════

class RandomForestSignalModel(_BaseModel):
    """
    RandomForestClassifier with Platt scaling calibration for well-calibrated
    probability estimates.
    """
    def __init__(self,
                 n_estimators: int = 30,
                 max_depth: int = 10,
                 min_samples_leaf: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42):
        self._rf = RandomForestClassifier(
            n_estimators    = n_estimators,
            max_depth       = max_depth,
            min_samples_leaf= min_samples_leaf,
            class_weight    = "balanced",
            n_jobs          = n_jobs,
            random_state    = random_state,
        )
        self._model = self._rf
        self.classes_ = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("[RandomForest] Training …")
        self._model.fit(X_train, y_train)
        self.classes_  = self._rf.classes_
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._rf.feature_importances_

    def top_features(self, feature_names: list[str], n: int = 15) -> list[tuple]:
        imp   = self.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        return pairs[:n]


# ════════════════════════════════════════════════════════════════
#  2. XGBoost / Gradient Boosting
# ════════════════════════════════════════════════════════════════

class XGBoostSignalModel(_BaseModel):
    """
    XGBClassifier with hyper-parameters tuned for financial time-series.
    """
    def __init__(self,
                 n_estimators: int = 600,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 subsample: float = 0.80,
                 colsample_bytree: float = 0.75,
                 gamma: float = 0.1,
                 reg_alpha: float = 0.1,
                 reg_lambda: float = 1.0,
                 early_stopping: int = 30,
                 n_jobs: int = -1,
                 random_state: int = 42):
        if XGB_AVAILABLE:
            self._model = XGBClassifier(
                n_estimators     = n_estimators,
                max_depth        = max_depth,
                learning_rate    = learning_rate,
                subsample        = subsample,
                colsample_bytree = colsample_bytree,
                gamma            = gamma,
                reg_alpha        = reg_alpha,
                reg_lambda       = reg_lambda,
                objective        = "multi:softprob",
                num_class        = 3,
                eval_metric      = "mlogloss",
                use_label_encoder= False,
                n_jobs           = n_jobs,
                random_state     = random_state,
                early_stopping_rounds = early_stopping,
                verbosity        = 0,
            )
        else:
            self._model = GradientBoostingClassifier(
                n_estimators  = min(n_estimators, 200),
                max_depth     = max_depth,
                learning_rate = learning_rate,
                subsample     = subsample,
                random_state  = random_state,
                verbose       = 0,
            )
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.utils.class_weight import compute_sample_weight
        sw = compute_sample_weight("balanced", y_train)
        print(f"[XGBoost] Training on {len(X_train)} samples …")

        if X_val is not None:
            self._model.fit(
                X_train, y_train,
                sample_weight = sw,
                eval_set      = [(X_val, y_val)],
                verbose       = False,
            )
            best = self._model.best_iteration
            print(f"[XGBoost] Best iteration: {best}")
        else:
            self._model.fit(X_train, y_train, sample_weight=sw)

        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_

    def top_features(self, feature_names: list[str], n: int = 15) -> list[tuple]:
        imp   = self.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        return pairs[:n]
