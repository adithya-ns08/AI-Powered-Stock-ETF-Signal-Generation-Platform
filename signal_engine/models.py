"""
signal_engine/models.py
========================
Three model families for signal generation.

Classes
-------
RandomForestSignalModel   — sklearn RandomForestClassifier on flat features
LSTMSignalModel           — Keras/TensorFlow sequential LSTM on time-series windows
XGBoostSignalModel        — XGBClassifier for gradient boosting on flat features

Each class implements:
    .fit(X_train, y_train, X_val, y_val)
    .predict_proba(X) → np.ndarray  shape (n, 3) probabilities [sell, hold, buy]
    .predict(X)       → np.ndarray  shape (n,)   discrete labels {0, 1, 2}
    .feature_importances_ (RF and XGBoost only)
"""

from __future__ import annotations

import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Scikit-learn ─────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# ── XGBoost ──────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier  # fallback
    XGB_AVAILABLE = False
    print("[models] xgboost not installed — using sklearn GradientBoosting as fallback.")

# ── TensorFlow / Keras ───────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[models] TensorFlow not installed — LSTM model will be unavailable.")


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

    Hyper-parameters chosen for quant finance:
      - n_estimators = 400  (enough trees for stable OOB error)
      - max_depth    = 12   (moderate depth prevents overfitting)
      - class_weight = 'balanced'  (handles Buy/Hold/Sell imbalance)
      - min_samples_leaf = 5  (smooths leaf probabilities)
    """

    def __init__(self,
                 n_estimators: int = 400,
                 max_depth: int = 12,
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
            oob_score       = True,
        )
        # Isotonic calibration wraps the RF for better P(class) estimates
        self._model = CalibratedClassifierCV(self._rf, method="isotonic", cv=3)
        self.classes_ = None
        self.is_fitted = False

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print("[RandomForest] Training …")
        self._model.fit(X_train, y_train)
        # OOB score is on the inner RF before calibration wrapping
        self._rf.fit(X_train, y_train)
        print(f"[RandomForest] OOB accuracy: {self._rf.oob_score_:.4f}")
        self.classes_  = self._rf.classes_
        self.is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Returns (n, 3) probabilities for [SELL=0, HOLD=1, BUY=2]."""
        return self._model.predict_proba(np.asarray(X, dtype=np.float32))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._rf.feature_importances_

    def top_features(self, feature_names: list[str], n: int = 15) -> list[tuple]:
        imp   = self.feature_importances_
        pairs = sorted(zip(feature_names, imp), key=lambda x: -x[1])
        return pairs[:n]


# ════════════════════════════════════════════════════════════════
#  2. LSTM (Keras / TensorFlow)
# ════════════════════════════════════════════════════════════════

class LSTMSignalModel(_BaseModel):
    """
    Stacked LSTM for multivariate time-series classification.

    Architecture
    ------------
    Input (lookback, n_features)
    → LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)
    → LSTM(64,  return_sequences=False, dropout=0.2, recurrent_dropout=0.1)
    → Dense(64, activation='relu', kernel_regularizer=L2)
    → Dropout(0.3)
    → Dense(3,  activation='softmax')   ← 3-class: sell / hold / buy

    Training
    --------
    - Adam with cosine decay learning rate
    - Sparse categorical cross-entropy (integer labels)
    - Early stopping on val_loss (patience=12, restore_best_weights)
    - ReduceLROnPlateau (factor=0.5, patience=5)
    - Class-weight balancing to avoid HOLD dominance
    """

    def __init__(self,
                 lstm_units: tuple[int, int] = (128, 64),
                 dense_units: int = 64,
                 dropout: float = 0.25,
                 learning_rate: float = 1e-3,
                 epochs: int = 80,
                 batch_size: int = 64,
                 n_classes: int = 3):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required for LSTMSignalModel.")
        self.lstm_units   = lstm_units
        self.dense_units  = dense_units
        self.dropout      = dropout
        self.lr           = learning_rate
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.n_classes    = n_classes
        self.model        = None
        self.history      = None
        self.is_fitted    = False

    def _build(self, input_shape: tuple):
        inp = keras.Input(shape=input_shape, name="ohlcv_sequence")

        x = layers.LSTM(self.lstm_units[0],
                        return_sequences=True,
                        dropout=self.dropout,
                        recurrent_dropout=0.1,
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="lstm1")(inp)
        x = layers.BatchNormalization()(x)

        x = layers.LSTM(self.lstm_units[1],
                        return_sequences=False,
                        dropout=self.dropout,
                        recurrent_dropout=0.1,
                        kernel_regularizer=regularizers.l2(1e-4),
                        name="lstm2")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Dense(self.dense_units,
                         activation="relu",
                         kernel_regularizer=regularizers.l2(1e-4),
                         name="dense1")(x)
        x = layers.Dropout(self.dropout + 0.05, name="dropout_head")(x)

        out = layers.Dense(self.n_classes, activation="softmax", name="signal_output")(x)

        model = keras.Model(inputs=inp, outputs=out, name="LSTMSignalModel")

        # Cosine annealing LR schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.lr,
            decay_steps=self.epochs * 10,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(lr_schedule),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def _class_weights(y: np.ndarray) -> dict:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        cw      = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes.astype(int), cw))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow is required.")
        print(f"[LSTM] Building model — input shape {X_train.shape[1:]}")
        self.model = self._build(input_shape=X_train.shape[1:])

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=5, min_lr=1e-6, verbose=0),
        ]

        val_data = (X_val, y_val) if X_val is not None else None
        cw       = self._class_weights(y_train)

        self.history = self.model.fit(
            X_train, y_train,
            validation_data = val_data,
            epochs          = self.epochs,
            batch_size      = self.batch_size,
            class_weight    = cw,
            callbacks       = cbs,
            verbose         = 1,
        )
        self.is_fitted = True
        best_epoch = np.argmin(self.history.history.get("val_loss", [0]))
        print(f"[LSTM] Best epoch: {best_epoch+1} | "
              f"val_loss: {min(self.history.history.get('val_loss', [0])):.4f}")
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=np.float32), verbose=0)

    def summary(self):
        if self.model:
            self.model.summary()


# ════════════════════════════════════════════════════════════════
#  3. XGBoost / Gradient Boosting
# ════════════════════════════════════════════════════════════════

class XGBoostSignalModel(_BaseModel):
    """
    XGBClassifier with hyper-parameters tuned for financial time-series.

    Key design choices
    ------------------
    - num_class = 3        (multiclass softmax)
    - use_label_encoder=False + eval_metric='mlogloss'
    - subsample + colsample_bytree  → variance reduction (bagging-like)
    - early_stopping_rounds via eval_set (uses validation data)
    - scale_pos_weight is handled via sample_weight in fit()
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
            # Fallback: sklearn GradientBoosting (no native multiclass softprob)
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

        if XGB_AVAILABLE and X_val is not None:
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
