# /world_engine/models/baselines.py
# ======================================================================================
# World Discovery Engine (WDE)
# Baselines — Lightweight, dependable tabular baselines for classification/regression
# --------------------------------------------------------------------------------------
# Purpose
#   Provide a single, convenient module to train/evaluate/save/load a suite of classic
#   scikit-learn baselines on tabular features (NumPy arrays or pandas DataFrames).
#
# Highlights (Upgraded)
#   • Single config to pick a model: 'logreg', 'logreg_l1', 'ridge', 'lasso', 'elasticnet',
#     'svm', 'rf' (RandomForest), 'gb' (GradientBoosting), 'knn'.
#   • Robust preprocessing: numeric/categorical handling, missing values, scaling,
#     optional PCA, variance thresholding, optional constant-column drop, explicit
#     numeric/categorical overrides, and safe handling of ±∞ (converted to NaN).
#   • Metrics: classification (acc, f1 macro/weighted, precision/recall macro/weighted,
#     roc-auc, logloss, ece, brier, per-class f1), regression (rmse, mae, r2, mape).
#     Handles binary/multiclass robustly.
#   • Calibration (optional) via CalibratedClassifierCV (sigmoid or isotonic).
#   • Threshold tuning utility for binary classification (optimize F1/BA/Youden-J).
#   • Feature importance: model-native (tree-based/linear) and permutation importance.
#   • Train/val/test splitting with stratification and sample weights support; optional CV.
#   • Save/load via joblib (single .joblib artifact with config + pipeline + metadata).
#   • Kaggle/CI friendly: CPU-only, deterministic seeds, modest dependencies.
#
# License
#   MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# Optional pandas for DataFrame column typing; module works with NumPy alone.
try:
    import pandas as pd  # type: ignore
    _PANDAS_AVAILABLE = True
except Exception:
    _PANDAS_AVAILABLE = False

# scikit-learn is the only hard dependency for modeling; fail gracefully if missing.
try:
    from sklearn.base import BaseEstimator, clone
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import (
        LogisticRegression, Ridge, Lasso, ElasticNet
    )
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, log_loss, brier_score_loss, confusion_matrix,
        r2_score, mean_absolute_error, mean_squared_error,
        balanced_accuracy_score
    )
    from sklearn.metrics import roc_curve
    from sklearn.inspection import permutation_importance
    import joblib  # type: ignore

    _SKLEARN_AVAILABLE = True
except Exception as _e:
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e  # stored to show helpful message later


# --------------------------------------------------------------------------------------
# Reproducibility & small utils
# --------------------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set NumPy RNG seed. (scikit-learn estimators will also use 'random_state'.)
    """
    np.random.seed(seed)


def _as_numpy(X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
    """
    Convert DataFrame to NumPy without losing numeric types; otherwise return as-is.
    """
    if _PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X)


def _sanitize_input(X: Union[np.ndarray, "pd.DataFrame"]) -> Union[np.ndarray, "pd.DataFrame"]:
    """
    Replace ±∞ with NaN to avoid failures in imputers/scalers.
    Operates in a copy-safe manner (does not mutate caller's object).
    """
    if _PANDAS_AVAILABLE and isinstance(X, pd.DataFrame):
        Xc = X.copy()
        return Xc.replace([np.inf, -np.inf], np.nan)
    Xn = np.array(X, copy=True)
    Xn[np.isinf(Xn)] = np.nan
    return Xn


def _nanmean_safe(x: np.ndarray) -> float:
    """
    Safe nanmean with fallback if array is empty.
    """
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


# --------------------------------------------------------------------------------------
# Expected Calibration Error (ECE) utility
# --------------------------------------------------------------------------------------

def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute ECE for binary/multiclass probabilities.

    Parameters
    ----------
    probs : np.ndarray, shape [N, C] or [N] for binary (interpreted as P(class 1))
    y_true : np.ndarray, shape [N] with integer labels
    n_bins : int, number of confidence bins in [0,1]

    Returns
    -------
    ece : float
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    if probs.ndim == 1:
        # binary: make [N,2] to reuse multiclass code
        probs = np.stack([1.0 - probs, probs], axis=1)

    conf = probs.max(axis=1)           # predicted confidence
    pred = probs.argmax(axis=1)        # predicted class
    correct = (pred == y_true).astype(float)
    # Include the left edge in the first bin and the right edge in the last bin
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == 0:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        w = mask.mean()
        ece += w * abs(avg_conf - acc)
    return float(ece)


# --------------------------------------------------------------------------------------
# Configuration dataclass
# --------------------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """
    Configuration for a baseline model and preprocessing.

    General
    -------
    problem_type : {"classification", "regression"}
        Type of task.
    model : str
        One of: 'logreg', 'logreg_l1', 'ridge', 'lasso', 'elasticnet', 'svm', 'rf', 'gb', 'knn'.
        For regression, 'logreg' variants are invalid (use ridge/lasso/elasticnet/svm/rf/gb/knn).
        For classification with 'svm', we use SVC (probability=True if calibrate=False).
    random_state : int
        RNG seed for reproducibility.

    Preprocessing
    -------------
    scale_numeric : bool
        Apply StandardScaler to numeric columns.
    pca_components : Optional[int or float]
        If int, number of components for PCA; if float in (0,1], keep components
        explaining that fraction of variance. None disables PCA.
    drop_missing_above_pct : float
        If using pandas DataFrame input, drop columns whose missingness percentage
        exceeds this threshold (0..100). Ignored for NumPy input.
    impute_numeric_strategy : {"median","mean"}
        Imputation strategy for numeric.
    impute_categorical_strategy : {"most_frequent","constant"}
        Imputation for categorical; constant value is "missing" if chosen.
    treat_all_as_numeric : bool
        If True, ignore categorical handling and treat everything as numeric
        (useful when your input is already numeric features).
    variance_threshold : Optional[float]
        If set, apply VarianceThreshold to remove near-constant features after preprocessing.
    drop_constant_cols : bool
        If True and pandas input, drop columns with a single unique value (post-NaN drop).
    numeric_cols : Optional[List[str]]
        Explicit list of numeric columns (pandas only). If provided, overrides detection.
    categorical_cols : Optional[List[str]]
        Explicit list of categorical columns (pandas only). If provided, overrides detection.

    Model hyperparameters (common)
    ------------------------------
    n_estimators : int
        For 'rf' and 'gb' number of trees.
    max_depth : Optional[int]
        Max depth for 'rf' and 'gb' (depth of individual trees); None = unlimited (RF).
    learning_rate : float
        For 'gb' gradient boosting.
    C : float
        Regularization inverse strength for 'logreg'/'svm'. For ridge/lasso/elasticnet we map to alpha=1/C.
    l1_ratio : float
        For ElasticNet [0..1].
    n_neighbors : int
        For 'knn'.

    Calibration (classification only)
    ---------------------------------
    calibrate : bool
        If True, wrap classifier with CalibratedClassifierCV.
    calibrate_method : {"sigmoid","isotonic"}
        Calibration method.
    calibrate_cv : int
        Number of CV folds used for calibration.

    Split
    -----
    test_size : float
        Fraction for test split.
    val_size : float
        Fraction of (train) reserved for validation metrics (optional).
    stratify : bool
        Stratify by y for classification splits.

    Cross-validation
    ----------------
    cv_folds : int
        If >1, .cross_validate() will run CV and report scores.
    scoring : Optional[str]
        scikit-learn scoring name for CV (e.g., "roc_auc", "neg_root_mean_squared_error").

    Misc
    ----
    class_weight : Optional[str]
        For classification models that support it (e.g., 'balanced').
    """
    # General
    problem_type: str = "classification"  # {"classification","regression"}
    model: str = "logreg"                 # see list above
    random_state: int = 42

    # Preprocessing
    scale_numeric: bool = True
    pca_components: Optional[Union[int, float]] = None
    drop_missing_above_pct: float = 100.0
    impute_numeric_strategy: str = "median"
    impute_categorical_strategy: str = "most_frequent"
    treat_all_as_numeric: bool = False
    variance_threshold: Optional[float] = None
    drop_constant_cols: bool = True
    numeric_cols: Optional[List[str]] = None
    categorical_cols: Optional[List[str]] = None

    # Model hypers
    n_estimators: int = 200
    max_depth: Optional[int] = None
    learning_rate: float = 0.05
    C: float = 1.0
    l1_ratio: float = 0.5
    n_neighbors: int = 25

    # Calibration
    calibrate: bool = False
    calibrate_method: str = "sigmoid"  # or "isotonic"
    calibrate_cv: int = 3

    # Split
    test_size: float = 0.2
    val_size: float = 0.0
    stratify: bool = True

    # CV
    cv_folds: int = 0
    scoring: Optional[str] = None

    # Misc
    class_weight: Optional[str] = None


# --------------------------------------------------------------------------------------
# BaselineModel wrapper
# --------------------------------------------------------------------------------------

class BaselineModel:
    """
    Unified wrapper around a scikit-learn baseline with robust preprocessing and metrics.

    Typical usage
    -------------
        cfg = BaselineConfig(problem_type="classification", model="rf", n_estimators=300)
        model = BaselineModel(cfg)

        X_train, X_test, y_train, y_test = model.train_test_split(X, y)
        model.fit(X_train, y_train)
        report = model.evaluate(X_test, y_test)
        model.save("artifacts/baseline_rf.joblib")

        # Later…
        model2 = BaselineModel.load("artifacts/baseline_rf.joblib")
        preds = model2.predict(X_new)
    """

    def __init__(self, config: BaselineConfig = BaselineConfig()):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"[baselines.py] scikit-learn is required. Import error: {_SKLEARN_IMPORT_ERROR}")
        self.cfg = config
        set_global_seed(self.cfg.random_state)
        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_in_: Optional[List[str]] = None  # for pandas input
        self.is_fitted_: bool = False
        self.binary_threshold_: Optional[float] = None  # optional tuned threshold for binary classification

    # ----------------------------- Public API -----------------------------------------

    def train_test_split(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Split X, y into train/test using config. Stratifies for classification.
        """
        X_arr, y_arr = _sanitize_input(X), np.asarray(y)
        strat = y_arr if (self.cfg.problem_type == "classification" and self.cfg.stratify) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_arr, y_arr,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=strat,
        )
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaselineModel":
        """
        Fit the baseline pipeline on data.
        """
        X_in, y = _sanitize_input(X), np.asarray(y)
        self.feature_names_in_ = list(X_in.columns) if (_PANDAS_AVAILABLE and isinstance(X_in, pd.DataFrame)) else None
        self.pipeline_ = self._build_pipeline_for(X_in)
        # Fit; pass sample_weight if supported by underlying estimator
        if sample_weight is not None:
            try:
                self.pipeline_.fit(X_in, y, **{"model__sample_weight": sample_weight})
            except Exception:
                # Some pipelines/estimators do not accept sample_weight; fallback
                self.pipeline_.fit(X_in, y)
        else:
            self.pipeline_.fit(X_in, y)
        self.is_fitted_ = True
        return self

    def fit_with_validation(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Fit with an internal train/validation split if cfg.val_size > 0 and return a validation report.
        If val_size == 0, just calls fit() and returns an empty dict.
        """
        if self.cfg.val_size and self.cfg.val_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                _sanitize_input(X), np.asarray(y),
                test_size=self.cfg.val_size,
                random_state=self.cfg.random_state,
                stratify=(y if (self.cfg.problem_type == "classification" and self.cfg.stratify) else None),
            )
            self.fit(X_train, y_train, sample_weight=sample_weight)
            return self.evaluate(X_val, y_val)
        else:
            self.fit(X, y, sample_weight=sample_weight)
            return {}

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Predict labels (classification) or values (regression).
        For binary classification, if a tuned threshold was set via tune_threshold(),
        returns class labels based on that threshold; otherwise uses estimator default.
        """
        self._assert_fitted()
        if self.cfg.problem_type != "classification":
            return self.pipeline_.predict(_sanitize_input(X))  # type: ignore
        # Classification path
        if self.binary_threshold_ is None:
            return self.pipeline_.predict(_sanitize_input(X))  # type: ignore
        # If we have a tuned threshold, use probabilities if available
        proba = self.predict_proba(X)
        if proba is None or proba.shape[1] != 2:
            # fallback to decision_function or default predict
            return self.pipeline_.predict(_sanitize_input(X))  # type: ignore
        p1 = proba[:, 1]
        return (p1 >= float(self.binary_threshold_)).astype(int)

    def predict_proba(self, X: Union[np.ndarray, "pd.DataFrame"]) -> Optional[np.ndarray]:
        """
        Predict class probabilities (classification only). Returns None for regression.
        """
        self._assert_fitted()
        if self.cfg.problem_type != "classification":
            return None
        Xs = _sanitize_input(X)
        model: Any = self.pipeline_.named_steps.get("model")  # type: ignore
        # Calibrated classifier wraps the pipeline and exposes predict_proba
        if hasattr(model, "predict_proba"):
            return self.pipeline_.predict_proba(Xs)  # type: ignore
        # SVC(probability=False) won't have predict_proba; try decision_function
        if hasattr(self.pipeline_, "decision_function"):
            scores = self.pipeline_.decision_function(Xs)  # type: ignore
            if scores.ndim == 1:
                # binary margin; map with logistic
                p1 = 1.0 / (1.0 + np.exp(-scores))
                return np.stack([1 - p1, p1], axis=1)
            else:
                # multiclass margins; softmax
                e = np.exp(scores - scores.max(axis=1, keepdims=True))
                return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
        return None

    def evaluate(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute a rich set of metrics depending on problem_type.
        """
        self._assert_fitted()
        y = np.asarray(y)
        Xs = _sanitize_input(X)
        report: Dict[str, Any] = {"model": asdict(self.cfg)}

        if self.cfg.problem_type == "classification":
            y_pred = self.predict(Xs)
            proba = self.predict_proba(Xs)
            # Metrics
            report["accuracy"] = float(accuracy_score(y, y_pred))
            report["balanced_accuracy"] = float(balanced_accuracy_score(y, y_pred))
            report["f1_macro"] = float(f1_score(y, y_pred, average="macro"))
            report["f1_weighted"] = float(f1_score(y, y_pred, average="weighted"))
            report["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0))
            report["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0))
            # per-class f1 (useful for imbalanced multiclass)
            try:
                unique_labels = np.unique(y)
                f1_per_class = f1_score(y, y_pred, average=None, labels=unique_labels)
                report["f1_per_class"] = {int(lbl): float(score) for lbl, score in zip(unique_labels, f1_per_class)}
            except Exception:
                report["f1_per_class"] = None

            # logloss/auc/brier if probs available
            if proba is not None:
                try:
                    if proba.shape[1] == 2:
                        auc = roc_auc_score(y, proba[:, 1])
                    else:
                        # If some classes are missing in y, roc_auc_score can fail; guard it
                        y_unique = np.unique(y)
                        if y_unique.size == proba.shape[1]:
                            auc = roc_auc_score(y, proba, multi_class="ovr")
                        else:
                            auc = float("nan")
                except Exception:
                    auc = float("nan")
                report["roc_auc"] = float(auc)
                # logloss
                try:
                    report["log_loss"] = float(log_loss(y, proba, labels=np.unique(y)))
                except Exception:
                    report["log_loss"] = float("nan")
                # brier score (binary only)
                if proba.shape[1] == 2:
                    try:
                        report["brier"] = float(brier_score_loss(y, proba[:, 1]))
                    except Exception:
                        report["brier"] = float("nan")
                # ECE
                try:
                    report["ece"] = expected_calibration_error(proba, y, n_bins=15)
                except Exception:
                    report["ece"] = float("nan")
            else:
                report["roc_auc"] = float("nan")
                report["log_loss"] = float("nan")
                report["brier"] = float("nan")
                report["ece"] = float("nan")

            # Confusion matrix
            try:
                cm = confusion_matrix(y, y_pred, labels=np.unique(y))
                report["confusion_matrix"] = cm.tolist()
            except Exception:
                report["confusion_matrix"] = None

            # If we used a tuned threshold, record it
            if self.binary_threshold_ is not None:
                report["binary_threshold"] = float(self.binary_threshold_)

        else:  # regression
            y_pred = self.predict(Xs)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            mae = float(mean_absolute_error(y, y_pred))
            r2 = float(r2_score(y, y_pred))
            # MAPE (careful with division by zero)
            denom = np.where(np.abs(y) < 1e-8, np.nan, np.abs(y))
            mape = float(np.nanmean(np.abs((y - y_pred) / denom)) * 100.0)
            # Residual diagnostics (basic)
            residuals = y - y_pred
            report.update({
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mape_pct": mape,
                "residual_mean": float(np.nanmean(residuals)),
                "residual_std": float(np.nanstd(residuals)),
                "residual_median_abs": float(np.nanmedian(np.abs(residuals))),
            })

        return report

    def cross_validate(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        scoring: Optional[str] = None,
        cv_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run K-fold (or StratifiedKFold) CV using the current config. Returns mean/std.
        """
        self._assert_not_fitted_yet("cross_validate")  # encourage CV on a fresh clone
        Xs, y = _sanitize_input(X), np.asarray(y)
        folds = int(self.cfg.cv_folds if cv_folds is None else cv_folds)
        if folds <= 1:
            return {"cv_folds": 0, "scores": [], "mean_score": float("nan"), "std_score": float("nan")}
        scoring_name = scoring if scoring is not None else self.cfg.scoring

        pipe = self._build_pipeline_for(Xs)  # fresh pipeline (unfitted)
        if self.cfg.problem_type == "classification":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.cfg.random_state)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=self.cfg.random_state)

        scores = cross_val_score(pipe, Xs, y, scoring=scoring_name, cv=cv, n_jobs=None)
        return {
            "cv_folds": folds,
            "scoring": scoring_name,
            "scores": [float(s) for s in scores],
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
        }

    def feature_importance(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Optional[np.ndarray] = None,
        top_k: int = 25,
        kind: str = "auto",  # {"auto","model","permutation"}
        n_repeats: int = 5,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute feature importances.

        Modes
        -----
        - "model": if underlying estimator exposes feature_importances_ or coef_.
        - "permutation": permutation_importance (requires y).
        - "auto": try model importance first, else fall back to permutation (if y provided).
        """
        self._assert_fitted()
        names = self._resolve_feature_names_after_preprocessing()
        result = {"names": [], "scores": [], "method": None}

        # Try model-native importance
        if kind in ("auto", "model"):
            est = self._get_final_estimator()
            try:
                if hasattr(est, "feature_importances_"):
                    scores = np.asarray(est.feature_importances_, dtype=float)
                    order = np.argsort(scores)[::-1]
                    order = order[: min(top_k, scores.size)]
                    result["names"] = [names[i] for i in order]
                    result["scores"] = [float(scores[i]) for i in order]
                    result["method"] = "model_feature_importances_"
                    return result
                if hasattr(est, "coef_"):
                    coef = np.asarray(est.coef_)
                    scores = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
                    order = np.argsort(scores)[::-1]
                    order = order[: min(top_k, scores.size)]
                    result["names"] = [names[i] for i in order]
                    result["scores"] = [float(scores[i]) for i in order]
                    result["method"] = "model_coef_abs"
                    return result
            except Exception:
                pass
            if kind == "model":
                return result  # model kind requested but not available

        # Fallback: permutation importance (requires y)
        if kind in ("auto", "permutation"):
            if y is None:
                return result  # cannot compute permutation without targets
            y = np.asarray(y)
            try:
                r = permutation_importance(
                    self.pipeline_, _sanitize_input(X), y,
                    n_repeats=n_repeats,
                    random_state=self.cfg.random_state if random_state is None else random_state,
                    n_jobs=None,
                )
                scores = r.importances_mean
                order = np.argsort(scores)[::-1]
                order = order[: min(top_k, scores.size)]
                result["names"] = [names[i] for i in order]
                result["scores"] = [float(scores[i]) for i in order]
                result["method"] = "permutation_importance"
                return result
            except Exception:
                return result

        return result  # empty if nothing worked

    def tune_threshold(
        self,
        X_val: Union[np.ndarray, "pd.DataFrame"],
        y_val: np.ndarray,
        metric: str = "f1",
    ) -> Dict[str, Any]:
        """
        Tune a decision threshold for **binary** classification using validation data.
        metric ∈ {"f1", "balanced_accuracy", "youden_j"}.

        Returns
        -------
        dict with keys: {"metric", "best_threshold", "best_score"}
        """
        self._assert_fitted()
        if self.cfg.problem_type != "classification":
            raise ValueError("Threshold tuning only applies to classification.")
        proba = self.predict_proba(X_val)
        if proba is None or proba.shape[1] != 2:
            raise ValueError("Threshold tuning requires binary probabilities (shape [N,2]).")
        y_val = np.asarray(y_val)
        p1 = proba[:, 1]

        fpr, tpr, thr = roc_curve(y_val, p1)
        # roc_curve returns thresholds aligned with tpr/fpr except the first element (inf) in some versions
        # We'll build a candidate grid including those thresholds plus a uniform grid for robustness.
        grid = np.unique(np.concatenate([
            thr[np.isfinite(thr)],
            np.linspace(0.0, 1.0, 101)
        ]))
        best_t, best_score = 0.5, -np.inf
        for t in grid:
            yp = (p1 >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_val, yp, zero_division=0)
            elif metric in ("balanced_accuracy", "ba"):
                score = balanced_accuracy_score(y_val, yp)
            elif metric in ("youden_j", "youden"):
                # J = TPR - FPR (use roc_curve pair closest to t)
                idx = np.argmin(np.abs(thr - t)) if thr.size > 0 else 0
                J = (tpr[idx] - fpr[idx]) if (0 <= idx < tpr.size and 0 <= idx < fpr.size) else float("-inf")
                score = float(J)
            else:
                raise ValueError(f"Unsupported metric for threshold tuning: {metric}")
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        self.binary_threshold_ = best_t
        return {"metric": metric, "best_threshold": float(best_t), "best_score": float(best_score)}

    def save(self, path: str) -> None:
        """
        Save the entire fitted BaselineModel (config + pipeline + metadata) to a .joblib file.
        """
        self._assert_fitted()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "config": asdict(self.cfg),
            "feature_names_in": self.feature_names_in_,
            "pipeline": self.pipeline_,
            "is_fitted": self.is_fitted_,
            "binary_threshold": self.binary_threshold_,
            "format_version": "1.1.0",
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """
        Load a BaselineModel from a .joblib file created by save().
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"[baselines.py] scikit-learn is required. Import error: {_SKLEARN_IMPORT_ERROR}")
        payload = joblib.load(path)
        cfg = BaselineConfig(**payload["config"])
        obj = cls(cfg)
        obj.feature_names_in_ = payload.get("feature_names_in")
        obj.pipeline_ = payload["pipeline"]
        obj.is_fitted_ = bool(payload.get("is_fitted", True))
        obj.binary_threshold_ = payload.get("binary_threshold", None)
        return obj

    def export_report_json(self, report: Dict[str, Any], path: str) -> None:
        """
        Save a metrics report (from evaluate/cross_validate) as JSON.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # ----------------------------- Internal: pipeline ---------------------------------

    def _build_pipeline_for(self, X: Union[np.ndarray, "pd.DataFrame"]) -> Pipeline:
        """
        Construct a scikit-learn Pipeline with preprocessing + selected model (+ optional calibration).
        The ColumnTransformer handles numeric/categorical when pandas is available.
        """
        # Identify column types if pandas and treat_all_as_numeric=False
        numeric_features: List[str] = []
        categorical_features: List[str] = []
        use_pandas = _PANDAS_AVAILABLE and isinstance(X, pd.DataFrame)

        # Potentially drop missingness-heavy and constant columns (pandas path only)
        X_work = X
        if use_pandas:
            X_work = _sanitize_input(X)  # replace infs with NaN
            df: pd.DataFrame = X_work  # type: ignore
            if self.cfg.drop_missing_above_pct < 100.0:
                pct = df.isna().mean() * 100.0
                keep_cols = pct[pct <= self.cfg.drop_missing_above_pct].index.tolist()
                df = df[keep_cols]
            if self.cfg.drop_constant_cols:
                nunq = df.nunique(dropna=True)
                keep_cols2 = nunq[nunq > 1].index.tolist()
                df = df[keep_cols2]
            X_work = df  # updated DataFrame

        if use_pandas and not self.cfg.treat_all_as_numeric:
            df = X_work  # type: ignore
            # Explicit overrides take precedence if provided
            if self.cfg.numeric_cols is not None or self.cfg.categorical_cols is not None:
                if self.cfg.numeric_cols is not None:
                    missing = [c for c in self.cfg.numeric_cols if c not in df.columns]
                    if missing:
                        warnings.warn(f"[baselines] numeric_cols missing in DataFrame: {missing}")
                    numeric_features = [c for c in (self.cfg.numeric_cols or []) if c in df.columns]
                else:
                    numeric_features = [c for c, dtype in df.dtypes.items() if pd.api.types.is_numeric_dtype(dtype)]
                if self.cfg.categorical_cols is not None:
                    missing = [c for c in self.cfg.categorical_cols if c not in df.columns]
                    if missing:
                        warnings.warn(f"[baselines] categorical_cols missing in DataFrame: {missing}")
                    categorical_features = [c for c in (self.cfg.categorical_cols or []) if c in df.columns]
                else:
                    categorical_features = [c for c, dtype in df.dtypes.items() if not pd.api.types.is_numeric_dtype(dtype)]
            else:
                # Automatic detection
                for col, dtype in df.dtypes.items():
                    if pd.api.types.is_numeric_dtype(dtype):
                        numeric_features.append(col)
                    else:
                        categorical_features.append(col)

            # Build pipelines per column type
            num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=self.cfg.impute_numeric_strategy))]
            if self.cfg.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            num_pipe = Pipeline(num_steps)

            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=self.cfg.impute_categorical_strategy, fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ])

            pre = ColumnTransformer(
                transformers=[
                    ("num", num_pipe, numeric_features),
                    ("cat", cat_pipe, categorical_features),
                ],
                remainder="drop",
                sparse_threshold=0.0,
            )

            steps: List[Tuple[str, Any]] = [("pre", pre)]

        else:
            # NumPy path or treat_all_as_numeric=True: treat everything as numeric
            num_steps = [("imputer", SimpleImputer(strategy=self.cfg.impute_numeric_strategy))]
            if self.cfg.scale_numeric:
                num_steps.append(("scaler", StandardScaler()))
            pre = Pipeline(num_steps)
            steps = [("pre", pre)]

        # Optional PCA
        if self.cfg.pca_components is not None:
            steps.append(("pca", PCA(n_components=self.cfg.pca_components, random_state=self.cfg.random_state)))

        # Optional Variance Threshold (post-scaling/PCA)
        if self.cfg.variance_threshold is not None:
            steps.append(("varth", VarianceThreshold(threshold=float(self.cfg.variance_threshold))))

        # Model selection
        model = self._build_model()
        steps.append(("model", model))

        pipe = Pipeline(steps)

        # Optional calibration for classification models
        if self.cfg.problem_type == "classification" and self.cfg.calibrate:
            # Wrap the whole pipeline in a CalibratedClassifierCV (note: calibration refits model)
            # We construct a new pipeline where the estimator to calibrate is the entire pipe.
            calibrated = CalibratedClassifierCV(
                base_estimator=pipe,
                method=self.cfg.calibrate_method,
                cv=self.cfg.calibrate_cv,
            )
            # Return a simple pipeline where 'model' is the calibrated wrapper
            return Pipeline([("model", calibrated)])

        # If we altered DataFrame columns (drop steps), keep feature_names_in_ in sync for later reference
        if use_pandas:
            self.feature_names_in_ = list(X_work.columns)  # type: ignore

        return pipe

    def _build_model(self) -> BaseEstimator:
        """
        Instantiate the chosen baseline estimator with config hyperparameters.
        """
        m = self.cfg.model.lower().strip()
        pt = self.cfg.problem_type.lower().strip()

        if pt == "classification":
            if m == "logreg":
                # Multinomial with lbfgs handles multi-class cleanly; class weights optional
                return LogisticRegression(
                    C=self.cfg.C,
                    solver="lbfgs",
                    multi_class="auto",
                    class_weight=self.cfg.class_weight,
                    random_state=self.cfg.random_state,
                    max_iter=1000,
                )
            if m == "logreg_l1":
                # L1-penalized logistic regression (liblinear or saga)
                solver = "liblinear"
                # Use saga for multinomial if needed; keep liblinear default for binary/small
                return LogisticRegression(
                    C=self.cfg.C,
                    penalty="l1",
                    solver=solver,
                    multi_class="auto",
                    class_weight=self.cfg.class_weight,
                    random_state=self.cfg.random_state,
                    max_iter=1000,
                )
            if m == "svm":
                # SVC with RBF; if calibration=True, probability flag is redundant (CalibratedCV wraps)
                return SVC(
                    C=self.cfg.C,
                    kernel="rbf",
                    gamma="scale",
                    probability=not self.cfg.calibrate,  # if calibrating, CalibratedCV handles probabilities
                    class_weight=self.cfg.class_weight,
                    random_state=self.cfg.random_state,
                )
            if m == "rf":
                return RandomForestClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    random_state=self.cfg.random_state,
                    class_weight=self.cfg.class_weight,
                    n_jobs=-1,
                )
            if m == "gb":
                # GradientBoostingClassifier does not support class_weight directly
                return GradientBoostingClassifier(
                    learning_rate=self.cfg.learning_rate,
                    n_estimators=self.cfg.n_estimators,
                    max_depth=3 if self.cfg.max_depth is None else self.cfg.max_depth,
                    random_state=self.cfg.random_state,
                )
            if m == "knn":
                # Some sklearn versions don't support n_jobs in KNN; keep None for portability
                return KNeighborsClassifier(
                    n_neighbors=self.cfg.n_neighbors,
                    weights="distance",
                )
            # Ridge/Lasso/ElasticNet are regression-only; fall through to error
            raise ValueError(f"Unsupported classification model: {self.cfg.model}")

        elif pt == "regression":
            if m == "ridge":
                return Ridge(alpha=1.0 / max(self.cfg.C, 1e-6), random_state=self.cfg.random_state)
            if m == "lasso":
                return Lasso(alpha=1.0 / max(self.cfg.C, 1e-6), random_state=self.cfg.random_state, max_iter=10000)
            if m == "elasticnet":
                return ElasticNet(
                    alpha=1.0 / max(self.cfg.C, 1e-6),
                    l1_ratio=self.cfg.l1_ratio,
                    random_state=self.cfg.random_state,
                    max_iter=10000,
                )
            if m == "svm":
                return SVR(C=self.cfg.C, kernel="rbf", gamma="scale")
            if m == "rf":
                return RandomForestRegressor(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    random_state=self.cfg.random_state,
                    n_jobs=-1,
                )
            if m == "gb":
                return GradientBoostingRegressor(
                    learning_rate=self.cfg.learning_rate,
                    n_estimators=self.cfg.n_estimators,
                    max_depth=3 if self.cfg.max_depth is None else self.cfg.max_depth,
                    random_state=self.cfg.random_state,
                )
            if m == "knn":
                return KNeighborsRegressor(
                    n_neighbors=self.cfg.n_neighbors,
                    weights="distance",
                )
            # LogisticRegression variants are invalid for regression
            raise ValueError(f"Unsupported regression model: {self.cfg.model}")

        else:
            raise ValueError(f"Unknown problem_type: {self.cfg.problem_type}")

    # ----------------------------- Internal: helpers ----------------------------------

    def _assert_fitted(self) -> None:
        if not self.is_fitted_ or self.pipeline_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit(X, y) first.")

    def _assert_not_fitted_yet(self, fn: str) -> None:
        if self.is_fitted_:
            raise RuntimeError(f"{fn} should be called before fit(). Start from a fresh model instance.")

    def _resolve_feature_names_after_preprocessing(self) -> List[str]:
        """
        Attempt to recover feature names after preprocessing (esp. OneHot).
        Falls back to generic names if not available.
        """
        pipe = self.pipeline_
        if pipe is None:
            return []
        # If calibrated, the pipeline is wrapped; try to reach inside
        try:
            if "model" in pipe.named_steps and isinstance(pipe.named_steps["model"], CalibratedClassifierCV):
                # Base estimator is our entire original pipeline
                base = pipe.named_steps["model"].base_estimator  # type: ignore
                if isinstance(base, Pipeline):
                    pipe = base
        except Exception:
            pass
        # ColumnTransformer names
        try:
            if isinstance(pipe, Pipeline) and "pre" in pipe.named_steps:
                pre = pipe.named_steps["pre"]
                if hasattr(pre, "get_feature_names_out"):
                    names = list(pre.get_feature_names_out())
                    if names:
                        return names
        except Exception:
            pass
        # PCA present -> PCs
        try:
            if isinstance(pipe, Pipeline) and "pca" in pipe.named_steps:
                pca = pipe.named_steps["pca"]
                n = int(getattr(pca, "n_components_", getattr(pca, "n_components", 0)))
                if n and n > 0:
                    return [f"PC{i+1}" for i in range(n)]
        except Exception:
            pass
        # Fallback: original pandas feature names
        if self.feature_names_in_:
            return list(self.feature_names_in_)
        # Generic names (unknown)
        est = self._get_final_estimator()
        try:
            d = int(est.n_features_in_)  # type: ignore
        except Exception:
            d = 0
        return [f"feat_{i}" for i in range(d)]

    def _get_final_estimator(self) -> Any:
        """
        Return the final estimator after preprocessing and optional wrappers.
        """
        if self.pipeline_ is None:
            return None
        est = self.pipeline_.named_steps.get("model")
        # If calibrated, estimator is nested
        if isinstance(est, CalibratedClassifierCV):
            base = est.base_estimator
            if isinstance(base, Pipeline):
                return base.named_steps.get("model")
            return base
        return est


# --------------------------------------------------------------------------------------
# Convenience builders
# --------------------------------------------------------------------------------------

def build_classifier(model: str = "rf", **kwargs: Any) -> BaselineModel:
    """
    Quick constructor for a classification baseline.
    Example:
        model = build_classifier("logreg", calibrate=True, calibrate_method="isotonic")
    """
    cfg = BaselineConfig(problem_type="classification", model=model, **kwargs)
    return BaselineModel(cfg)


def build_regressor(model: str = "rf", **kwargs: Any) -> BaselineModel:
    """
    Quick constructor for a regression baseline.
    Example:
        model = build_regressor("elasticnet", C=10.0, l1_ratio=0.3)
    """
    cfg = BaselineConfig(problem_type="regression", model=model, **kwargs)
    return BaselineModel(cfg)


# --------------------------------------------------------------------------------------
# Self-test (CPU) — synthetic demos for classification & regression
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    if not _SKLEARN_AVAILABLE:
        raise SystemExit(f"scikit-learn missing: '{_SKLEARN_IMPORT_ERROR}'. Install scikit-learn to run self-test.")

    set_global_seed(7)

    # -------------------- Classification demo (multiclass) --------------------
    N, D, C = 1200, 12, 3
    Xc = np.random.randn(N, D).astype(np.float32)
    w = np.random.randn(D, C)
    logits = Xc @ w + 0.5 * np.random.randn(N, C)
    yc = logits.argmax(axis=1)

    clf = build_classifier(
        model="rf",
        n_estimators=250,
        max_depth=None,
        calibrate=True,            # show calibrated pipeline path
        calibrate_method="sigmoid",
        calibrate_cv=3,
        stratify=True,
        random_state=7,
        cv_folds=3,
        scoring="roc_auc_ovr",
        variance_threshold=None,
        pca_components=None,
    )

    Xtr, Xte, ytr, yte = clf.train_test_split(Xc, yc)
    clf.fit(Xtr, ytr)
    report_c = clf.evaluate(Xte, yte)
    print("[classification] report:", json.dumps(report_c, indent=2)[:800], "...")

    imp_c = clf.feature_importance(Xte, yte, top_k=5, kind="auto")
    print("[classification] top-5 importance:", imp_c)

    # -------------------- Classification demo (binary with threshold tuning) --------------------
    Nb, Db = 1000, 8
    Xb = np.random.randn(Nb, Db).astype(np.float32)
    # Linear separator with noise
    beta = np.random.randn(Db)
    margin = Xb @ beta + 0.2 * np.random.randn(Nb)
    yb = (margin > 0.0).astype(int)

    clf_bin = build_classifier(model="logreg", calibrate=False, random_state=13)
    Xb_tr, Xb_te, yb_tr, yb_te = clf_bin.train_test_split(Xb, yb)
    clf_bin.fit(Xb_tr, yb_tr)
    # Tune threshold on the validation portion of the test split (toy example)
    tune = clf_bin.tune_threshold(Xb_te, yb_te, metric="f1")
    print("[binary] tuned threshold:", tune)
    rep_bin = clf_bin.evaluate(Xb_te, yb_te)
    print("[binary] post-tune report:", json.dumps(rep_bin, indent=2))

    # -------------------- Regression demo --------------------
    Nr, Dr = 1000, 10
    Xr = np.random.randn(Nr, Dr).astype(np.float32)
    beta_r = np.random.randn(Dr)
    yr = Xr @ beta_r + 0.5 * np.random.randn(Nr)

    reg = build_regressor(
        model="gb",
        n_estimators=300,
        learning_rate=0.05,
        random_state=11,
        cv_folds=3,
        scoring="neg_root_mean_squared_error",
        variance_threshold=None,
        pca_components=None,
    )

    Xtr, Xte, ytr, yte = reg.train_test_split(Xr, yr)
    reg.fit(Xtr, ytr)
    report_r = reg.evaluate(Xte, yte)
    print("[regression] report:", json.dumps(report_r, indent=2))

    imp_r = reg.feature_importance(Xte, yte, top_k=5, kind="auto")
    print("[regression] top-5 importance:", imp_r)

    # -------------------- Save/Load smoke test --------------------
    out_path = "./_baseline_rf_calibrated.joblib"
    clf.save(out_path)
    restored = BaselineModel.load(out_path)
    rep2 = restored.evaluate(Xte, yte)  # reuse regression test arrays just to validate call path shape differences
    print("[reload] keys:", list(rep2.keys())[:6], "...")
