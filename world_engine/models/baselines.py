```python
# /models/baselines.py
# ======================================================================================
# World Discovery Engine (WDE)
# Baselines — Lightweight, dependable tabular baselines for classification/regression
# --------------------------------------------------------------------------------------
# Purpose
#   Provide a single, convenient module to train/evaluate/save/load a suite of classic
#   scikit-learn baselines on tabular features (NumPy arrays or pandas DataFrames).
#
# Highlights
#   • One config to pick a model: 'logreg', 'ridge', 'lasso', 'elasticnet', 'svm',
#     'rf' (RandomForest), 'gb' (GradientBoosting), 'knn'.
#   • Robust preprocessing: numeric/categorical handling, missing values, scaling,
#     optional PCA, column dropping by missingness.
#   • Metrics: classification (acc, f1, roc-auc, logloss, ece, brier), regression
#     (rmse, mae, r2, mape). Handles binary/multiclass.
#   • Calibration (optional) via CalibratedClassifierCV (sigmoid or isotonic).
#   • Feature importance: model-native (tree-based) and permutation importance.
#   • Train/val/test splitting with stratification and sample weights support.
#   • Save/load via joblib (single .joblib artifact with everything).
#
# Design notes
#   • External deps kept modest: numpy + optional pandas; scikit-learn is required
#     for modeling; joblib for persistence. If scikit-learn is missing, we raise
#     a helpful ImportError.
#   • This file is Kaggle/CI-friendly: no GPU tooling; deterministic seeds.
#
# License
#   MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
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
    from sklearn.base import BaseEstimator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
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
        r2_score, mean_absolute_error, mean_squared_error
    )
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


def _nanmean_safe(x: np.ndarray) -> float:
    """
    Safe nanmean with fallback if array is empty.
    """
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _percent_missing_per_column(df: "pd.DataFrame") -> np.ndarray:
    """
    Percent missing per column (0..100). Returns zeros for non-pandas input.
    """
    if not (_PANDAS_AVAILABLE and isinstance(df, pd.DataFrame)):
        return np.zeros(0)
    return df.isna().mean(axis=0).values * 100.0


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
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
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
        One of: 'logreg', 'ridge', 'lasso', 'elasticnet', 'svm', 'rf', 'gb', 'knn'.
        For regression, 'logreg' is invalid (use ridge/lasso/elasticnet/svr/rf/gb/knn).
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

    Model hyperparameters (common)
    ------------------------------
    n_estimators : int
        For 'rf' and as subsampling iter in some models.
    max_depth : Optional[int]
        Max depth for 'rf' or as a guardrail for overfitting; None = unlimited (RF).
    learning_rate : float
        For 'gb' gradient boosting.
    C : float
        Regularization inverse strength for 'logreg'/'svm'.
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
        self.feature_names_in_: Optional[List[str]] = None  # for pandas
        self.is_fitted_: bool = False

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
        X_arr, y_arr = X, np.asarray(y)
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
        X_in, y = X, np.asarray(y)
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

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """
        Predict labels (classification) or values (regression).
        """
        self._assert_fitted()
        return self.pipeline_.predict(X)  # type: ignore

    def predict_proba(self, X: Union[np.ndarray, "pd.DataFrame"]) -> Optional[np.ndarray]:
        """
        Predict class probabilities (classification only). Returns None for regression.
        """
        self._assert_fitted()
        if self.cfg.problem_type != "classification":
            return None
        model: Any = self.pipeline_.named_steps["model"]  # type: ignore
        # If it's a calibrated classifier, it will expose predict_proba
        if hasattr(model, "predict_proba"):
            return self.pipeline_.predict_proba(X)  # type: ignore
        # SVC(probability=False) won't have predict_proba
        if hasattr(model, "decision_function"):
            # Map decision scores to probabilities via softmax over margins (approx)
            scores = self.pipeline_.decision_function(X)  # type: ignore
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
        report: Dict[str, Any] = {"model": asdict(self.cfg)}

        if self.cfg.problem_type == "classification":
            y_pred = self.predict(X)
            proba = self.predict_proba(X)
            # Metrics
            report["accuracy"] = float(accuracy_score(y, y_pred))
            report["f1_macro"] = float(f1_score(y, y_pred, average="macro"))
            report["precision_macro"] = float(precision_score(y, y_pred, average="macro", zero_division=0))
            report["recall_macro"] = float(recall_score(y, y_pred, average="macro", zero_division=0))
            # logloss/auc/brier if probs available
            if proba is not None:
                try:
                    if proba.shape[1] == 2:
                        auc = roc_auc_score(y, proba[:, 1])
                    else:
                        auc = roc_auc_score(y, proba, multi_class="ovr")
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
                cm = confusion_matrix(y, y_pred)
                report["confusion_matrix"] = cm.tolist()
            except Exception:
                report["confusion_matrix"] = None

        else:  # regression
            y_pred = self.predict(X)
            rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
            mae = float(mean_absolute_error(y, y_pred))
            r2 = float(r2_score(y, y_pred))
            # MAPE (careful with division by zero)
            denom = np.where(np.abs(y) < 1e-8, np.nan, np.abs(y))
            mape = float(np.nanmean(np.abs((y - y_pred) / denom)) * 100.0)
            report.update({"rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape})

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
        y = np.asarray(y)
        folds = int(self.cfg.cv_folds if cv_folds is None else cv_folds)
        if folds <= 1:
            return {"cv_folds": 0, "scores": [], "mean_score": float("nan"), "std_score": float("nan")}
        scoring_name = scoring if scoring is not None else self.cfg.scoring

        pipe = self._build_pipeline_for(X)  # fresh pipeline (unfitted)
        if self.cfg.problem_type == "classification":
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.cfg.random_state)
        else:
            cv = KFold(n_splits=folds, shuffle=True, random_state=self.cfg.random_state)

        scores = cross_val_score(pipe, X, y, scoring=scoring_name, cv=cv, n_jobs=None)
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
            est = self.pipeline_.named_steps["model"]  # type: ignore
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
            # Build a shallow clone (already-fitted) for permutation_importance
            try:
                r = permutation_importance(
                    self.pipeline_, X, y,
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
            "format_version": "1.0.0",
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
        return obj

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

        if use_pandas and not self.cfg.treat_all_as_numeric:
            df: pd.DataFrame = X  # type: ignore
            # Optionally drop columns by missingness
            if self.cfg.drop_missing_above_pct < 100.0:
                pct = df.isna().mean() * 100.0
                keep_cols = pct[pct <= self.cfg.drop_missing_above_pct].index.tolist()
                df = df[keep_cols]
                # NOTE: We return a view; caller's DataFrame is not mutated.

            # Detect types
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
            # Return a simple pipeline where 'model' is the calibrated wrapper,
            # but to maintain API we keep naming consistent:
            return Pipeline([("model", calibrated)])

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
                return GradientBoostingClassifier(
                    learning_rate=self.cfg.learning_rate,
                    n_estimators=self.cfg.n_estimators,
                    max_depth=3 if self.cfg.max_depth is None else self.cfg.max_depth,
                    random_state=self.cfg.random_state,
                )
            if m == "knn":
                return KNeighborsClassifier(
                    n_neighbors=self.cfg.n_neighbors,
                    weights="distance",
                    n_jobs=None,
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
            # LogisticRegression is invalid for regression
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
        # If we used calibration, the actual estimator is under 'model' step (CalibratedCV),
        # which wraps our original pipeline. In that case, names are hard to propagate;
        # we produce "feat_0..feat_{d-1}" as a fallback.
        pipe = self.pipeline_
        if pipe is None:
            return []
        try:
            # If we have ColumnTransformer with OneHotEncoder(s), extract names
            if "pre" in pipe.named_steps:
                pre = pipe.named_steps["pre"]
                names: List[str] = []
                if hasattr(pre, "get_feature_names_out"):
                    names = list(pre.get_feature_names_out())
                else:
                    # Fallback: try to infer from numeric/cat lists
                    names = []
                if names:
                    return names
        except Exception:
            pass
        # If PCA present, names are PC1..PCk
        try:
            if "pca" in pipe.named_steps:
                pca = pipe.named_steps["pca"]
                n = int(getattr(pca, "n_components_", getattr(pca, "n_components", 0)))
                if n and n > 0:
                    return [f"PC{i+1}" for i in range(n)]
        except Exception:
            pass
        # Fallback: if original pandas feature names known
        if self.feature_names_in_:
            return list(self.feature_names_in_)
        # Generic names (unknown)
        model = pipe.named_steps.get("model")
        try:
            d = int(model.n_features_in_)  # type: ignore
        except Exception:
            d = 0
        return [f"feat_{i}" for i in range(d)]


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
    )

    Xtr, Xte, ytr, yte = clf.train_test_split(Xc, yc)
    clf.fit(Xtr, ytr)
    report_c = clf.evaluate(Xte, yte)
    print("[classification] report:", json.dumps(report_c, indent=2)[:500], "...")

    imp_c = clf.feature_importance(Xte, yte, top_k=5, kind="auto")
    print("[classification] top-5 importance:", imp_c)

    # -------------------- Regression demo --------------------
    Nr, Dr = 1000, 10
    Xr = np.random.randn(Nr, Dr).astype(np.float32)
    beta = np.random.randn(Dr)
    yr = Xr @ beta + 0.5 * np.random.randn(Nr)

    reg = build_regressor(
        model="gb",
        n_estimators=300,
        learning_rate=0.05,
        random_state=11,
        cv_folds=3,
        scoring="neg_root_mean_squared_error",
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
    rep2 = restored.evaluate(Xte, yte)
    print("[reload] accuracy:", rep2.get("accuracy"), "roc_auc:", rep2.get("roc_auc"))
```
