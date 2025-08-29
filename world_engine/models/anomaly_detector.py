# /world_engine/models/anomaly_detector.py
# ======================================================================================
# World Discovery Engine (WDE)
# AnomalyDetector — Unified API for geospatial/tabular anomaly detection (Upgraded)
# --------------------------------------------------------------------------------------
# What this is
# ------------
# A single, configurable interface around multiple anomaly detection backends:
#   • 'zscore'          : mean/std Z-score (L2 aggregate)
#   • 'robust_zscore'   : median/MAD Z-score (outlier-robust)
#   • 'mahalanobis'     : covariance-based Mahalanobis distance
#   • 'isoforest'       : Isolation Forest (sklearn)
#   • 'lof'             : Local Outlier Factor (sklearn, novelty mode)
#   • 'ocsvm'           : One-Class SVM (sklearn)
#   • 'autoencoder_cnn' : tiny ConvAE on image patches [N, C, H, W] (PyTorch optional)
#
# It supports:
#   • Tabular arrays    X.shape == [N, D] for all non-neural backends
#   • Image patches     X.shape == [N, C, H, W] for the autoencoder backend
#   • Deterministic seeds, lightweight defaults, and CPU-only paths (Kaggle/CI-safe)
#   • Sliding-window raster scoring helper (CPU, NumPy) with batch control
#   • Save/Load to a directory with versioned metadata; joblib for sklearn, pt for torch
#   • Threshold utilities: percentile or fixed cutoff; FPR-target thresholding
#   • Explain helpers for Z-score/Robust/Mahalanobis (top contributing features)
#
# Design goals
# ------------
#   • Numpy-first API; gracefully handle missing/inf values (sanitize to NaN then impute)
#   • Optional StandardScaler on sklearn backends (config.use_scaler)
#   • Robust Z-score based on MAD = 1.4826 * median(|x - median|)
#   • Covariance via Ledoit-Wolf shrinkage when available (fallback to np.cov)
#   • Avoid GPU dependencies unless autoencoder is explicitly requested
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ----------------------------- Optional / required deps ------------------------------
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM
    from sklearn.covariance import LedoitWolf
    import joblib  # type: ignore

    _SKLEARN_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e


# ======================================================================================
# Utilities
# ======================================================================================

API_FORMAT_VERSION = "2.0.0"


def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility across numpy and torch (if available).
    """
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic algs can be set true if you accept perf tradeoffs:
        # torch.use_deterministic_algorithms(True)
        torch.use_deterministic_algorithms(False)


def _sanitize_array(X: np.ndarray) -> np.ndarray:
    """
    Replace ±∞ with NaN. Ensure float dtype. Do not modify caller input in-place.
    """
    Xc = np.array(X, copy=True)
    if not np.issubdtype(Xc.dtype, np.floating):
        Xc = Xc.astype(np.float32, copy=False)
    Xc[np.isinf(Xc)] = np.nan
    return Xc


def _impute_with_feature_median(X: np.ndarray) -> np.ndarray:
    """
    Simple, dependency-light imputation: per-column median for NaNs.
    Works on 2D arrays. Copies the data.
    """
    if X.ndim != 2:
        raise ValueError("Median imputer expects tabular [N, D] array.")
    Xc = np.array(X, copy=True)
    nan_mask = np.isnan(Xc)
    if not nan_mask.any():
        return Xc
    med = np.nanmedian(Xc, axis=0)
    # replace NaN columns that are all NaN with zeros to avoid NaNs lingering
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(nan_mask)
    Xc[inds] = np.take(med, inds[1])
    return Xc


def _safe_covariance(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (mean, covariance) using Ledoit-Wolf if available (sklearn), else np.cov.

    Returns
    -------
    mean : (D,)
    cov  : (D, D)
    """
    mu = np.nanmean(X, axis=0)
    Xc = X - mu
    Xc = np.where(np.isnan(Xc), 0.0, Xc)
    if _SKLEARN_AVAILABLE:
        try:
            lw = LedoitWolf().fit(Xc)
            return mu, lw.covariance_
        except Exception:
            pass
    # fallback
    cov = np.cov(Xc, rowvar=False)
    return mu, cov


def _pinv_psd(cov: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Pseudo-inverse with Tikhonov regularization for numerical stability.
    """
    # Ensure symmetric
    cov = 0.5 * (cov + cov.T)
    # Add small ridge on diagonal
    cov = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)
    try:
        return np.linalg.pinv(cov)
    except Exception:
        # ultra fallback
        return np.eye(cov.shape[0], dtype=cov.dtype) / max(eps, 1e-8)


# ======================================================================================
# Config
# ======================================================================================

@dataclass
class AnomalyDetectorConfig:
    """
    Parameters controlling the AnomalyDetector behavior.

    method : str
        {'zscore','robust_zscore','mahalanobis','isoforest','lof','ocsvm','autoencoder_cnn'}
    seed : int
        RNG seed.
    use_scaler : bool
        Apply StandardScaler for sklearn backends (IF/LOF/OCSVM) for stability.

    Z-score
    -------
    zscore_columns : Optional[List[int]]  Columns to use (default: all)
    zscore_clip_sigma : Optional[float]   Clip |z| to this before aggregation (None disables)

    Robust Z-score (median/MAD)
    ---------------------------
    rzs_columns : Optional[List[int]]
    rzs_clip_sigma : Optional[float]      Clip |z| prior to aggregation
    rzs_mad_eps : float                   Floor MAD to avoid divide-by-zero

    Mahalanobis
    -----------
    maha_columns : Optional[List[int]]
    maha_reg_eps : float                  Diagonal ridge added to covariance
    maha_whiten : bool                    If True, return norm in whitened space; else classic M-distance

    Isolation Forest
    ----------------
    isoforest_n_estimators : int
    isoforest_max_samples : Union[int,float,str]
    isoforest_contamination : Optional[float]
    isoforest_max_features : Union[int,float]

    LOF
    ---
    lof_n_neighbors : int
    lof_metric : str
    lof_novelty : bool

    One-Class SVM
    -------------
    ocsvm_kernel : str ('rbf','linear','poly','sigmoid')
    ocsvm_nu : float                       An upper bound on training outliers fraction (0..1]
    ocsvm_gamma : Union[float,str]         'scale' or 'auto' or float

    Autoencoder CNN (image patches)
    -------------------------------
    ae_epochs : int
    ae_batch_size : int
    ae_lr : float
    ae_weight_decay : float
    ae_hidden_mult : int
    ae_device : str ('cpu' or 'cuda')
    ae_patience : int
    ae_val_split : float (0..0.49)
    """
    method: str = "zscore"
    seed: int = 42
    use_scaler: bool = True

    # zscore
    zscore_columns: Optional[List[int]] = None
    zscore_clip_sigma: Optional[float] = 6.0

    # robust zscore
    rzs_columns: Optional[List[int]] = None
    rzs_clip_sigma: Optional[float] = 6.0
    rzs_mad_eps: float = 1e-6

    # mahalanobis
    maha_columns: Optional[List[int]] = None
    maha_reg_eps: float = 1e-6
    maha_whiten: bool = False

    # Isolation Forest
    isoforest_n_estimators: int = 200
    isoforest_max_samples: Union[int, float, str] = "auto"
    isoforest_contamination: Optional[float] = None
    isoforest_max_features: Union[int, float] = 1.0

    # LOF
    lof_n_neighbors: int = 20
    lof_metric: str = "minkowski"
    lof_novelty: bool = True

    # OCSVM
    ocsvm_kernel: str = "rbf"
    ocsvm_nu: float = 0.05
    ocsvm_gamma: Union[str, float] = "scale"

    # AE
    ae_epochs: int = 10
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 0.0
    ae_hidden_mult: int = 2
    ae_device: str = "cpu"
    ae_patience: int = 5
    ae_val_split: float = 0.0


# ======================================================================================
# Strategy Base
# ======================================================================================

class _BaseStrategy:
    """
    Minimal interface for anomaly backends.
    """

    def fit(self, X: np.ndarray) -> "_BaseStrategy":
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "_BaseStrategy":
        raise NotImplementedError

    # Optional: explainability (tabular) — return top-k contributing features per row
    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        raise NotImplementedError


# ======================================================================================
# Z-score (mean/std)
# ======================================================================================

class _ZScoreStrategy(_BaseStrategy):
    """
    Classic Z-score anomaly detector. Scores = L2 norm of Z across configured columns.
    """

    def __init__(self, columns: Optional[List[int]] = None, clip_sigma: Optional[float] = 6.0):
        self.columns = columns
        self.clip_sigma = clip_sigma
        self.mu_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.cols_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_ZScoreStrategy":
        if X.ndim != 2:
            raise ValueError("ZScore expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        cols = np.arange(Xs.shape[1]) if self.columns is None else np.array(self.columns, dtype=int)
        Xc = Xs[:, cols].astype(np.float64)
        self.mu_ = np.nanmean(Xc, axis=0)
        self.std_ = np.nanstd(Xc, axis=0, ddof=1)
        self.std_[self.std_ == 0.0] = 1e-8
        self.cols_ = cols
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.std_ is None or self.cols_ is None:
            raise RuntimeError("ZScoreStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Z = (Xs[:, self.cols_] - self.mu_) / self.std_
        if self.clip_sigma is not None:
            Z = np.clip(Z, -self.clip_sigma, self.clip_sigma)
        return np.linalg.norm(Z, ord=2, axis=1)

    def save(self, path: str) -> None:
        blob = {
            "columns": None if self.cols_ is None else self.cols_.tolist(),
            "mu": None if self.mu_ is None else self.mu_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "clip_sigma": self.clip_sigma,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "_ZScoreStrategy":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        obj = cls(columns=blob["columns"], clip_sigma=blob["clip_sigma"])
        obj.cols_ = None if blob["columns"] is None else np.array(blob["columns"], dtype=int)
        obj.mu_ = None if blob["mu"] is None else np.array(blob["mu"], dtype=float)
        obj.std_ = None if blob["std"] is None else np.array(blob["std"], dtype=float)
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        if self.mu_ is None or self.std_ is None or self.cols_ is None:
            raise RuntimeError("ZScoreStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Z = (Xs[:, self.cols_] - self.mu_) / self.std_
        if self.clip_sigma is not None:
            Z = np.clip(Z, -self.clip_sigma, self.clip_sigma)
        absZ = np.abs(Z)
        idxs = np.argsort(-absZ, axis=1)[:, :k]
        names = feature_names if feature_names is not None else [f"feat_{i}" for i in range(Xs.shape[1])]
        out: List[List[Tuple[str, float]]] = []
        for n, row in enumerate(idxs):
            pairs: List[Tuple[str, float]] = []
            for j in row:
                col = int(self.cols_[j])
                pairs.append((names[col], float(absZ[n, j])))
            out.append(pairs)
        return out


# ======================================================================================
# Robust Z-score (median/MAD)
# ======================================================================================

class _RobustZScoreStrategy(_BaseStrategy):
    """
    Robust Z-score using median and 1.4826*MAD (median absolute deviation).
    """

    def __init__(self, columns: Optional[List[int]] = None, clip_sigma: Optional[float] = 6.0, mad_eps: float = 1e-6):
        self.columns = columns
        self.clip_sigma = clip_sigma
        self.mad_eps = mad_eps
        self.med_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.cols_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_RobustZScoreStrategy":
        if X.ndim != 2:
            raise ValueError("RobustZScore expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        cols = np.arange(Xs.shape[1]) if self.columns is None else np.array(self.columns, dtype=int)
        Xc = Xs[:, cols].astype(np.float64)
        med = np.nanmedian(Xc, axis=0)
        mad = np.nanmedian(np.abs(Xc - med), axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale < self.mad_eps, self.mad_eps, scale)
        self.med_ = med
        self.scale_ = scale
        self.cols_ = cols
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.med_ is None or self.scale_ is None or self.cols_ is None:
            raise RuntimeError("RobustZScoreStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Z = (Xs[:, self.cols_] - self.med_) / self.scale_
        if self.clip_sigma is not None:
            Z = np.clip(Z, -self.clip_sigma, self.clip_sigma)
        return np.linalg.norm(Z, ord=2, axis=1)

    def save(self, path: str) -> None:
        blob = {
            "columns": None if self.cols_ is None else self.cols_.tolist(),
            "med": None if self.med_ is None else self.med_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
            "clip_sigma": self.clip_sigma,
            "mad_eps": self.mad_eps,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "_RobustZScoreStrategy":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        obj = cls(columns=blob["columns"], clip_sigma=blob["clip_sigma"], mad_eps=blob.get("mad_eps", 1e-6))
        obj.cols_ = None if blob["columns"] is None else np.array(blob["columns"], dtype=int)
        obj.med_ = None if blob["med"] is None else np.array(blob["med"], dtype=float)
        obj.scale_ = None if blob["scale"] is None else np.array(blob["scale"], dtype=float)
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        if self.med_ is None or self.scale_ is None or self.cols_ is None:
            raise RuntimeError("RobustZScoreStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Z = (Xs[:, self.cols_] - self.med_) / self.scale_
        if self.clip_sigma is not None:
            Z = np.clip(Z, -self.clip_sigma, self.clip_sigma)
        absZ = np.abs(Z)
        idxs = np.argsort(-absZ, axis=1)[:, :k]
        names = feature_names if feature_names is not None else [f"feat_{i}" for i in range(Xs.shape[1])]
        out: List[List[Tuple[str, float]]] = []
        for n, row in enumerate(idxs):
            pairs: List[Tuple[str, float]] = []
            for j in row:
                col = int(self.cols_[j])
                pairs.append((names[col], float(absZ[n, j])))
            out.append(pairs)
        return out


# ======================================================================================
# Mahalanobis distance
# ======================================================================================

class _MahalanobisStrategy(_BaseStrategy):
    """
    Mahalanobis distance with shrinkage covariance (Ledoit-Wolf) when available.
    Score = sqrt( (x - μ)^T Σ^{-1} (x - μ) ) if maha_whiten=False
    Or L2 norm in whitened space if maha_whiten=True (equivalent ordering).
    """

    def __init__(self, columns: Optional[List[int]] = None, reg_eps: float = 1e-6, whiten: bool = False):
        self.columns = columns
        self.reg_eps = reg_eps
        self.whiten = whiten
        self.mu_: Optional[np.ndarray] = None
        self.Sigma_inv_: Optional[np.ndarray] = None
        self.cols_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_MahalanobisStrategy":
        if X.ndim != 2:
            raise ValueError("Mahalanobis expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        cols = np.arange(Xs.shape[1]) if self.columns is None else np.array(self.columns, dtype=int)
        Xc = Xs[:, cols].astype(np.float64)
        mu, cov = _safe_covariance(Xc)
        inv = _pinv_psd(cov, eps=self.reg_eps)
        self.mu_ = mu
        self.Sigma_inv_ = inv
        self.cols_ = cols
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.Sigma_inv_ is None or self.cols_ is None:
            raise RuntimeError("MahalanobisStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Xc = Xs[:, self.cols_] - self.mu_
        if self.whiten:
            # Compute sqrt of inv via eig (Σ^{-1/2}) and apply; then L2 norm
            vals, vecs = np.linalg.eigh(self.Sigma_inv_)
            vals = np.clip(vals, 0.0, None)
            root_inv = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
            Y = Xc @ root_inv.T
            return np.linalg.norm(Y, ord=2, axis=1)
        # Classic M-distance
        tmp = Xc @ self.Sigma_inv_
        d2 = np.sum(tmp * Xc, axis=1)
        d2 = np.where(d2 < 0, 0, d2)
        return np.sqrt(d2)

    def save(self, path: str) -> None:
        blob = {
            "columns": None if self.cols_ is None else self.cols_.tolist(),
            "mu": None if self.mu_ is None else self.mu_.tolist(),
            "Sigma_inv": None if self.Sigma_inv_ is None else self.Sigma_inv_.tolist(),
            "reg_eps": self.reg_eps,
            "whiten": self.whiten,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "_MahalanobisStrategy":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        obj = cls(columns=blob["columns"], reg_eps=float(blob.get("reg_eps", 1e-6)), whiten=bool(blob.get("whiten", False)))
        obj.cols_ = None if blob["columns"] is None else np.array(blob["columns"], dtype=int)
        obj.mu_ = None if blob["mu"] is None else np.array(blob["mu"], dtype=float)
        Sigma_inv = blob.get("Sigma_inv", None)
        obj.Sigma_inv_ = None if Sigma_inv is None else np.array(Sigma_inv, dtype=float)
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        """
        Use absolute whitened contributions |(Σ^{-1/2}(x-μ))_j| as per-feature magnitudes.
        """
        if self.mu_ is None or self.Sigma_inv_ is None or self.cols_ is None:
            raise RuntimeError("MahalanobisStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        Xc = Xs[:, self.cols_] - self.mu_
        vals, vecs = np.linalg.eigh(self.Sigma_inv_)
        vals = np.clip(vals, 0.0, None)
        root_inv = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
        Y = Xc @ root_inv.T  # whitened coords
        absY = np.abs(Y)
        idxs = np.argsort(-absY, axis=1)[:, :k]
        names = feature_names if feature_names is not None else [f"feat_{i}" for i in range(Xs.shape[1])]
        out: List[List[Tuple[str, float]]] = []
        for n, row in enumerate(idxs):
            pairs: List[Tuple[str, float]] = []
            for j in row:
                col = int(self.cols_[j])
                pairs.append((names[col], float(absY[n, j])))
            out.append(pairs)
        return out


# ======================================================================================
# Isolation Forest (sklearn)
# ======================================================================================

class _IsoForestStrategy(_BaseStrategy):
    """
    Isolation Forest. sklearn's score_samples: higher => more normal. We invert.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[int, float, str] = "auto",
        contamination: Optional[float] = None,
        max_features: Union[int, float] = 1.0,
        use_scaler: bool = True,
        seed: int = 42,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        self.use_scaler = use_scaler
        self.pipeline_: Optional[Pipeline] = None
        self.params = dict(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray) -> "_IsoForestStrategy":
        if X.ndim != 2:
            raise ValueError("IsolationForest expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("iso", IsolationForest(**self.params)))
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("IsoForestStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        s = self.pipeline_.score_samples(Xs)  # higher => more normal
        return -s

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("IsoForestStrategy not fitted.")
        joblib.dump({"pipeline": self.pipeline_, "use_scaler": self.use_scaler}, path)

    @classmethod
    def load(cls, path: str) -> "_IsoForestStrategy":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        blob = joblib.load(path)
        obj = cls()
        obj.pipeline_ = blob["pipeline"]
        obj.use_scaler = bool(blob.get("use_scaler", True))
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        # Tree-based IF does not expose per-sample feature attribution in sklearn; return empty.
        return [[] for _ in range(X.shape[0])]


# ======================================================================================
# LOF (sklearn)
# ======================================================================================

class _LOFStrategy(_BaseStrategy):
    """
    LOF in novelty mode supports score_samples on new data.
    We negate and shift so that higher => more anomalous.
    """

    def __init__(self, n_neighbors: int = 20, metric: str = "minkowski", novelty: bool = True, use_scaler: bool = True):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.novelty = novelty
        self.use_scaler = use_scaler
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X: np.ndarray) -> "_LOFStrategy":
        if X.ndim != 2:
            raise ValueError("LOF expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("lof", LocalOutlierFactor(n_neighbors=self.n_neighbors, metric=self.metric, novelty=self.novelty, n_jobs=-1)))
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("LOFStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        s = self.pipeline_.score_samples(Xs)  # higher => more normal
        s = -s
        s = s - s.min()
        return s

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("LOFStrategy not fitted.")
        joblib.dump(
            {
                "pipeline": self.pipeline_,
                "n_neighbors": self.n_neighbors,
                "metric": self.metric,
                "novelty": self.novelty,
                "use_scaler": self.use_scaler,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "_LOFStrategy":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        blob = joblib.load(path)
        obj = cls(
            n_neighbors=int(blob["n_neighbors"]),
            metric=str(blob["metric"]),
            novelty=bool(blob["novelty"]),
            use_scaler=bool(blob["use_scaler"]),
        )
        obj.pipeline_ = blob["pipeline"]
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        # Nearest-neighbor density method; per-sample contributions are not exposed. Return empty.
        return [[] for _ in range(X.shape[0])]


# ======================================================================================
# One-Class SVM (sklearn)
# ======================================================================================

class _OCSVMStrategy(_BaseStrategy):
    """
    One-Class SVM. decision_function: + values => inliers, negative => outliers.
    We invert so that higher => more anomalous (i.e., -decision_function).
    """

    def __init__(self, kernel: str = "rbf", nu: float = 0.05, gamma: Union[str, float] = "scale", use_scaler: bool = True, seed: int = 42):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        self.use_scaler = use_scaler
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X: np.ndarray) -> "_OCSVMStrategy":
        if X.ndim != 2:
            raise ValueError("OCSVM expects [N, D].")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("ocsvm", OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)))
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(Xs)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("OCSVMStrategy not fitted.")
        Xs = _impute_with_feature_median(_sanitize_array(X))
        # decision_function: positive inliers, negative outliers
        d = self.pipeline_.decision_function(Xs)
        return -d  # higher => more anomalous

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("OCSVMStrategy not fitted.")
        joblib.dump({"pipeline": self.pipeline_, "kernel": self.kernel, "nu": self.nu, "gamma": self.gamma, "use_scaler": self.use_scaler}, path)

    @classmethod
    def load(cls, path: str) -> "_OCSVMStrategy":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        blob = joblib.load(path)
        obj = cls(kernel=str(blob["kernel"]), nu=float(blob["nu"]), gamma=blob["gamma"], use_scaler=bool(blob["use_scaler"]))
        obj.pipeline_ = blob["pipeline"]
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        # Not exposed in sklearn; return empty.
        return [[] for _ in range(X.shape[0])]


# ======================================================================================
# Autoencoder (Tiny Conv) for [N, C, H, W]
# ======================================================================================

class _SimpleConvAutoencoder(nn.Module):
    """
    Minimal ConvAE:
      Encoder: Conv->ReLU->Conv->ReLU (downsample /2 then /4)
      Decoder: ConvT->ReLU->ConvT mirrors to original size
    """

    def __init__(self, in_channels: int = 3, hidden_mult: int = 2):
        super().__init__()
        base = 16 * hidden_mult
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        return self.dec(z)


class _AutoencoderStrategy(_BaseStrategy):
    """
    Train a tiny ConvAE; score by per-sample MSE recon error.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_mult: int = 2,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
        patience: int = 5,
        val_split: float = 0.0,
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError(f"PyTorch not available: {_TORCH_IMPORT_ERROR}")
        set_global_seed(seed)
        self.model = _SimpleConvAutoencoder(in_channels=in_channels, hidden_mult=hidden_mult)
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model.to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_split = max(0.0, min(0.49, val_split))
        self.best_state_: Optional[Dict[str, Any]] = None

    @staticmethod
    def _loader(X: np.ndarray, batch_size: int, shuffle: bool) -> "torch.utils.data.DataLoader":
        t = torch.as_tensor(X, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(t, t)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def fit(self, X: np.ndarray) -> "_AutoencoderStrategy":
        if X.ndim != 4:
            raise ValueError("Autoencoder expects [N, C, H, W].")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # naive split at the tail
        n = X.shape[0]
        if self.val_split > 0.0 and n >= 10:
            n_val = max(1, int(n * self.val_split))
            Xtr, Xval = X[:-n_val], X[-n_val:]
        else:
            Xtr, Xval = X, None
        train_loader = self._loader(Xtr, self.batch_size, shuffle=True)
        val_loader = self._loader(Xval, self.batch_size, shuffle=False) if Xval is not None else None

        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        crit = nn.MSELoss()

        best_val = float("inf")
        bad = 0
        self.best_state_ = None

        for _ in range(self.epochs):
            self.model.train()
            total = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                rec = self.model(xb)
                loss = crit(rec, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * xb.size(0)
            train_loss = total / len(train_loader.dataset)

            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        rec = self.model(xb)
                        loss = crit(rec, yb)
                        total += float(loss.item()) * xb.size(0)
                val_loss = total / len(val_loader.dataset)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                bad = 0
                self.best_state_ = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if self.best_state_ is not None:
            self.model.load_state_dict(self.best_state_)
        self.model.eval()
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError("Autoencoder expects [N, C, H, W].")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        loader = self._loader(X, self.batch_size, shuffle=False)
        crit = nn.MSELoss(reduction="none")
        out = np.zeros(X.shape[0], dtype=np.float32)
        i0 = 0
        with torch.no_grad():
            for xb, _ in loader:
                bs = xb.size(0)
                xb = xb.to(self.device, non_blocking=True)
                rec = self.model(xb)
                err = crit(rec, xb).mean(dim=(1, 2, 3))  # per-sample MSE
                out[i0:i0 + bs] = err.detach().cpu().numpy()
                i0 += bs
        return out

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "ae_state.pt"))
        meta = {"in_channels": int(next(self.model.parameters()).shape[1]), "hidden_mult": int(getattr(self.model.enc[0], "out_channels", 16) // 16)}
        with open(os.path.join(path, "ae_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "_AutoencoderStrategy":
        if not _TORCH_AVAILABLE:
            raise ImportError(f"PyTorch not available: {_TORCH_IMPORT_ERROR}")
        with open(os.path.join(path, "ae_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            in_channels=int(meta.get("in_channels", 3)),
            hidden_mult=int(meta.get("hidden_mult", 2)),
            epochs=1,
            batch_size=64,
            lr=1e-3,
            device="cpu",
            seed=42,
            patience=1,
            val_split=0.0,
        )
        state = torch.load(os.path.join(path, "ae_state.pt"), map_location="cpu")
        obj.model.load_state_dict(state)
        obj.model.eval()
        return obj

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        # Image patches: no tabular feature names — return empty.
        return [[] for _ in range(X.shape[0])]


# ======================================================================================
# Public Wrapper
# ======================================================================================

class AnomalyDetector:
    """
    Unified wrapper with a strategy backend.

    Methods
    -------
    fit(X) -> self
    score_samples(X) -> np.ndarray[ N ]
    predict(X, threshold='p99' or quantile in [0,1] or absolute float) -> np.ndarray[bool]
    threshold_for_fpr(scores_inliers, scores_outliers, target_fpr=0.01) -> float
    explain_topk(X, k=5, feature_names=None) -> per-sample top-k contributors (when supported)
    save(dir_path), load(dir_path) -> persistence
    raster_sliding_window_scores(...) -> anomaly heatmap over rasters
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self._backend: Optional[_BaseStrategy] = None
        set_global_seed(self.config.seed)
        self._init_backend()

    def _init_backend(self) -> None:
        m = self.config.method.lower().strip()
        if m == "zscore":
            self._backend = _ZScoreStrategy(columns=self.config.zscore_columns, clip_sigma=self.config.zscore_clip_sigma)
        elif m == "robust_zscore":
            self._backend = _RobustZScoreStrategy(
                columns=self.config.rzs_columns, clip_sigma=self.config.rzs_clip_sigma, mad_eps=self.config.rzs_mad_eps
            )
        elif m == "mahalanobis":
            self._backend = _MahalanobisStrategy(
                columns=self.config.maha_columns, reg_eps=self.config.maha_reg_eps, whiten=self.config.maha_whiten
            )
        elif m == "isoforest":
            self._backend = _IsoForestStrategy(
                n_estimators=self.config.isoforest_n_estimators,
                max_samples=self.config.isoforest_max_samples,
                contamination=self.config.isoforest_contamination,
                max_features=self.config.isoforest_max_features,
                use_scaler=self.config.use_scaler,
                seed=self.config.seed,
            )
        elif m == "lof":
            self._backend = _LOFStrategy(
                n_neighbors=self.config.lof_n_neighbors,
                metric=self.config.lof_metric,
                novelty=self.config.lof_novelty,
                use_scaler=self.config.use_scaler,
            )
        elif m == "ocsvm":
            self._backend = _OCSVMStrategy(
                kernel=self.config.ocsvm_kernel,
                nu=self.config.ocsvm_nu,
                gamma=self.config.ocsvm_gamma,
                use_scaler=self.config.use_scaler,
                seed=self.config.seed,
            )
        elif m == "autoencoder_cnn":
            # Lazy create in fit() when in_channels known
            self._backend = None
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    # --------------------------------- Core API ---------------------------------------

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        if self.config.method == "autoencoder_cnn":
            if X.ndim != 4:
                raise ValueError("autoencoder_cnn expects [N, C, H, W] image patches.")
            in_ch = int(X.shape[1])
            self._backend = _AutoencoderStrategy(
                in_channels=in_ch,
                hidden_mult=self.config.ae_hidden_mult,
                epochs=self.config.ae_epochs,
                batch_size=self.config.ae_batch_size,
                lr=self.config.ae_lr,
                weight_decay=self.config.ae_weight_decay,
                device=self.config.ae_device,
                seed=self.config.seed,
                patience=self.config.ae_patience,
                val_split=self.config.ae_val_split,
            )
        if self._backend is None:
            raise RuntimeError("Backend not initialized.")
        self._backend.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("Detector not fitted.")
        return self._backend.score_samples(X)

    def predict(self, X: np.ndarray, threshold: Union[float, str] = "p99") -> np.ndarray:
        """
        threshold:
          - 'pXX': string percentile (e.g., 'p99' -> top 1%)
          - float in [0,1]: quantile
          - float > 1: absolute cutoff
        """
        s = self.score_samples(X)
        cutoff = self._resolve_threshold(s, threshold)
        return s >= cutoff

    @staticmethod
    def _resolve_threshold(scores: np.ndarray, threshold: Union[float, str]) -> float:
        if isinstance(threshold, str):
            st = threshold.lower().strip()
            if st.startswith("p"):
                try:
                    q = float(st[1:]) / 100.0
                except Exception:
                    raise ValueError(f"Invalid percentile string: {threshold}")
                return float(np.quantile(scores, q))
            else:
                raise ValueError(f"Unknown threshold format: {threshold}")
        elif isinstance(threshold, (int, float)):
            t = float(threshold)
            if 0.0 <= t <= 1.0:
                return float(np.quantile(scores, t))
            return t
        else:
            raise TypeError("threshold must be float or 'pXX'")

    @staticmethod
    def threshold_for_fpr(scores_inliers: np.ndarray, scores_outliers: np.ndarray, target_fpr: float = 0.01) -> float:
        """
        Choose a threshold such that FPR (inliers misclassified as outliers) ≈ target_fpr.
        Returns the score cutoff; higher scores => more anomalous.
        """
        s_in = np.asarray(scores_inliers, dtype=float)
        s_out = np.asarray(scores_outliers, dtype=float)
        if s_in.ndim != 1 or s_out.ndim != 1:
            raise ValueError("scores must be 1D.")
        # Sweep unique candidate thresholds from inliers+outliers
        cand = np.unique(np.concatenate([s_in, s_out]))
        if cand.size == 0:
            return float("inf")
        # Evaluate FPR at each threshold (score >= t => predicted outlier)
        # We choose the smallest t whose FPR <= target.
        cand_sorted = np.sort(cand)
        N_in = float(s_in.size)
        best_t = cand_sorted[-1]
        for t in cand_sorted:
            fp = float((s_in >= t).sum())
            fpr = fp / max(N_in, 1.0)
            if fpr <= target_fpr:
                best_t = t
                break
        return float(best_t)

    def explain_topk(self, X: np.ndarray, k: int = 5, feature_names: Optional[List[str]] = None) -> List[List[Tuple[str, float]]]:
        """
        Returns per-row list of (feature_name, magnitude) for the top-k contributors when supported.
        Implemented for: zscore, robust_zscore, mahalanobis. Others return empty lists.
        """
        if self._backend is None:
            raise RuntimeError("Detector not fitted.")
        try:
            return self._backend.explain_topk(X, k=k, feature_names=feature_names)
        except NotImplementedError:
            return [[] for _ in range(X.shape[0])]

    # -------------------------------- Persistence -------------------------------------

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        meta = {"format_version": API_FORMAT_VERSION, "config": asdict(self.config)}
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self._backend is None:
            raise RuntimeError("No backend to save. Call fit() first.")
        m = self.config.method
        if m == "zscore":
            self._backend.save(os.path.join(dir_path, "zscore.json"))
        elif m == "robust_zscore":
            self._backend.save(os.path.join(dir_path, "robust_zscore.json"))
        elif m == "mahalanobis":
            self._backend.save(os.path.join(dir_path, "mahalanobis.json"))
        elif m == "isoforest":
            self._backend.save(os.path.join(dir_path, "isoforest.joblib"))
        elif m == "lof":
            self._backend.save(os.path.join(dir_path, "lof.joblib"))
        elif m == "ocsvm":
            self._backend.save(os.path.join(dir_path, "ocsvm.joblib"))
        elif m == "autoencoder_cnn":
            self._backend.save(os.path.join(dir_path, "autoencoder"))
        else:
            raise ValueError(f"Unknown method: {m}")

    @classmethod
    def load(cls, dir_path: str) -> "AnomalyDetector":
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg_dict = meta.get("config", meta)  # backward compatible if older saved only config dict
        cfg = AnomalyDetectorConfig(**cfg_dict)
        obj = cls(cfg)
        m = cfg.method
        if m == "zscore":
            obj._backend = _ZScoreStrategy.load(os.path.join(dir_path, "zscore.json"))
        elif m == "robust_zscore":
            obj._backend = _RobustZScoreStrategy.load(os.path.join(dir_path, "robust_zscore.json"))
        elif m == "mahalanobis":
            obj._backend = _MahalanobisStrategy.load(os.path.join(dir_path, "mahalanobis.json"))
        elif m == "isoforest":
            obj._backend = _IsoForestStrategy.load(os.path.join(dir_path, "isoforest.joblib"))
        elif m == "lof":
            obj._backend = _LOFStrategy.load(os.path.join(dir_path, "lof.joblib"))
        elif m == "ocsvm":
            obj._backend = _OCSVMStrategy.load(os.path.join(dir_path, "ocsvm.joblib"))
        elif m == "autoencoder_cnn":
            obj._backend = _AutoencoderStrategy.load(os.path.join(dir_path, "autoencoder"))
        else:
            raise ValueError(f"Unknown method: {m}")
        return obj

    # -------------------------- Sliding-window raster scoring --------------------------

    @staticmethod
    def raster_sliding_window_scores(
        patcher: "RasterPatcherProtocol",
        detector: "AnomalyDetector",
        stride: int,
        normalize: bool = True,
        percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
        dtype: str = "float32",
        max_batch: int = 256,
    ) -> np.ndarray:
        """
        Build an anomaly heatmap over a raster by sliding window patches.
        Accumulates overlapping patch scores by mean.

        Parameters
        ----------
        patcher : RasterPatcherProtocol
            .shape() -> (H,W,C), .get_patch(y0,x0,ps) -> [C,ps,ps] or [ps,ps,C], .patch_size, .channels_first
        detector : AnomalyDetector
            Fitted detector. For 'autoencoder_cnn' expects channels-first tensors.
        stride : int
            Pixel step between patch top-left corners.
        normalize : bool
            Per-channel percentile normalization to [0,1].
        percentile_clip : Optional[Tuple[float,float]]
            (lo,hi) percentiles for normalization (ignored if normalize=False).
        dtype : str
            Output dtype for heatmap.
        max_batch : int
            Batch size of patches per scoring call (trade memory/speed).

        Returns
        -------
        heatmap : np.ndarray [H,W] of normalized anomaly intensities in [0,1].
        """
        H, W, C, channels_first = _infer_patcher_shape(patcher)
        ps = patcher.patch_size

        heat = np.zeros((H, W), dtype=np.float64)
        cnt = np.zeros((H, W), dtype=np.float64)

        patches: List[np.ndarray] = []
        coords: List[Tuple[int, int]] = []

        def _flush():
            if not patches:
                return
            X = np.stack(patches, axis=0)
            # optional normalization
            if normalize:
                Xn = _per_channel_percentile_normalize(X, channels_first, percentile_clip)
            else:
                Xn = X.astype(np.float32, copy=False)

            # convert to detector input
            if detector.config.method == "autoencoder_cnn":
                if not channels_first:
                    Xn = np.transpose(Xn, (0, 3, 1, 2))
            else:
                # tabular flatten
                if channels_first:
                    Xn = Xn.reshape(Xn.shape[0], -1)
                else:
                    Xn = np.transpose(Xn, (0, 3, 1, 2)).reshape(Xn.shape[0], -1)

            s = detector.score_samples(Xn)
            # accumulate to windows (uniform write to the window area)
            for (y0, x0), sv in zip(coords, s):
                ys = slice(y0, min(y0 + ps, H))
                xs = slice(x0, min(x0 + ps, W))
                heat[ys, xs] += float(sv)
                cnt[ys, xs] += 1.0

            patches.clear()
            coords.clear()

        for y0 in range(0, max(1, H - ps + 1), stride):
            for x0 in range(0, max(1, W - ps + 1), stride):
                p = patcher.get_patch(y0, x0, ps)
                patches.append(p)
                coords.append((y0, x0))
                if len(patches) >= max_batch:
                    _flush()
        _flush()

        cnt[cnt == 0] = 1.0
        hm = (heat / cnt).astype(dtype)
        # normalize to [0,1]
        mn, mx = float(hm.min()), float(hm.max())
        if mx > mn:
            hm = (hm - mn) / (mx - mn)
        return hm


# ======================================================================================
# Raster utilities & protocol
# ======================================================================================

class RasterPatcherProtocol:
    """
    Duck-typed protocol for raster patchers.

    Required:
      patch_size: int
      channels_first: bool
      shape() -> (H, W, C)
      get_patch(y0:int, x0:int, size:int) -> np.ndarray of [C,H,W] or [H,W,C] consistent with channels_first
    """

    patch_size: int
    channels_first: bool

    def shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    def get_patch(self, y0: int, x0: int, size: int) -> np.ndarray:
        raise NotImplementedError


def _infer_patcher_shape(patcher: RasterPatcherProtocol) -> Tuple[int, int, int, bool]:
    s = patcher.shape()
    if len(s) != 3:
        raise ValueError("patcher.shape() must return (H, W, C)")
    H, W, C = int(s[0]), int(s[1]), int(s[2])
    return H, W, C, bool(patcher.channels_first)


def _per_channel_percentile_normalize(
    X: np.ndarray, channels_first: bool, percentile_clip: Optional[Tuple[float, float]]
) -> np.ndarray:
    """
    Per-channel map to [0,1] using percentiles. Safe for both channel orders.
    """
    if channels_first:
        B, C, H, W = X.shape
        Xn = np.empty_like(X, dtype=np.float32)
        for c in range(C):
            flat = X[:, c, :, :].reshape(-1)
            lo, hi = (np.percentile(flat, percentile_clip) if percentile_clip is not None
                      else (float(flat.min()), float(flat.max())))
            if not np.isfinite(hi - lo) or hi <= lo:
                hi = lo + 1e-6
            Xn[:, c, :, :] = ((X[:, c, :, :] - lo) / (hi - lo)).astype(np.float32)
        return Xn
    else:
        B, H, W, C = X.shape
        Xn = np.empty_like(X, dtype=np.float32)
        for c in range(C):
            flat = X[:, :, :, c].reshape(-1)
            lo, hi = (np.percentile(flat, percentile_clip) if percentile_clip is not None
                      else (float(flat.min()), float(flat.max())))
            if not np.isfinite(hi - lo) or hi <= lo:
                hi = lo + 1e-6
            Xn[:, :, :, c] = ((X[:, :, :, c] - lo) / (hi - lo)).astype(np.float32)
        return Xn


# ======================================================================================
# Quick builders
# ======================================================================================

def build_zscore_detector(columns: Optional[List[int]] = None, clip_sigma: Optional[float] = 6.0, seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="zscore", zscore_columns=columns, zscore_clip_sigma=clip_sigma, seed=seed)
    return AnomalyDetector(cfg)


def build_robust_zscore_detector(columns: Optional[List[int]] = None, clip_sigma: Optional[float] = 6.0, seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="robust_zscore", rzs_columns=columns, rzs_clip_sigma=clip_sigma, seed=seed)
    return AnomalyDetector(cfg)


def build_mahalanobis_detector(columns: Optional[List[int]] = None, reg_eps: float = 1e-6, whiten: bool = False, seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="mahalanobis", maha_columns=columns, maha_reg_eps=reg_eps, maha_whiten=whiten, seed=seed)
    return AnomalyDetector(cfg)


def build_isoforest_detector(n_estimators: int = 200, contamination: Optional[float] = None, seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="isoforest", isoforest_n_estimators=n_estimators, isoforest_contamination=contamination, seed=seed)
    return AnomalyDetector(cfg)


def build_lof_detector(n_neighbors: int = 20, seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="lof", lof_n_neighbors=n_neighbors, seed=seed)
    return AnomalyDetector(cfg)


def build_ocsvm_detector(kernel: str = "rbf", nu: float = 0.05, gamma: Union[str, float] = "scale", seed: int = 42) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="ocsvm", ocsvm_kernel=kernel, ocsvm_nu=nu, ocsvm_gamma=gamma, seed=seed)
    return AnomalyDetector(cfg)


def build_autoencoder_detector(epochs: int = 10, device: str = "cpu", seed: int = 42, hidden_mult: int = 2) -> AnomalyDetector:
    cfg = AnomalyDetectorConfig(method="autoencoder_cnn", ae_epochs=epochs, ae_device=device, seed=seed, ae_hidden_mult=hidden_mult)
    return AnomalyDetector(cfg)


# ======================================================================================
# Self-test (optional)
# ======================================================================================

if __name__ == "__main__":  # pragma: no cover
    set_global_seed(123)
    print("[WDE] anomaly_detector.py self-test")

    # Tabular synthetic
    N, D = 300, 8
    X = np.random.randn(N, D).astype(np.float32)
    X[:5] += 6.0  # inject some outliers

    # Z-score
    det = build_zscore_detector()
    det.fit(X[5:])
    s = det.score_samples(X)
    print("zscore top5:", np.argsort(s)[-5:][::-1])
    print("zscore explain(3):", det.explain_topk(X[:3], k=3)[:1])

    # Robust Z-score
    det_r = build_robust_zscore_detector()
    det_r.fit(X[5:])
    sr = det_r.score_samples(X)
    print("robust top5:", np.argsort(sr)[-5:][::-1])

    # Mahalanobis
    det_m = build_mahalanobis_detector()
    det_m.fit(X[5:])
    sm = det_m.score_samples(X)
    print("maha top5:", np.argsort(sm)[-5:][::-1])
    print("maha explain(3):", det_m.explain_topk(X[:3], k=3)[:1])

    # Isolation Forest (if sklearn exists)
    if _SKLEARN_AVAILABLE:
        det_if = build_isoforest_detector(n_estimators=128)
        det_if.fit(X[5:])
        sif = det_if.score_samples(X)
        print("IF top5:", np.argsort(sif)[-5:][::-1])

        det_l = build_lof_detector(n_neighbors=20)
        det_l.fit(X[5:])
        sl = det_l.score_samples(X)
        print("LOF top5:", np.argsort(sl)[-5:][::-1])

        det_sv = build_ocsvm_detector(nu=0.05)
        det_sv.fit(X[5:])
        ssv = det_sv.score_samples(X)
        print("OCSVM top5:", np.argsort(ssv)[-5:][::-1])
    else:
        print("sklearn not available; skipping IF/LOF/OCSVM")

    # AE smoke (if torch exists)
    if _TORCH_AVAILABLE:
        C, H, W = 3, 32, 32
        Ximg = (np.random.rand(128, C, H, W).astype(np.float32) * 0.05)
        Ximg[:3] += 0.9
        det_ae = build_autoencoder_detector(epochs=2, device="cpu")
        det_ae.fit(Ximg[3:])
        sae = det_ae.score_samples(Ximg)
        print("AE top3:", np.argsort(sae)[-3:][::-1])
    else:
        print("torch not available; skipping AE")

    print("[WDE] self-test complete.")
