# /models/anomaly_detector.py
# ======================================================================================
# World Discovery Engine (WDE)
# AnomalyDetector — Unified API for geospatial/tabular anomaly detection
# --------------------------------------------------------------------------------------
# This module provides a single, configurable interface around multiple anomaly
# detection backends (statistical z-score, Isolation Forest, Local Outlier Factor,
# and optional neural autoencoder for image patches). It is designed for use in the
# WDE pipeline where inputs may be tabular features (e.g., soil/vegetation indices)
# or image patches (e.g., Sentinel-2 / SAR tiles).
#
# Key features
# ------------
# - Strategy pattern: ‘zscore’, ‘isoforest’, ‘lof’, ‘autoencoder_cnn’
# - Numpy-first API for easy interoperability with Kaggle notebooks
# - Optional PyTorch dependency for neural backends; gracefully degrades if missing
# - Deterministic seeds and lightweight defaults for reproducible runs
# - Works for:
#     (A) Tabular arrays:   X.shape == [N, D]
#     (B) Image patches:    X.shape == [N, C, H, W]   (neural backends)
# - Sliding-window raster scoring helper for large GeoTIFFs (CPU-only, numpy)
# - Save/Load support for sklearn strategies + torch models (if used)
#
# Notes
# -----
# - All outputs are standardized: anomaly_score (higher = more anomalous)
# - For LOF, the scikit-learn convention yields negative_outlier_factor_: lower is
#   more outlying. We convert to a positive “anomaly_score” by negation and
#   standardization to keep a consistent “higher is worse” interpretation.
# - For Z-score, you can choose which columns to z-score (default: all).
# - For ISOForest/LOF, StandardScaler is applied by default for stability.
# - Autoencoder uses MSE recon error; optionally channel-wise weighting can be added.
#
# Dependencies
# ------------
# - Required: numpy, scikit-learn (for non-neural strategies)
# - Optional: torch, torch.nn, torch.optim (only for ‘autoencoder_cnn’)
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
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# sklearn imports for classical anomaly detection
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    _SKLEARN_AVAILABLE = True
except Exception as _e:  # pragma: no cover - only hit if sklearn not installed
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _e

# Optional torch imports (only needed for neural backends)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except Exception as _e:  # pragma: no cover - only hit if torch not installed
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e


# --------------------------------------------------------------------------------------
# Utility: set global seeds for reproducibility (numpy and torch if present)
# --------------------------------------------------------------------------------------
def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility across numpy (and torch if available).

    Parameters
    ----------
    seed : int
        Seed value to set for RNGs.
    """
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms when possible for CPU-only inference
        torch.use_deterministic_algorithms(False)  # set True if you accept perf trade-offs


# --------------------------------------------------------------------------------------
# Config dataclass
# --------------------------------------------------------------------------------------
@dataclass
class AnomalyDetectorConfig:
    """
    Configuration for the AnomalyDetector.

    Attributes
    ----------
    method : str
        One of {'zscore', 'isoforest', 'lof', 'autoencoder_cnn'}.
    seed : int
        RNG seed for reproducibility.
    # Common preprocessing
    use_scaler : bool
        If True, apply StandardScaler before ISOForest / LOF.
    # Z-score settings
    zscore_columns : Optional[List[int]]
        Column indices to z-score (None means all columns).
    zscore_clip_sigma : Optional[float]
        If provided, clip z-scores to +/- this value before aggregation.
    # Isolation Forest
    isoforest_n_estimators : int
        Number of base estimators in the ensemble.
    isoforest_max_samples : Union[int, float, str]
        Number of samples or fraction; see sklearn docs.
    isoforest_contamination : Optional[float]
        Expected fraction of outliers; used to threshold if needed.
    isoforest_max_features : Union[int, float]
        Number or fraction of features per estimator.
    # LOF
    lof_n_neighbors : int
        Number of neighbors for LOF.
    lof_metric : str
        Distance metric for LOF.
    lof_novelty : bool
        If True, LOF supports predict on new data (novelty detection).
    # Autoencoder CNN (optional)
    ae_epochs : int
        Training epochs.
    ae_batch_size : int
        Batch size for training.
    ae_lr : float
        Learning rate for optimizer.
    ae_weight_decay : float
        L2 regularization.
    ae_hidden_mult : int
        Width multiplier for conv channels.
    ae_device : str
        'cpu' or 'cuda' if available.
    ae_patience : int
        Early stopping patience (epochs without improvement).
    ae_val_split : float
        Fraction for validation split during training (0 to disable).
    """
    method: str = "zscore"
    seed: int = 42
    # preprocessing
    use_scaler: bool = True
    # zscore
    zscore_columns: Optional[List[int]] = None
    zscore_clip_sigma: Optional[float] = 6.0
    # Isolation Forest
    isoforest_n_estimators: int = 200
    isoforest_max_samples: Union[int, float, str] = "auto"
    isoforest_contamination: Optional[float] = None
    isoforest_max_features: Union[int, float] = 1.0
    # LOF
    lof_n_neighbors: int = 20
    lof_metric: str = "minkowski"
    lof_novelty: bool = True
    # Autoencoder
    ae_epochs: int = 10
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_weight_decay: float = 0.0
    ae_hidden_mult: int = 2
    ae_device: str = "cpu"
    ae_patience: int = 5
    ae_val_split: float = 0.0


# --------------------------------------------------------------------------------------
# Strategy interface and concrete implementations
# --------------------------------------------------------------------------------------
class _BaseStrategy:
    """
    Abstract-ish base class for anomaly detection backends.
    Concrete classes should implement fit() and score_samples().
    """

    def fit(self, X: np.ndarray) -> "_BaseStrategy":
        raise NotImplementedError

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return a 1D array of anomaly scores (higher = more anomalous).
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """
        Save the model parameters to a directory or file path.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "_BaseStrategy":
        """
        Load the model parameters from a directory or file path.
        """
        raise NotImplementedError


# ---- Z-SCORE STRATEGY ---------------------------------------------------------------
class _ZScoreStrategy(_BaseStrategy):
    """
    Z-score based anomaly detector for tabular features.

    Scoring
    -------
    - Compute per-feature z = (x - mean) / std (across the training set).
    - Optionally clip extreme z values to maintain stability.
    - Aggregate into a scalar anomaly score per row via L2 norm of z.
      (Higher norm => more anomalous.)

    Notes
    -----
    - This is fast and simple, best used as a baseline or when distributions are
      approximately Gaussian after optional transforms.
    """

    def __init__(self, columns: Optional[List[int]] = None, clip_sigma: Optional[float] = 6.0):
        self.columns = columns
        self.clip_sigma = clip_sigma
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted_cols: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "_ZScoreStrategy":
        if X.ndim != 2:
            raise ValueError("ZScore expects tabular X with shape [N, D].")
        cols = np.arange(X.shape[1]) if self.columns is None else np.array(self.columns, dtype=int)
        Xc = X[:, cols].astype(np.float64)
        self.mean_ = Xc.mean(axis=0)
        self.std_ = Xc.std(axis=0, ddof=1)
        # Avoid division by zero — replace zeros with small epsilon
        self.std_[self.std_ == 0.0] = 1e-8
        self._fitted_cols = cols
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self._fitted_cols is None:
            raise RuntimeError("ZScoreStrategy is not fitted.")
        if X.ndim != 2:
            raise ValueError("ZScore expects tabular X with shape [N, D].")
        Xc = X[:, self._fitted_cols].astype(np.float64)
        z = (Xc - self.mean_) / self.std_
        if self.clip_sigma is not None:
            z = np.clip(z, -self.clip_sigma, self.clip_sigma)
        # L2 norm across features yields a single anomaly magnitude (higher = more anomalous)
        scores = np.linalg.norm(z, ord=2, axis=1)
        return scores

    def save(self, path: str) -> None:
        data = {
            "columns": None if self._fitted_cols is None else self._fitted_cols.tolist(),
            "mean": None if self.mean_ is None else self.mean_.tolist(),
            "std": None if self.std_ is None else self.std_.tolist(),
            "clip_sigma": self.clip_sigma,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "_ZScoreStrategy":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls(columns=data["columns"], clip_sigma=data["clip_sigma"])
        obj._fitted_cols = None if data["columns"] is None else np.array(data["columns"], dtype=int)
        obj.mean_ = None if data["mean"] is None else np.array(data["mean"], dtype=float)
        obj.std_ = None if data["std"] is None else np.array(data["std"], dtype=float)
        return obj


# ---- ISOLATION FOREST STRATEGY -------------------------------------------------------
class _IsoForestStrategy(_BaseStrategy):
    """
    Isolation Forest strategy with optional StandardScaler.

    Scoring
    -------
    - uses estimator.score_samples(X) which yields anomaly scores where *higher*
      means *less* abnormal in sklearn. We invert to make higher=more anomalous.

    Notes
    -----
    - We construct a sklearn Pipeline: [StandardScaler] -> [IsolationForest]
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
        self.pipeline_: Optional[Pipeline] = None
        self.params = dict(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        )
        self.use_scaler = use_scaler

    def fit(self, X: np.ndarray) -> "_IsoForestStrategy":
        if X.ndim != 2:
            raise ValueError("IsolationForest expects tabular X with shape [N, D].")
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("iso", IsolationForest(**self.params)))
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("IsoForestStrategy is not fitted.")
        # sklearn IF: higher score => more normal; we invert
        s = self.pipeline_.score_samples(X)
        return -s

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("IsoForestStrategy is not fitted.")
        joblib.dump({"pipeline": self.pipeline_, "use_scaler": self.use_scaler}, path)

    @classmethod
    def load(cls, path: str) -> "_IsoForestStrategy":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        obj = cls()
        blob = joblib.load(path)
        obj.pipeline_ = blob["pipeline"]
        obj.use_scaler = blob["use_scaler"]
        return obj


# ---- LOF STRATEGY --------------------------------------------------------------------
class _LOFStrategy(_BaseStrategy):
    """
    Local Outlier Factor strategy with optional StandardScaler.

    Scoring
    -------
    - sklearn exposes negative_outlier_factor_ (on fit_predict) or score_samples for novelty=True.
    - Convention is “lower = more outlier”. We convert to positive “higher = more anomalous”
      by returning -score_samples and shifting to >= 0 with min subtraction.

    Notes
    -----
    - With novelty=True, LOF can score new samples after fit().
    - With novelty=False, it cannot score new data (transductive only).
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = "minkowski",
        novelty: bool = True,
        use_scaler: bool = True,
    ):
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        self.use_scaler = use_scaler
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.novelty = novelty
        self.pipeline_: Optional[Pipeline] = None

    def fit(self, X: np.ndarray) -> "_LOFStrategy":
        if X.ndim != 2:
            raise ValueError("LOF expects tabular X with shape [N, D].")
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("lof", LocalOutlierFactor(n_neighbors=self.n_neighbors, metric=self.metric, novelty=self.novelty, n_jobs=-1)))
        self.pipeline_ = Pipeline(steps)
        # LOF requires fit() then optionally decision_function/score_samples (novelty=True)
        self.pipeline_.fit(X)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("LOFStrategy is not fitted.")
        # For novelty=True, score_samples is available.
        # LOF higher score typically indicates more normal; decision_function also exists.
        # We'll use score_samples, negate it, and shift to be >= 0.
        s = self.pipeline_.score_samples(X)
        s = -s
        s = s - s.min()  # shift to non-negative for convenience (optional)
        return s

    def save(self, path: str) -> None:
        if self.pipeline_ is None:
            raise RuntimeError("LOFStrategy is not fitted.")
        joblib.dump(
            {
                "pipeline": self.pipeline_,
                "use_scaler": self.use_scaler,
                "n_neighbors": self.n_neighbors,
                "metric": self.metric,
                "novelty": self.novelty,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "_LOFStrategy":
        if not _SKLEARN_AVAILABLE:
            raise ImportError(f"scikit-learn not available: {_SKLEARN_IMPORT_ERROR}")
        blob = joblib.load(path)
        obj = cls(
            n_neighbors=blob["n_neighbors"],
            metric=blob["metric"],
            novelty=blob["novelty"],
            use_scaler=blob["use_scaler"],
        )
        obj.pipeline_ = blob["pipeline"]
        return obj


# ---- AUTOENCODER CNN (IMAGE PATCHES) -------------------------------------------------
class _SimpleConvAutoencoder(nn.Module):
    """
    A small, fast convolutional autoencoder for image patches [N, C, H, W].

    - Encoder: Conv → ReLU → Conv → ReLU → flatten
    - Bottleneck size scales with ae_hidden_mult.
    - Decoder: ConvTranspose layers mirroring encoder.

    This is intentionally minimalistic to keep runtime low in Kaggle environments.
    """

    def __init__(self, in_channels: int = 3, hidden_mult: int = 2):
        super().__init__()
        base = 16 * hidden_mult
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base, kernel_size=3, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1),     # H/4
            nn.ReLU(inplace=True),
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 2, base, kernel_size=4, stride=2, padding=1),  # H/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, kernel_size=4, stride=2, padding=1),  # H
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.enc(x)
        x_rec = self.dec(z)
        return x_rec


class _AutoencoderStrategy(_BaseStrategy):
    """
    Autoencoder-based anomaly scoring for image patches (higher recon error = more anomalous).

    Requirements
    ------------
    - torch must be available.
    - Input X must be float32 numpy array with shape [N, C, H, W] and values in [0, 1] (recommended).

    Training
    --------
    - MSE loss between input and reconstruction.
    - Early stopping on validation loss (if ae_val_split > 0).
    - CPU by default; set ae_device='cuda' if available and desired.

    Scoring
    -------
    - Return per-sample mean squared error across all pixels/channels.
    - Output is a 1D float array of length N.
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
        self.val_split = max(0.0, min(0.49, val_split))  # keep small to avoid starving train
        self.best_state_: Optional[Dict[str, Any]] = None

    def _to_torch_loader(self, X: np.ndarray, shuffle: bool) -> "torch.utils.data.DataLoader":
        # Convert numpy [N, C, H, W] to torch dataset/loader
        tensor = torch.as_tensor(X, dtype=torch.float32)
        ds = torch.utils.data.TensorDataset(tensor, tensor)  # (input, target) same for AE
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def fit(self, X: np.ndarray) -> "_AutoencoderStrategy":
        if X.ndim != 4:
            raise ValueError("Autoencoder expects image patches with shape [N, C, H, W].")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        # Optional naive split for early stopping
        n = X.shape[0]
        if self.val_split > 0.0 and n >= 10:
            n_val = max(1, int(n * self.val_split))
            X_train, X_val = X[:-n_val], X[-n_val:]
        else:
            X_train, X_val = X, None

        train_loader = self._to_torch_loader(X_train, shuffle=True)
        val_loader = self._to_torch_loader(X_val, shuffle=False) if X_val is not None else None

        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        best_val = float("inf")
        bad_epochs = 0
        self.best_state_ = None

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                rec = self.model(xb)
                loss = criterion(rec, yb)
                loss.backward()
                opt.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validate ---
            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                total = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=True)
                        yb = yb.to(self.device, non_blocking=True)
                        rec = self.model(xb)
                        loss = criterion(rec, yb)
                        total += loss.item() * xb.size(0)
                val_loss = total / len(val_loader.dataset)

            # --- Early stopping ---
            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                bad_epochs = 0
                # Snapshot the best model weights
                self.best_state_ = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break  # stop early

        # Restore best weights if we captured any
        if self.best_state_ is not None:
            self.model.load_state_dict(self.best_state_)
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError("Autoencoder expects image patches with shape [N, C, H, W].")
        if X.dtype != np.float32:
            X = X.astype(np.float32)
        loader = self._to_torch_loader(X, shuffle=False)
        self.model.eval()
        scores = np.zeros(X.shape[0], dtype=np.float32)
        i0 = 0
        with torch.no_grad():
            for xb, _ in loader:
                bs = xb.size(0)
                xb = xb.to(self.device, non_blocking=True)
                rec = self.model(xb)
                # MSE per sample across all pixels/channels
                err = torch.mean((rec - xb) ** 2, dim=(1, 2, 3))
                scores[i0:i0 + bs] = err.detach().cpu().numpy()
                i0 += bs
        return scores

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "ae_state.pt"))
        meta = {
            "in_channels": next(self.model.parameters()).shape[1] if hasattr(self.model, "enc") else None,
            "hidden_mult": getattr(self.model, "enc")[0].out_channels // 16 if hasattr(self.model, "enc") else 2,
        }
        with open(os.path.join(path, "ae_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "_AutoencoderStrategy":
        if not _TORCH_AVAILABLE:
            raise ImportError(f"PyTorch not available: {_TORCH_IMPORT_ERROR}")
        with open(os.path.join(path, "ae_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(
            in_channels=int(meta["in_channels"]) if meta["in_channels"] is not None else 3,
            hidden_mult=int(meta["hidden_mult"]),
            epochs=1,  # dummy
            batch_size=64,
            lr=1e-3,
            device="cpu",
            seed=42,
        )
        state = torch.load(os.path.join(path, "ae_state.pt"), map_location="cpu")
        obj.model.load_state_dict(state)
        obj.model.eval()
        return obj


# --------------------------------------------------------------------------------------
# Public API: AnomalyDetector
# --------------------------------------------------------------------------------------
class AnomalyDetector:
    """
    Unified anomaly detector wrapper with multiple backends.

    Example
    -------
    >>> from models.anomaly_detector import AnomalyDetector, AnomalyDetectorConfig
    >>> cfg = AnomalyDetectorConfig(method="isoforest")
    >>> det = AnomalyDetector(cfg).fit(X_train)
    >>> scores = det.score_samples(X_test)   # higher = more anomalous

    Methods
    -------
    - fit(X): fit the chosen backend
    - score_samples(X): compute anomaly scores (1D)
    - predict(X, threshold): binary mask of anomalies based on quantile/float threshold
    - save(dir_path): persist model + config
    - load(dir_path): classmethod to restore a saved detector
    - raster_sliding_window_scores(...): build anomaly heatmap over large rasters
    """

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self._backend: Optional[_BaseStrategy] = None
        set_global_seed(self.config.seed)
        self._init_backend()

    # ----- Backend factory -------------------------------------------------------------
    def _init_backend(self) -> None:
        m = self.config.method.lower().strip()
        if m == "zscore":
            self._backend = _ZScoreStrategy(
                columns=self.config.zscore_columns, clip_sigma=self.config.zscore_clip_sigma
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
        elif m == "autoencoder_cnn":
            if not _TORCH_AVAILABLE:
                raise ImportError(f"Method 'autoencoder_cnn' requires PyTorch: {_TORCH_IMPORT_ERROR}")
            # Autoencoder requires the caller to pass image data to fit() so we can infer C
            # We’ll lazily construct AE after seeing input in fit(), but we pre-stage config.
            self._backend = None  # will be constructed in fit() once we know in_channels
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    # ----- Core API -------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """
        Fit the underlying backend on array X.

        Parameters
        ----------
        X : np.ndarray
            - For tabular methods (zscore/isoforest/lof): shape [N, D]
            - For autoencoder_cnn: shape [N, C, H, W], values ideally in [0, 1]
        """
        if self.config.method == "autoencoder_cnn":
            if X.ndim != 4:
                raise ValueError("autoencoder_cnn expects [N, C, H, W] image patches.")
            in_channels = int(X.shape[1])
            self._backend = _AutoencoderStrategy(
                in_channels=in_channels,
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
        """
        Compute anomaly scores for each row/sample in X (higher = more anomalous).
        """
        if self._backend is None:
            raise RuntimeError("Detector is not fitted/initialized.")
        return self._backend.score_samples(X)

    def predict(self, X: np.ndarray, threshold: Union[float, str] = "p99") -> np.ndarray:
        """
        Predict binary anomaly mask for X.

        Parameters
        ----------
        X : np.ndarray
            Input samples.
        threshold : Union[float, str]
            - If float in [0, 1], treated as quantile (e.g., 0.99 for top 1%).
            - If float > 1, treated as absolute score cutoff.
            - If str like 'p99', parsed as quantile 0.99.

        Returns
        -------
        mask : np.ndarray of bool
            True for anomaly, False otherwise.
        """
        scores = self.score_samples(X)
        if isinstance(threshold, str):
            # parse 'pXX' as percentile
            s = threshold.lower().strip()
            if s.startswith("p"):
                try:
                    q = float(s[1:]) / 100.0
                except Exception:
                    raise ValueError(f"Invalid percentile string: {threshold}")
                cutoff = np.quantile(scores, q)
            else:
                raise ValueError(f"Unknown threshold string format: {threshold}")
        elif isinstance(threshold, (int, float)):
            if 0.0 <= float(threshold) <= 1.0:
                cutoff = np.quantile(scores, float(threshold))
            else:
                cutoff = float(threshold)
        else:
            raise TypeError("threshold must be float or 'pXX' string (e.g., 'p99').")
        return scores >= cutoff

    # ----- Persistence ----------------------------------------------------------------
    def save(self, dir_path: str) -> None:
        """
        Save detector and config to a directory.

        Layout
        ------
        dir_path/
          config.json
          model.bin or model_dir/...
        """
        os.makedirs(dir_path, exist_ok=True)
        # Save config
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2)
        # Save backend
        if self._backend is None:
            raise RuntimeError("No backend to save; fit() must be called first.")
        # Choose filename based on method
        if self.config.method in {"zscore"}:
            self._backend.save(os.path.join(dir_path, "zscore.json"))
        elif self.config.method in {"isoforest"}:
            self._backend.save(os.path.join(dir_path, "isoforest.joblib"))
        elif self.config.method in {"lof"}:
            self._backend.save(os.path.join(dir_path, "lof.joblib"))
        elif self.config.method in {"autoencoder_cnn"}:
            self._backend.save(os.path.join(dir_path, "autoencoder"))
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    @classmethod
    def load(cls, dir_path: str) -> "AnomalyDetector":
        """
        Load detector and config from a directory created by save().
        """
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = AnomalyDetectorConfig(**cfg_dict)
        obj = cls(cfg)
        # Load backend
        m = cfg.method
        if m == "zscore":
            backend = _ZScoreStrategy.load(os.path.join(dir_path, "zscore.json"))
        elif m == "isoforest":
            backend = _IsoForestStrategy.load(os.path.join(dir_path, "isoforest.joblib"))
        elif m == "lof":
            backend = _LOFStrategy.load(os.path.join(dir_path, "lof.joblib"))
        elif m == "autoencoder_cnn":
            backend = _AutoencoderStrategy.load(os.path.join(dir_path, "autoencoder"))
        else:
            raise ValueError(f"Unknown method: {m}")
        obj._backend = backend
        return obj

    # ----- Sliding-window raster scoring ----------------------------------------------
    @staticmethod
    def raster_sliding_window_scores(
        patcher: "RasterPatcherProtocol",
        detector: "AnomalyDetector",
        stride: int,
        normalize: bool = True,
        percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
        dtype: str = "float32",
    ) -> np.ndarray:
        """
        Build an anomaly heatmap over a large raster by sliding window patches.

        Parameters
        ----------
        patcher : RasterPatcherProtocol
            An object exposing:
              - .shape -> Tuple[H, W, C] or (C, H, W)
              - .get_patch(y0:int, x0:int, size:int) -> np.ndarray [C, size, size] or [size, size, C]
              - .patch_size -> int
              - .channels_first -> bool
        detector : AnomalyDetector
            A fitted detector (ideally autoencoder for image patches).
        stride : int
            Sliding step in pixels between top-left corners.
        normalize : bool
            If True, per-channel percentile normalization is applied to [0,1].
        percentile_clip : Optional[Tuple[float, float]]
            Low/high percentiles for normalization.
        dtype : str
            Output array dtype.

        Returns
        -------
        heatmap : np.ndarray of shape [H, W] with anomaly intensities aggregated per patch center.
        """
        # Extract raster geometry
        H, W, C, channels_first = _infer_patcher_shape(patcher)
        ps = patcher.patch_size

        # Accumulators for overlapping patch scores
        heat = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        # Process in batches for speed when possible
        batch_patches: List[np.ndarray] = []
        batch_coords: List[Tuple[int, int]] = []
        max_batch = 256  # small batch to avoid memory blow-ups

        def _flush_batch():
            if not batch_patches:
                return
            X = np.stack(batch_patches, axis=0)  # [B, C, ps, ps] or [B, ps, ps, C]
            # Normalize
            if normalize:
                X = _per_channel_percentile_normalize(X, channels_first, percentile_clip)
            # Ensure channels-first for the detector if using AE
            if detector.config.method == "autoencoder_cnn":
                if not channels_first:
                    X = np.transpose(X, (0, 3, 1, 2))  # to [B, C, H, W]
                X = X.astype(np.float32)
            else:
                # For tabular/scikit methods, flatten patch to vector
                if channels_first:
                    X = X.reshape(X.shape[0], -1)
                else:
                    X = np.transpose(X, (0, 3, 1, 2)).reshape(X.shape[0], -1)
            scores = detector.score_samples(X)
            # Write scores into heatmap centered on patch
            for (yy, xx), s in zip(batch_coords, scores):
                y_slice = slice(yy, min(yy + ps, H))
                x_slice = slice(xx, min(xx + ps, W))
                heat[y_slice, x_slice] += float(s)
                count[y_slice, x_slice] += 1.0
            batch_patches.clear()
            batch_coords.clear()

        for y0 in range(0, H - ps + 1, stride):
            for x0 in range(0, W - ps + 1, stride):
                patch = patcher.get_patch(y0, x0, ps)  # either [C, ps, ps] or [ps, ps, C]
                batch_patches.append(patch)
                batch_coords.append((y0, x0))
                if len(batch_patches) >= max_batch:
                    _flush_batch()
        _flush_batch()

        # Avoid division by zero
        count[count == 0] = 1.0
        heatmap = (heat / count).astype(dtype)
        # Normalize heatmap to [0,1] for visualization convenience (optional)
        hm_min, hm_max = float(heatmap.min()), float(heatmap.max())
        if hm_max > hm_min:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        return heatmap


# --------------------------------------------------------------------------------------
# Helper protocols and utilities
# --------------------------------------------------------------------------------------
class RasterPatcherProtocol:
    """
    Minimal protocol for a raster patcher object. This is NOT an abstract base class
    and is intentionally duck-typed. Provide these attributes/methods:

    Attributes
    ----------
    patch_size : int
        Size (pixels) of the square patch to extract.
    channels_first : bool
        If True, patches are [C, H, W]; else [H, W, C].

    Methods
    -------
    shape -> Tuple[int, int, int]
        Returns (H, W, C)
    get_patch(y0:int, x0:int, size:int) -> np.ndarray
        Returns a patch of shape [C, size, size] or [size, size, C] consistent with channels_first.
    """

    patch_size: int
    channels_first: bool

    def shape(self) -> Tuple[int, int, int]:
        raise NotImplementedError

    def get_patch(self, y0: int, x0: int, size: int) -> np.ndarray:
        raise NotImplementedError


def _infer_patcher_shape(patcher: RasterPatcherProtocol) -> Tuple[int, int, int, bool]:
    """
    Return (H, W, C, channels_first) from a RasterPatcherProtocol instance.
    """
    s = patcher.shape()
    if len(s) != 3:
        raise ValueError("patcher.shape() must return (H, W, C).")
    H, W, C = int(s[0]), int(s[1]), int(s[2])
    return H, W, C, bool(patcher.channels_first)


def _per_channel_percentile_normalize(
    X: np.ndarray, channels_first: bool, percentile_clip: Optional[Tuple[float, float]]
) -> np.ndarray:
    """
    Normalize per-channel to [0,1] using percentile clipping.

    Parameters
    ----------
    X : np.ndarray
        [B, C, H, W] if channels_first else [B, H, W, C]
    channels_first : bool
        Whether channels dimension is at index 1.
    percentile_clip : Optional[Tuple[float, float]]
        (low, high) percentiles. If None, use min/max.

    Returns
    -------
    Xn : np.ndarray
        Same shape as X, normalized to [0,1] per channel.
    """
    Xf = X
    if channels_first:
        B, C, H, W = X.shape
        Xn = np.empty_like(Xf, dtype=np.float32)
        for c in range(C):
            chan = Xf[:, c, :, :].reshape(-1)
            if percentile_clip is not None:
                lo, hi = np.percentile(chan, [percentile_clip[0], percentile_clip[1]])
            else:
                lo, hi = float(chan.min()), float(chan.max())
            if hi <= lo:
                hi = lo + 1e-6
            Xn[:, c, :, :] = ((Xf[:, c, :, :] - lo) / (hi - lo)).astype(np.float32)
        return Xn
    else:
        B, H, W, C = X.shape
        Xn = np.empty_like(Xf, dtype=np.float32)
        for c in range(C):
            chan = Xf[:, :, :, c].reshape(-1)
            if percentile_clip is not None:
                lo, hi = np.percentile(chan, [percentile_clip[0], percentile_clip[1]])
            else:
                lo, hi = float(chan.min()), float(chan.max())
            if hi <= lo:
                hi = lo + 1e-6
            Xn[:, :, :, c] = ((Xf[:, :, :, c] - lo) / (hi - lo)).astype(np.float32)
        return Xn


# --------------------------------------------------------------------------------------
# Convenience: quick builders for common configurations
# --------------------------------------------------------------------------------------
def build_zscore_detector(
    columns: Optional[List[int]] = None,
    clip_sigma: Optional[float] = 6.0,
    seed: int = 42,
) -> AnomalyDetector:
    """
    Construct a Z-score detector for tabular data.
    """
    cfg = AnomalyDetectorConfig(method="zscore", zscore_columns=columns, zscore_clip_sigma=clip_sigma, seed=seed)
    return AnomalyDetector(cfg)


def build_isoforest_detector(
    n_estimators: int = 200,
    contamination: Optional[float] = None,
    seed: int = 42,
) -> AnomalyDetector:
    """
    Construct an Isolation Forest detector with sensible defaults.
    """
    cfg = AnomalyDetectorConfig(
        method="isoforest",
        isoforest_n_estimators=n_estimators,
        isoforest_contamination=contamination,
        seed=seed,
    )
    return AnomalyDetector(cfg)


def build_lof_detector(
    n_neighbors: int = 20,
    seed: int = 42,
) -> AnomalyDetector:
    """
    Construct a LOF detector (novelty=True) with StandardScaler enabled.
    """
    cfg = AnomalyDetectorConfig(
        method="lof",
        lof_n_neighbors=n_neighbors,
        lof_novelty=True,
        seed=seed,
    )
    return AnomalyDetector(cfg)


def build_autoencoder_detector(
    in_channels_hint: int = 3,
    epochs: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> AnomalyDetector:
    """
    Construct an autoencoder-based detector. Note: actual in_channels are bound in fit()
    from the provided training array shape; `in_channels_hint` is for documentation only.
    """
    cfg = AnomalyDetectorConfig(
        method="autoencoder_cnn",
        ae_epochs=epochs,
        ae_device=device,
        ae_hidden_mult=2,
        seed=seed,
    )
    return AnomalyDetector(cfg)


# --------------------------------------------------------------------------------------
# __main__ quick self-test (non-unit, CPU-only) — safe to remove if undesired
# --------------------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Minimal smoke test on synthetic data to validate core paths.
    set_global_seed(123)
    print("[WDE] anomaly_detector.py self-test")

    # 1) Tabular — Z-score
    X = np.random.normal(0, 1, size=(200, 5)).astype(np.float32)
    # Inject anomalies
    X[:5] += 8.0
    det = build_zscore_detector()
    det.fit(X[5:])
    s = det.score_samples(X)
    print("Z-score top-5 indices:", np.argsort(s)[-5:][::-1])

    # 2) Tabular — IsolationForest
    if _SKLEARN_AVAILABLE:
        det2 = build_isoforest_detector(n_estimators=100)
        det2.fit(X[5:])
        s2 = det2.score_samples(X)
        print("IF top-5 indices:", np.argsort(s2)[-5:][::-1])
    else:
        print("scikit-learn not available; skipping IF test")

    # 3) Image patches — Autoencoder (tiny)
    if _TORCH_AVAILABLE:
        # Fake “satellite” patches with a few anomalies
        N, C, H, W = 128, 3, 32, 32
        Ximg = np.random.rand(N, C, H, W).astype(np.float32) * 0.1  # mostly dark
        Ximg[:3] += 0.9  # bright anomalies
        det3 = build_autoencoder_detector(epochs=2, device="cpu")
        det3.fit(Ximg[3:])
        s3 = det3.score_samples(Ximg)
        print("AE top-3 indices:", np.argsort(s3)[-3:][::-1])
    else:
        print("PyTorch not available; skipping AE test")

    print("[WDE] self-test complete.")
```
