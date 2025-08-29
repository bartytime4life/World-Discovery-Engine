# /world_engine/models/ade_fingerprint.py
# ======================================================================================
# World Discovery Engine (WDE)
# ADEFingerprint — Multi-modal “Amazonian Dark Earth” (ADE) fingerprint scorer (Upgraded)
# --------------------------------------------------------------------------------------
# What this is
# ------------
# A compact, dependency-light module that turns heterogeneous geospatial signals into an
# interpretable ADE fingerprint score (0..1). It supports both a physics/soil-guided
# heuristic scorer and an optional supervised upgrade via LogisticRegression
# (plus optional probability calibration) when scikit-learn is available.
#
# Inputs (vectorized)
# -------------------
# You provide a nested dict of 1-D arrays (length N) — one array per feature:
#
#   sensors = {
#     "s2": { "B02": ..., "B03": ..., "B04": ..., "B08": ..., "B11": ..., "B12": ... },  # Sentinel-2
#     "sar": { "VV": ..., "VH": ... },                                                   # Sentinel-1
#     "lidar": { "chm": ..., "dtm": ..., "dem": ... },                                   # LiDAR / terrain (optional)
#     "soil": { "ph": ..., "soc": ..., "p": ..., "ca": ..., "k": ..., "n": ... },        # Soil/chemistry
#     "climate": { "precip": ..., "temp": ... },                                         # Optional
#   }
#
# Then:
#     model = ADEFingerprint()
#     X, names = model.engineer_features(sensors)   # X: [N, D], names: List[str]
#     scores = model.score(X)                       # 0..1 ADE likelihood scores
#
# Highlights (what’s new)
# -----------------------
# • Added robust_z scaling and per-feature clipping with configurable bounds
# • Expanded spectral feature set: NDMI, MSI, BSI (in addition to NDVI/EVI/SAVI/NDWI/NBR)
# • SAR handling auto-detects linear vs dB inputs and computes stable ratios/diffs
# • Terrain proxies (SLOPE, TPI) with safe behavior on tiny batches
# • pH bell transform (proximity to configurable center) + soil macros (SOC/P/Ca/K/N)
# • Optional supervised upgrade via LogisticRegression (+ optional CalibratedClassifierCV)
# • Explanations: returns per-feature contributions for both heuristic and logistic modes
# • Save/Load round-trip (config + scaler stats + features + weights + optional logistic)
# • Sliding-window raster heatmap with percentile normalization and batching
#
# Design goals
# ------------
# • Numpy-first (sklearn optional). No GPU dependencies. Deterministic seeds.
# • Stable on Kaggle/CI: avoids heavy libraries, defensive against NaN/inf, tiny memory use.
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# Optional scikit-learn (supervised head + calibration + persistence)
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    import joblib  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

API_FORMAT_VERSION = "2.1.0"


# ======================================================================================
# Utilities
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set global RNG seeds for reproducibility (numpy only; no torch dependency here).
    """
    np.random.seed(seed)


def _safe_div(n: np.ndarray, d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Element-wise safe division with epsilon.
    """
    return n / (np.abs(d) + eps)


def _nan_to_num(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Replace NaNs and ±inf with a finite fallback.
    """
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


def _as_float32(x: np.ndarray) -> np.ndarray:
    """
    Ensure float32 dtype with copy=False when possible.
    """
    if x.dtype == np.float32:
        return x
    return x.astype(np.float32, copy=False)


# ======================================================================================
# Robust feature scaler (median/IQR with clipping)
# ======================================================================================

@dataclass
class RobustScalerConfig:
    """
    Configuration for robust feature scaling.

    clip_low : float   Lower bound (in robust-z units) for clipping after scaling.
    clip_high: float   Upper bound (in robust-z units) for clipping after scaling.
    iqr_eps  : float   Minimum IQR to avoid division-by-zero.
    """
    clip_low: float = -6.0
    clip_high: float = +6.0
    iqr_eps: float = 1e-6


class RobustScaler:
    """
    Median/IQR scaler with clipping. Transforms each feature to a robust-z score:
      z = (x − median) / max(iqr, iqr_eps), then clip to [clip_low, clip_high].
    """

    def __init__(self, cfg: RobustScalerConfig = RobustScalerConfig()):
        self.cfg = cfg
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        if X.ndim != 2:
            raise ValueError("RobustScaler.fit expects 2D array [N, D].")
        X = _as_float32(X)
        self.median_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqr_ = np.maximum(q75 - q25, self.cfg.iqr_eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("RobustScaler not fitted.")
        X = _as_float32(X)
        Z = (X - self.median_) / self.iqr_
        Z = np.clip(Z, self.cfg.clip_low, self.cfg.clip_high)
        Z = _nan_to_num(Z, 0.0)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cfg": asdict(self.cfg),
            "median": None if self.median_ is None else self.median_.tolist(),
            "iqr": None if self.iqr_ is None else self.iqr_.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustScaler":
        obj = cls(RobustScalerConfig(**d["cfg"]))
        obj.median_ = None if d["median"] is None else np.array(d["median"], dtype=float)
        obj.iqr_ = None if d["iqr"] is None else np.array(d["iqr"], dtype=float)
        return obj


# ======================================================================================
# ADE fingerprint configuration
# ======================================================================================

@dataclass
class ADEFingerprintConfig:
    """
    Master configuration for ADEFingerprint.

    Heuristic defaults reflect qualitative ADE expectations:
      • SOC and P often elevated (positive weights)
      • pH closer to ~5.5–7 (we use a Gaussian bell proximity transform)
      • NDVI/EVI/SAVI tend to be moderate-high; NDWI lower; NBR/NDMI/BSI contextual
      • SAR VV/VH patterns can reflect structure/moisture differences
      • Terrain features included with small weights by default

    Prefer supervised mode (`fit(X, y)`) when labeled data are available.
    """
    # Randomness
    seed: int = 42

    # Robust scaling
    scaler: RobustScalerConfig = RobustScalerConfig()

    # Heuristic weights (feature name → weight). If empty, defaults are filled at init.
    heuristic_weights: Dict[str, float] = field(default_factory=dict)

    # Temperature for heuristic → probability mapping (sigmoid)
    heuristic_temperature: float = 1.0

    # Supervised LogisticRegression (+ optional probability calibration)
    use_logistic: bool = True
    logreg_C: float = 1.0
    logreg_max_iter: int = 300
    logreg_class_weight: Optional[str] = "balanced"  # or None
    calibrate: bool = False                           # if True, wrap with CalibratedClassifierCV
    calibrate_method: str = "sigmoid"                 # "sigmoid" or "isotonic"
    calibrate_cv: int = 3

    # Which engineered features to include; empty → compute all supported
    enabled_features: List[str] = field(default_factory=list)

    # pH bell transform
    ph_center: float = 6.0
    ph_width: float = 1.5  # controls bell spread


# ======================================================================================
# Feature engineering — vectorized transforms for each modality
# ======================================================================================

class FeatureEngineer:
    """
    Build per-sample engineered features from optional modality stacks.
    All inputs are arrays of length N; output X is [N, D].
    """

    # Canonical feature order (used if enabled_features is empty)
    DEFAULT_FEATURES: Tuple[str, ...] = (
        # Optical (Sentinel-2)
        "NDVI", "EVI", "SAVI", "NDWI", "NBR", "NDMI", "MSI", "BSI",
        # SAR (Sentinel-1)
        "VV_dB", "VH_dB", "VVVH_ratio", "VVVH_diff",
        # Terrain / LiDAR
        "CHM", "DTM", "DEM", "SLOPE", "TPI",
        # Soil/Chemistry
        "SOC", "P", "Ca", "K", "N", "pH_bell",
        # Climate (optional)
        "PRECIP", "TEMP",
    )

    @staticmethod
    def _compute_s2_indices(s2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute common Sentinel-2 spectral indices from bands dict.
        Keys: B02 (blue), B03 (green), B04 (red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
        """
        out: Dict[str, np.ndarray] = {}
        get = s2.get

        B02 = get("B02", None); B03 = get("B03", None); B04 = get("B04", None)
        B08 = get("B08", None); B11 = get("B11", None); B12 = get("B12", None)

        # Infer N (length) from any present band
        N = None
        for arr in (B02, B03, B04, B08, B11, B12):
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            # No bands → return empty vectors
            zeros = {
                "NDVI": np.zeros(0), "EVI": np.zeros(0), "SAVI": np.zeros(0), "NDWI": np.zeros(0),
                "NBR": np.zeros(0), "NDMI": np.zeros(0), "MSI": np.zeros(0), "BSI": np.zeros(0),
            }
            return zeros

        def _z(x):  # fill missing arrays with zeros of length N
            return np.zeros(N) if x is None else x

        B02 = _z(B02); B03 = _z(B03); B04 = _z(B04)
        B08 = _z(B08); B11 = _z(B11); B12 = _z(B12)

        # NDVI = (NIR - RED) / (NIR + RED)
        NDVI = _nan_to_num(_safe_div(B08 - B04, B08 + B04))
        # EVI = 2.5*(NIR-RED)/(NIR + 6*RED - 7.5*BLUE + 1)
        EVI = _nan_to_num(2.5 * _safe_div(B08 - B04, B08 + 6.0 * B04 - 7.5 * B02 + 1.0))
        # SAVI = (1+L)*(NIR-RED)/(NIR+RED+L), L=0.5
        L = 0.5
        SAVI = _nan_to_num((1.0 + L) * _safe_div(B08 - B04, B08 + B04 + L))
        # NDWI (McFeeters) ~ (GREEN - NIR) / (GREEN + NIR)
        NDWI = _nan_to_num(_safe_div(B03 - B08, B03 + B08))
        # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        NBR = _nan_to_num(_safe_div(B08 - B12, B08 + B12))
        # NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        NDMI = _nan_to_num(_safe_div(B08 - B11, B08 + B11))
        # MSI = SWIR1 / NIR (Moisture Stress Index)
        MSI = _nan_to_num(_safe_div(B11, B08))
        # BSI (Bare Soil Index) (per Qi 2019 style) ~ ((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))
        BSI = _nan_to_num(_safe_div((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02)))

        out["NDVI"] = NDVI
        out["EVI"] = EVI
        out["SAVI"] = SAVI
        out["NDWI"] = NDWI
        out["NBR"] = NBR
        out["NDMI"] = NDMI
        out["MSI"] = MSI
        out["BSI"] = BSI
        return out

    @staticmethod
    def _compute_sar_features(sar: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute SAR descriptors (VV/VH). Accepts either linear power or dB; converts to dB if
        inputs appear linear (heuristic: absolute values well below ~1000).
        """
        out: Dict[str, np.ndarray] = {}
        VV = sar.get("VV", None)
        VH = sar.get("VH", None)

        N = None
        for arr in (VV, VH):
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            return {"VV_dB": np.zeros(0), "VH_dB": np.zeros(0), "VVVH_ratio": np.zeros(0), "VVVH_diff": np.zeros(0)}

        VV = np.zeros(N) if VV is None else VV
        VH = np.zeros(N) if VH is None else VH

        def to_db(x: np.ndarray) -> np.ndarray:
            finite = np.isfinite(x)
            if not np.any(finite):
                return np.zeros_like(x)
            xmax = np.nanmax(np.abs(x[finite]))
            if xmax > 1000:  # assume already in dB (large absolute magnitudes)
                return x
            # treat as linear → convert to dB
            x_pos = np.clip(x, 1e-8, None)
            return 10.0 * np.log10(x_pos)

        VV_dB = to_db(VV)
        VH_dB = to_db(VH)
        VVVH_ratio = _nan_to_num(_safe_div(VV_dB, (np.abs(VH_dB) + 1e-6)))
        VVVH_diff = _nan_to_num(VV_dB - VH_dB)

        out["VV_dB"] = _nan_to_num(VV_dB)
        out["VH_dB"] = _nan_to_num(VH_dB)
        out["VVVH_ratio"] = VVVH_ratio
        out["VVVH_diff"] = VVVH_diff
        return out

    @staticmethod
    def _compute_terrain_features(lidar: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Terrain / canopy features (per-sample summaries):
          CHM: canopy height (m), DTM: bare earth (m), DEM: elevation (m)
          SLOPE: |∂DEM| proxy via 1-D gradient over sample order
          TPI: topographic position index vs. rolling median (window=5)
        """
        out: Dict[str, np.ndarray] = {}
        CHM = lidar.get("chm", None)
        DTM = lidar.get("dtm", None)
        DEM = lidar.get("dem", None)

        N = None
        for arr in (CHM, DTM, DEM):
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            return {"CHM": np.zeros(0), "DTM": np.zeros(0), "DEM": np.zeros(0), "SLOPE": np.zeros(0), "TPI": np.zeros(0)}

        CHM = np.zeros(N) if CHM is None else CHM
        DTM = np.zeros(N) if DTM is None else DTM
        DEM = np.zeros(N) if DEM is None else DEM

        out["CHM"] = _nan_to_num(CHM)
        out["DTM"] = _nan_to_num(DTM)
        out["DEM"] = _nan_to_num(DEM)

        # SLOPE proxy
        if N >= 3:
            d = np.gradient(DEM)
            out["SLOPE"] = _nan_to_num(np.abs(d))
        else:
            out["SLOPE"] = np.zeros(N)

        # TPI proxy vs. rolling median, window=5 (edges fallback)
        if N >= 5:
            med = np.copy(DEM)
            half = 2
            for i in range(N):
                i0 = max(0, i - half); i1 = min(N, i + half + 1)
                med[i] = np.median(DEM[i0:i1])
            TPI = DEM - med
        else:
            TPI = DEM - np.median(DEM)
        out["TPI"] = _nan_to_num(TPI)
        return out

    @staticmethod
    def _compute_soil_features(soil: Dict[str, np.ndarray], ph_center: float, ph_width: float) -> Dict[str, np.ndarray]:
        """
        Soil/chemistry features.
        Keys (optional subset): 'soc','p','ca','k','n','ph'
        pH_bell = exp( −0.5 * ((pH − center)/width)^2 ) in [0,1]
        """
        out: Dict[str, np.ndarray] = {}
        keys = ("soc", "p", "ca", "k", "n", "ph")
        N = None
        for k in keys:
            arr = soil.get(k, None)
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            return {"SOC": np.zeros(0), "P": np.zeros(0), "Ca": np.zeros(0), "K": np.zeros(0), "N": np.zeros(0), "pH_bell": np.zeros(0)}

        SOC = soil.get("soc", np.zeros(N))
        P = soil.get("p", np.zeros(N))
        Ca = soil.get("ca", np.zeros(N))
        K = soil.get("k", np.zeros(N))
        N_ = soil.get("n", np.zeros(N))
        pH = soil.get("ph", np.full(N, fill_value=ph_center))

        out["SOC"] = _nan_to_num(SOC)
        out["P"] = _nan_to_num(P)
        out["Ca"] = _nan_to_num(Ca)
        out["K"] = _nan_to_num(K)
        out["N"] = _nan_to_num(N_)
        z = (pH - ph_center) / max(ph_width, 1e-3)
        out["pH_bell"] = _nan_to_num(np.exp(-0.5 * z * z))
        return out

    @staticmethod
    def _compute_climate_features(climate: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Optional climate summaries: 'precip', 'temp'
        """
        out: Dict[str, np.ndarray] = {}
        PRECIP = climate.get("precip", None)
        TEMP = climate.get("temp", None)
        N = None
        for arr in (PRECIP, TEMP):
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            return {"PRECIP": np.zeros(0), "TEMP": np.zeros(0)}
        out["PRECIP"] = _nan_to_num(PRECIP if PRECIP is not None else np.zeros(N))
        out["TEMP"] = _nan_to_num(TEMP if TEMP is not None else np.zeros(N))
        return out

    @classmethod
    def engineer(
        cls,
        data: Dict[str, Dict[str, np.ndarray]],
        enabled_features: Optional[Iterable[str]] = None,
        ph_center: float = 6.0,
        ph_width: float = 1.5,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Engineer features from nested dict of modality → {feature: array[N]}.

        Returns
        -------
        X : np.ndarray [N, D]
        names : List[str]
        """
        s2_feat = cls._compute_s2_indices(data.get("s2", {}))
        sar_feat = cls._compute_sar_features(data.get("sar", {}))
        terr_feat = cls._compute_terrain_features(data.get("lidar", {}))
        soil_feat = cls._compute_soil_features(data.get("soil", {}), ph_center=ph_center, ph_width=ph_width)
        clim_feat = cls._compute_climate_features(data.get("climate", {}))

        merged: Dict[str, np.ndarray] = {}
        for d in (s2_feat, sar_feat, terr_feat, soil_feat, clim_feat):
            merged.update(d)

        if enabled_features is None:
            order = [f for f in cls.DEFAULT_FEATURES if f in merged]
        else:
            order = [f for f in enabled_features if f in merged]

        if not order:
            return np.zeros((0, 0), dtype=np.float32), []

        cols = [merged[name] for name in order]
        N = cols[0].shape[0]
        for c in cols:
            if c.shape[0] != N:
                raise ValueError("All input arrays must have the same length N.")
        X = np.stack(cols, axis=1)
        return _as_float32(_nan_to_num(X, 0.0)), order


# ======================================================================================
# ADE fingerprint core model
# ======================================================================================

class ADEFingerprint:
    """
    ADE fingerprint scorer with:
      • Robust feature engineering (FeatureEngineer)
      • Robust scaling (median/IQR with clipping)
      • Heuristic score (weighted sum → sigmoid) OR LogisticRegression (+ optional calibration)

    Typical usage
    -------------
        cfg = ADEFingerprintConfig()
        mdl = ADEFingerprint(cfg)

        X, names = mdl.engineer_features(data)
        mdl.fit(X, y)           # optional (needs sklearn & labels)
        s = mdl.score(X)        # 0..1
        mask = mdl.predict(X, threshold=0.5)
        exp = mdl.explain(X[:5])   # coefficients + per-feature contributions
    """

    def __init__(self, cfg: ADEFingerprintConfig = ADEFingerprintConfig()):
        self.cfg = cfg
        set_global_seed(cfg.seed)

        # Default heuristic weights if none provided
        if not self.cfg.heuristic_weights:
            self.cfg.heuristic_weights = {
                # Optical
                "NDVI": +0.60, "EVI": +0.35, "SAVI": +0.30, "NDWI": -0.20,
                "NBR": +0.10, "NDMI": +0.15, "MSI": -0.10, "BSI": -0.10,
                # SAR
                "VV_dB": +0.05, "VH_dB": +0.10, "VVVH_ratio": -0.05, "VVVH_diff": +0.05,
                # Terrain/LiDAR
                "CHM": +0.05, "DTM": -0.05, "DEM": 0.0, "SLOPE": -0.05, "TPI": +0.02,
                # Soil/Chemistry
                "SOC": +1.00, "P": +0.85, "Ca": +0.35, "K": +0.20, "N": +0.25, "pH_bell": +0.60,
                # Climate
                "PRECIP": +0.05, "TEMP": 0.0,
            }

        # Fitted components
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []
        self.logreg_: Optional[Any] = None        # LogisticRegression or CalibratedClassifierCV
        self._raw_logreg_: Optional[Any] = None   # Underlying LogisticRegression if calibrated

    # ----------------------------- Feature Engineering -----------------------------

    def engineer_features(self, data: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert nested dict → engineered feature matrix X and feature names.
        Initializes the robust scaler on first call; subsequent calls reuse fitted stats.
        """
        X_raw, names = FeatureEngineer.engineer(
            data,
            enabled_features=self.cfg.enabled_features if self.cfg.enabled_features else None,
            ph_center=self.cfg.ph_center,
            ph_width=self.cfg.ph_width,
        )
        self.feature_names_ = names
        if X_raw.size == 0:
            return X_raw, names

        if self.scaler_ is None:
            self.scaler_ = RobustScaler(self.cfg.scaler).fit(X_raw)
        X = self.scaler_.transform(X_raw)
        return X, names

    # ----------------------------- Heuristic score --------------------------------

    def _heuristic_score(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted sum of scaled features → sigmoid (temperature-scaled).
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D].")
        if not self.feature_names_:
            raise RuntimeError("Call engineer_features() first to set feature names.")

        w = np.array([self.cfg.heuristic_weights.get(f, 0.0) for f in self.feature_names_], dtype=np.float32)
        lin = (X * w.reshape(1, -1)).sum(axis=1)
        t = max(self.cfg.heuristic_temperature, 1e-6)
        prob = 1.0 / (1.0 + np.exp(-lin / t))
        return prob.astype(np.float32)

    # ----------------------------- Supervised (Logistic + optional calibration) -----

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> "ADEFingerprint":
        """
        Optionally train a LogisticRegression ADE classifier (if y labels are provided).
        If y is None or sklearn is unavailable or use_logistic=False, remains in heuristic mode.
        """
        if y is None or not _SKLEARN_AVAILABLE or not self.cfg.use_logistic:
            return self

        if X.ndim != 2:
            raise ValueError("fit expects X as [N, D].")
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("fit expects y as [N] matching X.")

        # Select solver based on data size; liblinear is robust for small datasets
        solver = "liblinear" if X.shape[0] < 50000 else "saga"
        raw = LogisticRegression(
            C=self.cfg.logreg_C,
            max_iter=self.cfg.logreg_max_iter,
            class_weight=self.cfg.logreg_class_weight,
            solver=solver,
            n_jobs=None if solver == "liblinear" else -1,
        )
        if self.cfg.calibrate:
            # Wrap entire logistic estimator with CalibratedClassifierCV
            calibrated = CalibratedClassifierCV(
                base_estimator=raw,
                method=self.cfg.calibrate_method,
                cv=self.cfg.calibrate_cv,
            )
            calibrated.fit(X, y.astype(int))
            self.logreg_ = calibrated
            self._raw_logreg_ = raw  # store the base schema (coef not directly available from calibrated)
        else:
            raw.fit(X, y.astype(int))
            self.logreg_ = raw
            self._raw_logreg_ = raw
        return self

    # ----------------------------- Inference --------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ADE probability per sample (0..1). Uses logistic+calibration if trained, else heuristic.
        """
        if self.logreg_ is not None and _SKLEARN_AVAILABLE:
            # Logistic/CalibratedClassifierCV returns [N, 2], take class 1
            proba = self.logreg_.predict_proba(X)[:, 1]
            return _as_float32(proba)
        return self._heuristic_score(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for predict_proba().
        """
        return self.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary ADE prediction (True if score >= threshold).
        """
        return self.predict_proba(X) >= float(threshold)

    # ----------------------------- Explainability --------------------------------

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Return a lightweight explanation payload:
          - names: List[str] feature names
          - coef:  np.ndarray[D] coefficients (logistic if trained; else heuristic weights)
          - bias:  float logistic intercept (0.0 for heuristic)
          - contrib: np.ndarray[N, D] = coef * X_scaled
        """
        if not self.feature_names_:
            raise RuntimeError("Feature names undefined. Call engineer_features() first.")

        if self._raw_logreg_ is not None and _SKLEARN_AVAILABLE and hasattr(self._raw_logreg_, "coef_"):
            coef = np.asarray(self._raw_logreg_.coef_).reshape(-1)
            bias = float(np.asarray(self._raw_logreg_.intercept_).reshape(-1)[0])
        else:
            coef = np.array([self.cfg.heuristic_weights.get(f, 0.0) for f in self.feature_names_], dtype=np.float32)
            bias = 0.0

        contrib = X * coef.reshape(1, -1)
        return {
            "names": list(self.feature_names_),
            "coef": _as_float32(coef),
            "bias": float(bias),
            "contrib": _as_float32(contrib),
        }

    # ----------------------------- Persistence ------------------------------------

    def save(self, dir_path: str) -> None:
        """
        Persist config, scaler stats, feature names, heuristic weights, and optional logistic head.

        Layout
        ------
        dir_path/
          config.json            (API format + cfg)
          scaler.json            (robust scaler stats)
          feature_names.json     (ordered names used at fit/inference time)
          heuristic.json         (heuristic weight table)
          logistic.joblib        (optional — raw or calibrated estimator)
        """
        os.makedirs(dir_path, exist_ok=True)

        # Config + API version wrapper (future-proofing)
        meta = {"format_version": API_FORMAT_VERSION, "config": asdict(self.cfg)}
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Scaler
        scaler_dict = None if self.scaler_ is None else self.scaler_.to_dict()
        with open(os.path.join(dir_path, "scaler.json"), "w", encoding="utf-8") as f:
            json.dump(scaler_dict, f)

        # Feature order + heuristic weights
        with open(os.path.join(dir_path, "feature_names.json"), "w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f, indent=2)
        with open(os.path.join(dir_path, "heuristic.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.heuristic_weights, f, indent=2)

        # Logistic head (optional)
        if self.logreg_ is not None and _SKLEARN_AVAILABLE:
            joblib.dump(self.logreg_, os.path.join(dir_path, "logistic.joblib"))

    @classmethod
    def load(cls, dir_path: str) -> "ADEFingerprint":
        """
        Load an ADEFingerprint from a directory created by save().
        """
        # Config
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg_dict = meta.get("config", meta)  # backward-compatible if older saved only cfg
        obj = cls(ADEFingerprintConfig(**cfg_dict))

        # Scaler
        with open(os.path.join(dir_path, "scaler.json"), "r", encoding="utf-8") as f:
            sdict = json.load(f)
        obj.scaler_ = None if sdict is None else RobustScaler.from_dict(sdict)

        # Feature names + heuristic
        with open(os.path.join(dir_path, "feature_names.json"), "r", encoding="utf-8") as f:
            obj.feature_names_ = json.load(f)
        with open(os.path.join(dir_path, "heuristic.json"), "r", encoding="utf-8") as f:
            obj.cfg.heuristic_weights = json.load(f)

        # Logistic head
        lg_path = os.path.join(dir_path, "logistic.joblib")
        if _SKLEARN_AVAILABLE and os.path.exists(lg_path):
            obj.logreg_ = joblib.load(lg_path)
            # Try to recover underlying LR if calibrated (for explainability coef/bias)
            raw = getattr(obj.logreg_, "base_estimator", None)
            obj._raw_logreg_ = raw if raw is not None else (obj.logreg_ if isinstance(obj.logreg_, LogisticRegression) else None)
        else:
            obj.logreg_ = None
            obj._raw_logreg_ = None
        return obj

    # ----------------------------- Raster sliding window ---------------------------

    @staticmethod
    def raster_sliding_window_scores(
        patcher: "ADEFingerprintRasterPatcherProtocol",
        model: "ADEFingerprint",
        stride: int,
        percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
        dtype: str = "float32",
        max_batch: int = 256,
    ) -> np.ndarray:
        """
        Build an ADE heatmap by sliding a fixed window over a raster.
        The `patcher` must provide per-patch feature dicts compatible with engineer_features().

        Required `patcher` interface (duck-typed)
        -----------------------------------------
        .shape() -> (H, W)                        # raster size in pixels
        .patch_size : int                         # square window size
        .get_patch_features(y0, x0, size) -> Dict[str, Dict[str, np.ndarray]]
                                                  # returns nested dict with arrays of length 1

        Aggregation
        -----------
        Patch scores are written uniformly over each patch’s area. Overlaps are averaged.
        Output is percentile-normalized (1..99 by default) to [0,1] for visualization.
        """
        H, W = patcher.shape()
        ps = int(patcher.patch_size)

        heat = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        # Small batching of patches reduces overhead when model scoring is fast
        batch_payload: List[Tuple[int, int, Dict[str, Dict[str, np.ndarray]]]] = []

        def _flush_batch():
            if not batch_payload:
                return
            # Engineer → score each patch (N=1); vectorization here is limited by patcher API
            for (y0, x0, feat_dict) in batch_payload:
                X, _ = model.engineer_features(feat_dict)
                score = float(model.score(X)[0]) if X.size else 0.0
                y1 = min(y0 + ps, H); x1 = min(x0 + ps, W)
                heat[y0:y1, x0:x1] += score
                count[y0:y1, x0:x1] += 1.0
            batch_payload.clear()

        for y0 in range(0, max(1, H - ps + 1), stride):
            for x0 in range(0, max(1, W - ps + 1), stride):
                feat_dict = patcher.get_patch_features(y0, x0, ps)
                batch_payload.append((y0, x0, feat_dict))
                if len(batch_payload) >= max_batch:
                    _flush_batch()
        _flush_batch()

        # Average and normalize
        count[count == 0] = 1.0
        heatmap = (heat / count).astype(dtype)

        if percentile_clip is not None:
            lo, hi = np.percentile(heatmap, [percentile_clip[0], percentile_clip[1]])
            if hi > lo:
                heatmap = np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)
            else:
                # degenerate case → min-max
                mn, mx = float(heatmap.min()), float(heatmap.max())
                if mx > mn:
                    heatmap = (heatmap - mn) / (mx - mn)
        else:
            mn, mx = float(heatmap.min()), float(heatmap.max())
            if mx > mn:
                heatmap = (heatmap - mn) / (mx - mn)

        return heatmap


# ======================================================================================
# Raster patcher protocol (duck typed)
# ======================================================================================

class ADEFingerprintRasterPatcherProtocol:
    """
    Protocol for a patcher usable with ADEFingerprint.raster_sliding_window_scores.

    Required members
    ----------------
    patch_size : int

    Methods
    -------
    shape() -> Tuple[int, int]
        Returns (H, W) pixel dimensions of the raster.
    get_patch_features(y0:int, x0:int, size:int) -> Dict[str, Dict[str, np.ndarray]]
        Returns a nested dict compatible with FeatureEngineer.engineer(), arrays length 1 (N=1).
    """
    patch_size: int

    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get_patch_features(self, y0: int, x0: int, size: int) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError


# ======================================================================================
# Convenience builders
# ======================================================================================

def build_default_ade_model(seed: int = 42, use_logistic: bool = True, calibrate: bool = False) -> ADEFingerprint:
    """
    Quick constructor using robust defaults; supervised mode toggled by flags.
    """
    cfg = ADEFingerprintConfig(seed=seed, use_logistic=use_logistic, calibrate=calibrate)
    return ADEFingerprint(cfg)


# ======================================================================================
# Self-test (CPU) — synthetic data demonstration
# ======================================================================================

if __name__ == "__main__":
    # Synthetic demo to validate end-to-end flow
    set_global_seed(123)

    N = 600
    # Create synthetic Sentinel-2 bands with simple patterns
    s2 = {
        "B02": np.random.uniform(0.02, 0.12, size=N),  # blue
        "B03": np.random.uniform(0.05, 0.20, size=N),  # green
        "B04": np.random.uniform(0.05, 0.25, size=N),  # red
        "B08": np.random.uniform(0.10, 0.60, size=N),  # NIR
        "B11": np.random.uniform(0.02, 0.35, size=N),  # SWIR1
        "B12": np.random.uniform(0.02, 0.35, size=N),  # SWIR2
    }
    # SAR (VV/VH) — linear magnitudes (we will auto-convert to dB)
    sar = {
        "VV": np.random.uniform(0.001, 0.02, size=N),
        "VH": np.random.uniform(0.0005, 0.01, size=N),
    }
    # LiDAR/Terrain
    lidar = {
        "chm": np.random.uniform(0.0, 25.0, size=N),
        "dtm": np.random.uniform(50.0, 200.0, size=N),
        "dem": np.random.uniform(50.0, 200.0, size=N),
    }
    # Soils (inject a pocket of ADE-like chemistry for positives)
    soil = {
        "soc": np.random.gamma(shape=2.5, scale=1.2, size=N),
        "p":   np.random.gamma(shape=2.0, scale=0.8, size=N),
        "ca":  np.random.gamma(shape=2.0, scale=1.0, size=N),
        "k":   np.random.gamma(shape=1.5, scale=0.7, size=N),
        "n":   np.random.gamma(shape=1.5, scale=0.6, size=N),
        "ph":  np.random.normal(loc=5.7, scale=0.6, size=N),
    }
    climate = {
        "precip": np.random.uniform(1500, 3000, size=N),
        "temp":   np.random.uniform(22, 30, size=N),
    }

    # Weak synthetic labels correlated with P + SOC + pH proximity (+ a terrain nuance)
    ph_center = 6.0
    ph_width = 1.2
    z_ph = np.exp(-0.5 * ((soil["ph"] - ph_center) / ph_width) ** 2)
    latent = 0.7 * soil["p"] + 1.0 * soil["soc"] + 0.8 * z_ph + 0.1 * lidar["chm"]
    y = (latent > np.percentile(latent, 70)).astype(int)

    data = {"s2": s2, "sar": sar, "lidar": lidar, "soil": soil, "climate": climate}

    # Initialize model and engineer features
    cfg = ADEFingerprintConfig(seed=123, ph_center=ph_center, ph_width=ph_width, use_logistic=True, calibrate=True)
    model = ADEFingerprint(cfg)

    X, names = model.engineer_features(data)
    print(f"[engineer] X shape = {X.shape}, D = {len(names)} features")

    # Heuristic scores (no training)
    s0 = model.score(X)
    print(f"[heuristic] score range: [{s0.min():.3f}, {s0.max():.3f}], mean={s0.mean():.3f}")

    # Train logistic (with calibration if available/configured)
    if _SKLEARN_AVAILABLE:
        model.fit(X, y)
        s1 = model.score(X)
        acc = (model.predict(X, threshold=0.5) == y).mean()
        print(f"[logistic] score range: [{s1.min():.3f}, {s1.max():.3f}], acc={acc:.3f}")

        # Explain first sample
        exp = model.explain(X[:1])
        print("[explain] first 8 names:", exp["names"][:8])
        print("[explain] first 8 coef:", exp["coef"][:8])
        print("[explain] first contrib row (first 8):", exp["contrib"][0][:8])

        # Save / load round-trip
        outdir = "_ade_fp_tmp"
        model.save(outdir)
        restored = ADEFingerprint.load(outdir)
        s2 = restored.score(X)
        delta = float(np.abs(s1 - s2).mean())
        print(f"[persistence] mean|Δ| = {delta:.6f}")
    else:
        print("[warn] scikit-learn not available; logistic/calibration disabled")

    print("[done] ADEFingerprint self-test complete.")
