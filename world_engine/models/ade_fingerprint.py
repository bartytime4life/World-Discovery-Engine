# /world_engine/models/ade_fingerprint.py
# ======================================================================================
# World Discovery Engine (WDE)
# ADEFingerprint — Multi-modal “Amazonian Dark Earth” (ADE) fingerprint scorer
# --------------------------------------------------------------------------------------
# Purpose
#   Compute an interpretable ADE fingerprint score (0..1) per sample from heterogeneous
#   geospatial features (optical indices, SAR, LiDAR/terrain, soils/chemistry, climate).
#   The module ships with:
#     • Robust feature engineering (Sentinel-2 indices, SAR ratios, canopy metrics, etc.)
#     • A deterministic heuristic scorer (physics/soil-guided weights)
#     • Optional supervised upgrade via LogisticRegression if scikit-learn is available
#     • Quantile/robust scaling with per-feature caps for stability
#     • Save / load, vectorized API, and sliding-window raster heatmap support
#
# Input conventions
#   This module is vectorized. You pass a dict of arrays (one per modality/feature),
#   each array of length N (one value per sample). Missing signals are allowed; they
#   simply drop from the engineered feature set (and are ignored by the scorer).
#
#   Example structure for N samples:
#     sensors = {
#       "s2": { "B02": np.ndarray[N], "B03": ..., "B04": ..., "B08": ..., "B11": ..., "B12": ... },
#       "sar": { "VV": np.ndarray[N], "VH": np.ndarray[N] },
#       "lidar": { "chm": np.ndarray[N], "dtm": np.ndarray[N], "dem": np.ndarray[N] },  # optional
#       "soil": { "ph": np.ndarray[N], "soc": np.ndarray[N], "p": np.ndarray[N], "ca": np.ndarray[N], "k": np.ndarray[N], "n": np.ndarray[N] },
#       "climate": { "precip": np.ndarray[N], "temp": np.ndarray[N] },  # optional
#     }
#
#   Then you call:
#     model = ADEFingerprint(ADEFingerprintConfig())
#     X, names = model.engineer_features(sensors)  # X: [N, D], names: List[str]
#     scores = model.score(X)                      # 0..1 ADE likelihood scores
#
# Design notes
#   • We keep external deps light: numpy required; scikit-learn optional; joblib optional.
#   • When sklearn is present, ADEFingerprint can “fit” (logistic regression) to labels.
#   • When sklearn is absent or y=None, we use the deterministic heuristic weights.
#   • RobustScaler (median/IQR) caps extreme values for numerical stability.
#   • Explanations: per-feature contribution vector is returned when logistic head is used.
#
# License
#   MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional scikit-learn logistic regression for supervised upgrade
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.utils.validation import check_is_fitted
    import joblib
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set global RNG seeds for reproducibility (numpy only; no torch dependency here).
    """
    np.random.seed(seed)


def _safe_div(n: np.ndarray, d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Elementwise safe division with epsilon.
    """
    out = n / (np.abs(d) + eps)
    return out


def _nan_to_num(x: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """
    Replace NaNs and infinite values with a finite fallback.
    """
    return np.nan_to_num(x, nan=fill, posinf=fill, neginf=fill)


# --------------------------------------------------------------------------------------
# Robust feature scaler (median/IQR with caps)
# --------------------------------------------------------------------------------------

@dataclass
class RobustScalerConfig:
    """
    Configuration for robust feature scaling.

    Attributes
    ----------
    clip_low : float
        Lower bound (in robust-z units) for clipping after scaling.
    clip_high : float
        Upper bound (in robust-z units) for clipping after scaling.
    iqr_eps : float
        Minimum IQR to avoid division-by-zero.
    """
    clip_low: float = -6.0
    clip_high: float = +6.0
    iqr_eps: float = 1e-6


class RobustScaler:
    """
    Median/IQR scaler with clipping. Transforms each feature to an approximately
    standard-scale robust z: z = (x - median) / max(iqr, iqr_eps), then clip.
    """

    def __init__(self, cfg: RobustScalerConfig = RobustScalerConfig()):
        self.cfg = cfg
        self.median_: Optional[np.ndarray] = None
        self.iqr_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustScaler":
        if X.ndim != 2:
            raise ValueError("RobustScaler.fit expects 2D array [N, D].")
        self.median_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqr_ = np.maximum(q75 - q25, self.cfg.iqr_eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("RobustScaler not fitted.")
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


# --------------------------------------------------------------------------------------
# ADE fingerprint configuration
# --------------------------------------------------------------------------------------

@dataclass
class ADEFingerprintConfig:
    """
    Configuration for ADEFingerprint.

    Notes on defaults (heuristic mode)
    ----------------------------------
    • Weights reflect qualitative expectations for ADE sites:
      - SOC and P typically elevated vs. baseline soils (positive weights).
      - pH often less acidic (toward ~5.5–7), so we use |pH - 6.0| in a bell transform.
      - NDVI/EVI positive but not extreme; indices help separate bare soil / water.
      - SAR VV/VH ratio and difference can reflect structure/moisture contrasts.
      - Canopy/terrain metrics included but low-weight by default.

    Important: These are heuristics supporting initial exploration and triage. You should
    prefer the supervised mode (`fit(X, y)`) once you have labels from field data,
    archives, or vetted candidate sets.
    """
    # Randomness
    seed: int = 42

    # Robust scaling
    scaler: RobustScalerConfig = RobustScalerConfig()

    # Heuristic weights (feature name → weight). If empty, defaults are filled at init.
    heuristic_weights: Dict[str, float] = field(default_factory=dict)

    # Sigmoid temperature for heuristic score → probability
    heuristic_temperature: float = 1.0

    # LogisticRegression settings (if sklearn is available and y is provided)
    use_logistic: bool = True
    logreg_C: float = 1.0
    logreg_max_iter: int = 200
    logreg_class_weight: Optional[str] = "balanced"

    # List of engineered features to compute; empty → compute all supported
    enabled_features: List[str] = field(default_factory=list)

    # Optional center for pH bell transform (closer to center → higher score)
    ph_center: float = 6.0
    ph_width: float = 1.5  # controls bell spread


# --------------------------------------------------------------------------------------
# Feature engineering — vectorized transforms for each modality
# --------------------------------------------------------------------------------------

class FeatureEngineer:
    """
    Build per-sample engineered features from optional sensor stacks. Everything is
    vectorized: input arrays should be shape [N], outputs [N].
    """

    # Canonical feature order (used if enabled_features is empty)
    DEFAULT_FEATURES: Tuple[str, ...] = (
        # Optical (Sentinel-2)
        "NDVI", "NDWI", "NBR", "EVI", "SAVI",
        # SAR
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
        Compute common Sentinel-2 spectral indices from bands dictionary.
        Expected keys: B02 (blue), B03 (green), B04 (red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
        """
        out: Dict[str, np.ndarray] = {}
        # Gracefully handle missing bands by returning NaNs; later replaced with 0 by scaler.
        get = lambda k: s2.get(k, None)

        B02 = get("B02"); B03 = get("B03"); B04 = get("B04")
        B08 = get("B08"); B11 = get("B11"); B12 = get("B12")

        # Helper to build zero-arrays when a band is missing (preserves N)
        N = None
        for arr in (B02, B03, B04, B08, B11, B12):
            if arr is not None:
                N = arr.shape[0]
                break
        if N is None:
            return {
                "NDVI": np.zeros(0), "NDWI": np.zeros(0), "NBR": np.zeros(0),
                "EVI": np.zeros(0), "SAVI": np.zeros(0),
            }

        def _z(x):  # turn None into zeros of len N
            return np.zeros(N) if x is None else x

        B02 = _z(B02); B03 = _z(B03); B04 = _z(B04)
        B08 = _z(B08); B11 = _z(B11); B12 = _z(B12)

        # NDVI = (NIR - RED) / (NIR + RED)
        out["NDVI"] = _nan_to_num(_safe_div(B08 - B04, B08 + B04))
        # NDWI (McFeeters) ~ (GREEN - NIR) / (GREEN + NIR)
        out["NDWI"] = _nan_to_num(_safe_div(B03 - B08, B03 + B08))
        # NBR = (NIR - SWIR2) / (NIR + SWIR2)
        out["NBR"] = _nan_to_num(_safe_div(B08 - B12, B08 + B12))
        # EVI (2.5*(NIR-RED)/(NIR + 6*RED - 7.5*BLUE + 1))
        out["EVI"] = _nan_to_num(2.5 * _safe_div(B08 - B04, B08 + 6 * B04 - 7.5 * B02 + 1.0))
        # SAVI ((1+L)*(NIR-RED)/(NIR+RED+L)) with L=0.5
        L = 0.5
        out["SAVI"] = _nan_to_num((1 + L) * _safe_div(B08 - B04, B08 + B04 + L))

        return out

    @staticmethod
    def _compute_sar_features(sar: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute simple SAR descriptors from VV/VH backscatter (linear or dB).
        If inputs appear in linear units (heuristic: max < 1e2), we convert to dB.
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

        def to_db(x: np.ndarray) -> np.ndarray:
            # If values look like linear power (0..something reasonable), convert to dB
            # Otherwise assume already in dB.
            finite = np.isfinite(x)
            if not np.any(finite):
                return np.zeros_like(x)
            xmax = np.nanmax(np.abs(x[finite]))
            if xmax > 1000:  # likely already dB, avoid log10(negative)
                return x
            # Ensure positive for log
            x_pos = np.clip(x, 1e-8, None)
            return 10.0 * np.log10(x_pos)

        VV = np.zeros(N) if VV is None else VV
        VH = np.zeros(N) if VH is None else VH

        VV_dB = to_db(VV)
        VH_dB = to_db(VH)
        out["VV_dB"] = _nan_to_num(VV_dB)
        out["VH_dB"] = _nan_to_num(VH_dB)
        # ratio and difference in dB space
        out["VVVH_ratio"] = _nan_to_num(_safe_div(VV_dB, (np.abs(VH_dB) + 1e-6)))
        out["VVVH_diff"] = _nan_to_num(VV_dB - VH_dB)
        return out

    @staticmethod
    def _compute_terrain_features(lidar: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute terrain / canopy features.
          - CHM: canopy height model (m)
          - DTM: digital terrain model (m)
          - DEM: (optional) digital elevation model (m)
          - SLOPE: approximated from DEM via finite difference in 1D (proxy)
          - TPI: topographic position index (DEM - median within neighborhood proxy)
        Inputs are per-sample summaries. If only DEM is present, CHM/DTM default to 0.
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

        # Simple slope proxy from DEM along sample order (not spatially aware; used only if you batch by tiles).
        # For vector batches from arbitrary ordering, SLOPE contributes little and is low-weight by default.
        if N >= 3:
            d = np.gradient(DEM)
            out["SLOPE"] = _nan_to_num(np.abs(d))
        else:
            out["SLOPE"] = np.zeros(N)

        # Simple TPI proxy vs. rolling median (window=5); at edges, fall back to DEM-median
        TPI = DEM.copy()
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
        Soil/chemistry features. Expected keys (any subset): 'soc','p','ca','k','n','ph'
        We add a bell-shaped pH proximity transform centered at `ph_center`:
          pH_bell = exp( -0.5 * ((pH - center)/width)^2 )
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

        # pH “closeness to center” using Gaussian bell; closer to ~6 (default) → higher value (0..1)
        z = (pH - ph_center) / max(ph_width, 1e-3)
        out["pH_bell"] = _nan_to_num(np.exp(-0.5 * z ** 2))
        return out

    @staticmethod
    def _compute_climate_features(climate: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Climate summaries (optional). Expected keys: 'precip', 'temp' (any unit/scale).
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
        Engineer features from a nested dict of modality → {feature: array[N]}.

        Returns
        -------
        X : np.ndarray [N, D]
            Engineered feature matrix (NaNs replaced with 0 later by scaler).
        names : List[str]
            Column names in X order.
        """
        # Compute modality-specific dictionaries (each mapping name → vector)
        s2_feat = cls._compute_s2_indices(data.get("s2", {}))
        sar_feat = cls._compute_sar_features(data.get("sar", {}))
        terr_feat = cls._compute_terrain_features(data.get("lidar", {}))
        soil_feat = cls._compute_soil_features(data.get("soil", {}), ph_center=ph_center, ph_width=ph_width)
        clim_feat = cls._compute_climate_features(data.get("climate", {}))

        # Merge dicts and filter by enabled list (or use default canonical order)
        merged: Dict[str, np.ndarray] = {}
        for d in (s2_feat, sar_feat, terr_feat, soil_feat, clim_feat):
            merged.update(d)

        # Decide order
        if enabled_features is None:
            order = [f for f in cls.DEFAULT_FEATURES if f in merged]
        else:
            order = [f for f in enabled_features if f in merged]

        if not order:
            # No features present — return empty matrix
            return np.zeros((0, 0), dtype=np.float32), []

        # Stack to [N, D]
        cols = [merged[name] for name in order]
        # Sanity check: all same length
        N = cols[0].shape[0]
        for c in cols:
            if c.shape[0] != N:
                raise ValueError("All input arrays must have the same length N.")
        X = np.stack(cols, axis=1).astype(np.float32)
        X = _nan_to_num(X, 0.0)
        return X, order


# --------------------------------------------------------------------------------------
# ADE fingerprint core model
# --------------------------------------------------------------------------------------

class ADEFingerprint:
    """
    ADE fingerprint scorer with:
      • Robust feature engineering (FeatureEngineer)
      • Robust scaling (median/IQR)
      • Heuristic score (weighted sum → sigmoid) OR LogisticRegression (if trained)

    Workflow
    --------
      model = ADEFingerprint()
      X, names = model.engineer_features(data)
      model.fit(X, y)        # optional (requires sklearn); otherwise heuristic mode
      scores = model.score(X)  # 0..1
      probs  = model.predict_proba(X)  # alias of score()
      mask   = model.predict(X, threshold=0.5)

    Explanations
    ------------
    • If logistic head is trained, `explain(X)` returns per-feature contributions using
      β * x (linear contributions) and a dict with 'coef', 'bias', 'contrib', 'names'.
    • In heuristic mode, `explain(X)` returns weight*scaled_value contributions.
    """

    def __init__(self, cfg: ADEFingerprintConfig = ADEFingerprintConfig()):
        self.cfg = cfg
        set_global_seed(cfg.seed)
        # Initialize default heuristic weights if none provided (positive → increases ADE score)
        if not self.cfg.heuristic_weights:
            self.cfg.heuristic_weights = {
                # Optical
                "NDVI": +0.6, "NDWI": -0.2, "NBR": +0.1, "EVI": +0.3, "SAVI": +0.3,
                # SAR
                "VV_dB": +0.05, "VH_dB": +0.10, "VVVH_ratio": -0.05, "VVVH_diff": +0.05,
                # Terrain/LiDAR
                "CHM": +0.05, "DTM": -0.05, "DEM": 0.0, "SLOPE": -0.05, "TPI": +0.02,
                # Soil/Chemistry
                "SOC": +1.00, "P": +0.85, "Ca": +0.35, "K": +0.20, "N": +0.25, "pH_bell": +0.60,
                # Climate
                "PRECIP": +0.05, "TEMP": 0.0,
            }

        # Fitted components (after fit() or first engineer_features())
        self.scaler_: Optional[RobustScaler] = None
        self.feature_names_: List[str] = []
        self.logreg_: Optional[Any] = None  # LogisticRegression if trained

    # ----------------------------- Feature Engineering -------------------------------

    def engineer_features(self, data: Dict[str, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, List[str]]:
        """
        Convert nested dict of modality arrays → engineered feature matrix X and names.
        Also primes the scaler if not already present (fit on first call).
        """
        X_raw, names = FeatureEngineer.engineer(
            data,
            enabled_features=self.cfg.enabled_features if self.cfg.enabled_features else None,
            ph_center=self.cfg.ph_center,
            ph_width=self.cfg.ph_width,
        )
        self.feature_names_ = names
        if X_raw.size == 0:
            # No features; return as is
            return X_raw, names

        # Initialize or update scaler
        if self.scaler_ is None:
            self.scaler_ = RobustScaler(self.cfg.scaler).fit(X_raw)
        X = self.scaler_.transform(X_raw)
        return X, names

    # ----------------------------- Heuristic score -----------------------------------

    def _heuristic_score(self, X: np.ndarray) -> np.ndarray:
        """
        Weighted sum of scaled features mapped through a sigmoid.
        Missing weights default to 0 (ignored).
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D [N, D].")
        if not self.feature_names_:
            raise RuntimeError("Call engineer_features() first to set feature names.")

        # Build weight vector aligned to feature_names_
        w = np.array([self.cfg.heuristic_weights.get(f, 0.0) for f in self.feature_names_], dtype=np.float32)
        # Linear score
        lin = (X * w.reshape(1, -1)).sum(axis=1)
        # Temperature-scaled sigmoid
        t = max(self.cfg.heuristic_temperature, 1e-6)
        prob = 1.0 / (1.0 + np.exp(-lin / t))
        return prob.astype(np.float32)

    # ----------------------------- Supervised (Logistic) ------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray]) -> "ADEFingerprint":
        """
        Optionally train a logistic regression ADE classifier (if y labels are provided).
        If y is None or scikit-learn is unavailable, does nothing and remains heuristic.
        """
        if y is None:
            return self

        if not _SKLEARN_AVAILABLE or not self.cfg.use_logistic:
            # Either sklearn not installed or disabled; keep heuristic mode
            return self

        if X.ndim != 2:
            raise ValueError("fit expects X as [N, D].")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("fit expects y as [N] matching X.")

        # Train a simple logistic regression (liblinear for stability or saga for larger N)
        solver = "liblinear" if X.shape[0] < 50000 else "saga"
        self.logreg_ = LogisticRegression(
            C=self.cfg.logreg_C,
            max_iter=self.cfg.logreg_max_iter,
            class_weight=self.cfg.logreg_class_weight,
            solver=solver,
            n_jobs=None if solver == "liblinear" else -1,
        )
        self.logreg_.fit(X, y.astype(int))
        return self

    # ----------------------------- Inference -----------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ADE probability per sample (0..1). Uses logistic model if present, else heuristic.
        """
        if self.logreg_ is not None and _SKLEARN_AVAILABLE:
            # scikit-learn predict_proba returns [N, 2]; class 1 at column 1
            proba = self.logreg_.predict_proba(X)[:, 1]
            return proba.astype(np.float32)
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

    # ----------------------------- Explainability ------------------------------------

    def explain(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Return a lightweight explanation payload:
          - names: List[str] feature names
          - coef:  np.ndarray[D] coefficients (logistic) or heuristic weights
          - bias:  float (logistic intercept) or 0.0 in heuristic mode
          - contrib: np.ndarray[N, D] contributions = coef * X_scaled
        """
        if not self.feature_names_:
            raise RuntimeError("Feature names are undefined. Call engineer_features() first.")

        if self.logreg_ is not None and _SKLEARN_AVAILABLE:
            coef = np.asarray(self.logreg_.coef_).reshape(-1)
            bias = float(self.logreg_.intercept_.reshape(-1)[0])
        else:
            coef = np.array([self.cfg.heuristic_weights.get(f, 0.0) for f in self.feature_names_], dtype=np.float32)
            bias = 0.0

        contrib = X * coef.reshape(1, -1)
        return {
            "names": list(self.feature_names_),
            "coef": coef.astype(np.float32),
            "bias": float(bias),
            "contrib": contrib.astype(np.float32),
        }

    # ----------------------------- Persistence ---------------------------------------

    def save(self, dir_path: str) -> None:
        """
        Persist config, scaler stats, (optional) logistic head, and feature names.
        Directory layout:
          dir_path/
            config.json
            scaler.json
            feature_names.json
            heuristic.json
            logreg.joblib   (optional; only if trained and sklearn available)
        """
        os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        # Scaler
        scaler_dict = None if self.scaler_ is None else self.scaler_.to_dict()
        with open(os.path.join(dir_path, "scaler.json"), "w", encoding="utf-8") as f:
            json.dump(scaler_dict, f)

        # Features and heuristic weights
        with open(os.path.join(dir_path, "feature_names.json"), "w", encoding="utf-8") as f:
            json.dump(self.feature_names_, f)
        with open(os.path.join(dir_path, "heuristic.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.heuristic_weights, f, indent=2)

        # Logistic head (optional)
        if self.logreg_ is not None and _SKLEARN_AVAILABLE:
            joblib.dump(self.logreg_, os.path.join(dir_path, "logreg.joblib"))

    @classmethod
    def load(cls, dir_path: str) -> "ADEFingerprint":
        """
        Load a saved ADEFingerprint from directory.
        """
        # Config
        with open(os.path.join(dir_path, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        obj = cls(ADEFingerprintConfig(**cfg_dict))

        # Scaler
        with open(os.path.join(dir_path, "scaler.json"), "r", encoding="utf-8") as f:
            sdict = json.load(f)
        obj.scaler_ = None if sdict is None else RobustScaler.from_dict(sdict)

        # Features and heuristic weights (weights override config to match training time)
        with open(os.path.join(dir_path, "feature_names.json"), "r", encoding="utf-8") as f:
            obj.feature_names_ = json.load(f)
        with open(os.path.join(dir_path, "heuristic.json"), "r", encoding="utf-8") as f:
            obj.cfg.heuristic_weights = json.load(f)

        # Logistic head (optional)
        logreg_path = os.path.join(dir_path, "logreg.joblib")
        if _SKLEARN_AVAILABLE and os.path.exists(logreg_path):
            obj.logreg_ = joblib.load(logreg_path)
        else:
            obj.logreg_ = None
        return obj

    # ----------------------------- Raster sliding window ------------------------------

    @staticmethod
    def raster_sliding_window_scores(
        patcher: "RasterPatcherProtocol",
        model: "ADEFingerprint",
        stride: int,
        percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
        dtype: str = "float32",
    ) -> np.ndarray:
        """
        Compute an ADE heatmap across a raster by sliding windows.
        This function is intentionally high-level and expects the caller's `patcher`
        to produce per-patch summary vectors compatible with FeatureEngineer.

        Expectations for `patcher`
        ---------------------------
        The patcher should expose:
          • .shape() -> (H, W)  (used to size the heatmap)
          • .patch_size : int   (sliding window size in pixels)
          • .get_patch_features(y0:int, x0:int, size:int) -> Dict[str, Dict[str, np.ndarray]]
              Returns a nested dict with the same structure expected by engineer_features(),
              but per-patch with arrays of length 1 (N=1).

        Aggregation
        -----------
        The patch score is written over the patch window [y0:y0+ps, x0:x0+ps], averaged
        in case of overlaps. Output heatmap is normalized to [0,1] for visualization.
        """
        H, W = patcher.shape()
        ps = patcher.patch_size

        heat = np.zeros((H, W), dtype=np.float64)
        count = np.zeros((H, W), dtype=np.float64)

        for y0 in range(0, max(1, H - ps + 1), stride):
            for x0 in range(0, max(1, W - ps + 1), stride):
                # Extract per-patch engineered features using the patcher's helper
                data = patcher.get_patch_features(y0, x0, ps)
                X, names = model.engineer_features(data)
                if X.size == 0:
                    score = 0.0
                else:
                    score = float(model.score(X)[0])  # N=1
                y1 = min(y0 + ps, H)
                x1 = min(x0 + ps, W)
                heat[y0:y1, x0:x1] += score
                count[y0:y1, x0:x1] += 1.0

        # Average overlapping scores
        count[count == 0] = 1.0
        heatmap = (heat / count).astype(dtype)

        # Normalize 1..99 percentile to [0,1] for display
        if percentile_clip is not None:
            lo, hi = np.percentile(heatmap, [percentile_clip[0], percentile_clip[1]])
            if hi > lo:
                heatmap = np.clip((heatmap - lo) / (hi - lo), 0.0, 1.0)
        else:
            # min-max fallback
            mn, mx = float(heatmap.min()), float(heatmap.max())
            if mx > mn:
                heatmap = (heatmap - mn) / (mx - mn)
        return heatmap


# --------------------------------------------------------------------------------------
# Raster patcher protocol (duck-typed expectation)
# --------------------------------------------------------------------------------------

class RasterPatcherProtocol:
    """
    Protocol for a patcher object usable with ADEFingerprint.raster_sliding_window_scores.
    This is a documentation stub; no runtime checks beyond duck typing.

    Required members
    ----------------
    patch_size : int

    Methods
    -------
    shape() -> Tuple[int, int]
        Returns (H, W) pixel dimensions of the raster.
    get_patch_features(y0:int, x0:int, size:int) -> Dict[str, Dict[str, np.ndarray]]
        Returns the nested dict for FeatureEngineer.engineer(), with arrays length 1.
    """
    patch_size: int

    def shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def get_patch_features(self, y0: int, x0: int, size: int) -> Dict[str, Dict[str, np.ndarray]]:
        raise NotImplementedError


# --------------------------------------------------------------------------------------
# Self-test (CPU) — synthetic data demonstration
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Synthetic demo to validate end-to-end flow
    set_global_seed(123)

    N = 500
    # Create synthetic Sentinel-2 bands with simple patterns
    s2 = {
        "B02": np.random.uniform(0.02, 0.12, size=N),  # blue
        "B03": np.random.uniform(0.05, 0.20, size=N),  # green
        "B04": np.random.uniform(0.05, 0.25, size=N),  # red
        "B08": np.random.uniform(0.10, 0.60, size=N),  # nir
        "B11": np.random.uniform(0.02, 0.35, size=N),  # swir1
        "B12": np.random.uniform(0.02, 0.35, size=N),  # swir2
    }
    # SAR (VV/VH)
    sar = {
        "VV": np.random.uniform(0.001, 0.02, size=N),  # linear units
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
        "ph":  np.random.normal(loc=5.6, scale=0.6, size=N),
    }
    climate = {
        "precip": np.random.uniform(1500, 3000, size=N),
        "temp":   np.random.uniform(22, 30, size=N),
    }

    # Create weak synthetic labels correlated with soil P + SOC + pH proximity
    # This is for demo of supervised mode.
    ph_center = 6.0
    ph_width = 1.2
    z_ph = np.exp(-0.5 * ((soil["ph"] - ph_center) / ph_width) ** 2)
    latent = 0.7 * soil["p"] + 1.0 * soil["soc"] + 0.8 * z_ph + 0.1 * lidar["chm"]
    y = (latent > np.percentile(latent, 70)).astype(int)

    # Assemble nested dict
    data = {"s2": s2, "sar": sar, "lidar": lidar, "soil": soil, "climate": climate}

    # Initialize model and engineer features
    cfg = ADEFingerprintConfig(seed=123, ph_center=ph_center, ph_width=ph_width, use_logistic=True)
    model = ADEFingerprint(cfg)

    X, names = model.engineer_features(data)
    print(f"[engineer] X shape = {X.shape}, features = {names}")

    # Heuristic scores (no training)
    s0 = model.score(X)
    print(f"[heuristic] score range: [{s0.min():.3f}, {s0.max():.3f}], mean={s0.mean():.3f}")

    # Train logistic regression if available
    if _SKLEARN_AVAILABLE:
        model.fit(X, y)
        s1 = model.score(X)
        acc = (model.predict(X, threshold=0.5) == y).mean()
        print(f"[logistic] score range: [{s1.min():.3f}, {s1.max():.3f}], acc={acc:.3f}")

        # Explain first 3 samples
        exp = model.explain(X[:3])
        print("[explain] names:", exp["names"][:6], "...")
        print("[explain] coef (first 6):", exp["coef"][:6])
        print("[explain] contrib[0] (first 6):", exp["contrib"][0][:6])

        # Save / load round-trip
        outdir = "_ade_fp_tmp"
        model.save(outdir)
        restored = ADEFingerprint.load(outdir)
        s2 = restored.score(X)
        delta = float(np.abs(s1 - s2).mean())
        print(f"[persistence] mean|Δ| = {delta:.6f}")
    else:
        print("[warn] scikit-learn not available; logistic mode disabled")

    print("[done] ADEFingerprint self-test completed.")
