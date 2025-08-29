# /world_engine/models/__init__.py
# ======================================================================================
# World Discovery Engine (WDE)
# models package — Unified import surface, registry, and convenience factories (Upgraded)
# --------------------------------------------------------------------------------------
# What this is
# ------------
# A single, stable import point for model classes/configs used across the WDE codebase
# and Kaggle notebooks. It exposes:
#
#   • Clean re-exports (import only what’s available; fail gracefully otherwise)
#   • A tiny string → factory registry to instantiate models by name
#   • Helpers to list what’s available/missing and to seed RNGs deterministically
#   • Light-weight utilities for config-driven creation
#
# Design principles
# -----------------
#   • Zero heavy side-effects on import (no GPU init, no large downloads)
#   • Optional dependencies are okay — missing modules won’t break import
#   • Clear error messages when attempting to create something unavailable
#   • Friendly for static analyzers while staying robust in dynamic CI/Kaggle envs
#
# Public surface (if deps present)
# --------------------------------
#   • AnomalyDetector, AnomalyDetectorConfig
#   • BaselineModel, BaselineConfig
#   • ADEFingerprint, ADEFingerprintConfig
#   • UncertaintyBGNN, UncertaintyBGNNConfig          (torch)
#   • CausalPAG                                       (numpy/scipy optional)
#   • (Optional) GNNFusion, GNNFusionConfig           (torch)    — if provided in repo
#
# Registry keys (aliases)
# -----------------------
#   "anomaly", "anomaly_detector"
#   "baseline", "baselines", "rf_classifier", "gb_regressor"
#   "ade_fingerprint", "ade", "ade_fp"
#   "ubgnn", "uncertainty_bgnn"
#   "causal_pag", "pag"
#   "gnn_fusion", "fusion_gnn"       (optional if file exists)
#
# License
# -------
# MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import inspect
import os
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

# ======================================================================================
# Optional imports — keep import-time light and fail gracefully
# ======================================================================================

# anomaly_detector (NumPy; sklearn optional; torch optional for AE)
_AnomalyDetector = None
_AnomalyDetectorConfig = None
try:
    from .anomaly_detector import (  # type: ignore
        AnomalyDetector as _AnomalyDetector,
        AnomalyDetectorConfig as _AnomalyDetectorConfig,
    )
except Exception:  # pragma: no cover
    _AnomalyDetector = None
    _AnomalyDetectorConfig = None

# baselines (requires scikit-learn; pandas optional)
_BaselineModel = None
_BaselineConfig = None
try:
    from .baselines import (  # type: ignore
        BaselineModel as _BaselineModel,
        BaselineConfig as _BaselineConfig,
    )
except Exception:  # pragma: no cover
    _BaselineModel = None
    _BaselineConfig = None

# ADE fingerprint (NumPy; sklearn optional)
_ADEFingerprint = None
_ADEFingerprintConfig = None
try:
    from .ade_fingerprint import (  # type: ignore
        ADEFingerprint as _ADEFingerprint,
        ADEFingerprintConfig as _ADEFingerprintConfig,
    )
except Exception:  # pragma: no cover
    _ADEFingerprint = None
    _ADEFingerprintConfig = None

# Uncertainty Bayesian GNN (requires torch)
_UncertaintyBGNN = None
_UncertaintyBGNNConfig = None
try:
    from .uncertainty_bgnn import (  # type: ignore
        UncertaintyBGNN as _UncertaintyBGNN,
        UncertaintyBGNNConfig as _UncertaintyBGNNConfig,
    )
except Exception:  # pragma: no cover
    _UncertaintyBGNN = None
    _UncertaintyBGNNConfig = None

# Constraint-based causal discovery to a PAG (numpy, scipy optional)
_CausalPAG = None
try:
    from .causal_pag import CausalPAG as _CausalPAG  # type: ignore
except Exception:  # pragma: no cover
    _CausalPAG = None

# Optional fusion GNN (only if present in repo)
_GNNFusion = None
_GNNFusionConfig = None
try:
    from .gnn_fusion import (  # type: ignore
        GNNFusion as _GNNFusion,
        GNNFusionConfig as _GNNFusionConfig,
    )
except Exception:  # pragma: no cover
    _GNNFusion = None
    _GNNFusionConfig = None


# ======================================================================================
# Public re-exports (only bind symbols that resolved successfully)
# ======================================================================================

if _AnomalyDetector is not None:
    AnomalyDetector = _AnomalyDetector  # type: ignore
if _AnomalyDetectorConfig is not None:
    AnomalyDetectorConfig = _AnomalyDetectorConfig  # type: ignore

if _BaselineModel is not None:
    BaselineModel = _BaselineModel  # type: ignore
if _BaselineConfig is not None:
    BaselineConfig = _BaselineConfig  # type: ignore

if _ADEFingerprint is not None:
    ADEFingerprint = _ADEFingerprint  # type: ignore
if _ADEFingerprintConfig is not None:
    ADEFingerprintConfig = _ADEFingerprintConfig  # type: ignore

if _UncertaintyBGNN is not None:
    UncertaintyBGNN = _UncertaintyBGNN  # type: ignore
if _UncertaintyBGNNConfig is not None:
    UncertaintyBGNNConfig = _UncertaintyBGNNConfig  # type: ignore

if _CausalPAG is not None:
    CausalPAG = _CausalPAG  # type: ignore

if _GNNFusion is not None:
    GNNFusion = _GNNFusion  # type: ignore
if _GNNFusionConfig is not None:
    GNNFusionConfig = _GNNFusionConfig  # type: ignore


def _collect_public() -> List[str]:
    """
    Build __all__ dynamically based on successfully imported symbols.
    """
    names: List[str] = [
        # registry helpers
        "ModelRegistry",
        "REGISTRY",
        "create",
        "get_available",
        "get_missing",
        "get_registry_table",
        "register_factory",
        # utils
        "set_global_seed",
        "get_version",
        "create_from_config",
    ]
    for n in [
        "AnomalyDetector",
        "AnomalyDetectorConfig",
        "BaselineModel",
        "BaselineConfig",
        "ADEFingerprint",
        "ADEFingerprintConfig",
        "UncertaintyBGNN",
        "UncertaintyBGNNConfig",
        "CausalPAG",
        "GNNFusion",
        "GNNFusionConfig",
    ]:
        if n in globals():
            names.append(n)
    return names


__all__ = _collect_public()


# ======================================================================================
# Version helper (manual bump or filled by CI if desired)
# ======================================================================================

def get_version() -> str:
    """
    Return a lightweight semantic-ish version for the models package.
    Honors env var WDE_MODELS_VERSION when set (CI stamping).
    """
    return os.environ.get("WDE_MODELS_VERSION", "1.1.0")


# ======================================================================================
# Global seeding convenience
# ======================================================================================

def set_global_seed(seed: int = 42) -> None:
    """
    Set NumPy and (if installed) PyTorch RNG seeds for reproducibility.
    Avoid importing torch unless available.
    """
    try:
        import numpy as _np  # local import
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # local import
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ======================================================================================
# Registry & Factories
# ======================================================================================

@dataclass
class _Entry:
    """Internal: registry entry describing one constructible thing."""
    name: str
    factory: Callable[..., Any]
    requires: Tuple[str, ...] = ()
    summary: str = ""


class ModelRegistry:
    """
    Minimal model registry for quick constructor lookup by string key.

    An entry maps a key → factory(**kwargs) → instance. We only register entries whose
    underlying modules imported successfully (i.e., deps available).

    Examples
    --------
        from world_engine.models import REGISTRY, create, get_available

        print(get_available())
        m = create("baseline", problem_type="classification", model="rf", n_estimators=300)
        det = create("anomaly", method="isoforest", isoforest_n_estimators=300)
        ade = create("ade_fingerprint")  # heuristic by default; can train with labels later
    """

    def __init__(self) -> None:
        self._map: Dict[str, _Entry] = {}

    def register(self, name: str, factory: Callable[..., Any], requires: Tuple[str, ...] = (), summary: str = "") -> None:
        key = name.strip().lower()
        self._map[key] = _Entry(name=key, factory=factory, requires=requires, summary=summary)

    def register_alias(self, alias: str, target: str) -> None:
        alias_key = alias.strip().lower()
        target_key = target.strip().lower()
        if target_key not in self._map:
            raise KeyError(f"Cannot alias unknown target '{target}'.")
        self._map[alias_key] = self._map[target_key]

    def create(self, name: str, /, **kwargs: Any) -> Any:
        key = name.strip().lower()
        if key not in self._map:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._map)}")
        entry = self._map[key]
        return entry.factory(**kwargs)

    def keys(self) -> List[str]:
        return sorted(self._map.keys())

    def entries(self) -> List[_Entry]:
        # unique by identity (aliases map to same object)
        seen: set[int] = set()
        uniq: List[_Entry] = []
        for k in sorted(self._map.keys()):
            e = self._map[k]
            if id(e) in seen:
                continue
            uniq.append(e)
            seen.add(id(e))
        return uniq

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._map

    def __len__(self) -> int:
        return len(self._map)


# Build the default registry with safe conditional registrations
REGISTRY = ModelRegistry()

# 1) Anomaly detector
if _AnomalyDetector is not None and _AnomalyDetectorConfig is not None:
    def _factory_anomaly(**kwargs: Any) -> Any:
        """Create an AnomalyDetector. Accepts keys from AnomalyDetectorConfig(**kwargs)."""
        cfg = _AnomalyDetectorConfig(**kwargs)  # type: ignore
        return _AnomalyDetector(cfg)  # type: ignore

    REGISTRY.register(
        "anomaly",
        _factory_anomaly,
        requires=("numpy", "sklearn (optional)", "torch (optional for AE)"),
        summary="Unified anomaly detection (zscore/robust/maha/isoforest/lof/ocsvm/AE)",
    )
    REGISTRY.register_alias("anomaly_detector", "anomaly")

# 2) Baselines
if _BaselineModel is not None and _BaselineConfig is not None:
    def _factory_baseline(**kwargs: Any) -> Any:
        """Create a BaselineModel. Accepts keys from BaselineConfig(**kwargs)."""
        cfg = _BaselineConfig(**kwargs)  # type: ignore
        return _BaselineModel(cfg)  # type: ignore

    REGISTRY.register(
        "baseline",
        _factory_baseline,
        requires=("numpy", "scikit-learn"),
        summary="Classic sklearn baselines with robust preprocessing & metrics",
    )
    REGISTRY.register_alias("baselines", "baseline")

    # Convenient specialized shortcuts
    def _factory_rf_classifier(**kwargs: Any) -> Any:
        cfg = _BaselineConfig(problem_type="classification", model="rf", **kwargs)  # type: ignore
        return _BaselineModel(cfg)  # type: ignore

    def _factory_gb_regressor(**kwargs: Any) -> Any:
        cfg = _BaselineConfig(problem_type="regression", model="gb", **kwargs)  # type: ignore
        return _BaselineModel(cfg)  # type: ignore

    REGISTRY.register("rf_classifier", _factory_rf_classifier, requires=("scikit-learn",), summary="RandomForestClassifier baseline")
    REGISTRY.register("gb_regressor", _factory_gb_regressor, requires=("scikit-learn",), summary="GradientBoostingRegressor baseline")

# 3) ADE Fingerprint
if _ADEFingerprint is not None and _ADEFingerprintConfig is not None:
    def _factory_ade(**kwargs: Any) -> Any:
        """Create an ADEFingerprint model. Accepts keys from ADEFingerprintConfig(**kwargs)."""
        cfg = _ADEFingerprintConfig(**kwargs)  # type: ignore
        return _ADEFingerprint(cfg)  # type: ignore

    REGISTRY.register(
        "ade_fingerprint",
        _factory_ade,
        requires=("numpy", "scikit-learn (optional)"),
        summary="Heuristic + optional logistic ADE scorer with robust feature engineering",
    )
    REGISTRY.register_alias("ade", "ade_fingerprint")
    REGISTRY.register_alias("ade_fp", "ade_fingerprint")

# 4) Uncertainty Bayesian GNN
if _UncertaintyBGNN is not None and _UncertaintyBGNNConfig is not None:
    def _factory_ubgnn(**kwargs: Any) -> Any:
        """Create an UncertaintyBGNN. Accepts keys from UncertaintyBGNNConfig(**kwargs)."""
        cfg = _UncertaintyBGNNConfig(**kwargs)  # type: ignore
        return _UncertaintyBGNN(cfg)  # type: ignore

    REGISTRY.register(
        "ubgnn",
        _factory_ubgnn,
        requires=("torch",),
        summary="Bayesian GNN with uncertainty head (torch)",
    )
    REGISTRY.register_alias("uncertainty_bgnn", "ubgnn")

# 5) Causal PAG
if _CausalPAG is not None:
    def _factory_causal_pag(**kwargs: Any) -> Any:
        """Create a CausalPAG instance; pass constructor kwargs directly."""
        return _CausalPAG(**kwargs)  # type: ignore

    REGISTRY.register(
        "causal_pag",
        _factory_causal_pag,
        requires=("numpy", "scipy (optional)"),
        summary="Constraint-based causal discovery to a PAG (partial ancestral graph)",
    )
    REGISTRY.register_alias("pag", "causal_pag")

# 6) Optional GNN fusion (if file present)
if _GNNFusion is not None and _GNNFusionConfig is not None:
    def _factory_gnn_fusion(**kwargs: Any) -> Any:
        """Create a GNNFusion model. Accepts keys from GNNFusionConfig(**kwargs)."""
        cfg = _GNNFusionConfig(**kwargs)  # type: ignore
        return _GNNFusion(cfg)  # type: ignore

    REGISTRY.register(
        "gnn_fusion",
        _factory_gnn_fusion,
        requires=("torch",),
        summary="Feature fusion GNN (optional component)",
    )
    REGISTRY.register_alias("fusion_gnn", "gnn_fusion")


# ======================================================================================
# Public helpers
# ======================================================================================

def create(name: str, /, **kwargs: Any) -> Any:
    """
    Create a model instance by registry key.

    Example
    -------
        model = create("baseline", problem_type="classification", model="rf")
    """
    return REGISTRY.create(name, **kwargs)


def register_factory(name: str, factory: Callable[..., Any], *, requires: Tuple[str, ...] = (), summary: str = "") -> None:
    """
    Allow external modules/notebooks to register new factories at runtime.
    """
    REGISTRY.register(name, factory, requires=requires, summary=summary)


def get_available() -> List[str]:
    """
    Return a sorted list of registry keys available in this environment.
    """
    return REGISTRY.keys()


def get_missing() -> Dict[str, List[str]]:
    """
    Return a map of registry key → missing dependency notes (best-effort).
    If everything for an entry is available, it's not listed here.

    Note: This is heuristic text for UX; we don't probe importability here (we only
    register entries when imports previously succeeded). Entries present in the registry
    are assumed available. This function is kept for symmetry/UX.
    """
    # Since we only register on successful imports, nothing is "missing" at runtime.
    # We still return an empty dict to make callers happy.
    return {}


def get_registry_table() -> List[Dict[str, Any]]:
    """
    Return a human-friendly table (list of dicts) describing the registry entries.
    Aliases are collapsed (one row per unique factory).
    """
    rows: List[Dict[str, Any]] = []
    # Build alias sets → unique entries
    # Reconstruct alias groups by identity
    by_id: Dict[int, Dict[str, Any]] = {}
    for k in REGISTRY._map:  # type: ignore[attr-defined]
        e = REGISTRY._map[k]  # type: ignore[attr-defined]
        if id(e) not in by_id:
            by_id[id(e)] = {
                "keys": [k],
                "requires": list(e.requires),
                "summary": e.summary,
            }
        else:
            by_id[id(e)]["keys"].append(k)
    for group in by_id.values():
        group["keys"] = sorted(set(group["keys"]))
        rows.append(group)
    rows.sort(key=lambda r: r["keys"][0])
    return rows


# ======================================================================================
# Config-driven creation
# ======================================================================================

def _is_config_like(obj: Any) -> bool:
    """
    Return True if obj looks like a dataclass or a dict suitable for constructing one.
    """
    if is_dataclass(obj):
        return True
    if isinstance(obj, dict):
        return True
    return False


def create_from_config(key: str, cfg: Any) -> Any:
    """
    Create a model from a registry key and a config/dataclass/dict.

    Rules
    -----
    • If cfg is a dataclass instance, we pass its fields as kwargs to the factory.
    • If cfg is a dict, we pass it as **kwargs.
    • Otherwise, we raise a TypeError.

    Example
    -------
        bm = create_from_config("baseline", {"problem_type":"classification","model":"rf"})
    """
    if is_dataclass(cfg):
        kwargs = asdict(cfg)  # type: ignore[arg-type]
    elif isinstance(cfg, dict):
        kwargs = dict(cfg)
    else:
        raise TypeError("cfg must be a dataclass instance or a dict of kwargs")
    return create(key, **kwargs)
