# /models/__init__.py
# ======================================================================================
# World Discovery Engine (WDE)
# models package — Upgraded unified import surface, registry, and convenience factories
# --------------------------------------------------------------------------------------
# Goals
#  • Provide a single, stable place to import common model classes/configs used across
#    the WDE codebase and Kaggle notebooks (clean UX: `from models import ...`).
#  • Offer a light-weight string registry (`create()`) for quick prototyping.
#  • Fail gracefully when optional dependencies (torch / scikit-learn / scipy) are not
#    present, while still allowing import of this package (lazy error on create()).
#  • Keep zero side effects on import; no heavy initialization.
#
# Contents (re-exported if available)
#  • Anomaly detection:
#      - AnomalyDetector, AnomalyDetectorConfig
#  • Baselines (scikit-learn):
#      - BaselineModel, BaselineConfig
#  • Bayesian GNN with uncertainty (PyTorch, no PyG):
#      - UncertaintyBGNN, UncertaintyBGNNConfig
#  • Constraint-based causal discovery to a PAG:
#      - CausalPAG
#
# Also includes:
#  • ModelRegistry class + default REGISTRY with canonical model keys.
#  • create(name, **kwargs) convenience to instantiate by name.
#  • get_available() helper that lists models available in the current environment.
#  • set_global_seed() thin wrapper to seed numpy/torch if present.
#
# License
#  MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import importlib
import os
import types
from dataclasses import is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

# --------------------------------------------------------------------------------------
# Optional imports — keep import-time light and fail gracefully
# --------------------------------------------------------------------------------------

# anomaly_detector (numpy + sklearn optional + torch optional for AE)
_AnomalyDetector = None
_AnomalyDetectorConfig = None
try:
    from .anomaly_detector import AnomalyDetector as _AnomalyDetector, AnomalyDetectorConfig as _AnomalyDetectorConfig
except Exception:  # pragma: no cover
    _AnomalyDetector = None
    _AnomalyDetectorConfig = None

# baselines (requires scikit-learn; pandas optional)
_BaselineModel = None
_BaselineConfig = None
try:
    from .baselines import BaselineModel as _BaselineModel, BaselineConfig as _BaselineConfig
except Exception:  # pragma: no cover
    _BaselineModel = None
    _BaselineConfig = None

# uncertainty_bgnn (requires torch)
_UncertaintyBGNN = None
_UncertaintyBGNNConfig = None
try:
    from .uncertainty_bgnn import UncertaintyBGNN as _UncertaintyBGNN, UncertaintyBGNNConfig as _UncertaintyBGNNConfig
except Exception:  # pragma: no cover
    _UncertaintyBGNN = None
    _UncertaintyBGNNConfig = None

# causal_pag (numpy, scipy optional)
_CausalPAG = None
try:
    from .causal_pag import CausalPAG as _CausalPAG
except Exception:  # pragma: no cover
    _CausalPAG = None


# --------------------------------------------------------------------------------------
# Public re-exports (only export names that resolved successfully)
# --------------------------------------------------------------------------------------

# We assign public names conditionally to keep `dir(models)` clean and to play nice
# with static analyzers while not breaking when optional deps are missing.

if _AnomalyDetector is not None:
    AnomalyDetector = _AnomalyDetector
if _AnomalyDetectorConfig is not None:
    AnomalyDetectorConfig = _AnomalyDetectorConfig

if _BaselineModel is not None:
    BaselineModel = _BaselineModel
if _BaselineConfig is not None:
    BaselineConfig = _BaselineConfig

if _UncertaintyBGNN is not None:
    UncertaintyBGNN = _UncertaintyBGNN
if _UncertaintyBGNNConfig is not None:
    UncertaintyBGNNConfig = _UncertaintyBGNNConfig

if _CausalPAG is not None:
    CausalPAG = _CausalPAG


def _collect_public() -> List[str]:
    """
    Build __all__ dynamically based on successfully imported symbols.
    """
    names: List[str] = ["create", "get_available", "REGISTRY", "ModelRegistry", "set_global_seed", "get_version"]
    for n in [
        "AnomalyDetector", "AnomalyDetectorConfig",
        "BaselineModel", "BaselineConfig",
        "UncertaintyBGNN", "UncertaintyBGNNConfig",
        "CausalPAG",
    ]:
        if n in globals():
            names.append(n)
    return names


__all__ = _collect_public()


# --------------------------------------------------------------------------------------
# Version helper (manual bump or filled by CI if desired)
# --------------------------------------------------------------------------------------

def get_version() -> str:
    """
    Return a lightweight semantic-ish version for the models package.
    Optionally honors an env var WDE_MODELS_VERSION for CI stamping.
    """
    return os.environ.get("WDE_MODELS_VERSION", "1.0.0")


# --------------------------------------------------------------------------------------
# Global seeding convenience
# --------------------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set NumPy and (if installed) PyTorch RNG seeds for reproducibility.
    This function intentionally avoids importing torch unconditionally.
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


# --------------------------------------------------------------------------------------
# Registry & Factories
# --------------------------------------------------------------------------------------

class ModelRegistry:
    """
    Minimal model registry for quick constructor lookup by string key.

    A registry entry maps a `name` → `factory`, where `factory(**kwargs)` returns an
    instance of a model (or its config). Factories are only registered if their underlying
    modules were imported successfully (i.e., deps available).

    Examples
    --------
        from models import REGISTRY, create, get_available

        # List available models for the current environment
        print(get_available())

        # Create a baseline RandomForest classifier (scikit-learn required)
        m = create("baseline", problem_type="classification", model="rf", n_estimators=300)

        # Create an Isolation Forest anomaly detector
        det = create("anomaly", method="isoforest", isoforest_n_estimators=300)

        # Create an UncertaintyBGNN (torch required)
        gnn = create("ubgnn", in_dim=32, hidden_dim=128, num_layers=2, task="regression")
    """

    def __init__(self) -> None:
        self._map: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        key = name.strip().lower()
        self._map[key] = factory

    def create(self, name: str, /, **kwargs: Any) -> Any:
        key = name.strip().lower()
        if key not in self._map:
            raise KeyError(f"Unknown model '{name}'. Known: {sorted(self._map)}")
        return self._map[key](**kwargs)

    def keys(self) -> List[str]:
        return sorted(self._map.keys())

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._map

    def __len__(self) -> int:
        return len(self._map)


# Build the default registry with safe conditional registrations
REGISTRY = ModelRegistry()

# 1) Anomaly detector
if _AnomalyDetector is not None and _AnomalyDetectorConfig is not None:
    def _factory_anomaly(**kwargs: Any) -> Any:
        """
        Create an AnomalyDetector. Accepts keys from AnomalyDetectorConfig(**kwargs).
        """
        cfg = _AnomalyDetectorConfig(**kwargs)
        return _AnomalyDetector(cfg)
    REGISTRY.register("anomaly", _factory_anomaly)
    REGISTRY.register("anomaly_detector", _factory_anomaly)

# 2) Baselines
if _BaselineModel is not None and _BaselineConfig is not None:
    def _factory_baseline(**kwargs: Any) -> Any:
        """
        Create a BaselineModel. Accepts keys from BaselineConfig(**kwargs).
        """
        cfg = _BaselineConfig(**kwargs)
        return _BaselineModel(cfg)
    REGISTRY.register("baseline", _factory_baseline)
    REGISTRY.register("baselines", _factory_baseline)
    # Special-case shortcuts for common baseline types
    def _factory_rf_classifier(**kwargs: Any) -> Any:
        cfg = _BaselineConfig(problem_type="classification", model="rf", **kwargs)
        return _BaselineModel(cfg)
    def _factory_gb_regressor(**kwargs: Any) -> Any:
        cfg = _BaselineConfig(problem_type="regression", model="gb", **kwargs)
        return _BaselineModel(cfg)
    REGISTRY.register("rf_classifier", _factory_rf_classifier)
    REGISTRY.register("gb_regressor", _factory_gb_regressor)

# 3) Uncertainty Bayesian GNN
if _UncertaintyBGNN is not None and _UncertaintyBGNNConfig is not None:
    def _factory_ubgnn(**kwargs: Any) -> Any:
        """
        Create an UncertaintyBGNN. Accepts keys from UncertaintyBGNNConfig(**kwargs).
        """
        cfg = _UncertaintyBGNNConfig(**kwargs)
        return _UncertaintyBGNN(cfg)
    REGISTRY.register("ubgnn", _factory_ubgnn)
    REGISTRY.register("uncertainty_bgnn", _factory_ubgnn)

# 4) Causal PAG
if _CausalPAG is not None:
    def _factory_causal_pag(**kwargs: Any) -> Any:
        """
        Create a CausalPAG instance (no config dataclass; pass constructor kwargs directly).
        """
        return _CausalPAG(**kwargs)
    REGISTRY.register("causal_pag", _factory_causal_pag)
    REGISTRY.register("pag", _factory_causal_pag)


def create(name: str, /, **kwargs: Any) -> Any:
    """
    Convenience wrapper: create a model instance by registry key.

    Example
    -------
        model = create("baseline", problem_type="classification", model="rf")
    """
    return REGISTRY.create(name, **kwargs)


def get_available() -> List[str]:
    """
    Return a sorted list of registry keys available in this environment.
    """
    return REGISTRY.keys()
```
