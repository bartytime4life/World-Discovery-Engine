# world_engine/__init__.py
# ======================================================================================
# 🌍 World Discovery Engine (WDE) — Core Python Package
#
# Implements the multi-stage "discovery funnel":
#   1) ingest   → AOI tiling & dataset preparation
#   2) detect   → coarse anomaly scan (CV/texture/VLM stubs)
#   3) evaluate → mid-scale evaluation (NDVI/EVI, hydro-geomorph, historical checks)
#   4) verify   → evidence fusion (ADE fingerprints, causal plausibility, uncertainty, SSIM what-if)
#   5) report   → candidate site dossiers (maps, overlays, narratives, JSON/GeoJSON)
#
# Design goals
# ------------
# - CLI-first orchestration (`world_engine.cli` via Typer).
# - Config-driven: all params defined in `/configs` YAMLs (Hydra-ready).
# - Reproducible: deterministic seeds, logged runs, artifact manifests.
# - Modular & testable: each stage isolated, composable, re-runnable:contentReference[oaicite:0]{index=0}.
# - Ethics-by-design: sovereignty checks & coordinate masking built-in:contentReference[oaicite:1]{index=1}.
#
# Refs: WDE Architecture Spec:contentReference[oaicite:2]{index=2}, Repository Structure:contentReference[oaicite:3]{index=3}, ADE Pipeline:contentReference[oaicite:4]{index=4}
#
# License: MIT (c) 2025 WDE contributors
# ======================================================================================

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    # Pipeline stages
    "ingest",
    "detect",
    "evaluate",
    "verify",
    "report",
    # CLI & utils
    "cli",
    "get_version",
    "get_package_root",
]

# --------------------------------------------------------------------------------------
# Versioning (manual bump or CI auto-injected via env var WDE_VERSION)
# --------------------------------------------------------------------------------------

def get_version() -> str:
    """
    Return WDE package version.
    Uses environment variable WDE_VERSION if present, else falls back to static.
    """
    return os.environ.get("WDE_VERSION", "1.0.0")


# --------------------------------------------------------------------------------------
# Package root path helper (for locating configs/data relative to source)
# --------------------------------------------------------------------------------------

def get_package_root() -> Path:
    """
    Return the root path of the WDE package (directory containing world_engine/).
    """
    return Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------
# Deterministic seeding helper (NumPy + torch if available)
# --------------------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Seed NumPy and torch (if installed) RNGs for reproducibility:contentReference[oaicite:5]{index=5}.
    """
    import numpy as np
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# Import key pipeline modules (lazy imports inside CLI for heavy deps)
# --------------------------------------------------------------------------------------

try:
    from . import ingest, detect, evaluate, verify, report, cli
except Exception as e:  # pragma: no cover
    # Allow docs build / partial installs to succeed without all deps
    import warnings
    warnings.warn(f"[WDE] Some submodules failed to import: {e}", RuntimeWarning)


# --------------------------------------------------------------------------------------
# Package metadata for reproducibility
# --------------------------------------------------------------------------------------

__version__ = get_version()
__doc__ = """
World Discovery Engine (WDE)
============================

The WDE is a **multi-stage scientific pipeline** for discovering archaeologically
significant sites (e.g. Anthropogenic Dark Earths, earthworks, geoglyphs) from open
geospatial, historical, and user-supplied data:contentReference[oaicite:6]{index=6}.

Stages
------
1. **Ingest**: AOI tiling, Sentinel/Landsat/SAR/DEM/LiDAR loading, overlay integration:contentReference[oaicite:7]{index=7}.
2. **Detect**: Coarse anomaly detection via CV, textures, DEM hillshades, VLM captions:contentReference[oaicite:8]{index=8}.
3. **Evaluate**: Mid-scale NDVI/EVI time-series, hydro-geomorph plausibility, historical overlays:contentReference[oaicite:9]{index=9}.
4. **Verify**: Multi-modal fusion, ADE fingerprints, causal PAG graphs, B-GNN uncertainty, SSIM what-if:contentReference[oaicite:10]{index=10}.
5. **Report**: Candidate dossiers with maps, overlays, causal graphs, uncertainty plots, narratives:contentReference[oaicite:11]{index=11}.

Features
--------
- **Reproducible**: config-driven, deterministic seeds, run manifests:contentReference[oaicite:12]{index=12}.
- **Explainable**: per-feature attributions, attention maps, causal rationales:contentReference[oaicite:13]{index=13}.
- **Ethical**: sovereignty hooks, coordinate masking, CARE-aligned reporting:contentReference[oaicite:14]{index=14}.
- **Kaggle-ready**: runs in 9h on Kaggle CPUs/GPUs; degrades gracefully to heuristics:contentReference[oaicite:15]{index=15}.

"""

