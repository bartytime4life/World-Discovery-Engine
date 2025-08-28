"""
World Discovery Engine (WDE) — Core Python Package

This package implements the multi-stage "discovery funnel":
1) ingest   → AOI tiling & dataset preparation
2) detect   → coarse anomaly scan (CV/texture/VLM stubs)
3) evaluate → mid-scale evaluation (NDVI/EVI, hydro-geomorph, checks)
4) verify   → evidence fusion (ADE fingerprints, causal plausibility, uncertainty, SSIM what-if)
5) report   → candidate site dossiers (maps, overlays, narrative, JSON/GeoJSON)

Design goals
------------
- CLI-first: the Typer CLI orchestrates the pipeline.
- Config-driven: everything is controlled via YAML in /configs.
- Reproducible: deterministic seeds, logged runs, JSON/GeoJSON artifacts.
- Ethics-by-design: sovereignty checks & coordinate masking are built-in hooks.

Author: WDE
License: MIT
"""
__all__ = [
    "cli",
    "ingest",
    "detect",
    "evaluate",
    "verify",
    "report",
]
