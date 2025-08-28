# FILE: CHANGELOG.md
# -------------------------------------------------------------------------------------------------
# 🌍 World Discovery Engine (WDE) — Changelog

This file tracks **all notable changes** to the WDE project.  
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- Draft design for **archaeological discovery funnel** (ingest → detect → evaluate → verify → report).
- Placeholder notebooks in `notebooks/` for Kaggle scaffold (`ade_discovery_pipeline.ipynb`).
- Initial hooks for ethics-aware masking and sovereignty checks.
- Continuous Integration (GitHub Actions) stubs.

### Changed
- Upgraded configs to YAML-based modular design (`configs/`).
- Improved logging configuration defaults (structured logs).
- Dockerfile enhanced with micromamba + GDAL/GEOS stack for geospatial reproducibility.

### Deprecated
- Legacy skeleton references; all functionality now routed through `world_engine/` package.

---

## [0.2.0] — YYYY-MM-DD
### Added
- **Pipeline stages implemented**:
  - `ingest.py` → AOI tiling, Sentinel/Landsat/DEM ingestion.
  - `detect.py` → Coarse anomaly detection (CV, textures).
  - `evaluate.py` → NDVI/EVI, LiDAR/DEM hillshades, hydro-geomorph plausibility.
  - `verify.py` → Bayesian GNN fusion + uncertainty.
  - `report.py` → Candidate dossiers (Markdown + HTML + per-site manifest).
- **CLI**: Typer-based (`wde`) with commands: `ingest`, `scan`, `evaluate`, `verify`, `report`, `full-run`, `selftest`.
- **Ethics guardrails**: Coordinate masking, Indigenous overlap checks, CARE integration.
- **Reproducibility**: Config hashing, run manifests, DVC integration, Kaggle lockfile exports.
- **CI/CD**: Lint/test/audit workflows, CODEOWNERS enforcement, artifact manifest check.

### Changed
- Default outputs written under `outputs/` with run-specific manifests.
- All outputs include SHA256 checksums for auditability.

### Fixed
- Ensure deterministic behavior by seeding NumPy, Python, Torch RNGs in CLI.

---

## [0.1.0] — Initial Skeleton
### Added
- Root files: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`, `.editorconfig`.
- Tooling: `pyproject.toml` (Poetry), `pre-commit`, `Makefile`, DVC skeleton, `Dockerfile`.
- Policy: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `ETHICS.md`, `datasets.md`, `CITATION.cff`.

---

## Legend
- **Added**: New features, modules, configs.
- **Changed**: Updates to existing functionality or configs.
- **Fixed**: Bug fixes and reproducibility patches.
- **Deprecated**: Features to be removed in future versions.
- **Removed**: Features already removed.
- **Security**: Vulnerability fixes or security enhancements.
