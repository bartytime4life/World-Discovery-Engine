# FILE: README.md
# =================================================================================================
# World Discovery Engine (WDE) — OpenAI → Z Challenge · Archaeology & Earth Systems
#
# Purpose
#   WDE is a multi-modal AI pipeline that surfaces archaeologically significant candidates
#   (e.g., ADE soils, earthworks, geoglyphs) by fusing satellite/radar/LiDAR imagery,
#   soil & vegetation layers, hydrology, historical archives, and core sampling resources.
#
# Why this repo?
#   • Single-command local setup with Poetry + pre-commit
#   • Kaggle-first: exportable notebook and CLI submission helpers
#   • Reproducible data stages with DVC (data/ → artifacts/)
#   • Clear contribution, ethics, security, and citation policies
#
# Quick Links
#   • CONTRIBUTING.md  – how to contribute safely & reproducibly
#   • ETHICS.md        – ethical constraints for discovery and publication
#   • SECURITY.md      – responsible vulnerability and secret handling
#   • datasets.md      – canonical data sources and access patterns
#   • CITATION.cff     – please cite WDE in research
#   • LICENSE          – Apache 2.0 (permissive)
# =================================================================================================

# 0) TL;DR (5 minutes)
# ---------------------------------------------------------------------------------
# 1. Install:    pipx install poetry  (or pip install --upgrade poetry)
# 2. Clone:      git clone <your-wde-repo> && cd wde
# 3. Env:        cp .env.example .env   # fill API keys (Kaggle, Sentinel, Planet, etc.)
# 4. Setup:      poetry install --with dev,geo,ml,viz,notebook
# 5. Hooks:      poetry run pre-commit install
# 6. Data init:  poetry run dvc init && git add .dvc && git commit -m "init dvc"
# 7. Smoke test: poetry run python -m wde --help   # (CLI placeholder until src is added)
# 8. Kaggle:     make kaggle-export   # builds notebooks/wde_kaggle.ipynb from templates
#
# NOTE: This skeleton ships root files only. Add src/ and notebooks/ next (see “Repo Layout”).


# 1) Features
# ---------------------------------------------------------------------------------
# • Multi-sensor fusion (Sentinel-1/2, Landsat, NICFI Planet mosaics, DEM/DTM, LiDAR/point clouds)
# • Thematic layers (soil/vegetation, land cover, climate, hydrology, historical/archival)
# • Candidate site detection with uncertainty + refutation (causal plausibility graphs)
# • Dossier generation (maps, overlays, references, core sampling context, provenance)
# • Kaggle-first delivery (single notebook artifact + CLI pack/submit helpers)
# • Reproducibility guardrails (Poetry, DVC, pre-commit, deterministic configs)


# 2) Repo Layout (thin skeleton; code/docs land later)
# ---------------------------------------------------------------------------------
# .                      # root (you are here)
# ├─ data/               # DVC-tracked data (raw/, interim/, processed/) — DO NOT COMMIT binaries
# ├─ artifacts/          # maps, overlays, reports, zipped submissions (DVC-optional)
# ├─ notebooks/          # exported Kaggle notebook(s) (built via Makefile/kaggle-export)
# ├─ configs/            # YAML configs (Hydra-style optional) for pipeline stages
# ├─ src/                # wde/ package (CLI, data, models, viz) — to be added
# ├─ tests/              # pytest suites (unit/integ/e2e)
# ├─ tools/              # small CLIs (tilers, converters, validators)
# └─ docs/               # architecture, design notes, ADRs (optional at start)


# 3) Installation
# ---------------------------------------------------------------------------------
# Prereqs: Python 3.11/3.12, GDAL runtime (handled via wheels/conda on most platforms)
#
# (A) Poetry (recommended)
#     pipx install poetry              # or: pip install --upgrade poetry
#     poetry install --with dev,geo,ml,viz,notebook
#
# (B) Pre-commit hooks
#     poetry run pre-commit install
#     # Run once locally:
#     poetry run pre-commit run --all-files
#
# (C) DVC (data versioning)
#     poetry run dvc init
#     git add .dvc .dvcignore dvc.yaml || true
#     git commit -m "init dvc pipeline" || true
#
# (D) Environment variables
#     cp .env.example .env
#     # Add keys: KAGGLE_USERNAME, KAGGLE_KEY, PLANET_API_KEY, SENTINELHUB_CLIENT_ID/SECRET, etc.


# 4) Makefile shortcuts
# ---------------------------------------------------------------------------------
# • make setup            – install deps + hooks
# • make lint             – ruff/black/isort/yaml/nbstripout
# • make test             – pytest quick
# • make data-pull        – fetch data via tools/ or src/wde/data/* (when added)
# • make kaggle-export    – build notebooks/wde_kaggle.ipynb from template(s)
# • make package          – bundle artifacts/ + manifest for release
# • make clean            – prune caches, __pycache__, orphaned artifacts


# 5) Kaggle (Notebook-first delivery)
# ---------------------------------------------------------------------------------
# • Author locally, export one canonical notebook via:
#     make kaggle-export
#   This writes: notebooks/wde_kaggle.ipynb
#
# • Upload on Kaggle (UI/API). For CLI submit pipelines, wire a CI job later.
#
# • Guidelines:
#   - Single notebook should (a) fetch data, (b) run detection, (c) emit outputs,
#     (d) show uncertainty/overlays, and (e) summarize ethics + provenance notes.


# 6) Contributing, Security, and Ethics
# ---------------------------------------------------------------------------------
# • Read CONTRIBUTING.md for PR, tests, DVC usage, and commit style.
# • See SECURITY.md for responsible reports & secrets handling.
# • Follow ETHICS.md for sensitive site handling, indigenous data respect, and publication norms.


# 7) License and Citation
# ---------------------------------------------------------------------------------
# • License: Apache-2.0 (see LICENSE).
# • Citation: See CITATION.cff. If your work uses WDE or its outputs, please cite accordingly.


# 8) Roadmap (initial)
# ---------------------------------------------------------------------------------
# [ ] Add src/wde package + Typer CLI (data pull, fuse, detect, dossier, export)
# [ ] Add configs/*.yaml + Hydra/OMEGACONF wiring
# [ ] Implement tools/tilers + tools/validators + tools/kaggle_pack
# [ ] Provide docs/architecture.md and docs/datasets/*.md mirrors
# [ ] Add CI: lint, tests, notebook export, (optional) Kaggle submit