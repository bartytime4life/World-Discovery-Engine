# 🛠️ Scripts — World Discovery Engine (WDE)

This folder hosts **utility scripts** that support the WDE pipeline.
Scripts are **thin orchestration layers** — all heavy lifting lives in `world_engine/` modules.
They provide **CLI-friendly entry points, reproducibility hooks, and automation shortcuts**.

---

## ✅ Principles

* **Reproducible:** Every script must run deterministically with config seeds and log manifests.
* **Config-driven:** No hard-coded parameters; always load from `configs/*.yaml`.
* **Modular:** Scripts only orchestrate pipeline stages; core logic lives in `world_engine/`.
* **Kaggle-ready:** Safe to run inside Kaggle notebooks, respecting runtime/memory constraints.
* **Ethical:** Scripts must honor sovereignty & CARE principles (e.g., masking coordinates in public outputs).
* **CLI-first:** Built around Typer CLI (`wde ingest`, `wde detect`, etc.).

---

## 📦 Included Scripts

### 🔹 `run_pipeline.sh`

* Orchestrates full pipeline: ingest → detect → evaluate → verify → report.
* Wraps `wde` CLI with config path and AOI arguments.
* Saves logs and outputs to `artifacts/` with timestamp.

### 🔹 `fetch_datasets.sh`

* Downloads open geospatial datasets (Sentinel, Landsat, DEM, SoilGrids, HydroSHEDS).
* Uses APIs (Copernicus, USGS, OpenTopography) or Kaggle Datasets when offline.
* Caches files under `data/raw/`.

### 🔹 `export_kaggle.sh`

* Prepares a Kaggle-ready bundle:

  * Copies notebook (`ade_discovery_pipeline.ipynb`), configs, and lightweight data samples.
  * Zips artifacts into `submission.zip`.
* Ensures Kaggle runtime compliance (≤8h, ≤19GB memory).

### 🔹 `validate_artifacts.py`

* Runs consistency checks on outputs (GeoTIFF validity, JSON schema, manifest completeness).
* Validates reproducibility by hashing configs + artifacts.
* Flags missing provenance (e.g., unlogged dataset source).

### 🔹 `profiling_tools.py`

* Lightweight runtime profilers (`with timer("stage"):`).
* Collects memory + runtime stats per pipeline stage.
* Outputs profiling log for CI and Kaggle notebooks.

### 🔹 `generate_reports.sh`

* Wraps `world_engine/report.py` to batch-produce candidate dossiers.
* Outputs human-readable `.md` and `.pdf` dossiers with figures, causal graphs, and uncertainty metrics.

---

## ⚙️ Usage Examples

```bash
# Run pipeline with default config
./scripts/run_pipeline.sh --config ./configs/default.yaml

# Fetch Sentinel-2 and DEM tiles for AOI
./scripts/fetch_datasets.sh --aoi data/aoi/brazil.geojson

# Export to Kaggle notebook bundle
./scripts/export_kaggle.sh --notebook notebooks/ade_discovery_pipeline.ipynb

# Validate artifacts from last run
python scripts/validate_artifacts.py --dir artifacts/last_run

# Profile pipeline on a demo AOI
python scripts/profiling_tools.py --config configs/kaggle.yaml
```

---

## 🧪 Testing & CI

* Every script is covered by tests in `tests/scripts/`.
* CI runs:

  * `run_pipeline.sh` on a dummy AOI (small tile)
  * `validate_artifacts.py` schema check
  * `export_kaggle.sh` dry-run
* Failures block merges (reproducibility & ethics guardrails).

---

## 🚀 Next Steps

* Add **data integrity hooks** (hash-check downloads against known checksums).
* Integrate **core sampling DB fetchers** (NOAA Paleoclim, Neotoma, IODP, WoSIS).
* Expand `profiling_tools.py` with **Monte Carlo runtime variance checks**.
* Add **CLI wrappers** for cloud batch jobs (Slurm, AWS Batch).

---

*Last updated: 2025-08-28*

---
