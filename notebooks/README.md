# 🗒️ Notebooks — World Discovery Engine (WDE)

This folder hosts the **Kaggle-ready pipeline notebook** and any **local exploration notebooks**.

👉 **Rule of thumb:** notebooks are **thin, documented entry points** — all core logic lives in `world_engine/`, configs in `configs/`, and tests in `tests/`.
Notebooks should be **reproducible narratives**, not where the heavy lifting lives.

---

## ✅ Goals

* **Kaggle-first:** must run end-to-end in a Kaggle kernel, with no hidden dependencies.
* **Reproducible:** deterministic seeds, containerized runtime, CI validation.
* **Modular:** all code imports from `world_engine/` (no inline re-implementation).
* **Config-driven:** behavior controlled by YAML/JSON in `configs/`, not hardcoded.
* **Artifacts-first:** save all outputs to `/kaggle/working/outputs/` for persistence and audit.

---

## 📦 Included

* **`ade_discovery_pipeline.ipynb`**
  Main Kaggle notebook (competition deliverable). Runs the full pipeline:

  1. AOI tiling & ingestion (Sentinel, SAR, DEM, optional LiDAR).
  2. Coarse anomaly scan (CV filters, VLM captions).
  3. Mid-scale evaluation (NDVI/EVI time-series, hydro-geomorph checks, historical overlays).
  4. Verification (ADE fingerprints, PAG causal graphs, Bayesian GNN, SSIM counterfactuals).
  5. Candidate dossiers (maps, overlays, uncertainty, refutation tests, narrative).

* **`WDE_Kaggle_Starter.ipynb`**
  Minimal demo: lists inputs, runs a stub pipeline, exports a toy CSV of demo candidates, and writes a run manifest.

---

## 📁 Paths (Kaggle runtime)

* **Input datasets**: `/kaggle/input/<competition-or-dataset>`
* **Working outputs**: `/kaggle/working/outputs/`
* **Notebook execution dir**: `/kaggle/working/`
* **Repo modules**: `./world_engine/`, `./configs/` (copied next to the notebook for Kaggle)

---

## ⚙️ Configuration

* Default: `./configs/kaggle.yaml` (or override with another YAML).
* Configs define: AOI, datasets, thresholds, model choices, output dirs.
* Helper: `world_engine/utils/config_loader.py` loads YAML/JSON and merges CLI overrides.

---

## 🧪 Quick Smoke Test

1. Open **`WDE_Kaggle_Starter.ipynb`** on Kaggle.
2. Attach a dataset under `/kaggle/input` (any CSV works for the demo).
3. Run all cells: you should see system info, input tree, and demo outputs in `/kaggle/working/outputs/`:

   * `demo_candidates.csv`
   * `run_manifest.json`

For the **full pipeline**, run **`ade_discovery_pipeline.ipynb`** — it will generate candidate dossiers in `/outputs/`.

---

## 🚀 Conventions for Future Notebooks

* Keep cells short, focused, and **narrative-first**.
* All heavy code → `world_engine/` modules (import, don’t duplicate).
* Use `with timer("step"):` or `%time` for lightweight profiling.
* Visuals: quick Matplotlib or inline plots; heavy GIS/3D viz belongs in local dev.
* Always version outputs (GeoJSON, JSON, reports) — Kaggle CI will validate artifacts.

---

*Last updated: 2025-08-27 19:35:00*

---
