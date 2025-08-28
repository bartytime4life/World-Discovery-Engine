# ⚙️ World Discovery Engine — Core Pipeline (`world_engine/`)

This directory contains the **core Python package** for the World Discovery Engine (WDE).
It implements the **multi-stage Discovery Funnel**:

1. **Ingest** → load AOI tiles, satellite/LiDAR/soil/historical overlays.
2. **Detect** → coarse anomaly scan (CV filters, texture, VLM captions).
3. **Evaluate** → mid-scale evaluation (NDVI/EVI time-series, hydro-geomorph plausibility, LiDAR canopy removal, historical concordance).
4. **Verify** → multi-modal fusion (ADE fingerprints, causal plausibility, Bayesian uncertainty, SSIM counterfactuals).
5. **Report** → candidate site dossiers (maps, overlays, uncertainty plots, refutation narratives).

---

## 📂 Module Overview

| File                    | Purpose                                                                                                                                                                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ingest.py`**         | **Step 1 — Tiling & Ingestion**. Splits AOI into \~0.05° tiles, loads Sentinel-2, Sentinel-1 SAR, DEM, optional LiDAR. Integrates user overlays (docs, maps, images). Outputs stacked rasters + metadata.                                                                  |
| **`detect.py`**         | **Step 2 — Coarse Scan**. Applies CV filters (edges, Hough, morphology), texture features (LBP, GLCM), DEM hillshades, and Vision-Language captions. Produces anomaly scores per tile.                                                                                     |
| **`evaluate.py`**       | **Step 3 — Mid-Scale Evaluation**. Deeper checks: NDVI/EVI seasonal stability, LiDAR canopy removal, hydro-geomorph plausibility (river terraces, bluffs), historical/archival concordance, user overlay cross-checks.                                                     |
| **`verify.py`**         | **Step 4 — Verification & Fusion**. Enforces multi-proof rule (≥2 modalities). Runs ADE fingerprint detection (NDVI peaks, floristic indicators, micro-topography, fractal analysis). Builds PAG causal graphs, Bayesian GNN uncertainty estimates, SSIM robustness tests. |
| **`report.py`**         | **Step 5 — Candidate Dossier Generation**. Compiles all evidence: maps, overlays, causal graphs, ADE checklist, uncertainty histograms, SSIM sensitivity maps, confidence narratives. Outputs reports in JSON, GeoJSON, PDF/HTML.                                          |
| **`utils/`**            | Shared helpers (geospatial reprojection, OCR/NLP for docs, image normalization, config loader, logging). Keeps pipeline DRY.                                                                                                                                               |
| **`models/`**           | ML model definitions (e.g. anomaly detector, Graph Neural Network for fusion). Trained only on open datasets, no proprietary weights.                                                                                                                                      |
| **`api/`** *(optional)* | FastAPI server for wrapping the pipeline into a web service. Not required for Kaggle; useful for integration with dashboards.                                                                                                                                              |
| **`ui/`** *(optional)*  | Placeholder for a thin web/GUI layer. Decoupled from pipeline logic (CLI-first principle).                                                                                                                                                                                 |
| **`cli.py`**            | Defines the **Typer CLI**. Subcommands: `ingest`, `scan`, `evaluate`, `verify`, `report`, `full-run`. Reads YAML configs from `/configs/`. Provides reproducibility by logging every run.                                                                                  |

---

## 🧪 Design Principles

* **Modular** — each stage is an independent file with clear I/O.
* **Reproducible** — configs control all parameters; random seeds are fixed.
* **CLI-first** — pipeline can be run via:

  ```bash
  wde full-run --config configs/default.yaml
  ```
* **Auditable** — every stage writes intermediate JSON/GeoTIFFs for traceability.
* **Extensible** — new models or detectors can be swapped in without breaking the notebook.

---

## 📑 References

* [Architecture Specification](../docs/architecture.md)
* [Repository Structure](../docs/repository_structure.md)
* [ADE Pipeline (Notebook Spec)](../docs/ADE_pipeline.md)

---
