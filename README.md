# 🌍 World Discovery Engine (WDE)

**OpenAI → Z Challenge · Archaeology & Earth Systems**
*Reproducible AI pipeline for archaeologically significant discovery in Amazonia and beyond*

---

## 📌 Overview

The **World Discovery Engine (WDE)** is a multi-modal AI pipeline that surfaces candidate archaeological sites (e.g. Anthropogenic Dark Earths (ADEs), earthworks, geoglyphs) from open geospatial data and historical archives.

It is designed to:

* **Run on Kaggle** as a single notebook (`notebooks/ade_discovery_pipeline.ipynb`).
* **Fuse heterogeneous sources** — Sentinel imagery, SAR, LiDAR, DEM, soils, vegetation, hydrology, colonial diaries, and user-uploaded maps/photos.
* **Validate discoveries** with ≥2 proofs: NDVI/EVI fingerprints, geomorphology, causal plausibility graphs, uncertainty estimates.
* **Generate candidate site dossiers** — maps, overlays, causal graphs, ADE fingerprints, refutation tests.
* **Respect ethics & sovereignty** (CARE Principles, FPIC, IPHAN law).
* **Be reproducible & auditable** — versioned configs, CI, Docker, DVC/Kaggle dataset tracking.

---

## ⚙️ Repository Structure

```text
├── README.md              # Project intro (this file)
├── LICENSE                # Open license (code), CC-0 (notebooks)
├── requirements.txt       # Python dependencies (pinned for reproducibility)
├── Dockerfile             # Containerized runtime (GDAL/PDAL, PyTorch, etc.)
├── .github/workflows/     # CI/CD pipelines (lint, tests, Kaggle notebook CI)
│   ├── kaggle_notebook_check.yml
│   ├── kaggle_notebook_ci.yml
│   └── lint.yml
├── world_engine/          # Core pipeline package
│   ├── ingest.py          # Step 1 – tiling & ingestion
│   ├── detect.py          # Step 2 – coarse anomaly scan
│   ├── evaluate.py        # Step 3 – mid-scale evaluation
│   ├── verify.py          # Step 4 – verification & fusion
│   ├── report.py          # Step 5 – dossier generation
│   └── utils/, models/    # Shared functions & ML models
├── configs/               # Config files (YAML/JSON for AOIs, datasets, models)
├── data/                  # (DVC/Kaggle-linked) raw, interim, output artifacts
├── notebooks/             # Kaggle notebook(s), exploration, experiments
│   └── ade_discovery_pipeline.ipynb
├── tests/                 # Unit & integration tests
└── docs/                  # Architecture, ethics, datasets, usage guides
```

See full structure in \[docs/repository\_structure.md].

---

## 🚀 Quickstart

### 1. Kaggle Notebook

Run the pipeline end-to-end on Kaggle:

```bash
# In Kaggle Notebook terminal
!pip install -r requirements.txt
!papermill notebooks/ade_discovery_pipeline.ipynb notebooks/out.ipynb -k python3
```

Outputs will be in `/kaggle/working/outputs/`, including:

* `candidates.json`, `candidates.geojson`
* `reports/` (PDF/HTML dossiers)
* `pag/` (causal graphs)
* `uncertainty/` (histograms, JSON)
* `ssim/` (counterfactual what-if tests)

### 2. CLI (local use)

```bash
# Install
pip install -e .
# Run pipeline
wde full-run --config configs/default.yaml
# Or stepwise
wde ingest --aoi data/aoi/brazil.geojson
wde scan
wde evaluate
wde verify
wde report
```

---

## 🔍 Pipeline (Discovery Funnel)

1. **Ingestion** — Tile AOI (0.05°), load Sentinel-2, Sentinel-1, DEM, SoilGrids, MapBiomas, HydroSHEDS, GEDI, + user overlays.
2. **Coarse Scan** — CV filters (edges, Hough), texture (LBP/GLCM), DEM hillshades, VLM captioning.
3. **Mid-Scale Evaluation** — LiDAR canopy removal, NDVI/EVI time-series, hydro-geomorph plausibility, historical concordance.
4. **Verification** — Multi-proof rule, ADE fingerprints (seasonal NDVI, flora, geomorph), PAG causal graphs, Bayesian GNN uncertainty, SSIM what-if tests.
5. **Reporting** — Candidate dossiers: maps, NDVI plots, SAR, historical snippets, causal graphs, uncertainty histograms, SSIM overlays, ADE checklist, confidence narratives.

---

## 📊 Data Sources

All datasets are open, CC-0/CC-BY, reproducible via Kaggle or APIs:

* **Satellite**: Sentinel-2 (optical), Sentinel-1 (SAR), Landsat, NICFI Planet mosaics.
* **Elevation & LiDAR**: SRTM, Copernicus DEM, GEDI, OpenTopography.
* **Soils**: ISRIC SoilGrids, SISLAC, RADAMBRASIL legacy.
* **Vegetation & Land Cover**: MapBiomas, MODIS NDVI/EVI, floristic indicators.
* **Hydrology**: HydroSHEDS (rivers, basins, floodplains).
* **Historical**: Colonial maps, missionary diaries (OCR+NLP), archaeological site DBs.
* **User Overlays**: Uploaded docs/maps/images, auto-OCR + georeference.

See full \[docs/datasets.md] registry.

---

## 🧭 Ethics & Governance

* **CARE Principles** (Collective Benefit, Authority to Control, Responsibility, Ethics).
* **FPIC & IPHAN compliance** for Brazil — mandatory archaeological authorization.
* **Ethics-by-Design** — auto-flags Indigenous lands, sovereignty notices in outputs.
* **Data colonialism safeguards** — discoveries are not auto-published; outputs intended for expert review & local collaboration.

See \[docs/ETHICS.md] for full governance.

---

## 🔬 Reproducibility

* **CausalOps lifecycle** — Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document.
* **Deterministic runs** — fixed seeds, logged configs, Docker reproducibility.
* **CI/CD** — GitHub Actions run lint, unit/integration tests, Kaggle notebook validation.
* **Artifacts** — All intermediate outputs logged: anomaly scores, PAG graphs, refutation reports.

---

## 📑 References

* Architecture & pipeline spec
* Enrichment datasets & anomaly detection methods
* Repository structure
* ADE discovery pipeline (Kaggle notebook)
* Data connection guide
* Core sampling databases

---
