# 🌍 World Discovery Engine (WDE) — Architecture

`/docs/ARCHITECTURE.md`

---

## 0. Purpose & Scope

The **World Discovery Engine (WDE)** is a **multi-modal AI pipeline** developed for the **OpenAI → Z Challenge**.
Its mission: detect **archaeologically significant sites** (e.g., **Anthropogenic Dark Earths (ADEs)**, earthworks, geoglyphs) in the Amazon and beyond using **open geospatial datasets**, **computer vision**, **causal inference**, and **simulation frameworks**.

The pipeline outputs **candidate site dossiers** that combine:

* 📡 Multi-sensor overlays (Sentinel, Landsat, SAR, LiDAR, DEMs)
* 🌱 Soil & vegetation fingerprints (ADE indicators, floristic species maps)
* 📜 Historical & archival references (OCR-processed maps, diaries, site DBs)
* 🔗 Causal plausibility graphs & Bayesian uncertainty estimates
* 🧩 Refutation & counterfactual tests (SSIM, multi-proof validation)

This architecture ensures **Kaggle-ready reproducibility**, **NASA-grade simulation rigor**, and **CARE-aligned ethics**.

---

## 1. Data Ecosystem

The WDE ingests **heterogeneous open datasets**:

### Core Geospatial

* **Optical**: Sentinel-2, Landsat (AWS/USGS), NICFI Planet mosaics
* **Radar**: Sentinel-1 SAR (ASF DAAC, Copernicus)
* **Elevation**: SRTM, Copernicus DEM, LiDAR (GEDI, OpenTopography)
* **Climate & Landcover**: MODIS, MapBiomas, HydroSHEDS

### Soil & Vegetation Proxies

* **SoilGrids**: nutrients (esp. phosphorus, carbon) → ADE proxy
* **Vegetation communities**: Brazil nut, peach palm, cacao → correlated with earthworks

### Historical & Archival

* **OCR-processed diaries & maps** → georeferenced overlays
* **Ethnographic databases & colonial surveys**

### Core Samples & Geochemistry

* NOAA Paleoclimatology, PANGAEA, Neotoma, ICDP, IODP, LacCore
* Used to validate **paleo-environmental context** of detected anomalies.

### User-Uploaded Overlays

* Docs (OCR → entity tags), images (geoaligned), shapefiles/GeoJSON

All sources are **open/CC-0** or user-contributed, ensuring Kaggle reproducibility.

---

## 2. Discovery Funnel (Pipeline)

The **Discovery Funnel** is a **five-stage modular pipeline**, each stage refining candidates.

### Step 1: Tiling & Ingestion

* AOI gridded into 0.05° tiles (\~5 km).
* Core rasters stacked (optical, SAR, DEM).
* User overlays attached.
* Output: `tile_data` (GeoTIFFs, JSON, GeoJSON).

### Step 2: Coarse Anomaly Scan

* **CV filters**: edges, Hough transforms, morphology
* **Texture metrics**: LBP, GLCM
* **DEM relief**: hillshades, Local Relief Models (LRM)
* **VLM captioning**: CLIP/PaLI-Gemma, detect “rectangular clearing” etc.
* Output: anomaly scores + ranked candidates.

### Step 3: Mid-Scale Evaluation

* **LiDAR/DEM refinement**: canopy removal, micro-relief detection
* **NDVI/EVI time-series**: confirm persistent anomalies (seasonal ADE vigor)
* **Hydro-geomorph plausibility**: terraces near rivers, not swamps
* **Historical overlays**: diary/map concordance
* Output: evidence-enriched candidates.

### Step 4: Verification & Fusion

* **Multi-proof rule**: ≥2 independent modalities (e.g., vegetation + geomorph)
* **ADE fingerprints**: dry-season NDVI peaks, floristic indicators, ring middens
* **Causal graphs**: PAG via FCI (soil → veg → anomaly)
* **Bayesian GNN**: uncertainty histograms
* **Counterfactual SSIM tests**: ablation by removing modalities
* Output: high-confidence discoveries.

### Step 5: Candidate Dossier

Each verified site gets a **dossier**:

1. Map + bounding box
2. DEM/LiDAR panels
3. Vegetation time-series
4. SAR overlays
5. Historical snippets
6. Causal graph visualization
7. B-GNN uncertainty plots
8. SSIM sensitivity heatmap
9. ADE fingerprint confirmation
10. Sovereignty/ethics flags

---

## 3. Repository Structure

The repo mirrors the pipeline:

```
world-discovery-engine/
├── world_engine/              # Core Python package
│   ├── ingest.py              # Step 1
│   ├── detect.py              # Step 2
│   ├── evaluate.py            # Step 3
│   ├── verify.py              # Step 4
│   ├── report.py              # Step 5
│   ├── models/                # ML models (CNN, GNN, BNN)
│   ├── utils/                 # Shared geospatial/text helpers
│   └── cli.py                 # Typer CLI: ingest → report
├── notebooks/
│   └── ade_discovery_pipeline.ipynb  # Kaggle-ready notebook:contentReference[oaicite:28]{index=28}
├── configs/
│   ├── default.yaml           # AOI, datasets, thresholds
│   ├── dataset/*.yaml         # Sentinel, SoilGrids, LiDAR configs
│   └── model/*.yaml           # ML/GNN/Bayesian configs
├── tests/                     # Unit + integration tests
├── docs/                      # Architecture, ethics, datasets
└── .github/workflows/         # CI/CD, lint, security, submission
```

---

## 4. CLI, GUI & Simulation Integration

* **CLI (Typer)**: `wde ingest`, `wde detect`, `wde evaluate`, `wde verify`, `wde report`
* **GUI**: Optional (React, Qt, Electron, Flutter), consuming CLI outputs
* **Simulation**: Monte Carlo, System Dynamics, ABM → test causal models & uncertainty
* **Physics Integration**: DEM/vegetation anomalies validated against geophysical models (erosion, hydrology)

---

## 5. Ethics & Governance

The WDE explicitly enforces **CARE Principles**:

* **Collective Benefit** → community sharing, open CC-0 artifacts.
* **Authority to Control** → sovereignty notices on Indigenous land detections.
* **Responsibility** → masking of coordinates (≥2 decimals) unless authorized.
* **Ethics** → refutation logs, governance overlays.

Safeguards:

* 🚫 No release of precise site coordinates in public Kaggle runs.
* ⚠ Candidate dossiers carry sovereignty flags if within Indigenous lands.
* ✅ Compliance hooks for IPHAN/Brazil and other national laws.

---

## 6. Reproducibility Backbone

* **Hydra configs** → experiment parameterization
* **DVC or Kaggle Datasets** → versioned open datasets
* **Dockerfile** → reproducible environment (GDAL, PDAL, PyTorch)
* **GitHub Actions** → CI/CD: lint, test, ethics guard, submission bundle
* **Hash logging** → manifest + checksum logs for every run.

---

## 7. Success Criteria

* **Archaeological Impact** → ADEs, geoglyphs, or earthworks surfaced.
* **Evidence Depth** → ≥2 independent modalities per candidate.
* **Causal Plausibility** → narrative graphs align with scientific reasoning.
* **Uncertainty Quantified** → Bayesian histograms, SSIM sensitivity maps.
* **Ethically Defensible** → CARE-aligned, sovereignty-aware outputs.
* **Reproducibility** → Kaggle rerun produces identical dossiers.

---

📖 **References**

* WDE Architecture Specification
* WDE Repository Structure
* ADE Discovery Pipeline Notebook
* Enriching WDE for Archaeology & Earth Systems
* Core Sampling Databases
* Connecting to Remote Sensing & Environmental Data Sources
* Scientific Modeling & Simulation Guide
* GUI & CLI Technical Guides
* Physics & Astrophysics Simulation Reference

---
