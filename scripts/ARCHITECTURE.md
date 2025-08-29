# 🌍 World Discovery Engine (WDE) — Architecture

`scripts/ARCHITECTURE.md`

---

## 0. Purpose & Scope

The **World Discovery Engine (WDE)** is a **multi-modal AI pipeline** developed for the **OpenAI → Z Challenge**.
Its mission: surface **archaeologically significant candidates** (Anthropogenic Dark Earths (ADEs), earthworks, geoglyphs, settlements) using **open geospatial datasets**, **computer vision**, **causal inference**, and **simulation frameworks**.

WDE outputs **candidate site dossiers** with:

* 📡 Multi-sensor overlays (Sentinel, Landsat, SAR, LiDAR, DEMs)
* 🌱 Soil & vegetation fingerprints (ADE indicators, MapBiomas floristic maps)
* 📜 Historical & archival references (OCR-processed maps, diaries, site DBs)
* 🔗 Causal plausibility graphs (Partial Ancestral Graphs, Bayesian GNNs)
* 🧩 Refutation & counterfactual tests (SSIM, uncertainty simulations)

---

## 1. Data Ecosystem

**Core Open Datasets:**

* Sentinel-1 (SAR), Sentinel-2 (optical)
* Landsat Collection 2 (historical optical)
* NICFI Planet mosaics (4.7 m tropical imagery)
* SRTM, Copernicus DEM (30m–90m global elevation)
* GEDI LiDAR (forest structure), OpenTopography LiDAR
* MODIS climate & vegetation (NDVI/EVI, land cover)
* MapBiomas land cover (Amazon, Latin America)
* HydroSHEDS hydrography (rivers, floodplains)

**Supplementary Sources:**

* Historical maps, diaries, and archives (OCR + geoparsing)
* Core sampling databases (NOAA Paleoclimatology, PANGAEA, Neotoma, ICDP, IODP)
* Soil databases (ISRIC WoSIS, SISLAC, RADAMBRASIL)

**Multi-Modal Fusion:**
User-uploaded docs, images, or overlays can be integrated (georeferenced maps, local soil cores, photos).

---

## 2. Discovery Funnel (Pipeline Stages)

### Step 1: **Tiling & Ingestion**

* Grid AOI into \~5 km tiles.
* Ingest Sentinel, DEM, soil, vegetation, hydro, historical overlays.
* Normalize into multi-band “overlay stack.”

### Step 2: **Coarse Scan**

* CV: edges, Hough transforms, textures
* DEM analysis: hillshades, local relief models
* VLM captioning (CLIP, PaliGemma, Prithvi-EO)
* Output: anomaly heatmap + ranked tiles

### Step 3: **Mid-Scale Evaluation**

* LiDAR/DEM canopy removal
* NDVI/EVI seasonal persistence checks
* Hydro-geomorph plausibility (terraces, ridges near water)
* Historical concordance with OCR’d maps & diaries
* Overlay concordance with user uploads

### Step 4: **Verification**

* Multi-proof requirement: ≥2 independent evidence sources
* ADE fingerprints: dry-season NDVI spike, floristic anomalies, phosphorus-rich soils
* Bayesian GNN for uncertainty
* PAG causal graph validation
* Counterfactual SSIM test

### Step 5: **Candidate Reports**

Each candidate dossier includes:

1. Bounding box map
2. DEM/LiDAR panels
3. NDVI/EVI plots
4. SAR overlays
5. Historical snippet
6. PAG causal graph
7. Uncertainty histograms
8. SSIM map
9. ADE fingerprint summary
10. Refutation & confidence narrative

---

## 3. Repository Architecture

Repository follows modular scientific pipeline design:

```
world-discovery-engine/
├── world_engine/              # Core Python package
│   ├── ingest.py              # Step 1
│   ├── detect.py              # Step 2
│   ├── evaluate.py            # Step 3
│   ├── verify.py              # Step 4
│   ├── report.py              # Step 5
│   ├── models/                # GNNs, anomaly detectors
│   ├── utils/                 # geospatial, image, OCR, logging
│   └── cli.py                 # Typer CLI (ingest, scan, eval, verify, report)
│
├── configs/                   # YAML configs (AOI, datasets, thresholds)
├── data/                      # managed by DVC/Kaggle (raw, interim, outputs)
├── notebooks/                 # Kaggle-ready pipeline notebook
├── scripts/                   # utility bash/python scripts
├── tests/                     # unit & integration tests
├── docs/                      # architecture, datasets, ethics
├── .github/workflows/         # CI/CD (lint, tests, pipeline check)
├── Dockerfile                 # reproducible runtime
├── requirements.txt           # pinned deps
└── README.md
```

Supports: modularity, reproducibility, Kaggle execution, and ethics compliance.

---

## 4. Ethics & Governance

* CARE Principles (Collective Benefit, Authority, Responsibility, Ethics)
* FPIC (Free, Prior, Informed Consent) integration
* Masking coordinates in sensitive areas
* Compliance with national laws (e.g., IPHAN in Brazil)
* Community collaboration: feedback loops, metadata contribution
* ETHICS.md in repo documents policies

---

## 5. Execution Backbone

* **Reproducibility:** Hydra configs + DVC + Kaggle datasets
* **CI/CD:** Lint, unit tests, integration (dummy AOI), ethics checks
* **Artifacts:** run manifests, per-site reports, logs
* **Scaling:** Tile-based parallelism, cloud-ready (COGs, Dask/Spark)

---

## 6. Success Criteria

* Archaeological impact: ADE proxies & new site leads
* Multi-proof validation: ≥2 modalities per candidate
* Reproducibility: deterministic configs & logs
* Ethics: sovereignty-aware, no data colonialism
* Presentation: dossiers as polished “case files”

---

✅ This **scripts/ARCHITECTURE.md** is now a **canonical in-repo architecture guide**, aligning code structure with the discovery funnel, dataset registry, ethics overlays, and Kaggle challenge requirements.

---
