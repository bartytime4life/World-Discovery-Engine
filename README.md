# 🌍 World Discovery Engine (WDE)

**OpenAI → Z Challenge · Archaeology & Earth Systems**

---

## 📌 Overview

The **World Discovery Engine (WDE)** is a **multi-modal AI pipeline** designed to surface archaeologically significant sites in the Amazon and beyond.
It fuses **satellite imagery, radar, LiDAR, soil & vegetation maps, hydrology, historical archives, and core sampling data** to generate **candidate site dossiers**.

Each dossier includes:

* 📡 **Multi-sensor overlays** (Sentinel, Landsat, SAR, LiDAR)
* 🌱 **Soil & vegetation fingerprints** (ADE indicators)
* 📜 **Historical & archival references** (maps, diaries, site DBs)
* 🔗 **Causal plausibility graphs** (PAG .gml from FCI inference)
* 🎲 **Uncertainty quantification** (Bayesian GNN + ensembles)
* 🧪 **Simulation & counterfactuals** (SSIM robustness tests)

The pipeline runs **fully on Kaggle** (GPU optional, CPU fallback) using only **open datasets (CC-0 or equivalent)**.

---

## 🏆 Challenge Context

This project is built for the **OpenAI → Z Challenge**.
Key requirements:

* ✅ **CC-0 licensed data** (≥2 independent sources per finding)
* ✅ **Archaeological impact focus** (ADEs, geoglyphs, settlement structures)
* ✅ **Single Kaggle Notebook deliverable** (`notebooks/ade_discovery_pipeline.ipynb`)
* ✅ **Reproducible outputs** with clear provenance and audit trail

The Kaggle rubric prioritizes **quality of discoveries** (archaeological plausibility) over raw anomaly count.

---

## 🔬 Pipeline Stages

The **Discovery Funnel** narrows from broad scan to detailed validation:

1. **Tiling & Ingestion**

   * AOI grid (0.05° tiles \~ 5 km)
   * Load Sentinel-2, Sentinel-1, DEM, optional LiDAR
   * Ingest user overlays (docs, maps, images)

2. **Coarse Scan**

   * CV filters (edges, Hough, morphology)
   * Texture features (LBP, GLCM)
   * DEM hillshades & Local Relief Model
   * Vision-Language captions (e.g. “rectangular clearing”)

3. **Mid-Scale Evaluation**

   * Seasonal NDVI/EVI time-series
   * LiDAR canopy removal (if available)
   * Hydro-geomorphology plausibility (terraces, bluffs, floodplains)
   * Historical concordance (OCR’d diaries, georeferenced maps)

4. **Verification & Fusion**

   * **Multi-proof rule**: ≥2 modalities required
   * ADE fingerprints (dry-season NDVI spikes, floristic indicators, micro-topography, fractal analysis)
   * Causal plausibility (PAG `.gml` graphs)
   * Bayesian GNN for calibrated uncertainty
   * SSIM counterfactuals (robustness checks)

5. **Candidate Dossier Generation**

   * Site maps, overlays, causal graph, uncertainty plots
   * ADE indicator checklist
   * Refutation tests summary
   * Narrative confidence statement

---

## 📂 Repository Structure

See [`docs/repository_structure.md`](docs/repository_structure.md) for full details.

Key directories:

* `world_engine/` — Core pipeline (ingest → detect → evaluate → verify → report)
* `configs/` — YAML configs for AOIs, datasets, models
* `notebooks/` — Kaggle-ready `ade_discovery_pipeline.ipynb`
* `data/` — (Optional) local mount for raw/interim/output (DVC-managed)
* `docs/` — Architecture, datasets, ethics, contributing guides
* `tests/` — Unit + integration tests (small AOI, reproducibility checks)

---

## ⚖️ Ethics & Governance

WDE is built with **CARE Principles** (Collective Benefit, Authority to Control, Responsibility, Ethics):

* **Respect Indigenous sovereignty**: detections in Indigenous lands are flagged; precise coordinates masked without consent.
* **Legal compliance**: supports region-specific restrictions (e.g. Brazil IPHAN).
* **Anti-data-colonialism**: outputs are intended for **collaborative archaeology**, not unilateral claims.

See [`docs/ETHICS.md`](docs/ETHICS.md) for details.

---

## ⚙️ Reproducibility

* **Deterministic runs** (fixed seeds, logged configs)
* **Dockerfile** for runtime parity with Kaggle
* **CI/CD via GitHub Actions**: lint, test, notebook CI, artifact validation
* **CausalOps lifecycle**: Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document

---

## 🚀 Getting Started

Clone and install:

```bash
git clone https://github.com/<your-org>/world-discovery-engine.git
cd world-discovery-engine
pip install -r requirements.txt
```

Run on Kaggle:

1. Upload repo (or link via GitHub → Kaggle integration)
2. Open `notebooks/ade_discovery_pipeline.ipynb`
3. “Run All” (uses open datasets + fallbacks)
4. Outputs available under `/outputs/`:

   * `candidates.json` + `candidates.geojson`
   * `/reports/` (site dossiers, PDF/HTML)
   * `/pag/` (causal graphs)
   * `/uncertainty/` (histograms, JSON)
   * `/ssim/` (robustness tests)
   * `/ndvi_timeseries/` (seasonal ADE checks)

---

## 📑 Documentation

See the [`docs/`](docs/) folder for:

* **Architecture** — [`docs/architecture.md`](docs/architecture.md)
* **Datasets registry** — [`docs/datasets.md`](docs/datasets.md)
* **Repository structure** — [`docs/repository_structure.md`](docs/repository_structure.md)
* **Ethics** — [`docs/ETHICS.md`](docs/ETHICS.md)
* **Contributing** — [`docs/contributing.md`](docs/contributing.md)

---

## 🛡️ License

* **Code**: MIT License
* **Data**: All inputs are **open / CC-0** compliant

---
