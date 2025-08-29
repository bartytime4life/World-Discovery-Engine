<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/wde_banner_dark.svg">
    <img src="assets/wde_banner_light.svg" alt="World Discovery Engine Banner" width="100%">
  </picture>
</p>

# 🌍 World Discovery Engine (WDE)

**OpenAI → Z Challenge · Archaeology & Earth Systems**

[![Kaggle](https://img.shields.io/badge/Kaggle-OpenAI→Z%20Challenge-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/openai-to-z-challenge)
[![CI](https://github.com/<your-org>/world-discovery-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/world-discovery-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/Docs-Architecture-blueviolet?logo=readthedocs&logoColor=white)](docs/architecture.md)

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/DVC-tracked-945DD6?logo=dvc&logoColor=white)](https://dvc.org/)
[![Contributions](https://img.shields.io/badge/Contributions-welcome-orange.svg)](docs/contributing.md)

---

## 📌 Overview

The **World Discovery Engine (WDE)** is a **multi-modal AI pipeline** for discovering archaeologically significant landscapes across the Amazon and beyond.

It fuses **satellite imagery, radar, LiDAR, soils & vegetation maps, hydrology layers, historical archives, and core sampling data** into a reproducible system that outputs **candidate site dossiers** — combining quantitative evidence with interpretive narrative.

Each dossier integrates:

- 📡 **Multi-sensor overlays** (Sentinel, Landsat, SAR, LiDAR)  
- 🌱 **Soil & vegetation fingerprints** (ADE / terra preta indicators)  
- 📜 **Historical concordance** (archival maps, expedition diaries, site DBs)  
- 🔗 **Causal plausibility graphs** (Partial Ancestral Graphs from FCI inference)  
- 🎲 **Uncertainty quantification** (Bayesian GNN ensembles + calibrated scores)  
- 🧪 **Simulation & counterfactuals** (SSIM falsification tests)  

WDE runs **entirely on Kaggle** (GPU optional, CPU fallback) with **open / CC-0 datasets only**, ensuring **transparent reproducibility**.

---

## 🏆 Challenge Context

Built for the **OpenAI → Z Challenge**, WDE satisfies all rubric pillars:

- ✅ **Open/CC-0 data only**, with ≥2 independent modalities per finding  
- ✅ **Archaeological impact focus** — ADEs, geoglyphs, settlement networks, ancient hydrological engineering  
- ✅ **Single Kaggle Notebook deliverable** (`notebooks/ade_discovery_pipeline.ipynb`)  
- ✅ **Reproducible outputs** with deterministic configs & audit logs  

> 🧭 **Key metric:** The rubric prioritizes *plausibility & significance of discoveries*, not raw anomaly counts.

---

## 🔬 Pipeline Stages — The Discovery Funnel

```mermaid
flowchart LR
  classDef stage fill:#0ea5e9,stroke:#0369a1,color:#fff,rx:14,ry:14;
  A[Tiling & Ingestion]:::stage --> B[Coarse Scan]:::stage --> C[Mid-Scale Evaluation]:::stage --> D[Verification & Fusion]:::stage --> E[Report & Dossiers]:::stage

1. Tiling & Ingestion
	•	AOI grid (0.05° ≈ 5 km tiles)
	•	Load Sentinel-2, Sentinel-1, DEMs, LiDAR (if available)
	•	Ingest user overlays (maps, registries, field docs)

2. Coarse Scan
	•	CV primitives (edges, Hough, morphology)
	•	Texture features (LBP, GLCM)
	•	DEM hillshades & Local Relief Models
	•	Vision–Language tags (e.g., “rectangular clearing”)

3. Mid-Scale Evaluation
	•	NDVI/EVI seasonal time-series
	•	LiDAR canopy removal
	•	Hydro-geomorphology plausibility checks
	•	Archival concordance (OCR’d maps, diaries)

4. Verification & Fusion
	•	Multi-proof rule: ≥2 modalities required
	•	ADE fingerprints (floristic spikes, fractal landforms)
	•	PAG causal graphs for plausibility
	•	Bayesian GNN with calibrated uncertainty
	•	SSIM counterfactuals for robustness

5. Report Generation
	•	Maps & overlays
	•	ADE indicator checklist
	•	Uncertainty plots
	•	Narrative confidence statement

⸻

🏗️ Architecture Diagram

flowchart TB
  %% 🌍 World Discovery Engine
  classDef stage fill:#0ea5e9,stroke:#0369a1,color:#ffffff,rx:14,ry:14;
  classDef store fill:#f59e0b,stroke:#92400e,color:#111827,rx:10,ry:10;
  classDef side  fill:#10b981,stroke:#065f46,color:#ffffff,rx:12,ry:12;
  classDef guard fill:#ef4444,stroke:#7f1d1d,color:#ffffff,rx:12,ry:12;
  classDef io    fill:#e5e7eb,stroke:#374151,color:#111827,rx:10,ry:10;

  U[Researcher / Collaborator]:::io
  K[Kaggle Notebook<br/>GPU or CPU]:::io
  GH[GitHub Actions CI/CD]:::io

  subgraph S[Open / CC-0 Data Sources]
    direction TB
    S2[Sentinel-2 Optical]:::store
    S1[Sentinel-1 SAR]:::store
    DEM[DEMs (SRTM / Copernicus)]:::store
    LIDAR[LiDAR / GEDI / OpenTopography]:::store
    SOIL[Soils & Vegetation<br/>(ADE indicators)]:::store
    HYDRO[Hydrography (HydroSHEDS)]:::store
    HIST[Archives: maps, diaries,<br/>registries (OCR)]:::store
    CORE[Core sampling DBs]:::store
  end

  U -->|Define AOI & configs| K
  S --> IN

  subgraph P[World Discovery Engine Pipeline]
    direction TB
    IN[Tiling & Ingestion]:::stage
    SC[Coarse Scan]:::stage
    EV[Mid-Scale Evaluation]:::stage
    VF[Verification & Fusion]:::stage
    RP[Report Generator]:::stage
    IN --> SC --> EV --> VF --> RP
  end

  subgraph G[Governance & Reproducibility]
    direction TB
    ETH[Ethics & Sovereignty]:::guard
    DVC[DVC & Manifests]:::side
    CONF[Determinism & Seeds]:::side
  end

  ETH --- P
  DVC --- P
  CONF --- P

  OUT1[candidates.json / geojson]:::io
  OUT2[/reports/*  dossiers]:::io
  OUT3[/pag/*  PAG graphs]:::io
  OUT4[/uncertainty/* JSON+plots]:::io
  OUT5[/ssim/*  counterfactuals]:::io
  OUT6[/ndvi_timeseries/*]:::io
  RP --> OUT1 & OUT2 & OUT3 & OUT4 & OUT5 & OUT6

  GH -->|lint • test • validate| K
  K -->|Run All| P


⸻

📂 Repository Structure

See docs/repository_structure.md for the complete layout.

Key directories:
	•	world_engine/ — Core pipeline (ingest → detect → evaluate → verify → report)
	•	configs/ — YAML configs (datasets, AOIs, models)
	•	notebooks/ — Kaggle-ready notebook(s)
	•	data/ — Optional local staging (DVC-managed)
	•	docs/ — Architecture, datasets, ethics, contributing
	•	tests/ — Unit & integration tests with mini-AOI reproducibility

⸻

⚖️ Ethics & Governance

WDE is aligned with the CARE Principles:
	•	🪶 Respect Indigenous sovereignty — sites in Indigenous lands flagged; coordinates masked without consent
	•	📜 Regional compliance — e.g., Brazil IPHAN protections
	•	🌐 Anti-data colonialism — results support collaborative archaeology, not unilateral extraction

See docs/ETHICS.md.

⸻

⚙️ Reproducibility & CI/CD
	•	Deterministic runs — fixed seeds, logged configs
	•	Docker parity — Kaggle ↔ local reproducibility
	•	GitHub Actions — lint, unit tests, notebook CI, artifact validation
	•	CausalOps lifecycle: Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document

⸻

🚀 Getting Started

Clone locally:

git clone https://github.com/<your-org>/world-discovery-engine.git
cd world-discovery-engine
pip install -r requirements.txt

Run on Kaggle:
	1.	Upload repo (or sync via GitHub → Kaggle integration)
	2.	Open notebooks/ade_discovery_pipeline.ipynb
	3.	Select “Run All”
	4.	Outputs appear under /outputs/:
	•	candidates.json / candidates.geojson
	•	/reports/ → site dossiers (PDF/HTML)
	•	/pag/ → causal graphs
	•	/uncertainty/ → calibration plots + JSON
	•	/ssim/ → robustness checks
	•	/ndvi_timeseries/ → vegetation fingerprints

⸻

📑 Documentation

The docs/ folder contains:
	•	Architecture — docs/architecture.md
	•	Datasets — docs/datasets.md
	•	Repository structure — docs/repository_structure.md
	•	Ethics — docs/ETHICS.md
	•	Contributing — docs/contributing.md

⸻

🛡️ License
	•	Code: MIT License
	•	Data: All inputs are open-access / CC-0 compliant

⸻

🏁 Project Footer

██████╗ ██╗   ██╗███████╗
██╔══██╗██║   ██║██╔════╝
██████╔╝██║   ██║███████╗
██╔═══╝ ██║   ██║╚════██║
██║     ╚██████╔╝███████║
╚═╝      ╚═════╝ ╚══════╝

🌍 World Discovery Engine (WDE) · OpenAI → Z Challenge
📖 Docs · 🛠️ Issues · 💬 Discussions · ⚙️ CI/CD

Made with ❤️ by the WDE Team · Contributions welcome

⸻

✨ WDE transforms open geospatial chaos into archaeological insight — reproducible, ethical, and scientifically defensible.

---