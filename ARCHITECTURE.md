🌍 World Discovery Engine (WDE) — Architecture

0. Purpose & Scope

The World Discovery Engine (WDE) is a multi-modal AI pipeline designed for the OpenAI → Z Challenge.
Its mission is to detect archaeologically significant sites (e.g., Anthropogenic Dark Earths (ADEs), earthworks, geoglyphs) using open geospatial data, computer vision, causal inference, and simulation.

WDE outputs candidate site dossiers that combine:
	•	Satellite/LiDAR imagery overlays
	•	Soil/vegetation anomalies
	•	Historical and archival references
	•	Causal plausibility graphs
	•	Uncertainty distributions
	•	Refutation/validation tests

The system is Kaggle-ready (single notebook deliverable), ethically aligned (CARE + FAIR), and reproducible (CI/CD + DVC + Docker).

⸻

1. Data Ecosystem

WDE ingests heterogeneous, open datasets:
	•	Satellite & Remote Sensing
Sentinel-2 optical, Sentinel-1 SAR radar, Landsat archives, NICFI Planet mosaics ￼.
	•	Elevation & LiDAR
SRTM, Copernicus DEM, GEDI LiDAR, OpenTopography ￼.
	•	Soil & Geochemistry
ISRIC SoilGrids (ADE proxy: phosphorus, pH, carbon) ￼.
	•	Vegetation & Ecology
MODIS, MapBiomas, domesticated species (Brazil nut, cacao) ￼.
	•	Hydrology & Terrain
HydroSHEDS rivers, watersheds, terrain derivatives (slope, hillshade, LRM) ￼.
	•	Historical & Archival
Colonial diaries, missionary maps, site databases via OCR + NLP ￼.
	•	Core Sampling
NOAA, PANGAEA, Neotoma, IODP, ICDP repositories (soil, sediment, lake, ice) ￼.
	•	User-Uploaded Overlays
Docs → OCR/entity extraction
Images → CV anomaly detection
Links → metadata classification ￼

All sources are open-access or CC-0; citations/logging ensure reproducibility.

⸻

2. Discovery Funnel

A five-stage pipeline progressively narrows from broad anomaly scans to high-confidence candidate sites:

Step 1: Tiling & Ingestion
	•	AOI divided into ~5km tiles ￼.
	•	Load Sentinel-2, Sentinel-1, DEM, Soil, Hydrology layers.
	•	Integrate user-uploaded overlays (docs, maps, images).

Step 2: Coarse Scan
	•	Computer Vision: Edge detection, Hough transforms, morphological filters ￼.
	•	Texture Analysis: LBP, GLCM ￼.
	•	Terrain Relief: Hillshade + Local Relief Model (LRM).
	•	VLM Captioning: Zero-shot anomaly captions (“rectangular clearing in canopy”).
	•	Scoring: Aggregate anomalies into ranked candidate tiles.

Step 3: Mid-Scale Evaluation
	•	LiDAR canopy removal (PDAL).
	•	NDVI/EVI time-series stability — persistent ADE anomalies ￼.
	•	Hydro-geomorph plausibility — terraces, river proximity ￼.
	•	Historical overlays — diary/map snippets cross-referenced.
	•	User overlay concordance — uploaded confirmations/refutations.

Step 4: Verification & Fusion
	•	Multi-proof rule: ≥2 independent modalities required ￼.
	•	ADE Fingerprints:
	•	Seasonal NDVI peaks (dry-season fertility)
	•	Floristic species markers (palms, cacao)
	•	Micro-topography (ring ditches, mounds) ￼
	•	Causal Graphs: FCI → PAG .gml (cause-effect plausibility).
	•	Uncertainty: Bayesian GNN probability + histograms.
	•	Counterfactual Tests: SSIM ablations (remove modality → check robustness).

Step 5: Candidate Dossiers
	•	Map + bounding box
	•	DEM/LiDAR panels
	•	Vegetation indices over time
	•	SAR overlays
	•	Historical text snippets
	•	PAG causal graph visualization
	•	Bayesian uncertainty plots
	•	SSIM sensitivity heatmaps
	•	ADE checklist
	•	Refutation summary
	•	Confidence narrative ￼

⸻

3. Scientific & Simulation Backbone

WDE integrates NASA-grade modeling and simulation practices ￼ ￼:
	•	Modeling Paradigms:
Agent-based (settlements), System Dynamics (erosion), Finite Element/DEM (terrain), Monte Carlo (uncertainty).
	•	Verification & Validation:
NASA-STD-7009 compliance, MIL-STD-3022 VV&A templates ￼.
	•	Simulation:
Counterfactual SSIM validation, dynamical plausibility checks ￼.
	•	CausalOps Lifecycle:
Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document ￼.

⸻

4. UI & Interaction Modes

Three user interaction layers ￼ ￼:
	1.	CLI (Typer-based)
calibrate, train, validate, diagnose, submit — scriptable, reproducible.
	2.	Web Dashboard (FastAPI + React)
Interactive spectra plots, SHAP overlays, FFT diagnostics, causal graph explorer.
	3.	Notebooks (Kaggle-first)
Exploratory, tutorial-friendly, pipeline importable as wde Python library.

All three share the same underlying pipeline code and configs (Hydra + YAML).

⸻

5. Patterns, Algorithms & Fractals

The WDE explicitly encodes geometric and fractal principles for anomaly detection ￼:
	•	Geometric patterns — circles, rectangles, symmetry groups.
	•	Temporal patterns — NDVI/EVI seasonal oscillations.
	•	Fractals & scaling laws — distinguishing natural irregularity from anthropogenic regularity ￼.
	•	Algorithmic patterns — CV filters, cellular automata, clustering, recursive causal graphs.

This aligns with the HIA–Geodetic Codex: multi-scale fractal tiling, nested symmetry, and causal overlays.

⸻

6. Ethics & Governance

WDE is built with ethical archaeology as a first-class principle ￼:
	•	CARE Principles — Collective Benefit, Authority to Control, Responsibility, Ethics.
	•	Indigenous Data Sovereignty — detections flagged if overlapping Indigenous territories.
	•	Legal Compliance — Brazil (IPHAN), Peru, etc. — site coordinates masked unless permissions obtained ￼.
	•	Anti-Data-Colonialism — collaborative archaeology, not extractive; outputs = dossiers for expert review, not public site maps.

⸻

7. Reproducibility & CI/CD
	•	Versioning: Git + DVC for data, Hydra for configs, MLflow/W&B for experiments.
	•	Artifacts: All outputs (GeoTIFFs, GeoJSONs, JSON logs) versioned and timestamped.
	•	Containers: Docker + Poetry/Conda → reproducible environments.
	•	CI/CD: GitHub Actions: lint → unit tests → anomaly scan → validation suite → dossier build ￼.

⸻

8. Success Criteria
	•	Archaeological impact: ADE proxies & geoglyphs surfaced.
	•	Evidence depth: ≥2 independent modalities per site.
	•	Clarity: Transparent overlays, interpretable causal graphs.
	•	Reproducibility: Rerunnable on Kaggle & Docker.
	•	Ethics: CARE-aligned, sovereignty-respecting.

⸻