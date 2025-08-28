# 🌍 World Discovery Engine — Models Architecture

`world_engine/ARCHITECTURE.md`

---

## 0. Purpose

The `world_engine` package contains the **core scientific and AI/ML models** that power the World Discovery Engine (WDE).
These models are designed to:

* Detect anomalies in satellite/remote sensing data.
* Fuse multi-modal evidence (imagery, radar, DEM, soil, vegetation, historical overlays).
* Identify **Anthropogenic Dark Earth (ADE) fingerprints** as proxies for ancient habitation.
* Quantify **uncertainty** via Bayesian Graph Neural Networks (B-GNNs).
* Encode **causal relationships** with Partial Ancestral Graphs (PAGs).
* Provide **baseline, benchmark, and ensemble models** for comparative evaluation.

This document describes the **architecture, roles, and interconnections** of all models under `/world_engine/models`.

---

## 1. Design Principles

The WDE models follow six architectural principles:

1. **Multi-Modality** — Fuse satellite, radar, DEM, LiDAR, soil, vegetation, and textual overlays.
2. **Explainability** — Models must produce interpretable outputs: feature maps, graphs, uncertainty distributions.
3. **Reproducibility** — All models run on **open datasets**, with deterministic seeds and Kaggle-ready runtime.
4. **Scalability** — Design supports both Kaggle (GPU-optional) and local/HPC runs.
5. **Modularity** — Each model is isolated in its own file, with standardized `fit()`, `predict()`, and `explain()` APIs.
6. **Ethics** — All outputs integrate sovereignty markers, CARE principles, and coordinate masking where appropriate.

---

## 2. Model Inventory

| File                      | Purpose                                                      | Key Methods                                                            | Inputs                                      | Outputs                                                   |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| **`anomaly_detector.py`** | Classical CV + AI anomaly scan (Step 2 of Discovery Funnel). | Edge detection, Hough transforms, texture features, VLM embeddings.    | Sentinel-2, Sentinel-1, DEM, NICFI mosaics. | Ranked anomaly list per tile.                             |
| **`gnn_fusion.py`**       | Graph Neural Network for **evidence fusion**.                | GATConv / RGCN with edge features (soil, hydro, vegetation, geomorph). | Multi-modal tile features.                  | Node embeddings + fused site probability.                 |
| **`ade_fingerprint.py`**  | ADE-specific detection module.                               | NDVI seasonality, floristic indicators, fractal dimension analysis.    | NDVI/EVI time-series, SoilGrids, MapBiomas. | ADE fingerprint likelihood.                               |
| **`uncertainty_bgnn.py`** | Bayesian GNN for uncertainty calibration.                    | Pyro + PyTorch Geometric, Monte Carlo dropout.                         | Evidence graphs from `gnn_fusion`.          | Probability distribution, entropy, confidence histograms. |
| **`causal_pag.py`**       | Causal graph builder.                                        | Fast Causal Inference (FCI), constraint-based discovery.               | Multi-modal features per candidate.         | `.gml` PAG causal graphs, edge interpretation.            |
| **`baselines.py`**        | Benchmark models.                                            | Random Forest, PCA, Logistic Regression.                               | Soil + elevation + vegetation covariates.   | Comparative site probability (baseline vs advanced).      |
| **`__init__.py`**         | Unified API.                                                 | Exports all model constructors.                                        | Import layer.                               | Standardized model registry.                              |

---

## 3. Evidence Flow

The models operate sequentially through the **Discovery Funnel**:

1. **Anomaly Scan:** `anomaly_detector.py` → candidate anomalies.
2. **Evidence Fusion:** `gnn_fusion.py` merges signals.
3. **ADE Fingerprints:** `ade_fingerprint.py` validates nutrient/vegetation patterns.
4. **Causal Verification:** `causal_pag.py` builds causal graphs.
5. **Uncertainty:** `uncertainty_bgnn.py` quantifies reliability.
6. **Baselines:** `baselines.py` provide sanity checks.
7. **Reporting Layer:** Outputs flow into `world_engine/report.py` as dossiers.

---

## 4. Data & Integration

**Core Inputs:**

* Sentinel-2 optical (RGB, NIR, SWIR).
* Sentinel-1 SAR radar.
* SRTM / Copernicus DEM.
* LiDAR (GEDI, OpenTopography).
* SoilGrids (phosphorus, carbon, pH).
* MODIS / MapBiomas vegetation layers.
* HydroSHEDS rivers/floodplains.
* Historical diaries, maps (OCR + NLP overlays).
* User-uploaded photos, geospatial files.

**Outputs:**

* Candidate GeoJSON with anomaly scores.
* ADE fingerprint CSVs.
* Causal graph `.gml` files.
* B-GNN uncertainty histograms.
* Final **candidate site dossiers**.

---

## 5. Mathematical & Scientific Basis

* **Patterns & Fractals:** ADE fingerprinting leverages fractal dimensions and scaling laws.
* **Graph Theory:** GNN fusion uses spectral/attention message passing.
* **Bayesian Modeling:** B-GNN uncertainty integrates Monte Carlo inference.
* **Physics-Informed Priors:** Terrain + hydrology modeled via DEM-based PDEs.
* **Causal Inference:** PAG ensures discoveries are causally plausible.

---

## 6. CLI & Notebook Integration

All models are accessible via:

* **CLI (`world_engine/cli.py`)**

  ```bash
  wde detect --aoi amazon.json
  wde verify --config configs/default.yaml
  ```
* **Kaggle Notebook (`ade_discovery_pipeline.ipynb`)**
  Direct `import world_engine.models as models` calls in pipeline cells.

This ensures both **scripted reproducibility (CLI)** and **interactive explainability (Notebook)**.

---

## 7. Ethics & Governance

* All outputs integrate **CARE principles**.
* Candidate coordinates are masked by default (2 decimal precision).
* Reports include sovereignty warnings where detections overlap Indigenous territories.
* Repository ships with `ETHICS.md` for explicit governance guidance.

---

## 8. Extensibility

Future model additions:

* **Foundation models for remote sensing** (Prithvi, SatCLIP).
* **Causal counterfactual simulators** beyond SSIM.
* **Reinforcement-learning anomaly prioritizers.**
* **Community-in-the-loop validation** (crowdsourced overlays, GlobalXplorer-style).

---
