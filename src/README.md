# 📂 `src/` — World Discovery Engine (WDE) Source Code

The **`src/` directory** contains the **core implementation of the World Discovery Engine (WDE)** — a reproducible AI/ML pipeline designed for the **OpenAI → Z Challenge**. It fuses **open geospatial data, anomaly detection, and causal verification** to surface candidate archaeological sites (e.g., Anthropogenic Dark Earths (ADEs), geoglyphs, earthworks) in Amazonia and beyond.

This directory provides **all source code** needed to run the discovery funnel end-to-end in Kaggle notebooks, local environments, or Dockerized CI pipelines.

---

## 🏛️ Pipeline Overview

The WDE pipeline is implemented in **modular stages**:

1. **Ingest (`ingest.py`)**

   * Divides AOI into tiles.
   * Loads core datasets: Sentinel-2, Sentinel-1 SAR, Landsat, DEMs (SRTM/Copernicus), LiDAR (GEDI/OpenTopography).
   * Integrates user overlays (documents, images, maps).

2. **Detect (`detect.py`)**

   * Coarse anomaly scan using:

     * CV filters (edges, Hough, morphology).
     * Texture features (LBP, GLCM).
     * DEM relief/hillshade models.
     * Vision-Language captions (e.g., CLIP).
   * Outputs ranked anomaly list per tile.

3. **Evaluate (`evaluate.py`)**

   * Mid-scale analysis of anomalies:

     * Seasonal NDVI/EVI consistency.
     * LiDAR canopy removal (if available).
     * Hydro-geomorphic plausibility (terraces, floodplains).
     * Historical overlays (diaries, maps).
   * Eliminates false positives, strengthens candidates.

4. **Verify (`verify.py`)**

   * Multi-proof rule: require ≥2 modalities of evidence.
   * ADE fingerprint detection: persistent NDVI peaks, floristic communities, micro-topography, fractal metrics.
   * Causal plausibility graphs (FCI → PAG).
   * Bayesian GNN uncertainty scoring.
   * SSIM counterfactual tests.

5. **Report (`report.py`)**

   * Generates **candidate site dossiers**:

     * AOI map + bounding box.
     * DEM/LiDAR panels.
     * NDVI/EVI time-series.
     * SAR overlays.
     * Historical references.
     * PAG causal graphs.
     * B-GNN uncertainty plots.
     * SSIM sensitivity maps.
   * Outputs PDFs, PNGs, GeoJSON, and JSON logs.

---

## 📂 Directory Layout

```
src/
├── world_engine/          # Core WDE package
│   ├── __init__.py
│   ├── ingest.py          # Step 1: Ingestion & tiling
│   ├── detect.py          # Step 2: Coarse anomaly detection
│   ├── evaluate.py        # Step 3: Mid-scale evaluation
│   ├── verify.py          # Step 4: Verification & fusion
│   ├── report.py          # Step 5: Candidate dossier generation
│   ├── utils/             # Shared utilities (geospatial, CV, NLP)
│   ├── models/            # ML models (GNNs, CNNs, anomaly detectors)
│   ├── cli.py             # Typer-based CLI entrypoint
│   ├── api/               # (Optional) FastAPI backend
│   └── ui/                # (Optional) lightweight UI stubs
```

---

## ⚙️ Design Principles

* **Modularity**: Each stage is a standalone module with clear I/O contracts.
* **Reproducibility**: Config-driven (`configs/` YAMLs, Hydra-ready). No hardcoded paths.
* **Transparency**: All inputs/outputs documented in logs & reports.
* **Fallbacks**: Kaggle-compatible — falls back to sample data when APIs unavailable.
* **Ethics**: Built-in safeguards (CARE principles, Indigenous sovereignty flags, coordinate rounding).

---

## 🚀 Usage

From Kaggle Notebook or CLI:

```bash
# Run entire pipeline on Amazonia AOI
python -m world_engine.cli run --config configs/amazon.yaml

# Or run individual stages
python -m world_engine.cli ingest --aoi configs/aoi_examples.yaml
python -m world_engine.cli detect --input data/tiles/
python -m world_engine.cli verify --candidates data/candidates_step3.json
```

Outputs are saved in `/outputs/`, including **GeoJSON maps, PNG overlays, JSON logs, and site dossier PDFs**.

---

## 📚 References

* **WDE Architecture Specification**
* **Repository Structure Guide**
* **ADE Discovery Pipeline (Kaggle Notebook Scaffold)**
* **Enriching WDE for Archaeology & Earth Systems**

---
