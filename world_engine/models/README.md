# 🧠 `models/` — Machine Learning & Simulation Models

This directory contains **all machine learning, statistical, and simulation models** used by the **World Discovery Engine (WDE)**. These models power anomaly detection, multi-modal evidence fusion, ADE fingerprint analysis, and uncertainty quantification.

---

## 📌 Purpose

* Provide **modular model definitions** for each stage of the WDE pipeline.
* Support **classical ML, deep learning, and Bayesian GNN approaches**.
* Ensure **reproducibility** (config-driven, no hard-coded paths).
* Remain **Kaggle-ready**: models either run directly on Kaggle GPUs/CPUs or degrade gracefully to heuristics if unavailable.

---

## 📂 Directory Layout

```
models/
├── anomaly_detector.py     # CNN/Autoencoder/CLIP-based anomaly detection
├── gnn_fusion.py           # Graph Neural Network for evidence fusion
├── ade_fingerprint.py      # ADE fingerprint classifier (NDVI, floristic, fractal metrics)
├── uncertainty_bgnn.py     # Bayesian Graph Neural Network for calibrated uncertainty
├── causal_pag.py           # FCI → PAG causal inference wrapper
├── baselines.py            # Random forest, PCA, and heuristic fallback models
└── __init__.py
```

---

## 🧩 Model Roles

### 1. **Anomaly Detector** (`anomaly_detector.py`)

* Detects coarse anomalies from imagery tiles.
* Methods:

  * CNN (ResNet/U-Net) for segmentation.
  * Autoencoder for unsupervised anomaly detection.
  * CLIP embedding similarity (semantic anomaly scoring).
* Outputs ranked anomaly heatmaps.

---

### 2. **Evidence Fusion GNN** (`gnn_fusion.py`)

* Fuses multi-modal features (imagery, DEM, soil, historical text overlays).
* Implements **Graph Neural Network** (PyTorch Geometric).
* Supports edge features (distance, modality type).
* Outputs candidate plausibility scores.

---

### 3. **ADE Fingerprint Model** (`ade_fingerprint.py`)

* Specialized classifier for **Anthropogenic Dark Earths** (ADEs).
* Features:

  * Seasonal NDVI/EVI persistence.
  * Floristic community indicators.
  * Fractal geometry from micro-topography.
* Outputs ADE likelihood + feature importance.

---

### 4. **Uncertainty Estimator** (`uncertainty_bgnn.py`)

* Implements a **Bayesian Graph Neural Network (B-GNN)**.
* Provides:

  * Probabilities (site vs. non-site).
  * Uncertainty histograms & confidence intervals.
* Enables calibrated, interpretable outputs.

---

### 5. **Causal Inference Model** (`causal_pag.py`)

* Wrapper for **Fast Causal Inference (FCI)** algorithm.
* Builds **Partial Ancestral Graph (PAG)** from site evidence.
* Validates causal plausibility (e.g., “elevation anomaly → soil → vegetation”).
* Outputs `.gml` graphs + causal narrative.

---

### 6. **Baselines & Fallbacks** (`baselines.py`)

* Lightweight, CPU-friendly alternatives:

  * Random Forest (environmental layers).
  * PCA/GLCM anomaly scores.
  * Rule-based ADE heuristics.
* Ensures the pipeline runs even without GPU/torch dependencies.

---

## ⚙️ Integration with Pipeline

* **`detect.py`** → calls **`anomaly_detector.py`**.
* **`evaluate.py`** → augments candidates with **ADE fingerprint models**.
* **`verify.py`** → fuses evidence via **GNN + B-GNN + PAG causal models**.
* **`report.py`** → retrieves model outputs for site dossiers.

All models load parameters via **YAML configs** (see `/configs/`), ensuring reproducibility and tunability.

---

## 🧪 Training & Reproducibility

* Training scripts (if needed) live in `notebooks/` or `scripts/`.
* Pretrained weights are **not stored in Git** (download via Kaggle Datasets or APIs).
* Models log their config + hash during runs, stored in `/outputs/` for audit trails.

---

📖 **References**

* WDE Architecture Specification
* Repository Structure Guide
* ADE Discovery Pipeline (Kaggle Scaffold)
* Enriching WDE for Archaeology & Earth Systems

---
