# 🧠 `models/` — Machine Learning, Causal Inference & Simulation Models

This directory contains **all machine learning, statistical, and simulation models** that power the **World Discovery Engine (WDE)**. These models are used to detect anomalies, compute ADE fingerprints, fuse multi-modal evidence, quantify uncertainty, and validate causal plausibility.

---

## 📌 Purpose

* Provide **modular model definitions** for each stage of the WDE pipeline.
* Support **classical ML, deep learning, and causal/Bayesian approaches**.
* Ensure **reproducibility** (config-driven, no hard-coded paths).
* Remain **Kaggle-ready**: models either run on Kaggle CPUs/GPUs or gracefully degrade to heuristics if unavailable.

---

## 📂 Directory Layout

```bash
models/
├── anomaly_detector.py     # Unified anomaly detection (Z-score, IsoForest, LOF, Autoencoder-CNN)
├── ade_fingerprint.py      # ADE fingerprint scorer (soil, NDVI, SAR ratios, fractals)
├── gnn_fusion.py           # Graph Neural Network for multi-modal evidence fusion
├── uncertainty_bgnn.py     # Bayesian Graph Neural Network (Pyro + PyTorch)
├── causal_pag.py           # Constraint-based causal discovery → PAG graphs
├── baselines.py            # scikit-learn baselines (RF, GB, Ridge, ElasticNet, SVM, etc.)
└── __init__.py             # Unified import surface, registry, and factories
```

---

## 🧩 Model Roles

### 1. **Anomaly Detector** (`anomaly_detector.py`)

* Detects **coarse anomalies** in satellite, radar, and LiDAR data.
* Backends:

  * **Z-Score** — statistical baseline.
  * **Isolation Forest** — ensemble-based anomaly isolation.
  * **Local Outlier Factor (LOF)** — density-based outlier detection.
  * **Autoencoder CNN** — unsupervised reconstruction errors on image patches.
* Supports **sliding-window raster scoring** for GeoTIFFs.
* Outputs **per-sample anomaly scores** (higher = more anomalous).

---

### 2. **ADE Fingerprint Scorer** (`ade_fingerprint.py`)

* Specialized for **Anthropogenic Dark Earths (ADEs)**.
* Features engineered from:

  * **Sentinel-2**: NDVI, EVI, NDWI, NBR, SAVI.
  * **SAR**: VV/VH ratios & differences.
  * **LiDAR/DEM**: slope, TPI, canopy height.
  * **Soils**: SOC, P, Ca, K, N, pH bell-curve transform.
  * **Climate**: precipitation, temperature.
* Modes:

  * **Heuristic scoring** (physics/soil-guided weights + sigmoid).
  * **Logistic regression upgrade** (if scikit-learn available).
* Outputs **ADE probability (0–1)** + **per-feature contributions**.

---

### 3. **Evidence Fusion GNN** (`gnn_fusion.py`)

* Fuses multi-modal features into a **graph of evidence**.
* Nodes: anomaly features (optical, radar, LiDAR, soils, climate).
* Edges: semantic relations (distance, modality type).
* Implements **Graph Neural Network** with support for:

  * Edge features (distance, correlation).
  * Multi-hop reasoning across modalities.
* Outputs **plausibility score per candidate site**.

---

### 4. **Uncertainty Estimator (Bayesian GNN)** (`uncertainty_bgnn.py`)

* Implements a **Bayesian Graph Neural Network**.
* Provides:

  * **Site probability** (classification or regression).
  * **Uncertainty histograms** (confidence intervals).
* Enables **calibrated outputs**, critical for scientific credibility.
* Based on **Pyro + PyTorch**, GPU-enabled but CPU-fallback available.

---

### 5. **Causal Inference Model** (`causal_pag.py`)

* Wrapper for **constraint-based causal discovery** (PC/FCI-style).
* Outputs **Partial Ancestral Graphs (PAGs)** from tabular features.
* Ensures **causal plausibility** (e.g., *elevation → soil nutrients → vegetation*).
* Provides `.gml` graphs and causal narratives.

---

### 6. **Baselines & Fallbacks** (`baselines.py`)

* Lightweight, CPU-friendly scikit-learn models:

  * Logistic Regression, Ridge, Lasso, ElasticNet.
  * Random Forest, Gradient Boosting.
  * SVM, KNN.
* Support for:

  * **Calibration** (Platt scaling, isotonic regression).
  * **Feature importances** (native + permutation).
  * **Classification & regression metrics** (GLL, RMSE, F1, ROC-AUC).
* Ensures pipeline runs **without GPU dependencies**.

---

### 7. **Unified Registry** (`__init__.py`)

* Provides a clean import surface:

  ```python
  from models import AnomalyDetector, ADEFingerprint, BaselineModel, CausalPAG
  ```
* Includes **ModelRegistry** with string factories:

  ```python
  from models import create
  det = create("anomaly", method="isoforest")
  clf = create("baseline", problem_type="classification", model="rf")
  ```
* Enables quick prototyping, reproducibility, and CLI integration.

---

## ⚙️ Integration with Pipeline

* **`ingest.py`** → calls **Baselines + AnomalyDetector** for coarse scan.
* **`evaluate.py`** → enriches with **ADE Fingerprint features**.
* **`verify.py`** → runs **Fusion GNN + Bayesian GNN + PAG causal checks**.
* **`report.py`** → assembles **site dossiers** with all model outputs.

---

## 🧪 Training & Reproducibility

* Config-driven via `/configs/` (YAML/JSON, Hydra-compatible).
* All models save/load via **joblib** or **torch.save**.
* Random seeds set for **deterministic runs**.
* Pretrained weights (if any) must be **open-access** and Kaggle-compatible.
* Outputs log config + hash for **audit trails**.

---

📖 **References**

* **WDE Architecture Specification**
* **Enriching WDE for Archaeology & Earth Systems**
* **ADE Discovery Pipeline (Kaggle Notebook)**
* **Patterns, Algorithms & Fractals Reference**
* **Physics & Simulation Reference**
* **CLI Technical Reference (MCP)**

---
