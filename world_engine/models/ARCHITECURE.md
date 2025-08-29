# 🧠 World Discovery Engine (WDE) · `world_engine/models/`

> **Mission:** Robust, explainable, and reproducible models that fuse multi-sensor Earth data to surface archaeologically significant candidates (e.g., ADE soils, earthworks, geoglyphs) with calibrated uncertainty and causal plausibility.

---

## 0) Scope & Design Principles

**Scope.**
Defines the architecture of the **Models Layer** (`world_engine/models/`). Covers taxonomy, data/feature contracts, fusion graphs, uncertainty modeling, causal reasoning, evaluation, registry, and CI/testing hooks.

**Principles.**

* **Multi-sensor first.** Optical, radar, DEM/LiDAR, soils, hydro, climate, archives.
* **Explainable by design.** SHAP, attention, prototypes, causal graphs as first-class outputs.
* **Uncertainty everywhere.** Predictive distributions, calibration, and coverage checks.
* **Reproducible & Kaggle-ready.** Hydra configs, CLI-first, 9-hour Kaggle runtime guardrails.
* **Modular & swappable.** Baselines live next to GNNs/transformers for sanity checks.
* **Evidence-to-dossier.** Models output not just scores but also rationale (overlays + causal plausibility).

---

## 1) Repository Map (Models Layer)

```bash
world_engine/
└── models/
    ├── __init__.py          # Registry & public API surface (factory/imports)
    ├── baselines.py         # sklearn baselines (LR, RF, GB, SVM, KNN)
    ├── ade_fingerprint.py   # ADE soil–vegetation fingerprint scorer
    ├── gnn_fusion.py        # Multi-sensor graph fusion GNN
    ├── uncertainty_bgnn.py  # Bayesian/evidential GNN + conformal wrappers
    ├── causal_pag.py        # Causal discovery (PAG/FCI-style)
    ├── anomaly_detector.py  # Anomaly discovery (Z-score, IsoForest, LOF, AE-CNN)
    └── README.md            # Quickstart + developer notes
```

*Future slots:* `temporal_transformer.py`, `prototype_retriever.py`, `segmentation_heads.py`.

---

## 2) Model Taxonomy

* **Baselines (`baselines.py`):** Logistic Regression, Random Forest, Gradient Boosting, Ridge/ElasticNet. Smoke tests, CI comparisons.
* **ADE Fingerprint (`ade_fingerprint.py`):** Hybrid soil/NDVI/SAR features; heuristic + logistic regression.
* **GNN Fusion (`gnn_fusion.py`):** Multi-modal graph with nodes = features, edges = relations (distance, modality).
* **Uncertainty BGNN (`uncertainty_bgnn.py`):** Bayesian (ensembles, MC-dropout), Evidential (Dirichlet), Conformal sets.
* **Causal PAG (`causal_pag.py`):** Constraint-based discovery → Partial Ancestral Graphs; outputs plausibility scores.
* **Anomaly Detector (`anomaly_detector.py`):** Z-score, Isolation Forest, LOF, Autoencoder CNN.

---

## 3) Data & Feature Contracts

All models consume a **standardized `sample` dict**:

```python
sample = {
  "tile_id": str,
  "coords": {"lat": float, "lon": float},
  "raster": {
    "s2": np.ndarray,    # Sentinel-2 bands
    "s1": np.ndarray,    # Sentinel-1 SAR VV/VH
    "dem": np.ndarray,   # DEM/LiDAR
  },
  "tabular": {
    "soil": np.ndarray,
    "climate": np.ndarray,
    "history": np.ndarray,
  },
  "mask": Optional[np.ndarray],
  "label": Optional[dict],
  "meta": dict,
}
```

**Defaults:** `256×256` patches, multi-band aligned, masks for clouds/water.

---

## 4) Fusion Graph Schema (`gnn_fusion.py`)

* **Nodes:** `{tile, sensor_patch, aux (soil/hydro/history)}`.
* **Edges:** co-location, temporal adjacency, cross-sensor alignment, hydro proximity.
* **Edge attributes:** distance, dt\_days, incidence angle, band-pair, cloud/speckle quality.
* **Backbones:** GAT, R-GCN, NNConv, GraphSAGE.

---

## 5) Uncertainty & Calibration (`uncertainty_bgnn.py`)

* **Aleatoric:** learned variance heads.
* **Epistemic:** dropout, ensembles.
* **Evidential:** Dirichlet, Gaussian evidential networks.
* **Conformal prediction:** per-AOI coverage sets.
* **Outputs:** unified `UncertaintyReport` JSON.

---

## 6) Causal Plausibility (`causal_pag.py`)

* Graph discovery with constraint-based tests.
* Bootstrap stability across subsamples.
* Counterfactual stress tests (e.g., remove hydro proximity).
* Outputs `.gml/.graphml` + plausibility scores.

---

## 7) Anomaly Modeling (`anomaly_detector.py`)

* **Reconstruction:** Autoencoder, Variational AE.
* **Density:** Normalizing flows.
* **Classical:** Isolation Forest, One-Class SVM, LOF.
* Outputs fused anomaly scores, residual maps, top-k ranked tiles.

---

## 8) Public API & Registry (`__init__.py`)

All models discoverable via `create()` factory:

```python
from world_engine.models import create

det = create("anomaly", method="isoforest")
ade = create("ade_fingerprint")
gnn = create("fusion_gnn", backbone="gat", layers=3)
```

Registry entries carry **requires** + **summary** for clarity.

---

## 9) Config & CLI

* **Hydra configs** (`/configs/model/*.yaml`) manage hyperparams, backbones, uncertainty toggles.
* **CLI integration:**

  ```bash
  wde train model=fusion_gnn data=amazon
  wde predict model=ade_fingerprint ckpt=... outputs=...
  wde explain model=causal_pag outputs=graph/
  ```

---

## 10) Evaluation & Metrics

* **Classification:** AUROC, AUPRC, F1\@k.
* **Segmentation:** IoU, Dice.
* **Anomaly:** Average Precision, PRO score.
* **Uncertainty:** ECE, Brier, coverage.
* **Ops:** runtime throughput vs Kaggle 9h budget.

---

## 11) Explainability

* **Feature attributions:** SHAP, Integrated Gradients.
* **Attention maps:** GNN attention weights, edge importances.
* **Causal rationales:** top v-structures.
* **Prototype retrieval:** nearest neighbor exemplars (future).
* All runs emit **diagnostics JSON/HTML** for dashboards.

---

## 12) MLOps & Reproducibility

* **Checkpoints:** `ckpt/model.pt`, `metrics.json`, `config.yaml`.
* **Hashes:** logged for config/data.
* **Seeds:** fixed for numpy/torch.
* **CI/CD:** smoke-train tiny AOI, assert metrics > floor, runtime < 9h.

---

## 13) Integration with Notebooks

* Import: `from world_engine.models import make_model`.
* Hydra overrides inline in Kaggle cells.
* Outputs: candidate dossiers with overlays, uncertainty bands, causal notes.
* Fallback: degrade gracefully if GPU/sensor missing.

---

## 14) Roadmap

* Temporal transformer encoder for seasonal NDVI sequences.
* Prototype retriever for case-based reasoning.
* SegFormer/U-Net++ heads for segmentation.
* Active learning hooks (uncertainty × plausibility).

---

## 15) Quickstart Snippets

```python
# ADE fingerprint scorer
from world_engine.models import create
ade = create("ade_fingerprint")
X, names = ade.engineer_features(sensors)
scores = ade.score(X)

# Anomaly detection with IsoForest
det = create("anomaly", method="isoforest", n_estimators=200)
det.fit(X_train)
anom_scores = det.score_samples(X_test)

# GNN fusion with Bayesian uncertainty
gnn = create("fusion_gnn", backbone="gat", hidden_dim=128, layers=3,
             uncertainty={"enable": True, "method": "evidential"})
gnn.fit(train_ds, val_ds)
rep = gnn.predict_with_uncertainty(test_ds)
```
