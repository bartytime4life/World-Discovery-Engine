

**World Discovery Engine (WDE) · `world_engine/models/`**

> Mission: robust, explainable, and reproducible models that fuse multi-sensor Earth data to surface archaeologically significant candidates (e.g., ADE soils, earthworks, geoglyphs) with calibrated uncertainty and causal plausibility.

---

## 0) Scope & Design Principles

**Scope.** This document defines the architecture of the **Models Layer** housed under `world_engine/models/`. It covers model taxonomy, data/feature contracts, fusion graphs, uncertainty modeling, causal reasoning, evaluation, registry/wiring, and CI/testing hooks.

**Principles.**

* **Multi-sensor first.** Optical, radar, elevation, soils, hydro, and archives are fused via explicit cross-sensor graphs.
* **Explainable by construction.** SHAP, attention, prototypes, rule masks, and causal graphs are first-class outputs.
* **Uncertainty everywhere.** Predictive distributions and coverage checks ship with every model head.
* **Reproducible & Kaggle-ready.** CLI + Hydra configs, single-notebook fallback, and 9-hour guardrails.
* **Modular & swappable.** Early/mid/late fusion pluggable; baselines live next to SOTA for sanity checks.
* **Evidence-to-dossier.** Models output not just scores but the *why* (evidence overlays + causal plausibility).

---

## 1) Repository Map (Models Layer)

```
world_engine/
└── models/
    ├── __init__.py                 # Registry & public API surface (factory/resolve/import side effects)
    ├── baselines.py                # Simple, reliable models: LR/MLP, RandomForest, LightGBM, U-Net/DeepLab, ResNet
    ├── gnn_fusion.py               # Heterogeneous multi-sensor graph fusion (GNN with edge features)
    ├── ade_fingerprint.py          # ADE/soil-vegetation fingerprint models (1D-CNN/TabNet/MLP + indices)
    ├── uncertainty_bgnn.py         # Bayesian / evidential GNN heads + ensembling + conformal wrappers
    ├── causal_pag.py               # Causal discovery & plausibility scoring via PAG/FCI-style structures
    ├── anomaly_detector.py         # Un/semisupervised anomaly discovery (AE/Flow/IsolationForest/OneClassSVM)
    └── README.md                   # Local quickstart & dev notes (separate from this ARCHITECTURE.md)
```

> Future slots (optional): `temporal_transformer.py` (multi-season sequences), `prototype_retriever.py` (case-based reasoning), `segmentation_heads.py` (U-Net++ / SegFormer).

---

## 2) Model Taxonomy & Responsibilities

### 2.1 Baselines (`baselines.py`)

* **Why:** Establish floor/ceil, detect data drifts, and provide fast ablations.
* **What:**

  * **Tabular:** Logistic Regression, RandomForest, LightGBM/XGBoost over engineered features.
  * **CV:** ResNet-family image classifier; U-Net/DeepLab for pixel/patch segmentation.
  * **Time series (optional):** 1D-CNN over spectral/temporal stacks for quick wins.
* **Outputs:** Score(s), optional pixel maps, feature importances/SHAP.
* **Use:** Sanity, smoke tests, CI comparisons, weak supervision.

### 2.2 ADE Fingerprint (`ade_fingerprint.py`)

* **Why:** Encode known soil/vegetation/hydrology signatures of ADEs and related phenomena.
* **What:** Hybrid feature pipeline → **1D-CNN / TabNet / MLP**. Inputs include indices (NDVI, NDWI, NBR, EVI), SWIR ratios, texture, canopy metrics, topographic derivatives, drought/seasonality features, soil class probabilities, historical/anthro proximity features.
* **Outputs:** ADE likelihood, per-feature attributions, optional class activation maps for image inputs.
* **Use:** Candidate ranking; interpretable priors for fusion graph; weak labels for anomaly training.

### 2.3 GNN Fusion (`gnn_fusion.py`)

* **Why:** Explicitly model **relationships** across sensors, times, and layers (optical↔radar↔terrain↔soil).
* **Graph:**

  * **Nodes:** `{tile, band, season, sensor-derived superpixels, auxiliary sources (soil/hydro/historic)}`
  * **Edges:** `{co-location, temporal adjacency, cross-sensor alignment, river proximity, slope/aspect similarity}`
  * **Edge features:** distance, time lag, band compatibility, landcover similarity, cloud/SAR-speckle quality.
* **Backbones:** GAT / GraphSAGE / R-GCN / NNConv (selectable), residual + edge-aware message passing.
* **Fusion stage:** **Mid-fusion** by default; supports early/late ablations for research.
* **Outputs:** Node/graph-level scores; attention weights; edge importance; embeddings for retrieval/explanations.

### 2.4 Uncertainty BGNN (`uncertainty_bgnn.py`)

* **Why:** Decision-grade **calibrated** uncertainty for candidate triage and follow-up planning.
* **Heads:**

  * **Bayesian:** MC-Dropout, ensembles.
  * **Evidential:** Dirichlet/Gaussian evidential regression/classification.
  * **Conformal:** Split/inductive conformal prediction; per-region coverage tracking.
* **Outputs:** Predictive intervals/sets, coverage reports (global/per-AOI/per-class), reliability diagrams.
* **Integration:** Wraps any base model (baseline, ADE, GNN) → unified `predict_with_uncertainty()` API.

### 2.5 Causal PAG (`causal_pag.py`)

* **Why:** Separate correlation from plausible cause using **Partial Ancestral Graphs (PAG)** / FCI-style discovery.
* **Vars:** Evidence nodes (indices, soil/terrain, hydro proximity, seasonal signals, radar backscatter, human activity proxies).
* **Functions:** constraint-based discovery, bootstrapped stability selection, causal plausibility scoring, counterfactual stress tests (e.g., *remove river proximity*; *alter slope class*).
* **Outputs:** Causal graph artifacts, per-candidate plausibility score, rationale (v-structures, (in)dependencies).

### 2.6 Anomaly Detector (`anomaly_detector.py`)

* **Why:** Discover novel/rare patterns—potential sites or preservation/erosion anomalies.
* **Methods:**

  * **Reconstruction:** Autoencoder/Variational AE/Masked-AE on multi-sensor stacks.
  * **Density:** Normalizing Flows.
  * **Classical:** Isolation Forest, One-Class SVM, LOF.
  * **Change:** Seasonal/temporal change-point detectors (if sequences provided).
* **Outputs:** Anomaly score maps, residual stacks, top-k tiles with evidence previews.

---

## 3) Data & Feature Contracts

### 3.1 Common Sample Abstraction

All models consume a **standardized sample dict** (Torch-style or NumPy):

```python
sample = {
  "tile_id": str,                   # unique key (AOI/grid/timestamp)
  "coords": {"lon": float, "lat": float},  # centroid or polygon ref
  "raster": {                       # per-sensor tensors (C×H×W)
    "s2": Tensor[C_opt, H, W],      # Sentinel-2 L2A (selected bands or indices)
    "s1": Tensor[C_sar, H, W],      # Sentinel-1 GRD (VV/VH, filters applied)
    "dem": Tensor[1..K, H, W],      # elevation + derivatives (slope/aspect/TPI/TRI/flowacc)
    "lidar": Optional[Tensor[...,]],# GEDI/point-derived rasters if available
  },
  "tabular": {                      # non-image features
    "soil": np.ndarray[D_soil],     # soil class probs/props
    "hydro": np.ndarray[D_hydro],   # distance to rivers/streams/floodplain class
    "climate": np.ndarray[D_clim],  # seasonality/drought indices
    "history": np.ndarray[D_hist],  # archives/proximity/anthropogenic priors
  },
  "mask": Optional[Tensor[H, W]],   # valid pixels mask (cloud/shadow/water)
  "label": Optional[dict],          # {"class": int/float, "seg": Tensor[H,W], ...}
  "meta": dict                      # timestamps, AOI id, quality flags, provenance, etc.
}
```

**Shapes.** Defaults assume **`H=W=256`** (configurable), meters/pixel and band sets are set by the datasource config.

### 3.2 Normalization & Quality

* **Optical:** per-band scaling to reflectance or z-scores; cloud/shadow masks applied.
* **SAR:** log-scaling, speckle filtering (Lee/Gamma-MAP), incidence angle normalization (if provided).
* **DEM/LiDAR:** min-max or standardization; derived features pre-computed at datasource stage.
* **Missingness:** explicit NaNs or masks; models must branch appropriately.

---

## 4) Fusion Graph Schema (for `gnn_fusion.py`)

**Nodes.**

* `Tile`: unit of inference; attaches AOI/time metadata.
* `SensorPatch`: derived per-sensor embeddings (optical/sar/dem/lidar).
* `AuxNode`: soil/hydro/historic *non-raster* features.

**Edges (typed).**

* `COLLOCATED(sensor_a, sensor_b)`: same tile/time window.
* `TEMPORAL_ADJ(t, t±Δ)`: sequence neighbors.
* `AUX_LINK(sensor, aux)`: attach auxiliary factors.
* `TERRAIN_SIM(sim)`: slope/aspect/roughness similarity between neighbor tiles.
* `HYDRO_PROX(d)`: distance-weighted proximity to rivers/streams.

**Edge attributes.** `{distance, dt_days, angle_diff, band_pair_code, cloud_score, speckle_score, landcover_sim}`.

**Backbones.** Choose via Hydra:

* `gat`: multi-head attention; export attention maps.
* `rgcn`: relation-type aware propagation (edge types).
* `nnconv`: edge-network that learns filters from edge attributes.
* `graphsage`: inductive baseline.

---

## 5) Uncertainty & Calibration (`uncertainty_bgnn.py`)

**Predictive Uncertainty.**

* **Aleatoric:** learned variance heads for regression; Dirichlet/Evidential for classification.
* **Epistemic:** MC-Dropout; deep ensembles.

**Calibration.**

* **Reliability:** ECE/MCE, reliability diagrams.
* **Conformal Sets/Intervals:** per-AOI coverage, conditional coverage where feasible.
* **Outputs:** unified `UncertaintyReport` JSON (coverage, interval sizes, failure regions).

---

## 6) Causal Plausibility (`causal_pag.py`)

1. **Graph discovery:** constraint-based (FCI/PAG) with independence tests over engineered features & embeddings.
2. **Bootstrap stability:** repeat discovery across resamples → edge frequency/stability.
3. **Plausibility score:** combine (i) presence of domain-expected edges (e.g., floodplain→vegetation), (ii) absence of known artifacts (e.g., cloud→class), and (iii) counterfactual stress outcomes.
4. **Export:** `.graphml`/`.json` plus a human-readable rationale list attached to each candidate.

---

## 7) Anomaly Modeling (`anomaly_detector.py`)

* **Reconstruction pathway:** Masked-AE on multi-sensor stacks; residual maps as anomaly scores.
* **Density pathway:** Normalizing Flows (tile embedding space).
* **Classical:** Isolation Forest / One-Class SVM over tabular+embedding concatenations.
* **Ranking:** fused anomaly score with uncertainty & causal plausibility to avoid over-flagging artifacts.

---

## 8) Public API & Registry (`__init__.py`)

**Factory.** All models are discoverable via string keys:

```python
Model = Protocol(
    fit=Callable[..., "Model"],
    predict=Callable[..., np.ndarray],
    predict_proba=Callable[..., np.ndarray] | None,
    predict_with_uncertainty=Callable[..., dict] | None,
    explain=Callable[..., dict] | None,
)

def make_model(name: str, **kwargs) -> Model:
    """
    name ∈ {
      "baseline/tabular", "baseline/resnet", "baseline/unet",
      "ade/fingerprint",
      "fusion/gnn",
      "uncertainty/bgnn",
      "causal/pag",
      "anomaly/autoencoder", "anomaly/flow", "anomaly/iforest"
    }
    """
```

**Conventions.**

* `fit(X, y, **cfg)` → returns self (or is no-op for pretrained).
* `predict(samples)` → scores or maps.
* `predict_with_uncertainty(samples)` → dict with `{"mean", "pi_low", "pi_high", "coverage", ...}`.
* `explain(samples)` → SHAP/attention/feature importances, graph edges, prototypes.

---

## 9) Configuration (Hydra) & CLI

**Hydra groups.**

```
conf/
  model/
    baseline.yaml
    ade_fingerprint.yaml
    fusion_gnn.yaml
    uncertainty_bgnn.yaml
    causal_pag.yaml
    anomaly.yaml
```

**Example:** `fusion_gnn.yaml`

```yaml
model:
  name: fusion/gnn
  backbone: gat            # [gat, rgcn, nnconv, graphsage]
  hidden_dim: 128
  layers: 3
  heads: 4
  dropout: 0.15
  edge_features: [distance, dt_days, angle_diff, band_pair_code, cloud_score, speckle_score, landcover_sim]
  readout: mean            # [mean, attention, set2set]
  loss: bce                # or focal, dice (if segmentation)
  class_weights: null
  optimizer: adamw
  lr: 2.5e-4
  weight_decay: 0.01
  epochs: 30
  early_stopping_patience: 5
  uncertainty:
    enable: true
    method: "evidential"   # or "mc_dropout", "ensemble"
    conformal:
      enable: true
      alpha: 0.1
```

**CLI usage (example).**

```
# Train a fusion GNN with uncertainty + causal plausibility export
wde train model=fusion_gnn data=amazon_west aoi=amazonia_west
wde predict model=fusion_gnn ckpt=... outputs=... +model.uncertainty.enable=true
wde explain  model=fusion_gnn ckpt=... outputs=... +explain.save_graph=true
```

---

## 10) Training & Evaluation

**Tasks.** Binary site classification, patch segmentation, anomaly ranking, retrieval (prototype nearest neighbors), multi-task (site class + uncertainty).

**Metrics.**

* **Classification:** AUROC, AUPRC, F1\@k, PR\@k, Top-K discovery rate, Lift curves.
* **Segmentation:** IoU/Dice/Recall\@IoU thresholds; boundary-aware metrics (Hausdorff distance).
* **Anomaly:** Average Precision, PRO score, PAUPRC (pixel AP), R\@K under review budgets.
* **Uncertainty:** ECE/MCE, NLL/Brier, calibration curves, coverage vs. target α (conformal).
* **Spatial:** Spatial precision/recall against reference sites; cluster hit-rate within search radius.
* **Ops:** Throughput (tiles/min), GPU memory, wall time vs. Kaggle limit (9h budget awareness).

**Validation strategies.**

* **AOI-wise splits** to avoid leakage (train on some basins, validate on others).
* **Time-wise splits** when sequences exist.
* **Bootstrapped confidence** on critical metrics.

---

## 11) Explainability & Reports

* **Feature attributions** (SHAP/Integrated Gradients) for tabular & CNN models.
* **Attention maps** for GAT/R-GCN, edge-type importances and path saliency.
* **Prototype retrieval** (optional): highlight nearest historical exemplars.
* **Causal rationales**: top v-structures/edges contributing to plausibility.
* **Unified diagnostics JSON/HTML:** each `predict` run emits explainability artifacts suitable for the dashboard/notebook.

---

## 12) MLOps: Checkpoints, Reproducibility, CI

* **Checkpoints:** standardized `ckpt/` layout with `model.pt`, `config.yaml`, `metrics.json`, `schema.json`.
* **Hashes:** data & config hashes recorded into run manifests.
* **Determinism:** seeds for NumPy/PyTorch; controlled non-determinism flags with opt-outs documented.
* **CI:** smoke-train on tiny AOI, run inference/explain/uncertainty, assert metrics > floor, time < budget.
* **Artifacts:** `submission.csv`/`candidates.geojson`, `uncertainty_report.json`, `causal_graph.json`, `explain/` plots.

---

## 13) Integration with the Notebook (Kaggle / GitHub)

* **One-cell import:** `from world_engine.models import make_model`
* **In-notebook config:** small Hydra override dictionary pattern for reproducible toggles.
* **Output widgets:** display top-K candidates with uncertainty bands, attention overlays, and causal notes.
* **Fail-safes:** if GPU absent, automatic baseline fallback; if sensor missing, degrade gracefully via masks.

---

## 14) Development Guide

**Add a new model.**

1. Create `world_engine/models/<new_model>.py`.
2. Implement `fit/predict/(predict_with_uncertainty)/explain`.
3. Register in `__init__.py` factory with a unique `name`.
4. Add Hydra config under `conf/model/<new_model>.yaml`.
5. Write tests in `tests/models/test_<new_model>.py` (fit small dummy, predict, explain, uncertainty).
6. Update docs (`README.md`, this file if architectural changes).
7. Run CI (lint, unit tests, tiny AOI smoke).

**Performance tips.**

* Cache sensor embeddings; prefer small graph batches; prune low-quality edges; mixed precision on CNN stages; use neighbor sampling (GraphSAGE) for large graphs.

---

## 15) Risks & Mitigations

* **Cloud/shadow leakage** → strict masks & QA flags; SAR/DEM triangulation.
* **Speckle/terrain confounds** → SAR preprocessing, terrain-aware controls.
* **Spatial leakage** → AOI-wise splits; buffer zones.
* **Overconfidence** → evidential heads + conformal; calibrate per-AOI.
* **Spurious correlations** → causal PAG checks; counterfactual stress tests baked into scoring.

---

## 16) Roadmap (Short-Term)

* **Temporal encoder:** (Seasonal transformer) add `temporal_transformer.py` and time-aware edges.
* **Prototype memory:** integrate case-based retrieval + human feedback loop.
* **SegFormer head:** for higher-fidelity site boundaries where labels exist.
* **Active learning hooks:** surface highest-value tiles with high uncertainty × high plausibility.

---

## 17) Glossary

* **ADE:** Anthropogenic Dark Earths; fertile Amazonian soils linked to past human activity.
* **PAG:** Partial Ancestral Graph; encodes causal relations with possible latent confounding.
* **ECE:** Expected Calibration Error; lower is better.
* **Conformal prediction:** framework providing finite-sample predictive sets/intervals with coverage guarantees.

---

## 18) Quickstart Snippets

**Make & train a fusion model with uncertainty:**

```python
model = make_model("fusion/gnn", backbone="gat", hidden_dim=128, layers=3, uncertainty={"enable": True, "method": "evidential"})
model.fit(train_ds, valid_ds, epochs=30)
rep = model.predict_with_uncertainty(test_ds)  # dict: scores + intervals + coverage
```

**Rank anomalies then filter by causal plausibility:**

```python
anom = make_model("anomaly/autoencoder", patch_size=256, masked=True)
anom.fit(unlabeled_ds)
anom_scores = anom.predict(test_ds)

cpl = make_model("causal/pag", vars_config="conf/causal/vars.yaml")
cpl_graphs = cpl.fit_discover(train_ds)
plaus = cpl.score(test_ds)

final = fuse_rank(anom_scores, plaus, weights={"anom": 0.6, "causal": 0.4})
```

---

## 19) Maintainers & Conventions

* **Code style:** ruff/black/isort/mypy; docstrings with parameter/returns; type hints everywhere.
* **Docs:** this file reflects the *intended* architecture; keep in sync with implementations.
* **Commit messages:** include model + change type (feat/fix/perf/docs/refactor) and metric deltas if relevant.

---
