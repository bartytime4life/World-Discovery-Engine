# 🧪 Tests for Notebooks — World Discovery Engine (WDE)

This directory contains **automated tests for Jupyter notebooks** in the WDE repository.  
The goal is to ensure that all notebooks (especially **`ade_discovery_pipeline.ipynb`**) run **end-to-end, reproducibly, and deterministically** — matching the architecture spec and Kaggle challenge requirements:contentReference[oaicite:2]{index=2}.

---

## ✅ Purpose
- Guarantee notebooks run **without manual edits** on Kaggle or local dev.
- Validate that key **artifacts** are produced (JSON, GeoJSON, reports).
- Check for **deterministic outputs** (random seeds, fixed configs).
- Confirm integration with the **`world_engine/` pipeline modules** (no hidden logic inside notebooks).

---

## 📂 Scope of Testing
Tests here target:
- **Kaggle pipeline notebook**:
  - `notebooks/ade_discovery_pipeline.ipynb`  
  Runs the full Discovery Funnel: ingestion → detection → evaluation → verification → dossier generation:contentReference[oaicite:3]{index=3}.
- **Demo starter notebook**:
  - `notebooks/WDE_Kaggle_Starter.ipynb`  
  Ensures the quickstart demo executes on Kaggle with minimal dependencies:contentReference[oaicite:4]{index=4}.
- **Experimental notebooks** (optional):
  - Any under `notebooks/experiments/` are tested for syntax and imports but may skip heavy compute.

---

## ⚙️ Test Strategy

### 1. Execution Tests
We use **`nbmake`** (pytest plugin) to run notebooks cell-by-cell:
```bash
pytest --nbmake notebooks/
````

This ensures:

* No runtime errors.
* All cells complete within Kaggle’s 9h limit (most within minutes).
* Outputs are generated in `outputs/` as per pipeline contract.

### 2. Artifact Validation

After execution, tests confirm presence of required outputs:

* `outputs/candidates.json`
* `outputs/candidates.geojson`
* `outputs/reports/` (candidate dossiers)
* `outputs/pag/*.gml` (causal graphs)
* `outputs/uncertainty/`
* `outputs/ssim/`
* `outputs/ndvi_timeseries/`

### 3. Reproducibility Checks

* Random seeds fixed (`numpy`, `torch`, `random`).
* Environment logged at runtime (Python, library versions).
* Test ensures that repeated runs yield identical hash of key outputs (e.g., `candidates.json`).

---

## 🚀 Running Locally

Install test deps:

```bash
pip install pytest nbmake papermill
```

Run tests:

```bash
pytest tests/notebooks/ --maxfail=1 --disable-warnings -q
```

---

## 📝 Notes

* Notebooks are **exploration front-ends** only. All heavy logic must live in `/world_engine` and be tested separately.
* If you add a new notebook, also add a **smoke test** here (basic execution, artifact validation).
* CI/CD: GitHub Actions workflows (`.github/workflows/kaggle_notebook_ci.yml`) run these tests on every main-branch push.

---

*Last updated: 2025-08-27*

```
