# 🧪 Testing Guide — World Discovery Engine (WDE)

This folder contains the **unit, integration, and CI tests** for the World Discovery Engine (WDE).
The goal is to ensure the pipeline is **reproducible, robust, and scientifically credible**.

---

## 📌 Testing Strategy

WDE uses a **layered testing approach**:

1. **Unit Tests**

   * Validate individual functions in `world_engine/` modules.
   * Examples:

     * `ingest`: tiling logic, CRS reprojection.
     * `detect`: anomaly scoring, CV filters.
     * `evaluate`: NDVI/EVI calculation, LiDAR fallback.
     * `verify`: ADE fingerprint rules, causal graph builder.
     * `report`: dossier template generation.

2. **Integration Tests**

   * Run the **full pipeline** on a **dummy AOI** with **mock datasets**.
   * Ensures all modules work together and produce outputs:

     * `candidates.json`, `candidates.geojson`
     * ADE fingerprint CSV/JSON
     * PAG graphs (.gml)
     * Dossiers (PDF/HTML/Markdown)

3. **Continuous Integration (CI)**

   * GitHub Actions runs tests automatically:

     * **Lint & static analysis** (Ruff, Black, isort, mypy).
     * **Unit tests** on every PR.
     * **Kaggle notebook execution**:

       * PRs → light profile (smoke test, minimal tiles).
       * Main branch → heavy profile (ADE, PAG, uncertainty, SSIM).

---

## 🎯 Dummy AOI & Mock Data

For fast, reproducible tests:

* **AOI**: A 0.05° box in the Amazon (`data/aoi/test_bbox.geojson`).
* **Mock imagery**: Synthetic Sentinel-2 tiles with injected “bright anomalies”.
* **Mock DEM**: Small raster (50×50) with artificial bumps for geomorphology tests.
* **Mock soil**: Tiny GeoTIFF with elevated phosphorus pixel.
* **Mock NDVI/EVI**: CSV time-series simulating ADE fingerprints (dry-season spike).
* **Mock PAG**: A 3-node causal graph (elevation → soil → NDVI).

These ensure deterministic results even without external data access.

---

## 🧾 Test Coverage

| File                        | Purpose                                             |
| --------------------------- | --------------------------------------------------- |
| `test_ingest.py`            | AOI tiling, CRS alignment, dataset stubs.           |
| `test_detect.py`            | CV filters, anomaly scoring, mock anomalies.        |
| `test_evaluate.py`          | NDVI/EVI time-series, LiDAR fallback.               |
| `test_verify.py`            | ADE fingerprint rules, PAG mock validation.         |
| `test_report.py`            | Dossier generator (PDF/HTML/MD, with mock figures). |
| `test_cli.py`               | CLI entrypoints (`wde ingest`, `wde full-run`).     |
| `test_notebook.py`          | Runs `ade_discovery_pipeline.ipynb` with papermill. |
| `test_integration_dummy.py` | End-to-end run on dummy AOI, verify output dirs.    |

---

## ⚖️ ADE Fingerprint Mocks

ADE fingerprints are **domain-critical**:

* Elevated NDVI in dry season.
* Floristic signatures (mocked via categorical raster).
* Soil phosphorus spike.
* Circular anomaly geometry (synthetic DEM bump).

Unit tests validate that the verification stage correctly flags ADE sites under these conditions.

---

## 🚦 CI Modes

* **PR Workflow** (`kaggle_notebook_check.yml`)

  * Light mode: 2 tiles, no LiDAR, minimal ADE checks.
  * Validates that the notebook runs end-to-end on Kaggle.

* **Main Workflow** (`kaggle_notebook_ci.yml`)

  * Heavy mode: ADE fingerprints, PAG `.gml`, Bayesian uncertainty, SSIM counterfactuals.
  * Produces candidate dossiers + Markdown run summary.

---

## 🧰 Running Tests Locally

```bash
# Unit + integration tests
pytest -v

# Run specific test
pytest tests/test_verify.py::test_ade_fingerprint

# With coverage report
pytest --cov=world_engine
```

---

## 📑 References

* Architecture & pipeline spec
* Repository structure
* ADE pipeline (Kaggle notebook)
