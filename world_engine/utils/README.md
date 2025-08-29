# 🛠️ `utils/` — Shared Utilities & Helpers

This directory contains **shared functions and utilities** used across the **World Discovery Engine (WDE)** pipeline. Utilities abstract away repetitive geospatial, image, text, and math operations so that core pipeline modules (`ingest.py`, `detect.py`, `evaluate.py`, `verify.py`, `report.py`, `models/*`) remain clean, modular, and reproducible.

---

## 📌 Purpose

* Provide **common helper functions** for geospatial/image/text/scientific tasks.
* Ensure **DRY** (Don’t Repeat Yourself) code across pipeline stages.
* Guarantee **determinism and reproducibility** by centralizing seeding, hashing, and logging.
* Make **testing and CI** easier with small, isolated functions.

---

## 📂 Directory Layout

```bash
world_engine/utils/
├── __init__.py           # Marks package, may re-export most used utilities
├── geospatial.py         # Coordinate transforms, raster reprojection, tiling logic
├── image_utils.py        # Image normalization, patch extraction, filters
├── text_utils.py         # OCR/text cleaning, entity extraction
├── math_utils.py         # Stats, z-scores, safe divs, calibration helpers
├── viz_utils.py          # Plotting overlays, raster previews, heatmaps
├── logging_utils.py      # Run logger, config hashing, provenance capture
└── README.md             # (this file)
```

*Optional future slots:* `parallel.py` (multiprocessing helpers), `download.py` (data fetch wrappers), `cli_helpers.py` (Typer helpers).

---

## 🧩 Roles of Each Module

### 1. **Geospatial (`geospatial.py`)**

* Coordinate transforms (WGS84 ↔ UTM).
* Raster reprojection & resampling.
* Tile/grid generators (0.05° Kaggle default).
* Distance-to-hydro & slope/aspect calculations.

### 2. **Image Utils (`image_utils.py`)**

* Cloud/shadow masking.
* Patch extraction (`256×256` default).
* Histogram equalization, normalization, z-scaling.
* Edge detection & morphology helpers for anomaly scan.

### 3. **Text Utils (`text_utils.py`)**

* OCR + cleaning for historical maps/diaries.
* Keyword/entity extraction (river names, cultural terms).
* Tokenization + lightweight embeddings for overlays.

### 4. **Math Utils (`math_utils.py`)**

* Safe division & nan-to-num wrappers.
* Statistical z-score, robust z (median/MAD).
* Calibration error (ECE, Brier score).
* Bootstrap confidence intervals.

### 5. **Viz Utils (`viz_utils.py`)**

* Quick plots for GeoTIFF patches, DEM hillshades.
* NDVI/EVI/SAR overlay previews.
* Heatmap visualizations for anomaly and ADE scores.
* Candidate dossier figures for `report.py`.

### 6. **Logging Utils (`logging_utils.py`)**

* Consistent run logging (timestamps, seeds, configs).
* Config/data hashing for audit trails.
* JSON/Markdown run manifests.
* CI/CD hooks for runtime + reproducibility checks.

---

## ⚙️ Integration

* **`ingest.py`** → uses `geospatial.py` for tiling/reprojection.
* **`detect.py`** → calls `image_utils.py` for edge/texture ops.
* **`evaluate.py`** → calls `math_utils.py` for ADE fingerprints.
* **`verify.py`** → integrates with `viz_utils.py` for fusion explainability.
* **`report.py`** → uses `logging_utils.py` + `viz_utils.py` for candidate dossiers.
* **`models/*`** → re-use `math_utils.py` for calibration & ECE checks.

---

## 🧪 Testing

Each module has **unit tests** under `tests/utils/`:

* `test_geospatial.py`: reprojection sanity, AOI tiling.
* `test_image_utils.py`: patch extraction, cloud mask.
* `test_math_utils.py`: ECE values on synthetic probs.
* `test_logging_utils.py`: hash consistency, manifest logs.

Tests run in CI to guarantee reproducibility.

---

## 🚀 Quick Examples

```python
from world_engine.utils import geospatial, image_utils, math_utils

# Tile an AOI into 0.05° grids
tiles = geospatial.tile_aoi((-3.5, -60.5, -3.4, -60.4), size_deg=0.05)

# Extract 256×256 patches from a GeoTIFF
patches = image_utils.extract_patches("sentinel2_tile.tif", size=256)

# Compute z-scores robustly
z = math_utils.robust_zscore(values, clip=6.0)

# Log a config hash for reproducibility
from world_engine.utils.logging_utils import log_run
log_run(cfg="configs/default.yaml", output_dir="outputs/")
```

---

📖 **References**

* WDE Repository Structure
* ADE Discovery Pipeline
* Enriching WDE for Archaeology & Earth Systems

---
