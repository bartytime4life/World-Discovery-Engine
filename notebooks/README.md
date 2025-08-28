# 🗒️ Notebooks — World Discovery Engine (WDE)

This folder hosts the **Kaggle-ready pipeline notebook** and any **local exploration notebooks**.

> **Rule of thumb:** notebooks are **thin, documented entry points** — all core logic lives in `world_engine/`, configs in `configs/`, and tests in `tests/`.
> Notebooks should be **reproducible narratives**, not where the heavy lifting lives.

---

## ✅ Goals

* **Kaggle-first:** must run end-to-end in a Kaggle kernel, with no hidden dependencies.
* **Reproducible:** deterministic seeds, pinned/runtime-pinned env, CI validation of artifacts.
* **Modular:** all code **imports** from `world_engine/` (no inline re-implementation).
* **Config-driven:** behavior controlled by YAML/JSON in `configs/`, not hardcoded.
* **Artifacts-first:** save all outputs to a writable work dir for persistence/audit.

> **Path policy (Kaggle):**
>
> * Inputs: `/kaggle/input/<dataset-or-competition>/`
> * Working dir: `/kaggle/working/`
> * Outputs (canonical): `/kaggle/working/outputs/`
>   (Notebooks should respect `OUT_DIR` env or config key and default to this path.)

---

## 📦 Included

### `ade_discovery_pipeline.ipynb`

**Main Kaggle notebook (competition deliverable).** Runs the full WDE pipeline:

1. **AOI tiling & ingestion** (Sentinel-2 optical, Sentinel-1 SAR, DEM; optional LiDAR/GEDI).
2. **Coarse anomaly scan** (CV filters, shape cues, optional VLM captions).
3. **Mid-scale evaluation** (NDVI/EVI time-series, hydro-geomorph checks, historical overlays).
4. **Verification & fusion** (ADE fingerprints, PAG causal graph, Bayesian UQ, SSIM counterfactuals).
5. **Candidate dossiers** (maps, overlays, uncertainty, refutation tests, narrative + artifacts).

### `WDE_Kaggle_Starter.ipynb`

**Minimal demo.** Lists inputs, runs a stub pipeline (synthetic AOI tile with a planted anomaly), exports a toy CSV of demo candidates, and writes a run manifest.

---

## 📁 Paths (Kaggle runtime)

* **Input datasets:** `/kaggle/input/<competition-or-dataset>/`
* **Working outputs (canonical):** `/kaggle/working/outputs/`
* **Notebook CWD:** `/kaggle/working/`
* **Repo modules (copied alongside notebook):** `./world_engine/`, `./configs/`

> **Tip:** In local runs, set `OUT_DIR=./outputs` and mirror the same relative layout for parity.

---

## ⚙️ Configuration

* **Default:** `./configs/kaggle.yaml` (override by setting `WDE_CONFIG` env or passing a path in the notebook’s first cell).
* **What configs define:** AOI bounds/shape, datasets & sources, thresholds, model choices, output dirs, seeds, runtime guards.
* **Loader:** `world_engine/utils/config_loader.py` should support:

  * YAML/JSON load
  * env var overlays (e.g., `OUT_DIR`, `WDE_SEED`)
  * programmatic overrides (dict merge)
  * validation (required keys, types)

**Example bootstrap (first code cell):**

```python
import os, json
from world_engine.utils.config_loader import load_config

cfg_path = os.environ.get("WDE_CONFIG", "./configs/kaggle.yaml")
CFG = load_config(cfg_path, overrides={"outputs.outputs_dir": os.environ.get("OUT_DIR", "/kaggle/working/outputs/")})
print(json.dumps(CFG, indent=2))
```

---

## 🧪 Quick Smoke Test

1. Open **`WDE_Kaggle_Starter.ipynb`** on Kaggle.
2. (Optional) Attach any dataset under `/kaggle/input/...` to confirm the tree listing.
3. **Run all cells** — expect:

   * System info & input tree
   * Synthetic AOI tile generated
   * Edge-based anomaly scored
   * Artifacts written to `/kaggle/working/outputs/`:

     * `demo_candidates_top50.csv`
     * `run_manifest.json`

For the **full pipeline**, run **`ade_discovery_pipeline.ipynb`** — it will produce **candidate dossiers** (PNG/MD/JSON) under the configured outputs dir.

---

## 🧩 Notebook Design Contract

* **Cells = narrative orchestration.** All computation is imported:

  * `from world_engine.ingest import run_ingest`
  * `from world_engine.detect import run_detect`
  * `from world_engine.evaluate import run_evaluate`
  * `from world_engine.verify import run_verify`
  * `from world_engine.report import build_dossiers`
* **Step timing:** wrap major calls with a light timer (context manager) or `%time`.
* **Visualization:** quick Matplotlib/inline plots; heavy GIS/3D belongs to local tooling.
* **State & paths:** never hardcode absolute paths; always route through config/env.
* **Idempotence:** re-running must not corrupt previous artifacts; version or overwrite deterministically.
* **Logging:** print concise, structured status lines; detailed logs go to a file in `outputs/`.

---

## 🧾 Artifact Policy

Each run must write:

* A **manifest** (`run_manifest.json`) with:

  * config hash (and/or the subset of config used)
  * seed(s)
  * timestamps
  * produced files (relative paths + sizes)
* Stage outputs written to stable subdirs, e.g.:

  * `outputs/candidates/` (JSON, PNG previews)
  * `outputs/dossiers/` (MD/HTML + figures)
  * `outputs/manifests/` (`run_manifest.json`, stage manifests)

> **Do** include minimal previews (PNGs) for candidates; **don’t** embed giant arrays in notebooks.

---

## 🧰 Execution Order (typical)

1. **Config** → load/echo
2. **Ingest** → tiles + stacks + overlay registry
3. **Detect** → anomaly ranking & artifacts
4. **Evaluate** → NDVI/EVI, terrain, plausibility, historical overlays
5. **Verify** → ADE fingerprints, PAG, Bayesian UQ, SSIM ablations
6. **Report** → per-candidate dossiers + run manifest
7. **Recap** → print short file tree for `/kaggle/working/outputs/`

---

## 🔒 Ethics & Safety (always)

* Mask or generalize **precise coordinates** in public outputs unless explicit permission exists.
* Include sovereignty notices for detections intersecting Indigenous territories.
* Keep all inputs **open-licensed** or user-provided with consent.
* Treat notebooks as **transparent narratives**: show what was done, with enough detail to audit.

---

## 🚀 Conventions for Future Notebooks

* Keep cells short, **narrative-first**, and literate (why → what → how).
* All heavy code → `world_engine/` modules (import; never duplicate).
* Use `OUT_DIR` and config keys; avoid path literals.
* Version output file names or embed run UUIDs when appropriate.
* Validate artifacts (basic schema/shape) before concluding the run.

---

## 🧱 Troubleshooting

* **Module import fails:** ensure `world_engine/` is in the same directory as the notebook (Kaggle copy step) or `sys.path.append('.')`.
* **No write permission:** only write under `/kaggle/working/` on Kaggle.
* **Missing packages:** prefer built-ins; if unavoidable, install at the top with `pip` and document versions.
* **Large outputs:** keep under Kaggle limits; offload heavy figures/data to local runs or compress.

---

*Last updated: 2025-08-27 19:35:00*
