# 🗒️ Notebooks — World Discovery Engine (WDE)

This folder hosts **reproducible Kaggle notebooks** and local development notebooks.
All notebooks must follow the project rule: **CLI-first**. Notebooks are thin, documented
entry points for **exploration and demos**; the core logic lives in `/tools` and configs in `/configs`.

---

## ✅ Goals
- Run cleanly on **Kaggle** (no internet installs; only use datasets provided in `/kaggle/input`).
- Save all artifacts to **`/kaggle/working/wde_outputs`** for persistence.
- Import everything from **`/tools`** and configure behavior with **`/configs/*.yaml`**.
- Deterministic: set seeds and prefer pure-Python fallbacks when GPUs/GIS are unavailable.

---

## 📦 Included
- `WDE_Kaggle_Starter.ipynb` — minimal starter that lists inputs, reads a config, runs a demo pipeline,
  exports a CSV of top-K demo candidates, and writes a small run manifest.

---

## 📁 Paths (Kaggle runtime)
- **Input datasets**: `/kaggle/input/<competition-or-dataset>`
- **Working outputs**: `/kaggle/working/wde_outputs/`
- **Notebook directory**: `/kaggle/working/` (execution happens here)
- **Repo modules**: `./tools`, `./configs` (place these next to the notebook when committing to Kaggle)

---

## ⚙️ Configuration
- The notebook expects `./configs/kaggle.yaml` in the working directory where the notebook runs.
- A helper loader in `tools/kaggle_utils.py` supports `.yaml`, `.json`, or simple `KEY=VALUE` formats.

---

## 🧪 Quick Smoke Test
1. Open **`WDE_Kaggle_Starter.ipynb`** in Kaggle.
2. Ensure a dataset is attached under `/kaggle/input` (any CSV is fine for the demo).
3. Run the first few cells — you should see environment info and a file tree listing of `/kaggle/input`.
4. The demo will write to `/kaggle/working/wde_outputs`:
   - `demo_candidates_top50.csv`
   - `run_manifest.json`

---

## 🚀 Conventions for Future Notebooks
- Keep cells short and focused; avoid long monolithic cells.
- All expensive work goes into `/tools` and is unit-testable.
- Wrap ad‑hoc code in small functions for clarity; use `with timer("step")` for simple profiling.
- Visuals should be quick: use Matplotlib scatter/line for maps & points; defer heavy GIS to local dev.

---

_Last updated: 2025-08-27 19:06:13_