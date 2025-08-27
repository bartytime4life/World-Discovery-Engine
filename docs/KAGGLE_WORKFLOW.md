# 🧭 Kaggle Workflow — World Discovery Engine (WDE)

This guide explains **how to run the WDE notebooks on Kaggle**, what **artifacts** to expect, and how the repository’s **CI/CD** keeps notebooks reproducible and publishable.

---

## 0) What you need

- A Kaggle account (free).
- The repo notebooks (you can copy/paste, upload, or attach via Kaggle Datasets).
- (Optional) Internet access in the notebook (toggle **Settings → Internet → On**) if you plan to fetch open data on the fly. Otherwise use attached datasets or the notebook’s small demo inputs.

> WDE uses **open datasets only** (Sentinel-1/2, Landsat, SRTM/Copernicus DEM, SoilGrids, etc.) and is structured to run end‑to‑end on Kaggle with CPU; **GPU is optional** for heavier models.

---

## 1) Notebooks you can run

- `notebooks/WDE_Kaggle_Starter.ipynb` — tiny, self‑contained starter; lists inputs and writes a minimal artifact set.
- `notebooks/WDE_End_to_End.ipynb` — end‑to‑end demo (ingest → detect → evaluate → verify → report) on a small AOI slice.
- `notebooks/WDE_Features_Scoring_Demo.ipynb` — focused demo for features + scoring.
- `notebooks/ade_discovery_pipeline.ipynb` — full ADE Discovery Pipeline (main deliverable), producing candidate dossiers and an exportable bundle.

> Each notebook prints environment info, saves artifacts in `/kaggle/working/outputs/`, and falls back to small demo data if external APIs are unavailable.

---

## 2) Step‑by‑step on Kaggle

1. **Create Notebook** → new Python notebook.
2. In the file browser, **Upload** the notebook (`*.ipynb`) and (optionally) the `tools/` folder if you want export utilities inline.
3. Open **Settings**:
   - **Accelerator**: None/CPU is OK; GPU speeds up deep models.
   - **Internet**: turn **On** if you want to fetch open data via APIs. If **Off**, attach a Kaggle Dataset containing your inputs.
4. **Run All**. Artifacts will be written to `/kaggle/working/outputs/` and shown in the right sidebar’s **Output** panel.

> If you see API/auth prompts, either disable those features (they’re optional) or provide keys via **Kaggle Secrets**. The demo path requires no keys.

---

## 3) What artifacts you’ll see

| Path | Description |
|---|---|
| `/kaggle/working/outputs/run_manifest.json` | Run manifest (timestamps, versions, artifact list). |
| `/kaggle/working/outputs/*dossier*/` | One folder per candidate site with figures, maps, and a short confidence narrative. |
| `/kaggle/working/outputs/verified_candidates.geojson` | Final site candidates with attributes (scores, evidence flags). |
| `/kaggle/working/outputs/submission.csv` | Lightweight CSV export (useful for leaderboard-style submissions or downstream apps). |
| `/kaggle/working/outputs/bundle.zip` | Zipped bundle of dossiers + manifest for easy download/archival. |

> Use `tools/wde_export.py` to regenerate `submission.csv`, `manifest.json`, and a zipped bundle from a candidates file.

---

## 4) Reproducibility checklist

- **Config‑driven**: parameters are loaded from `configs/*.yaml` where possible.
- **Deterministic seeds**: notebooks set NumPy/Python/Torch seeds by default.
- **Saved artifacts**: every run writes a `run_manifest.json` and version snapshot of key libs (printed at top of notebook).
- **CI path smoke**: the repo includes a `tests/notebooks/test_kaggle_path.py` to validate imports and notebook convertibility.

---

## 5) Ethics‑by‑default in notebooks

- Coordinates in public HTML exports can be **rounded or masked** (see the ethics section in the notebooks).  
- Dossiers warn when a detection falls within **Indigenous territories or protected areas**; share privately with authorities/research partners first.
- All sources are **open** and credited in the report footer.

---

## 6) CI/CD badges (GitHub)

- **Lint & Static Analysis**: `.github/workflows/lint.yml`
- **Kaggle Notebook Check**: `.github/workflows/kaggle_notebook_check.yml` (executes first cells for import wiring)
- **Notebook Publish (this guide)**: `.github/workflows/notebook_publish.yml` → converts notebooks to HTML and uploads as build artifacts

You’ll see green checks in PRs and the default branch when: code style passes, imports are clean, and notebooks render.

---

## 7) Troubleshooting

- **Kernel timed out**: Reduce AOI (tile count), disable heavy models, or run only the demo path.
- **No Internet**: Attach a Kaggle Dataset with input rasters or rely on the included demo tiles.
- **GDAL/Geo libs missing**: The notebooks avoid system dependencies where possible; for local runs use the Docker image to match our CI environment.

---

## 8) Export quickstart

From a notebook cell (or local terminal) you can generate standard exports:

```bash
# submission CSV
python tools/wde_export.py submission   --candidates /kaggle/working/outputs/verified_candidates.geojson   --out /kaggle/working/outputs/submission.csv

# manifest
python tools/wde_export.py manifest   --outputs /kaggle/working/outputs   --out /kaggle/working/outputs/run_manifest.json

# zip bundle
python tools/wde_export.py bundle   --outputs /kaggle/working/outputs   --zip /kaggle/working/outputs/bundle.zip
```

---

**Ready to go.** Open `ade_discovery_pipeline.ipynb`, run the cells, and download `bundle.zip` from the Output panel 🎒