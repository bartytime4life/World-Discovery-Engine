# 🏁 WDE — Kaggle Run Guide

This README documents **how to run the World Discovery Engine (WDE) on Kaggle** with the
notebooks and stubs provided in this repository.

## Quick Start
1. Upload the following folders/files to Kaggle alongside your notebook(s):
   - `tools/` (`kaggle_utils.py`, `wde_pipeline.py`, `wde_features.py`, `wde_scoring.py`, `geo_io.py`)
   - `configs/` (`kaggle.yaml`, `pipeline.yaml`, `features.yaml`, `scoring.yaml`, `datasources.yaml`)
   - `notebooks/` (the notebook you intend to run, or copy content into a Kaggle notebook)
2. Attach the competition dataset(s) under `/kaggle/input` as needed.
3. Run the notebook. Artifacts will be saved to `/kaggle/working/wde_outputs/`.

## Notebooks
- `notebooks/WDE_Kaggle_Starter.ipynb` — environment checks, input listing, config load, demo export.
- `notebooks/WDE_End_to_End.ipynb` — wires the pipeline stubs to configs and exports artifacts.
- `notebooks/WDE_Features_Scoring_Demo.ipynb` — focuses on features + scoring with minimal mocks.

## Paths (Kaggle)
- Input datasets: `/kaggle/input/<dataset>`
- Working dir: `/kaggle/working/`
- Outputs: `/kaggle/working/wde_outputs/`

## Configuration
- `configs/kaggle.yaml`: runtime parameters (seed, sample rows, output dir, demo size).
- `configs/pipeline.yaml`: toggles for steps and top-K export.
- `configs/features.yaml`: feature computation parameters and switches.
- `configs/scoring.yaml`: scoring strategy and hyperparameters.
- `configs/datasources.yaml`: AOI examples and public data catalog links.

## Artifacts
The minimal stubs export:
- `demo_candidates_top50.csv`
- `demo_candidates_scatter.png` (if Matplotlib available)
- `run_manifest.json`

## CI (GitHub Actions)
- `.github/workflows/kaggle_notebook_check.yml` executes the first cells of the starter notebook
  and validates that modules in `tools/` import successfully. This keeps the Kaggle path green.

_Last updated: 2025-08-27 19:17:40_