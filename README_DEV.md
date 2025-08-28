# 🔧 WDE — Developer Quickstart (`README_DEV.md`)

Local developer guide for the **World Discovery Engine (WDE)**.
If you’re looking for the Kaggle judge view, see `README_KAGGLE.md`.
For the full project overview, see `README.md`.

WDE is a **config-driven, CLI-first** pipeline with a thin Kaggle notebook on top. Core code lives in `world_engine/`, configs in `configs/`, tests in `tests/`, docs in `docs/` . The discovery funnel (ingest → detect → evaluate → verify → report) and outputs (GeoJSON, dossiers, PAG graphs, uncertainty, SSIM) are specified in the architecture/docs set .

---

## 1) Prerequisites

* **Python**: 3.10–3.11
* **System libs** (for geospatial): `gdal`, `proj`, `geos` (install via OS pkg manager)
* **Optional** (for LiDAR & ML extras): `pdal`, CUDA toolkit/GPU drivers
* **Make** (optional), **Git**, and (recommended) **direnv/pyenv**

> Why GDAL/PDAL? The pipeline ingests Sentinel/Landsat/SAR/DEM, and optionally LiDAR (GEDI/OpenTopography) that benefit from GDAL/PDAL bindings .

---

## 2) Setup

```bash
git clone https://github.com/<your-org>/world-discovery-engine.git
cd world-discovery-engine

# Create and activate a fresh virtualenv
python -m venv .venv && source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

Docker (optional but recommended for parity with CI/Kaggle):

```bash
docker build -t wde:dev .
docker run --rm -it -v $PWD:/app -w /app wde:dev bash
```

The repo’s structure and Dockerfile align with the documented reproducibility model (deterministic seeds, pinned env, artifact logging) .

---

## 3) Configuration

All runs are **config-driven**. Put YAML in `configs/`:

```yaml
# configs/default.yaml
aoi: "data/aoi/brazil_polygon.geojson"     # or a bbox in another file
datasets:
  sentinel2: { source: "aws://sentinel-2/...", bands: ["B4","B8","B11"] }
  sentinel1: { source: "aws://sentinel-1/..." }
  dem:       { source: "aws://copernicus-dem-30m/..." }
  soilgrid:  { source: "aws://isric/soilgrids/phosphorus.tif" }
  # optional: lidar/gedi/opentopo
pipeline:
  tile_size_deg: 0.05
  anomaly_threshold: 0.8
  use_gnn: true
output:
  dir: "outputs"
```

> Datasets and access patterns follow the platform/data guide (Sentinel-2/1, Landsat, SRTM/Copernicus DEM, NICFI, SoilGrids, HydroSHEDS, GEDI/OT) .
> Repo config/structure guidance: `docs/repository_structure.md` and `docs/architecture.md` .

Secrets (if needed) should be set via env vars (e.g., `SENTINEL_API_KEY`, `USGS_API_KEY`) — never hard-code in configs .

---

## 4) CLI — run the pipeline locally

The CLI is Typer-based and maps 1:1 to the discovery funnel:

```bash
# Ingest → Detect → Evaluate → Verify → Report (single shot)
python -m world_engine.cli full-run --config configs/default.yaml

# Or stage-by-stage
python -m world_engine.cli ingest  --config configs/default.yaml
python -m world_engine.cli scan    --config configs/default.yaml
python -m world_engine.cli evaluate --config configs/default.yaml
python -m world_engine.cli verify  --config configs/default.yaml
python -m world_engine.cli report  --config configs/default.yaml
```

**Outputs (default):** `outputs/`

* `candidates.json`, `candidates.geojson`
* `reports/` (PDF/HTML site dossiers)
* `pag/` (`*.gml` partial ancestral graphs)
* `uncertainty/` (JSON/plots)
* `ssim/` (robustness artifacts)
* `ndvi_timeseries/` (seasonal ADE checks)

This mirrors the architecture/dossier contract (evidence stacks, PAG, calibrated uncertainty, SSIM) .

---

## 5) Notebook (optional, local)

You can run the main notebook with **papermill** (CI does this):

```bash
pip install papermill
papermill notebooks/ade_discovery_pipeline.ipynb \
          outputs/ade_discovery_pipeline_out.ipynb
```

On Kaggle, place the notebook under `notebooks/`; it imports the same `world_engine/` modules and uses `configs/` for settings .

---

## 6) Data notes

* **Sentinel-2/1**: open access via Copernicus Data Space, ASF (SAR), or public cloud buckets; STAC APIs supported .
* **Landsat**: USGS EarthExplorer/API & AWS public buckets; C2 Level-2 products (GeoTIFF) .
* **DEM**: SRTM (public domain) & Copernicus DEM (AWS COGs) .
* **NICFI**: Planet basemaps (tropics) via API (non-commercial terms) .
* **SoilGrids / HydroSHEDS / GEDI**: see `docs/datasets.md` for endpoints and licensing; ADE proxy layers supported .

When in doubt, consult `docs/datasets.md` and `docs/architecture.md` for the exact sources, licensing, and ingestion advice .

---

## 7) Testing & Linting

```bash
# Unit/integration tests (small dummy AOI)
pytest -q

# Lint & static analysis (matches CI)
pre-commit run -a
```

CI workflows validate notebook execution and artifacts (candidates, GeoJSON, PAG, uncertainty, SSIM, NDVI) per the pipeline contract .

---

## 8) Developer tips

* **Keep code in `world_engine/`**, not notebooks; import from notebooks/CLI only .
* **Determinism**: fix random seeds; avoid time-based randomness; log config hashes.
* **Artifacts as truth**: write intermediate JSON/GeoTIFFs (ingest, detect, evaluate) for traceability and re-runs .
* **Configs over kwargs**: add new options to YAML; reflect in CLI help.
* **Respect ethics**: never publish precise coordinates in sensitive regions; follow CARE/IPHAN guards baked into the pipeline .

---

## 9) Troubleshooting

* **GDAL/PROJ issues**: ensure system libs are present before `pip install -r requirements.txt`.
* **Long LiDAR runs**: skip or downsample (set `use_lidar: false` or point to DEM derivatives) if PDAL isn’t available.
* **Kaggle parity**: if a feature fails locally but works on Kaggle, compare library versions (pin with Docker for parity) .
* **Empty outputs**: check AOI & dataset availability; run `ingest` standalone and inspect cached tiles/interim artifacts.

---

## 10) References

* **Architecture & Dossier Contract** — `docs/architecture.md`
* **Repository & Module Layout** — `docs/repository_structure.md`
* **Notebook Spec** — `notebooks/ade_discovery_pipeline.ipynb` (see `docs/ADE_pipeline.md`)
* **Datasets & Access** — `docs/datasets.md` (Sentinel/Landsat/DEM/NICFI/SoilGrids/HydroSHEDS/GEDI)
* **Ethics & Governance** — `docs/ETHICS.md` (CARE/IPHAN, masking, anti-colonial practice)

---

Happy digging — and keep the pipeline **config-driven, auditable, and Kaggle-reproducible**.
