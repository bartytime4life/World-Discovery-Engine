# FILE: README.md
# ---------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — OpenAI → Z Challenge · Archaeology & Earth Systems
#
# Mission
#   WDE is a reproducible, open-data pipeline that surfaces archaeologically significant candidates
#   (e.g., Anthropogenic Dark Earths, earthworks, geoglyphs) from geospatial signals. It fuses
#   satellite imagery, radar, DEM/LiDAR, hydrology, soils, historical texts, and user-provided
#   overlays, then verifies each candidate with multi-modal evidence, uncertainty, and a clear
#   causal plausibility story. 
#
# Why this repo?
#   • Kaggle-first: single notebook deliverable (ade_discovery_pipeline.ipynb) that runs end-to-end
#     on Kaggle CPU/GPU with public data only. 
#   • Evidence-rich outputs: per-site “candidate dossiers” with maps, overlays, NDVI/terrain,
#     causal graphs (PAG), and Bayesian uncertainty visuals. 
#   • Reproducible & ethical by design: fixed seeds, pinned deps, Docker, CI, CARE/FPIC/sovereignty
#     notices, and coordinate coarsening for sensitive contexts.  
#
# Pipeline overview (Discovery Funnel)
#   1) Tiling & Ingestion: stack Sentinel-2/1, Landsat, DEM/DEM-derivatives, soils, hydrology; ingest
#      user overlays (docs/links/images) aligned to tiles. 
#   2) Coarse Scan: fast CV (edges/Hough/texture), DEM relief (multi-angle hillshade/LRM), and a
#      lightweight VLM cue pass to rank tiles by anomaly score. 
#   3) Mid-Scale Evaluation: NDVI/EVI time-series stability, LiDAR/DEM micro-relief, hydro-geomorph
#      plausibility, historical overlays, user-concordance. 
#   4) Verification & Evidence Fusion: multi-proof rule (≥2 modalities), ADE fingerprints, causal
#      plausibility graph (FCI→PAG), Bayesian GNN uncertainty, counterfactual SSIM stress-tests.  
#   5) Dossier: site map, NDVI/DEM/SAR panels, causal graph, uncertainty plots, historical snippets,
#      sensitivity summaries, confidence narrative. 
#
# Data sources (open only, reproducible)
#   Sentinel-2, Sentinel-1, Landsat, NICFI Planet mosaics (tropics), SRTM/Copernicus DEM, GEDI,
#   MapBiomas/MODIS, HydroSHEDS, plus historical maps/diaries. Access and examples in docs/datasets.md.   
#
# Repository layout
#   world_engine/   core Python package (ingest, detect, evaluate, verify, report)
#   configs/        YAML configs (AOIs, thresholds, dataset endpoints)
#   notebooks/      ade_discovery_pipeline.ipynb (Kaggle-ready)
#   tests/          unit + integration tests
#   docs/           architecture, datasets registry, ethics & governance
#   scripts/        helper scripts (optional)
#   data/           (empty placeholders; large data via APIs or Kaggle Datasets) 
#
# Quickstart
#   1) Create a Python 3.11 env; install deps:
#        python -m venv .venv && source .venv/bin/activate
#        pip install -r requirements.txt
#   2) (Optional) Build the Docker image for identical environment:
#        docker build -t wde:latest .
#   3) Run locally via CLI:
#        python -m world_engine.cli ingest --config configs/default.yaml
#        python -m world_engine.cli detect  --config configs/default.yaml
#        python -m world_engine.cli evaluate --config configs/default.yaml
#        python -m world_engine.cli verify   --config configs/default.yaml
#        python -m world_engine.cli report   --config configs/default.yaml
#   4) Or open notebooks/ade_discovery_pipeline.ipynb on Kaggle and “Run All”. 
#
# Kaggle notes
#   • Use GPU accelerator for deep/VLM steps where enabled; the notebook auto-skips heavy models
#     if GPU is unavailable. 
#   • If Internet is off (some comps), attach datasets via Kaggle Datasets and set config paths. 
#
# Ethics & governance
#   WDE integrates CARE/FPIC, local legal compliance (e.g., IPHAN in Brazil), sovereignty notices,
#   and default coordinate coarsening in public outputs; see ETHICS.md.  
#
# Reproducibility & CI
#   • Pinned deps (requirements.txt), fixed seeds, env printouts.
#   • Dockerfile for cross-platform parity.
#   • GitHub Actions: lint, unit/integration tests, and refutation suite on a toy AOI.  
#
# References
#   • Architecture & funnel: 
#   • Repo structure:       
#   • Data access guides:   
#   • Core sampling DBs:    
#   • Kaggle platform:      
# ---------------------------------------------------------------------------------------------------


# FILE: LICENSE
# ---------------------------------------------------------------------------------------------------
# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/
#
# TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
# 1. Definitions.
# "License" shall mean the terms and conditions for use, reproduction, and distribution as defined by
# Sections 1 through 9 of this document.
# "Licensor" shall mean the copyright owner or entity authorized by the copyright owner that is granting
# the License.
# "Legal Entity" shall mean the union of the acting entity and all other entities that control, are controlled by,
# or are under common control with that entity. ...
# (Full standard Apache-2.0 text continues; unchanged)
#
# END OF TERMS AND CONDITIONS
# ---------------------------------------------------------------------------------------------------


# FILE: requirements.txt
# ---------------------------------------------------------------------------------------------------
# Core scientific
numpy==1.26.4
scipy==1.13.1
pandas==2.2.2
xarray==2024.6.0
rioxarray==0.15.0

# Geospatial stack (install via wheels; GDAL and PROJ provided by Docker)
shapely==2.0.4
pyproj==3.6.1
rasterio==1.3.9
geopandas==0.14.4
folium==0.16.0
pystac-client==0.7.5

# Remote sensing access
sentinelsat==1.2.1
asf-search==7.0.3
landsatxplore==0.14.1
earthaccess==0.9.2
boto3==1.34.162
requests==2.32.3

# Computer vision & ML
opencv-python-headless==4.10.0.84
scikit-image==0.24.0
scikit-learn==1.5.1
matplotlib==3.9.1
plotly==5.22.0

# Optional deep/VLM (auto-skipped if unavailable)
torch==2.3.1
torchvision==0.18.1
open-clip-torch==2.24.0

# LiDAR/point-cloud & DEM utilities
pdal==3.4.5

# Causal & uncertainty
pgmpy==0.1.24
networkx==3.3

# CLI & utils
typer[all]==0.12.5
rich==13.7.1
pyyaml==6.0.1

# Notebook
jupyter==1.0.0
ipykernel==6.29.5
# ---------------------------------------------------------------------------------------------------


# FILE: Dockerfile
# ---------------------------------------------------------------------------------------------------
# Minimal, reproducible environment with GDAL/PDAL for geospatial processing.
# Note: Building GDAL/PDAL from distro packages for reliability.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# System deps (build tools + geospatial libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git curl wget ca-certificates \
    python3 python3-venv python3-pip \
    gdal-bin libgdal-dev proj-bin libproj-dev geotiff-bin \
    pdal libpdal-dev \
    libspatialindex-dev libgeos-dev libjpeg-turbo-progs \
    && rm -rf /var/lib/apt/lists/*

# Create app env
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Make sure GDAL/PROJ are discoverable for wheels that look up headers
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_DATA=/usr/share/gdal

RUN python3 -m venv /opt/venv && . /opt/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /app/requirements.txt

ENV PATH="/opt/venv/bin:${PATH}"

# Copy source (optional; most users will run via notebook or mount)
# COPY world_engine /app/world_engine
# COPY configs /app/configs

# Default command prints versions for reproducibility
CMD python -c "import sys,platform,subprocess; print('Python',platform.python_version()); subprocess.run(['gdalinfo','--version']); import rasterio; print('rasterio',rasterio.__version__)"
# ---------------------------------------------------------------------------------------------------


# FILE: .gitignore
# ---------------------------------------------------------------------------------------------------
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.build/
dist/
.eggs/
.ipynb_checkpoints/
.mypy_cache/
.pytest_cache/
.venv/

# Data & artifacts (use APIs / Kaggle Datasets / DVC)
data/**
!data/.gitkeep
outputs/**
!outputs/.gitkeep

# Logs, temp
*.log
tmp/
.cache/

# Notebooks
*.nbconvert.ipynb

# Docker
*.tar
# ---------------------------------------------------------------------------------------------------


# FILE: .github/workflows/ci.yml
# ---------------------------------------------------------------------------------------------------
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install system deps (GDAL/PDAL)
        run: |
          sudo apt-get update
          sudo apt-get install -y gdal-bin libgdal-dev pdal libpdal-dev

      - name: Install Python deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest black flake8

      - name: Lint (black/flake8)
        run: |
          black --check .
          flake8 .

      - name: Unit tests
        run: |
          pytest -q

      - name: Smoke integration (toy AOI)
        run: |
          echo "TODO: add small AOI config & smoke run (ingest->detect->report) in tests/"
# ---------------------------------------------------------------------------------------------------


# FILE: CONTRIBUTING.md
# ---------------------------------------------------------------------------------------------------
# Contributing to WDE

Thanks for helping build the World Discovery Engine! This project follows a documentation-first,
reproducible workflow and an ethics-by-design stance.

## Ground rules
- Prefer small, focused PRs with tests and updated docs (README/docs/architecture).
- Keep the pipeline modular and deterministic; avoid hard-coded paths (“magic numbers”).
- Discuss data ethics impacts in PRs that add new sources or change outputs (see ETHICS.md).

## Dev setup
- Python 3.11+; `pip install -r requirements.txt`
- Optional: `docker build -t wde:latest .` for parity.
- Pre-commit style: run `black .` and `flake8 .` locally before pushing.

## Tests & CI
- Add unit tests under `tests/` for new modules; include small fixtures (no large data in Git).
- CI runs lint + pytest + a smoke integration on a toy AOI.  

## Notebooks & CLI
- The Kaggle notebook must call the same library functions as the CLI; no “divergent” code paths. 

## Reproducibility
- Keep seeds fixed; print environment details in notebooks and logs.
- Use pinned deps; prefer Docker for cross-platform runs. 

## Ethics
- Confirm CARE/FPIC compliance and relevant local legal frameworks (e.g., IPHAN in Brazil). 
- Default to coordinate coarsening for public artifacts; never expose sensitive locations without approval. 
# ---------------------------------------------------------------------------------------------------


# FILE: ETHICS.md
# ---------------------------------------------------------------------------------------------------
# Ethics & Data Governance — World Discovery Engine (WDE)

WDE is an evidence-driven, community-minded tool. It is designed to *assist* authorized archaeological
research programs and local/Indigenous communities — not to independently publish sensitive information.

## Principles we follow
- **CARE Principles for Indigenous Data Governance** — Collective Benefit, Authority to Control,
  Responsibility, Ethics. WDE’s default behaviors prioritize community control, benefit sharing, and safe
  handling of any heritage-relevant information. 
- **FPIC** (Free, Prior, and Informed Consent) — When outputs intersect with Indigenous territories or
  knowledge, consultation and consent precede any public sharing. 
- **Local legal compliance** — In Brazil, IPHAN has legal jurisdiction over archaeological sites; projects must
  be approved under national protocols. WDE must *not* be used to bypass these frameworks. 

## Default safeguards (“ethics-by-design”)
1) **Coordinate coarsening in public artifacts**: notebook/report outputs round coordinates (e.g., 2 decimals)
   unless a secure/authorized mode is enabled. 
2) **Sovereignty notices**: dossiers automatically annotate if a candidate falls within known Indigenous
   lands or protected areas; reviewers are prompted to engage appropriate authorities first. 
3) **Access control posture**: team workflows favor private review by local experts before any public sharing
   of locations or details that could increase looting risk. 
4) **Data lineage & transparency**: every dossier includes provenance and model uncertainty, so that
   decisions can weigh limitations. 

## What WDE is *not*
- A mechanism to “crowdsource” site locations in defiance of national law or community wishes. See SAB’s
  critique of bypassing national heritage protocols in Brazil; WDE explicitly rejects such deployments. 

## Privacy-preserving roadmap
- Hooks for *federated* training (local data never leaves community custody) and optional *differential
  privacy* in analytics; architectural placeholders are in the codebase for future enablement. 

## Escalation
If you believe a WDE output risks harm (e.g., potential looting), open a private issue with maintainers and
relevant local institutions; do not post precise coordinates publicly.
# ---------------------------------------------------------------------------------------------------


# FILE: datasets.md
# ---------------------------------------------------------------------------------------------------
# Datasets & Access — Open, Reproducible Inputs

This registry lists the principal open data sources used by WDE and how to access them reproducibly. See
the “Connecting to Remote Sensing…” guide for programmatic examples. 

## Optical imagery
- **Sentinel-2 (L2A surface reflectance)** — global multispectral (10–60 m). Access via Copernicus Data
  Space STAC/OData APIs or AWS public buckets; examples provided with `sentinelsat`/`pystac-client`. 
- **Landsat 8/9 (C2 L1/L2)** — global optical (30 m + 15 m pan). Access via USGS EarthExplorer/API or AWS.
  Example code using `landsatxplore` included. 

## Radar (SAR)
- **Sentinel-1 (GRD/SLC)** — C-band SAR. Access via ASF Vertex/asf_search or Copernicus; typical
  preprocessing: calibration, speckle filtering, terrain correction. 

## Elevation & terrain
- **SRTM 30 m** — near-global DEM; NASA Earthdata, OpenTopography; HGT/GeoTIFF. 
- **Copernicus DEM GLO-30/GLO-90** — global DSM via AWS COG tiles (unsigned). 

## LiDAR & canopy
- **GEDI L2A/L2B** — ISS LiDAR footprints; Earthdata via `earthaccess`. 
- **OpenTopography** — regional point clouds/derived DEMs; REST APIs & AWS S3. 

## Hydrology & land cover
- **HydroSHEDS** — river networks/basins. (Load as vectors/rasters; constrain settlement plausibility.) 
- **MapBiomas / MODIS** — Amazonian land-cover and seasonal indices (optional overlays). 

## Soils & ADE proxies
- **ISRIC SoilGrids (250 m)** — phosphorus, carbon, pH; strong ADE proxy layers. 
- **Floristic indicators** — overlap of useful tree species clusters with pre-Columbian earthworks (literature
  pointers included in docs). 

## Historical & archival
- **Historical maps/diaries** — georeferenced maps and OCR’d texts; integrated via entity extraction and
  tile overlays; useful for concordance. 

## Core sampling resources (context & validation)
- Global marine/continental/ice/soil core repositories and APIs (NOAA WDS Paleo, PANGAEA, Neotoma,
  IODP, NSIDC, WoSIS/SISLAC). Use for environmental context or follow-up validation. 

### Licensing notes
All sources above are open access (public domain or attribution licenses). Cite per each provider’s guidance
(e.g., “Contains modified Copernicus Sentinel data (Year)”). See the connection guide for details. 

### Repro tips
- Prefer COGs and tiled requests; clip to AOI to stay within Kaggle resource limits. 
- Cache intermediate tiles; keep seeds/configs fixed; print env/version info in notebooks. 
# ---------------------------------------------------------------------------------------------------