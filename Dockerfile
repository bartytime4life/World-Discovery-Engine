# FILE: Dockerfile
# -------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — Reproducible Geo/ML Runtime (CPU)
#
# Why micromamba?
#   Geo stacks (GDAL/PROJ/GEOS) are notoriously brittle with wheels-only installs.
#   Using conda-forge ensures ABI-compatible native libs for rasterio/fiona/shapely/geopandas.
#
# What you get:
#   - Python 3.12 on conda-forge (CPU)
#   - GDAL / PROJ / GEOS and friends
#   - Poetry-managed project install (no nested virtualenv; installs into the conda env)
#   - Non-root user
#
# Build:
#   docker build -t wde:cpu .
#
# Run (interactive shell):
#   docker run --rm -it -v $PWD:/app wde:cpu bash
#
# Run CLI (example):
#   docker run --rm -v $PWD:/app wde:cpu wde selftest
#   docker run --rm -v $PWD:/app wde:cpu wde report -c configs/default.yaml \
#       --in artifacts/verified/verified_candidates.geojson \
#       --out artifacts/dossiers --mask-precision 2 --public
# -------------------------------------------------------------------------------------------------

FROM mambaorg/micromamba:1.5.8-bookworm

# Micromamba layer settings
ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --------------------------------------------------------------------------------------
# System deps (lightweight) — curl/ca-certificates for Poetry download, git optional
# --------------------------------------------------------------------------------------
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash curl ca-certificates git tini \
 && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------------------
# Create non-root user
# --------------------------------------------------------------------------------------
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} app && useradd -m -u ${UID} -g app -s /bin/bash app
WORKDIR /app
RUN chown -R app:app /app
USER app

# --------------------------------------------------------------------------------------
# Conda-forge geo/ML base (CPU)
#   NOTE: micromamba is already on PATH; this installs into the "base" env.
# --------------------------------------------------------------------------------------
# Core scientific stack + geospatial libs with ABI-compatible native deps
RUN micromamba install -y -n base -c conda-forge \
    python=3.12 \
    gdal proj geos geotiff libspatialindex \
    rasterio fiona shapely pyproj rtree geopandas \
    numpy pandas scipy \
    pip \
 && micromamba clean -a -y

# Helpful environment hints (conda takes care of these paths internally)
ENV GDAL_DISABLE_READDIR_ON_OPEN=TRUE \
    OMP_NUM_THREADS=1 \
    PROJ_NETWORK=ON

# --------------------------------------------------------------------------------------
# Poetry (no nested virtualenv — install into conda env)
# --------------------------------------------------------------------------------------
ENV POETRY_VERSION=1.8.3
RUN python -m pip install --upgrade pip && \
    python -m pip install "poetry==${POETRY_VERSION}" && \
    poetry config virtualenvs.create false

# --------------------------------------------------------------------------------------
# Layer-cached install: copy manifests first, then install
# --------------------------------------------------------------------------------------
# Copy only files needed to resolve deps; add poetry.lock if present for reproducibility
COPY --chown=app:app pyproject.toml README.md LICENSE ./
# If you maintain a lock file, uncomment the next line:
# COPY --chown=app:app poetry.lock ./

# Install project (main deps only; no dev extras inside image)
RUN poetry install --only main --no-root

# --------------------------------------------------------------------------------------
# Copy the rest of the repo
# --------------------------------------------------------------------------------------
COPY --chown=app:app . .

# (Optional) install extras inside the image if you intend to run notebooks/plots in-container:
# RUN poetry install --with geo,ml,viz,notebook

# --------------------------------------------------------------------------------------
# Runtime: use tini as init; expose CLI entrypoint installed by Poetry
#   - If your console script is "wde" (from pyproject [tool.poetry.scripts]),
#     we can invoke it directly without 'poetry run'.
# --------------------------------------------------------------------------------------
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["wde", "--help"]