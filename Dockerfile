# FILE: Dockerfile
# -------------------------------------------------------------------------------------------------
# NOTE: This is a minimal, CPU-only base. Geo stacks are tricky; prefer mamba/conda for GDAL.
# For production, consider micromamba with an environment.yml that pins GDAL-compatible builds.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Basic build deps (adjust as needed for rasterio/geopandas wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install --upgrade pip && pip install "poetry>=1.8.3"

# Copy only manifests first (layer caching)
COPY pyproject.toml README.md LICENSE ./
RUN poetry config virtualenvs.in-project true && poetry install --only main --no-root

# Copy the rest (src when added)
COPY . .

# Dev extras optional in container
# RUN poetry install --with dev,geo,ml,viz,notebook

ENTRYPOINT ["poetry", "run", "python", "-m", "wde"]