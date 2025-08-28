# FILE: Makefile
# -------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — Project Makefile (Upgraded)
#
# Goals:
#  - One-command setup with Poetry + pre-commit
#  - Deterministic, provenance-rich runs (manifest with run/config hashes + seeds)
#  - Typer CLI pipeline: selftest → ingest → scan → evaluate → verify → report
#  - DVC orchestration, Kaggle-friendly lock exports, optional nb execution
#  - Ethics guardrails hook + supply-chain hygiene (lint/fmt/audit)
#  - Clean packaging with optional artifact inclusion
#
# Usage:
#   make help
#   make setup            # install deps (+dev extras) & pre-commit hooks
#   make run              # selftest → ingest → scan → evaluate → verify → report (+ manifest)
#   make dvc-repro        # run full DVC pipeline
#   make export-locks     # export requirements*.txt for Kaggle
#   make kaggle-zip       # build Kaggle bundle under artifacts/
#   make lint fmt test ci audit  # hygiene & tests
#   make manifest         # write artifacts/metrics/run_manifest.json
#   make package          # tarball of results & docs under artifacts/
# -------------------------------------------------------------------------------------------------

SHELL      := /bin/bash -eo pipefail
MAKEFLAGS  += --warn-undefined-variables
.ONESHELL:

# ---------- Variables ----------
PY          := poetry run
DATE        := $(shell date -u +%Y%m%d)
TIME        := $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
ARTIFACTS   := artifacts
NB_DIR      := notebooks
CFG         := configs/default.yaml

# Derivable metadata (override via environment if needed)
GIT_SHA     := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
WDE_VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo v0.0.0)
RUN_HASH    := sha256:$(shell printf "%s|%s|%s\n" "$(WDE_VERSION)" "$(GIT_SHA)" "$(TIME)" | sha256sum | awk '{print $$1}')
CONFIG_HASH := $(shell test -f "$(CFG)" && sha256sum "$(CFG)" | awk '{print $$1}' || echo unknown)

# Seeds (can be overridden by CI/env)
SEED_NUMPY  ?= 42
SEED_RANDOM ?= 42
SEED_TORCH  ?= 42

# Optional AOI bbox for manifest (min_lat,min_lon,max_lat,max_lon)
WDE_AOI_BBOX ?= -3.5,-60.5,-3.4,-60.4

# Color helpers
C_GREEN := \033[32m
C_BLUE  := \033[34m
C_YELL  := \033[33m
C_RED   := \033[31m
C_RST   := \033[0m

# ---------- Phony ----------
.PHONY: help setup setup-min fmt lint test audit ci \
        selftest ingest scan evaluate verify report run \
        dvc-repro dvc-status data-pull \
        export-locks kaggle-export kaggle-run kaggle-zip \
        manifest ethics-check package \
        clean clean-artifacts clean-dvc version hash

# ---------- Help ----------
help:
	@echo "WDE Make targets"
	@echo "  setup           : Install deps (+dev, geo, ml, viz, notebook) & pre-commit hooks"
	@echo "  setup-min       : Minimal deps (no extras) for CI or slim envs"
	@echo "  fmt             : Format code (black, isort) and fix simple issues"
	@echo "  lint            : Run pre-commit (ruff/black/isort/bandit/etc) on all files"
	@echo "  test            : Run pytest"
	@echo "  audit           : Supply-chain audit (pip-audit), report only"
	@echo "  ci              : Convenience (fmt + lint + test + audit)"
	@echo "  selftest        : CLI self-test"
	@echo "  ingest          : Fetch & tile AOI, assemble raw overlay stack"
	@echo "  scan            : Coarse anomaly detection"
	@echo "  evaluate        : Mid-scale evaluation (NDVI/EVI, geomorph/hydro, overlays)"
	@echo "  verify          : Multi-proof fusion (ADE fingerprints, PAG, Bayesian GNN)"
	@echo "  report          : Generate candidate dossiers (then write manifest)"
	@echo "  run             : End-to-end (selftest → ingest → scan → evaluate → verify → report + manifest)"
	@echo "  dvc-repro       : Reproduce full DVC pipeline"
	@echo "  dvc-status      : Show DVC status"
	@echo "  export-locks    : Export requirements*.txt for Kaggle"
	@echo "  kaggle-export   : (Placeholder) Render notebook(s) from templates"
	@echo "  kaggle-run      : (Optional) Run a notebook with nbconvert (if template present)"
	@echo "  kaggle-zip      : Build Kaggle bundle zip under artifacts/"
	@echo "  ethics-check    : Optional ethics guardrails dry-run (no-op if not implemented)"
	@echo "  manifest        : Write artifacts/metrics/run_manifest.json (provenance)"
	@echo "  package         : Tarball of results & docs under artifacts/"
	@echo "  version         : Print version metadata"
	@echo "  hash            : Print run/config hashes"
	@echo "  clean           : Remove caches/__pycache__/logs"
	@echo "  clean-artifacts : Remove artifacts/"
	@echo "  clean-dvc       : Remove DVC caches/locks (careful)"

# ---------- Setup ----------
setup:
	@echo "$(C_BLUE)[WDE] Installing deps (dev, geo, ml, viz, notebook) & pre-commit hooks$(C_RST)"
	poetry install --with dev,geo,ml,viz,notebook
	$(PY) pre-commit install

setup-min:
	@echo "$(C_BLUE)[WDE] Installing minimal deps (no extras)$(C_RST)"
	poetry install --only main

# ---------- Quality ----------
fmt:
	@echo "$(C_BLUE)[WDE] Formatting code$(C_RST)"
	$(PY) isort world_engine tests
	$(PY) black world_engine tests

lint:
	@echo "$(C_BLUE)[WDE] Linting with pre-commit$(C_RST)"
	$(PY) pre-commit run --all-files || true

test:
	@echo "$(C_BLUE)[WDE] Running tests$(C_RST)"
	$(PY) pytest -q

audit:
	@echo "$(C_BLUE)[WDE] pip-audit (advisory only)$(C_RST)"
	$(PY) pip-audit || true

ci: fmt lint test audit

# ---------- CLI pipeline ----------
selftest:
	@echo "$(C_GREEN)[WDE] Selftest$(C_RST)"
	$(PY) wde selftest

ingest:
	@echo "$(C_GREEN)[WDE] Ingest → $(CFG)$(C_RST)"
	$(PY) wde ingest --config $(CFG) --out data/raw

scan:
	@echo "$(C_GREEN)[WDE] Scan$(C_RST)"
	$(PY) wde scan --config $(CFG) --in data/raw --out artifacts/candidates

evaluate:
	@echo "$(C_GREEN)[WDE] Evaluate$(C_RST)"
	$(PY) wde evaluate --config $(CFG) --in artifacts/candidates --out artifacts/evaluated

verify:
	@echo "$(C_GREEN)[WDE] Verify$(C_RST)"
	$(PY) wde verify --config $(CFG) --in artifacts/evaluated --out artifacts/verified

report:
	@echo "$(C_GREEN)[WDE] Report$(C_RST)"
	$(PY) wde report --config $(CFG) --in artifacts/verified --out artifacts/dossiers
	@$(MAKE) --no-print-directory manifest

run: selftest ingest scan evaluate verify report
	@echo "$(C_GREEN)[WDE] Done.$(C_RST)"

# ---------- DVC ----------
dvc-repro:
	@echo "$(C_BLUE)[WDE] DVC repro$(C_RST)"
	dvc repro

dvc-status:
	dvc status -c

# ---------- Data ----------
data-pull:
	@echo "$(C_YELL)[WDE] placeholder: implement data pulls in world_engine/ingest.py and document in datasets.md$(C_RST)"

# ---------- Kaggle ----------
export-locks:
	@echo "$(C_BLUE)[WDE] Exporting requirements*.txt for Kaggle$(C_RST)"
	mkdir -p $(ARTIFACTS)
	poetry export -f requirements.txt --without-hashes -o requirements.txt
	poetry export -f requirements.txt --without-hashes --with geo -o requirements-geo.txt
	poetry export -f requirements.txt --without-hashes --with ml -o requirements-ml.txt
	poetry export -f requirements.txt --without-hashes --with viz -o requirements-viz.txt
	poetry export -f requirements.txt --without-hashes --with notebook -o requirements-notebook.txt

kaggle-export:
	@echo "$(C_BLUE)[WDE] (Placeholder) Render/export Kaggle notebooks$(C_RST)"
	mkdir -p $(NB_DIR)
	@# Example if you maintain a template notebook:
	@# $(PY) jupyter nbconvert --to notebook --execute templates/ade_discovery_pipeline.ipynb \
	@#   --output $(NB_DIR)/ade_discovery_pipeline.ipynb

kaggle-run:
	@echo "$(C_BLUE)[WDE] (Optional) Execute a notebook via nbconvert (set NB?=path/to.ipynb)$(C_RST)"
	@test -n "$(NB)" || (echo "$(C_RED)NB variable not set (e.g., make kaggle-run NB=$(NB_DIR)/ade_discovery_pipeline.ipynb)$(C_RST)"; exit 1)
	$(PY) jupyter nbconvert --to notebook --execute "$(NB)" --output "$(NB:.ipynb=.executed.ipynb)"

kaggle-zip: export-locks
	@echo "$(C_BLUE)[WDE] Building Kaggle bundle zip$(C_RST)"
	mkdir -p $(ARTIFACTS)
	zip -r $(ARTIFACTS)/wde_kaggle_$(DATE).zip \
	  $(NB_DIR)/*.ipynb \
	  requirements.txt requirements-geo.txt requirements-ml.txt requirements-viz.txt requirements-notebook.txt \
	  README.md LICENSE datasets.md ETHICS.md SECURITY.md \
	  $(CFG) world_engine

# ---------- Ethics (optional hook) ----------
ethics-check:
	@echo "$(C_BLUE)[WDE] Ethics guardrails dry-run (no-op if not implemented)$(C_RST)"
	@# If you provide a checker (e.g., world_engine/ethics_guardrails.py with a CLI), call it here:
	@# $(PY) python -m world_engine.ethics_guardrails --check artifacts/dossiers || true
	@true

# ---------- Manifest (provenance) ----------
manifest:
	@echo "$(C_BLUE)[WDE] Generating run manifest$(C_RST)"
	@mkdir -p $(ARTIFACTS)/metrics
	@$(PY) python - << 'PY'
import os, json, time, hashlib
from pathlib import Path

WDE_VERSION = os.environ.get("WDE_VERSION", "$(WDE_VERSION)")
RUN_HASH    = os.environ.get("WDE_RUN_HASH", "$(RUN_HASH)")
TIMESTAMP   = os.environ.get("WDE_TIMESTAMP", "$(TIME)")
CFG         = os.environ.get("WDE_CONFIG", "$(CFG)")
CFG_HASH    = os.environ.get("WDE_CONFIG_HASH", "$(CONFIG_HASH)")
AOI         = os.environ.get("WDE_AOI_BBOX", "$(WDE_AOI_BBOX)")
SEED_NUMPY  = int(os.environ.get("SEED_NUMPY", "$(SEED_NUMPY)"))
SEED_RANDOM = int(os.environ.get("SEED_RANDOM", "$(SEED_RANDOM)"))
SEED_TORCH  = int(os.environ.get("SEED_TORCH", "$(SEED_TORCH)"))

manifest = {
  "pipeline_version": WDE_VERSION,
  "run_hash": RUN_HASH,
  "timestamp_utc": TIMESTAMP,
  "config_path": CFG,
  "config_hash": CFG_HASH if CFG_HASH.startswith("sha") else f"sha256:{CFG_HASH}",
  "seed_numpy": SEED_NUMPY,
  "seed_random": SEED_RANDOM,
  "seed_torch": SEED_TORCH,
  "aoi_bbox": [float(x) for x in AOI.split(",")]
}
Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
with open("artifacts/metrics/run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Wrote artifacts/metrics/run_manifest.json")
PY

# ---------- Packaging ----------
package:
	@echo "$(C_BLUE)[WDE] Packaging release artifacts$(C_RST)"
	mkdir -p $(ARTIFACTS)
	tar -czf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz \
	  README.md LICENSE SECURITY.md ETHICS.md datasets.md \
	  configs \
	  world_engine \
	  notebooks 2>/dev/null || true
	@# Optionally append artifacts if present:
	@if [ -d artifacts/dossiers ] || [ -d artifacts/metrics ] || [ -d artifacts/verified ]; then \
	  echo "$(C_YELL)[WDE] Appending artifacts (dossiers/metrics/verified)$(C_RST)"; \
	  tar -rzf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz artifacts/dossiers 2>/dev/null || true; \
	  tar -rzf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz artifacts/metrics 2>/dev/null || true; \
	  tar -rzf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz artifacts/verified 2>/dev/null || true; \
	fi
	@echo "$(C_GREEN)[WDE] Package: $(ARTIFACTS)/wde_$(DATE)_bundle.tgz$(C_RST)"

# ---------- Metadata ----------
version:
	@echo "WDE_VERSION=$(WDE_VERSION)"
	@echo "GIT_SHA=$(GIT_SHA)"

hash:
	@echo "RUN_HASH=$(RUN_HASH)"
	@echo "CONFIG_HASH=$(CONFIG_HASH)"

# ---------- Clean ----------
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache __pycache__ **/__pycache__ \
	       .coverage coverage.xml .cache .tox *.log
	find . -name '*.pyc' -delete
	find . -name '*.pyo' -delete

clean-artifacts:
	rm -rf $(ARTIFACTS)/*

clean-dvc:
	rm -rf .dvc/tmp .dvc/cache .dvc/lock 2>/dev/null || true
	@echo "$(C_YELL)[WDE] DVC cache cleared (ensure remote is backed up if needed)$(C_RST)"
