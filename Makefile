# FILE: Makefile
# -------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — Project Makefile
#
# Key goals:
#  - One-command setup with Poetry + pre-commit
#  - Reproducible runs via Typer CLI (world_engine.cli:app) and DVC stages
#  - Kaggle-friendly lock exports (requirements*.txt)
#  - Supply-chain checks (lint, fmt, audit), clean targets
#
# Usage:
#   make help            # list targets
#   make setup           # install deps (+dev extras) & pre-commit hooks
#   make run             # selftest → ingest → scan → evaluate → verify → report (CLI)
#   make dvc-repro       # run full DVC pipeline
#   make export-locks    # export requirements*.txt for Kaggle
#   make kaggle-zip      # build a Kaggle-ready bundle under artifacts/
#   make lint fmt test audit  # hygiene & tests
# -------------------------------------------------------------------------------------------------

SHELL := /bin/bash -eo pipefail
MAKEFLAGS += --warn-undefined-variables
.ONESHELL:

# ---------- Variables ----------
PY        := poetry run
DATE      := $(shell date +%Y%m%d)
ARTIFACTS := artifacts
NB_DIR    := notebooks
CFG       := configs/default.yaml

# ---------- Phony ----------
.PHONY: help setup setup-min fmt lint test audit clean clean-artifacts clean-dvc \
        selftest ingest scan evaluate verify report run dvc-repro dvc-status \
        export-locks kaggle-export kaggle-zip package data-pull

# ---------- Help ----------
help:
	@echo "WDE Make targets"
	@echo "  setup           : Install deps (+dev, geo, ml, viz, notebook) & pre-commit hooks"
	@echo "  setup-min       : Minimal deps (no extras) for CI or slim envs"
	@echo "  fmt             : Format code (black, isort) and fix simple issues"
	@echo "  lint            : Run pre-commit (ruff/black/isort/bandit/etc) on all files"
	@echo "  test            : Run pytest"
	@echo "  audit           : Supply-chain audit (pip-audit), report only"
	@echo "  selftest        : CLI self-test"
	@echo "  ingest          : Fetch & tile AOI, assemble raw overlay stack"
	@echo "  scan            : Coarse anomaly detection"
	@echo "  evaluate        : Mid-scale evaluation (NDVI/EVI, geomorph/hydro, overlays)"
	@echo "  verify          : Multi-proof fusion (ADE fingerprints, PAG, Bayesian GNN)"
	@echo "  report          : Generate candidate dossiers"
	@echo "  run             : End-to-end (selftest → ingest → scan → evaluate → verify → report)"
	@echo "  dvc-repro       : Reproduce full DVC pipeline"
	@echo "  dvc-status      : Show DVC status"
	@echo "  export-locks    : Export requirements*.txt for Kaggle"
	@echo "  kaggle-export   : (Placeholder) Render notebook(s) from templates"
	@echo "  kaggle-zip      : Build Kaggle bundle zip under artifacts/"
	@echo "  package         : Tarball of results & docs under artifacts/"
	@echo "  data-pull       : Placeholder for data pulls (documented in datasets.md)"
	@echo "  clean           : Remove caches/__pycache__/logs"
	@echo "  clean-artifacts : Remove artifacts/"
	@echo "  clean-dvc       : Remove DVC caches/locks (careful)"

# ---------- Setup ----------
setup:
	@echo "[WDE] Installing deps (dev, geo, ml, viz, notebook) & pre-commit hooks"
	poetry install --with dev,geo,ml,viz,notebook
	$(PY) pre-commit install

setup-min:
	@echo "[WDE] Installing minimal deps (no extras)"
	poetry install --only main

# ---------- Quality ----------
fmt:
	@echo "[WDE] Formatting code"
	$(PY) isort world_engine tests
	$(PY) black world_engine tests

lint:
	@echo "[WDE] Linting with pre-commit"
	$(PY) pre-commit run --all-files || true

test:
	@echo "[WDE] Running tests"
	$(PY) pytest -q

audit:
	@echo "[WDE] pip-audit (advisory only)"
	# Prefer a Poetry virtualenv; fallback to pip-audit via python -m if needed
	$(PY) pip-audit || true

# ---------- CLI pipeline ----------
selftest:
	@echo "[WDE] Selftest"
	$(PY) wde selftest

ingest:
	@echo "[WDE] Ingest → $(CFG)"
	$(PY) wde ingest --config $(CFG) --out data/raw

scan:
	@echo "[WDE] Scan"
	$(PY) wde scan --config $(CFG) --in data/raw --out artifacts/candidates

evaluate:
	@echo "[WDE] Evaluate"
	$(PY) wde evaluate --config $(CFG) --in artifacts/candidates --out artifacts/evaluated

verify:
	@echo "[WDE] Verify"
	$(PY) wde verify --config $(CFG) --in artifacts/evaluated --out artifacts/verified

report:
	@echo "[WDE] Report"
	$(PY) wde report --config $(CFG) --in artifacts/verified --out artifacts/dossiers

run: selftest ingest scan evaluate verify report
	@echo "[WDE] Done."

# ---------- DVC ----------
dvc-repro:
	@echo "[WDE] DVC repro"
	dvc repro

dvc-status:
	dvc status -c

# ---------- Data ----------
data-pull:
	@echo "[WDE] placeholder: implement data pulls in world_engine/ingest.py and document in datasets.md"

# ---------- Kaggle ----------
export-locks:
	@echo "[WDE] Exporting requirements*.txt for Kaggle"
	mkdir -p $(ARTIFACTS)
	poetry export -f requirements.txt --without-hashes -o requirements.txt
	poetry export -f requirements.txt --without-hashes --with geo -o requirements-geo.txt
	poetry export -f requirements.txt --without-hashes --with ml -o requirements-ml.txt
	poetry export -f requirements.txt --without-hashes --with viz -o requirements-viz.txt
	poetry export -f requirements.txt --without-hashes --with notebook -o requirements-notebook.txt

kaggle-export:
	@echo "[WDE] (Placeholder) Render/export Kaggle notebooks"
	mkdir -p $(NB_DIR)
	@# Example if you maintain a template notebook:
	@# $(PY) jupyter nbconvert --to notebook --execute templates/ade_discovery_pipeline.ipynb \
	@#   --output $(NB_DIR)/ade_discovery_pipeline.ipynb

kaggle-zip: export-locks
	@echo "[WDE] Building Kaggle bundle zip"
	mkdir -p $(ARTIFACTS)
	zip -r $(ARTIFACTS)/wde_kaggle_$(DATE).zip \
	  $(NB_DIR)/*.ipynb \
	  requirements.txt requirements-geo.txt requirements-ml.txt requirements-viz.txt requirements-notebook.txt \
	  README.md LICENSE datasets.md ETHICS.md SECURITY.md \
	  configs/default.yaml world_engine

# ---------- Packaging ----------
package:
	@echo "[WDE] Packaging release artifacts"
	mkdir -p $(ARTIFACTS)
	tar -czf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz \
	  README.md LICENSE SECURITY.md ETHICS.md datasets.md \
	  configs default.yaml 2>/dev/null || true
	@echo "[WDE] If DVC outputs exist, include them as needed:"
	@echo "      tar -rzf $(ARTIFACTS)/wde_$(DATE)_bundle.tgz artifacts/verified artifacts/dossiers"

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
	@echo "[WDE] DVC cache cleared (ensure remote is backed up if needed)"