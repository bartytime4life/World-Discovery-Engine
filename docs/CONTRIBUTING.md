# 🤝 Contributing Guide — World Discovery Engine (WDE)

Welcome to the **World Discovery Engine (WDE)** project!
This guide explains how to contribute code, data, or documentation while maintaining **scientific rigor, reproducibility, and ethical compliance**.

---

## 📌 Principles

1. **Reproducibility First** — Every contribution must be reproducible on Kaggle and via CLI.
2. **Open Data Only** — All datasets must be CC-0/CC-BY or user-supplied with explicit consent.
3. **Ethics-by-Design** — Follow CARE Principles, FPIC/IPHAN compliance, and sovereignty rules.
4. **Modularity** — Code lives in `world_engine/` by pipeline stage; avoid one-off notebook hacks.
5. **Transparency** — Every config, dataset, and assumption must be documented.

---

## 🛠 Development Workflow

### 1. Fork & Branch

* Fork the repo, clone locally.
* Use **feature branches**:

  ```
  git checkout -b feature/<short-description>
  ```

### 2. Environment Setup

* Use Python 3.10/3.11 (Kaggle-compatible).
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Optional: build the Docker image for reproducibility:

  ```bash
  docker build -t wde:dev .
  ```

### 3. Running the Pipeline

* **Kaggle Notebook**: `notebooks/ade_discovery_pipeline.ipynb`.
* **CLI**:

  ```bash
  wde full-run --config configs/default.yaml
  ```

### 4. Tests & Lint

* Run unit tests before commit:

  ```bash
  pytest -v
  ```
* Lint with pre-commit hooks:

  ```bash
  pre-commit run --all-files
  ```

### 5. Commit & PR

* Follow commit style:

  ```
  [component]: [short change summary]
  ```

  Example:

  ```
  ingest: add DEM hillshade normalization
  detect: improve anomaly score aggregation
  docs: update dataset registry with MapBiomas v7
  ```
* Open a PR and fill the PR template completely.

---

## 📂 Where to Put Things

* **Core code** → `world_engine/`

  * `ingest.py` (data tiling, ingestion)
  * `detect.py` (coarse anomaly scan)
  * `evaluate.py` (NDVI/EVI, LiDAR, mid-scale)
  * `verify.py` (fusion, GNN, PAG, ADE fingerprints)
  * `report.py` (site dossiers)

* **Configs** → `configs/` (YAML for AOIs, datasets, models).

* **Notebooks** → `notebooks/` (only Kaggle pipeline + exploration).

* **Docs** → `docs/` (architecture, datasets, ethics, contributing).

* **Tests** → `tests/` (unit + integration; see below).

---

## ✅ Testing Standards

* **Unit tests** — for every function (tile creation, NDVI calculation, anomaly detection).
* **Integration tests** — pipeline on a dummy AOI with sample data.
* **CI** — PRs must pass:

  * Lint
  * Unit tests
  * Kaggle notebook execution (light mode)
  * Main branch CI (heavy mode: ADE fingerprints, PAG, uncertainty, SSIM)

---

## 📊 Data Contributions

* Update `docs/datasets.md` when adding a new dataset.
* Include:

  * Source link
  * License
  * Access method (API/Kaggle dataset)
  * Example usage snippet

**Never commit raw data**. Use DVC pointers or Kaggle datasets.

---

## ⚖️ Ethics Checklist

Before merging, confirm:

* [ ] Data is open-access & licensed correctly.
* [ ] Indigenous sovereignty respected (flag if in protected lands).
* [ ] Candidate outputs include sovereignty notice if required.
* [ ] No sensitive coordinates published without review.

---

## 📢 Communication

* Use GitHub Issues for bugs, features, and tasks.
* Tag issues with:

  * `pipeline` (core code)
  * `data` (datasets, ingestion)
  * `docs` (documentation)
  * `ethics` (compliance/sovereignty)
* Use Discussions for design debates.

---

## 🏆 Contributor Recognition

We follow the [All Contributors](https://allcontributors.org/) model: code, data, docs, design, and review contributions are all recognized.

---
