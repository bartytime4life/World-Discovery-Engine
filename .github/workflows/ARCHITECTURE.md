# 🌍 .github/workflows/ARCHITECTURE.md

World Discovery Engine (WDE) — CI/CD & Workflow Architecture

---

## 0. Purpose

This document defines the **architecture of GitHub Actions workflows** (`.github/workflows/`) supporting the WDE.
It ensures that every code change, dataset update, or Kaggle notebook run is:

* **Reproducible** (deterministic environments, version-pinned deps)
* **Tested** (unit + integration coverage of geospatial/ML pipeline)
* **Validated** (scientific credibility checks, ADE fingerprint rules, uncertainty estimates)
* **Deployable** (Kaggle-ready notebooks + Dockerized runtime)
* **Ethical** (CARE principles, indigenous data governance)

---

## 1. Workflow Inventory

| Workflow                     | File             | Trigger            | Purpose                                                               |
| ---------------------------- | ---------------- | ------------------ | --------------------------------------------------------------------- |
| **Lint & Static Analysis**   | `lint.yml`       | push / PR          | Enforce code quality (ruff, black, mypy, yamllint, nbqa)              |
| **Unit & Integration Tests** | `ci.yml`         | push / PR          | Run pytest suite: ingest → detect → evaluate → verify → report        |
| **Security Scan**            | `security.yml`   | scheduled, PR      | Dependency scan, secret leakage, CVE audit                            |
| **Submission Pipeline**      | `submission.yml` | workflow\_dispatch | Kaggle run: predict → validate → package → (optional) submit          |
| **Release Build**            | `release.yml`    | tag push           | Build Docker, archive outputs, update `outputs/` + candidate dossiers |

---

## 2. Workflow Topology

```mermaid
flowchart TB
  subgraph Dev [Developer Actions]
    A[Push / PR] --> B[lint.yml]
    A --> C[ci.yml]
    A --> D[security.yml]
  end

  subgraph Gate [Merge Gate]
    B --> E{Checks Passing?}
    C --> E
    D --> E
  end

  E -- no --> X[Fail fast<br/>Block merge]
  E -- yes --> F[submission.yml<br/>(manual dispatch)]

  subgraph Submission [Kaggle Pipeline]
    F --> G{Branch=main<br/>Secrets present<br/>submit=yes?}
    G -- no --> H[Bundle only<br/>submission.csv + manifest.json]
    G -- yes --> I[Kaggle Leaderboard Submission]
  end

  H --> J[[Release Artifacts<br/>dossiers, reports, CSVs]]
  I --> K[[Kaggle Leaderboard<br/>public/private split]]
```

---

## 3. Workflow Design Principles

### 3.1 Modularity

Each stage of the pipeline (ingest, detect, evaluate, verify, report) is **separated into modules** and tested independently.
This mirrors the WDE Discovery Funnel.

### 3.2 Scientific Reproducibility

* All runs pin **random seeds, library versions, and data hashes**
* DVC or Kaggle Datasets are used to ensure stable dataset access
* Outputs (GeoJSON, ADE fingerprints, candidate dossiers) are logged and versioned

### 3.3 Multi-Modal Verification

Workflows validate anomalies using:

* Optical + Radar (Sentinel-2 / Sentinel-1)
* DEM/LiDAR
* Soil & vegetation ADE proxies
* Historical overlays
* Causal plausibility graphs

### 3.4 Ethics & Governance

* Candidate reports auto-include CARE principle notices
* Sensitive AOIs trigger warnings (e.g., IPHAN in Brazil)
* Outputs redact precise coordinates unless ethical flags are cleared

---

## 4. Workflow-to-Pipeline Mapping

| Workflow       | WDE Stage(s)                                 | Outputs                                               |
| -------------- | -------------------------------------------- | ----------------------------------------------------- |
| lint.yml       | Pre-ingest                                   | Code quality logs                                     |
| ci.yml         | ingest → detect → evaluate → verify → report | Test artifacts (`/tests/artifacts/`)                  |
| security.yml   | Global                                       | CVE scan logs, dependency manifest                    |
| submission.yml | Full pipeline                                | `submission.csv`, `manifest.json`, candidate dossiers |
| release.yml    | Full pipeline + docs                         | Docker image, archived `outputs/`, PDF/HTML reports   |

---

## 5. NASA-Grade Simulation Standards

Borrowing from **NASA M\&S credibility guidelines**:

* **Verification**: each workflow stage is unit-tested (e.g., anomaly detector gets synthetic inputs → known outputs).
* **Validation**: results compared to external datasets (known ADE sites, geoglyphs).
* **Uncertainty Quantification**: B-GNN histograms + Monte Carlo dropout checks.
* **Refutation Tests**: counterfactual SSIM removal tests ensure robustness.

---

## 6. File Placement

```
.github/
 └── workflows/
      ├── README.md              # Diagram & overview
      ├── ARCHITECTURE.md        # This file (detailed workflow design)
      ├── lint.yml
      ├── ci.yml
      ├── security.yml
      ├── submission.yml
      └── release.yml
```
