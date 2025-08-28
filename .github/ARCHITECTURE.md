# 🏛️ `.github/ARCHITECTURE.md` — GitHub Workflows & Governance Architecture

This document defines the **architecture of GitHub-native automation** for the **World Discovery Engine (WDE)**.
It ensures **reproducibility, scientific rigor, ethical compliance, and Kaggle-ready deployment**.

---

## 🎯 Purpose

* **CI/CD Backbone**: Validate code, configs, and datasets at every commit.
* **Reproducibility**: Guarantee that results can be regenerated across environments (Kaggle, Docker, local).
* **Security & Ethics**: Scan dependencies, enforce CARE/FAIR principles, and redact sensitive data.
* **Governance**: Provide structured issue/PR templates, branch protection, and contributor guidance.

---

## 📂 Directory Layout

```
.github/
├── README.md                # Overview of configs & workflows
├── ARCHITECTURE.md          # (this file) Governance & CI/CD architecture
├── workflows/               # GitHub Actions (automation layer)
│   ├── ci.yml               # Unit + integration tests
│   ├── lint.yml             # Style, static analysis, notebook lint
│   ├── security.yml         # Dependency + secret scanning
│   ├── submission.yml       # Kaggle submission automation
│   ├── release.yml          # GitHub release bundling + Docker push
│   └── docker.yml           # (optional) Docker image CI/CD
├── ISSUE_TEMPLATE/          # Issue templates (structured YAML forms)
│   ├── bug_report.yml
│   ├── performance_issue.yml
│   ├── security_report.yml
│   ├── config_update.yml
│   └── task_tracking.yml
├── PULL_REQUEST_TEMPLATE.md # PR rigor & reproducibility checklist
└── CODEOWNERS               # Ownership & review rules
```

---

## ⚙️ Workflow Architecture

### 1. **Lint & Static Analysis** (`lint.yml`)

* Runs `black`, `isort`, `ruff`, `mypy`, `yamllint`, `nbqa`.
* Skips large data paths (`data/`, `outputs/`, `logs/`).
* Two-pass strategy: autofix → enforce clean.

### 2. **Continuous Integration** (`ci.yml`)

* **Unit tests**: pytest modules in `tests/`.
* **Integration tests**: run WDE pipeline on a demo AOI (small tile set).
* Produces artifacts: coverage reports, anomaly masks, candidate dossiers.

### 3. **Security & Ethics** (`security.yml`)

* Dependency scans (`pip-audit`), license checks.
* Secret scanning & coordinate redaction (no raw lat/lon leaks).
* CARE principles enforced via “ethical mode” flags.

### 4. **Submission Automation** (`submission.yml`)

* Dry-run by default; opt-in submit via `workflow_dispatch`.
* Guardrails: only runs on `main`, requires Kaggle API secrets.
* Pipeline: `predict → validate → package → (optional) submit`.
* Produces: `submission.csv`, `manifest.json`, `outputs/` dossiers.

### 5. **Release Automation** (`release.yml`)

* Triggered by tags or manual dispatch.
* Bundles artifacts into `.tar.gz` + `SHA256SUMS.txt`.
* Publishes to GitHub Releases; optionally pushes Docker images.

### 6. **Docker Build** (`docker.yml`)

* Builds WDE runtime (Python, GDAL, PDAL, PyTorch, rasterio).
* Ensures parity between local/Kaggle/Docker runs.

---

## 🔄 Workflow Graph

```mermaid
flowchart TB
  A[Push / PR]:::evt --> B[lint.yml]
  A --> C[ci.yml]
  A --> D[security.yml]
  B --> E{All checks pass?}
  C --> E
  D --> E
  E -- yes --> F[submission.yml<br/>Predict → Validate → Package<br/>(dry-run default)]
  E -- yes --> R[release.yml<br/>Bundle → Checksums → Release]
  E -- no --> X[Fail fast<br/>Block merge]:::fail
  F --> H{Branch=main<br/>Secrets present<br/>submit=yes?}
  H -- no --> I[Bundle only]
  H -- yes --> J[Kaggle Submit]:::ship
  R --> M[[GitHub Release:<br/>tar.gz + SHA256SUMS + Docker]]
  I --> M
  J --> L[[Kaggle Leaderboard]]
  classDef evt fill:#eef,stroke:#88f,color:#000;
  classDef fail fill:#fee,stroke:#f55,color:#600;
  classDef ship fill:#efe,stroke:#5a5,color:#060;
```

---

## 📜 Governance Rules

* **Branch Protection**: PRs must pass lint, CI, and security before merge.
* **CODEOWNERS**: WDE core modules require archaeologist + ML lead review.
* **PR Template**: Forces contributors to justify scientific validity, reproducibility, and ethical compliance.
* **Issue Templates**: All bugs, tasks, and updates use structured forms for traceability.

---

## 🔒 Ethics & Reproducibility

* **CARE-compliant outputs**: coordinate rounding, sovereignty notices.
* **CausalOps Lifecycle**: Arrange → Validate → Test → Publish → Monitor.
* **Audit Trail**: CI logs + manifests provide hash-traceable provenance for every run.
* **Kaggle Compliance**: Notebook runs entirely on open datasets; Docker parity ensures identical results locally.

---

## ✅ Contribution Flow

1. Fork & branch (`feature/<name>`).
2. Run `pytest` + `make lint`.
3. Submit PR with filled template.
4. CI/CD validates (lint → tests → security).
5. Reviewer sign-off (scientific + reproducibility).
6. Merge → Release automation → Kaggle-ready artifacts.

---

📖 **References**

* WDE Architecture Specification
* WDE Repository Structure
* ADE Discovery Pipeline
* Kaggle Platform Technical Guide
* Enriching WDE for Archaeology & Earth Systems

---
