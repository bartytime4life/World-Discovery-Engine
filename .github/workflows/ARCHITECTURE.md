
⸻

🌍 World Discovery Engine (WDE) — CI/CD & Workflow Architecture

/.github/workflows/ARCHITECTURE.md

⸻

0. Purpose

This document defines the GitHub Actions workflow architecture (.github/workflows/) that drives the WDE’s CI/CD pipeline.
It ensures that every code change, dataset update, or Kaggle notebook run is:
	•	Reproducible — deterministic environments, version-pinned dependencies, stable data hashes ￼
	•	Tested — unit + integration coverage of geospatial/ML pipeline ￼
	•	Validated — ADE fingerprint rules, causal plausibility graphs, and uncertainty quantification ￼
	•	Deployable — Kaggle-ready notebooks + Dockerized runtime ￼
	•	Ethical — CARE principles, Indigenous sovereignty, and heritage compliance ￼

⸻

1. Workflow Inventory

Workflow	File	Trigger	Purpose
Lint & Static Analysis	lint.yml	push / PR	Enforce code quality (ruff, black, mypy, yamllint, nbqa, prettier) ￼
Unit & Integration Tests	ci.yml	push / PR	Run pytest suite: ingest → detect → evaluate → verify → report ￼
Security Scan	security.yml	scheduled, PR	Dependency scan, secret leakage detection, CVE audit ￼
Submission Pipeline	submission.yml	workflow_dispatch	Kaggle run: predict → validate → package → (optional) submit ￼
Release Build	release.yml	tag push	Build Docker, archive outputs, publish candidate dossiers ￼


⸻

2. Workflow Topology

flowchart TB
  subgraph Dev [Developer Actions]
    A[Push / PR] --> B[lint.yml]
    A --> C[ci.yml]
    A --> D[security.yml]
  end

  subgraph Gate [Merge Gate]
    B --> E{All checks passing?}
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


⸻

3. Workflow Design Principles

3.1 Modularity

Each stage of the Discovery Funnel (ingest, detect, evaluate, verify, report) is isolated in its own module and workflow ￼ ￼.
This ensures unit-testable steps and Kaggle-friendly modularity.

3.2 Scientific Reproducibility
	•	Seeds, library versions, dataset hashes are pinned ￼
	•	DVC + Kaggle Datasets ensure stable inputs ￼
	•	Candidate dossiers (GeoJSON, ADE fingerprints, uncertainty graphs) are logged for each run ￼

3.3 Multi-Modal Verification

Workflows confirm anomalies using at least two modalities ￼:
	•	Optical + Radar (Sentinel-2 / Sentinel-1)
	•	DEM/LiDAR (SRTM, GEDI, OpenTopography)
	•	Soil & vegetation ADE proxies (SoilGrids, MapBiomas)
	•	Historical overlays (OCR diaries, georeferenced maps)
	•	Causal plausibility graphs + Bayesian GNN uncertainty ￼

3.4 Ethics & Governance
	•	Candidate reports auto-include CARE principle disclaimers ￼
	•	Sites intersecting Indigenous land trigger sovereignty notices ￼
	•	Coordinates are rounded / masked unless ethical flags are cleared ￼

⸻

4. Workflow-to-Pipeline Mapping

Workflow	WDE Stage(s)	Outputs
lint.yml	Pre-ingest	Code quality logs
ci.yml	ingest → detect → evaluate → verify → report	Test artifacts (/tests/artifacts/)
security.yml	Global	CVE scan logs, SBOM manifest
submission.yml	Full pipeline	submission.csv, manifest.json, candidate dossiers
release.yml	Full pipeline + docs	Docker image, archived outputs/, PDF/HTML reports


⸻

5. NASA-Grade Simulation Standards

Borrowed from NASA-STD-7009 credibility guidelines ￼ ￼:
	•	Verification — unit tests for each module (e.g., anomaly detector on synthetic input)
	•	Validation — cross-checks against known ADE sites & geoglyph datasets ￼
	•	Uncertainty Quantification — Bayesian GNN histograms, Monte Carlo dropout ￼
	•	Refutation Tests — counterfactual SSIM removal (remove vegetation signal, re-check anomaly) ￼

⸻

6. File Placement

.github/
 └── workflows/
      ├── README.md              # Diagram & quick overview
      ├── ARCHITECTURE.md        # This file (detailed workflow design)
      ├── lint.yml
      ├── ci.yml
      ├── security.yml
      ├── submission.yml
      └── release.yml


⸻