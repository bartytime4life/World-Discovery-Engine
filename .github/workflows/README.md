# Workflows Overview (WDE CI/CD)

This repo uses a **modular, defense-in-depth** CI/CD system. The diagram below shows how triggers fan out into lint/test/security, how PR gates enforce reproducibility & datasets governance, and how validated changes can be packaged/released or published to Pages / Kaggle.

> Tip: click the file names in the table to jump to their workflows.

---

## High-Level Flow

```mermaid
flowchart TB
  %% ===================== TRIGGERS =====================
  A([Push / PR / Dispatch]):::evt --> B[lint.yml<br/>Lint & Static Analysis]
  A --> C[ci.yml<br/>Unit & Integration Tests]
  A --> D[security.yml<br/>Vuln & Secret Scan]
  A --> T[PR Title Lint<br/>pr_title_lint.yml]
  A --> U[PR Labeler & Checklist Guard<br/>pr_label_guard.yml]
  A --> V[Datasets Registry Gate<br/>datasets_registry_gate.yml]
  A --> W[Validate New Dataset Issues<br/>issues_dataset_validate.yml]
  A --> N[Publish Notebooks (HTML)<br/>notebook_publish.yml]
  A --> Q[Kaggle Notebook Check<br/>kaggle_notebook_check.yml]

  %% ===================== CI FAN-IN =====================
  B --> E{All core checks<br/>passing?}:::gate
  C --> E
  D --> E
  T --> E
  U --> E
  V --> E
  Q --> E

  %% ===================== SUBMISSION / RELEASE =====================
  E -- yes --> F[submission.yml (manual or auto)<br/>Predict → Validate → Package<br/>(dry-run by default)]
  E -- yes --> R[release.yml (tag/dispatch)<br/>Mini E2E → Redaction → Bundle<br/>Build sdist/wheel + Docker + SBOM]
  E -- no --> X[Fail fast<br/>Block merge]:::fail

  %% ===================== ARTIFACTS FROM CI =====================
  C --> G[[CI Artifacts:<br/>• test reports<br/>• coverage<br/>• mini-AOI outputs]]
  B --> G
  D --> G
  Q --> G
  N --> Y[[Notebooks Site:<br/>HTML pages (Pages)]]

  %% ===================== SUBMISSION GUARDS =====================
  F --> H{Branch = main<br/>Secrets present<br/>submit=yes?}
  H -- no --> I[Create bundle only:<br/>submission.csv · manifest.json<br/>/outputs/ dossiers]
  H -- yes --> J[Kaggle Submit<br/>(CLI/API)]:::ship

  %% ===================== RELEASE OUTPUTS =====================
  R --> K[[Release Artifacts:<br/>• submission.csv<br/>• manifest.json<br/>• /outputs/ dossiers<br/>• sdist/wheel<br/>• checksums (SHA256)<br/>• Docker image (GHCR)<br/>• SBOMs]]
  I --> K
  J --> L[[Kaggle Leaderboard<br/>(public/private)]]

  %% ===================== STYLING =====================
  classDef evt fill:#eef,stroke:#88f,color:#000,stroke-width:1px;
  classDef fail fill:#fee,stroke:#f55,color:#600,stroke-width:1px;
  classDef ship fill:#efe,stroke:#5a5,color:#060,stroke-width:1px;
  classDef gate fill:#fff9e6,stroke:#e6b800,color:#3a2a00,stroke-width:1px;


⸻

Workflow Index

Area	Workflow	File	Purpose
Lint	Lint & Static Analysis	.github/workflows/lint.yml	ruff/black/isort/mypy; basic YAML/MD checks
Test	Unit & Integration Tests	.github/workflows/ci.yml	pytest (+mini AOI smoke) & coverage artifacts
Security	Vulnerability & Secret Scan	.github/workflows/security.yml	Trivy/OSV + secret scanner (fail-safe)
PR Quality	PR Title Lint (Conventional Commits)	.github/workflows/pr_title_lint.yml	Enforces type(scope?): subject & imperative mood
PR Governance	PR Labeler & Checklist Guard — Ultimate	.github/workflows/pr_label_guard.yml	Auto-labels by content/paths; enforces Ethics/Reproducibility/Datasets/Tests/CI Green checklist
Data Governance	Datasets Registry Gate — Ultimate	.github/workflows/datasets_registry_gate.yml	Requires datasets.md/registry update or no-datasets-needed label when data paths change
Issues Intake	Validate New Dataset Issues — Ultimate	.github/workflows/issues_dataset_validate.yml	Lints dataset intake issues (license/source/coverage/transforms/etc.)
Notebooks	Publish Notebooks (HTML) — Ultimate	.github/workflows/notebook_publish.yml	Convert .ipynb → HTML (optionally execute) and deploy to Pages
Kaggle	Kaggle Notebook Check (Ultimate)	.github/workflows/kaggle_notebook_check.yml	Execute ade_discovery_pipeline.ipynb with papermill, verify outputs & redaction guard
Submission	Submission	.github/workflows/submission.yml	Predict → Validate → Package (dry-run default; gated submit)
Release	Release & Supply-Chain (Ultimate)	.github/workflows/release.yml	Version guard, mini E2E, coordinate redaction, bundle, sdist/wheel, SBOM, GHCR image

Coordinate Redaction Guard: multiple workflows (Kaggle Check, Release) fail if high-precision coordinates leak into artifacts. Keep public outputs rounded/redacted.

⸻

Badges

Add these to your main README.md (adjust repo path):

![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)
![Lint](https://github.com/<org>/<repo>/actions/workflows/lint.yml/badge.svg)
![Security](https://github.com/<org>/<repo>/actions/workflows/security.yml/badge.svg)
![Kaggle Notebook Check](https://github.com/<org>/<repo>/actions/workflows/kaggle_notebook_check.yml/badge.svg)
![Publish Notebooks](https://github.com/<org>/<repo>/actions/workflows/notebook_publish.yml/badge.svg)
![Release](https://github.com/<org>/<repo>/actions/workflows/release.yml/badge.svg)


⸻

Gates & Policies
	•	Title Policy: PR titles must follow Conventional Commits. The linter gives a suggested fix in a sticky comment.
	•	Checklist Policy: PRs that affect code/config/docs must check: Ethics · Reproducibility · Datasets · Tests · CI Green.
	•	Datasets Registry: Changes in world_engine/ingest|evaluate|verify, configs/, data/, notebooks/, docs/ must update the registry (datasets.md / datasets/registry.yml) or add label no-datasets-needed.
	•	Notebook CI: The Kaggle notebook must execute in a Kaggle-like environment (fast/offline modes supported) and emit outputs/… artifacts. High-precision coords are blocked.

⸻

Quick Links
	•	Pages (Rendered Notebooks): https://<org>.github.io/<repo>/
	•	GHCR Images: ghcr.io/<org>/<repo>:latest
	•	Releases: See Code → Releases for bundled artifacts (CSV/manifest/dossiers/sdist/wheel/SBOM).

⸻

Local CI Tips
	•	Run linters & tests locally:

uv pip install -r requirements.txt  # or: pip install -r requirements.txt
ruff check . && black --check . && pytest -q


	•	Dry-run the submission pipeline:

gh workflow run submission.yml -f dry_run=true



⸻

This page documents the workflows under .github/workflows/. Keep it updated when adding or renaming pipelines.

