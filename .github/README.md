# 📂 `.github/` — GitHub Configurations & Workflows

This directory contains all **GitHub-native configuration** files for the **World Discovery Engine (WDE)** repository. It governs **CI/CD pipelines, security scanning, issue/PR templates, and automation rules**. Together, these ensure the project remains **reproducible, auditable, and ethically compliant**.

---

## 📌 Purpose

* **Continuous Integration (CI/CD):** Run tests, linting, security scans, and submission validation on every commit/PR.
* **Workflow Automation:** Build & publish Docker images, check Kaggle submissions, and run reproducibility tests.
* **Community Standards:** Enforce structured issue templates, PR templates, and branch protection policies.
* **Security & Ethics:** Automate dependency scanning, artifact checks, and ensure compliance with CARE principles.

---

## 📂 Directory Layout

```
.github/
├── README.md                   # Overview of GitHub configs (this file)
├── workflows/                  # GitHub Actions (CI/CD, security, submission, linting)
│   ├── ci.yml                   # Unit & integration tests
│   ├── lint.yml                 # Code style (black, isort, ruff, mypy, yamllint)
│   ├── security.yml             # Dependency + secret scanning
│   ├── submission.yml           # Kaggle submission automation
│   └── docker.yml               # Optional: Docker image build & push
├── ISSUE_TEMPLATE/              # Structured issue templates
│   ├── bug_report.yml
│   ├── performance_issue.yml
│   ├── security_report.yml
│   ├── config_update.yml
│   └── task_tracking.yml
├── PULL_REQUEST_TEMPLATE.md     # PR standards (scientific rigor, reproducibility)
└── CODEOWNERS                   # Ownership and review requirements
```

---

## ⚙️ Workflows

* **`ci.yml`** — Runs pytest-based unit/integration tests on each PR/commit.
* **`lint.yml`** — Enforces style & static analysis (Python, YAML, Notebooks).
* **`security.yml`** — Triggers dependency & secret scanning.
* **`submission.yml`** — Automates Kaggle competition submissions in a **dry-run mode by default**; requires explicit opt-in to submit.
* **`docker.yml`** — Builds and caches the Docker image (ensures reproducibility between Kaggle & local runs).

---

## 🐛 Issue Templates

Located in `.github/ISSUE_TEMPLATE/`:

* **`bug_report.yml`** — Standard debugging template.
* **`performance_issue.yml`** — Track runtime/memory regressions.
* **`security_report.yml`** — Handle vulnerabilities and data governance concerns.
* **`config_update.yml`** — Propose changes to pipeline configs (AOI, datasets, models).
* **`task_tracking.yml`** — Lightweight task planning & tracking.

---

## 🔒 Governance & Ethics

* **Branch Protection:** Enforced via GitHub rules + workflow checks before merge.
* **CARE Compliance:** Ethics guardrails integrated (e.g., masking site coordinates in PR builds).
* **Reproducibility:** Every workflow pins dependencies, logs hashes, and verifies outputs.

---

## ✅ Contribution Flow

1. Fork & branch (`feature/<name>`).
2. Run tests locally (`pytest tests/`).
3. Submit PR with filled template.
4. CI validates linting, tests, security, and reproducibility.
5. Reviewers check scientific/archaeological integrity before merge.

---

📖 **References**

* WDE Architecture Specification
* Repository Structure Guide
* Enriching WDE for Archaeology & Earth Systems

---
