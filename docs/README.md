# 📖 World Discovery Engine — Documentation Index

This folder contains the **technical documentation, specifications, and guides** for the **World Discovery Engine (WDE)**.
It complements the root `README.md` by providing deeper detail on architecture, datasets, ethics, and development practices.

---

## 📂 Contents

* **architecture.md**
  Canonical WDE architecture document — discovery funnel, pipeline design, CausalOps lifecycle, and reproducibility rules.

* **repository\_structure.md**
  File-by-file and directory-by-directory breakdown of the repo structure, including code, configs, tests, data handling, and CI/CD.

* **datasets.md**
  Registry of all datasets used in the pipeline (Sentinel, Landsat, LiDAR, soils, hydrology, historical archives). Includes sources, licenses, and access methods.

* **ETHICS.md**
  Ethical principles, governance, and safeguards. Covers CARE Principles, FPIC/IPHAN compliance in Brazil, data sovereignty, and anti-data-colonialism guidelines.

* **ADE\_pipeline.md** (notebook spec)
  Detailed breakdown of the Kaggle notebook `ade_discovery_pipeline.ipynb` — cell structure, fallback behavior, anomaly detection methods, and output artifacts.

* **causalops.md**
  Lifecycle specification: Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document. Ties directly to CI/CD and artifact logging.

* **contributing.md**
  Developer onboarding, coding conventions, testing strategy, and GitHub workflow expectations.

---

## 🔍 How to Use This Docs Folder

* **New contributors** — Start with `repository_structure.md` and `contributing.md`.
* **Pipeline developers** — Reference `architecture.md` and `causalops.md`.
* **Data wranglers** — Check `datasets.md` for all sources and access instructions.
* **Ethics reviewers** — See `ETHICS.md`.
* **Challenge submitters** — Use `ADE_pipeline.md` for the Kaggle notebook details.

---

## 📑 Cross-References

* The **root `README.md`** provides a high-level overview for external audiences (judges, Kaggle users).
* The **`docs/` folder** provides detailed specifications and developer guidance.
* All documents cite back to the project’s research & design files with `【message_idx†source】` markers for traceability.

---
