🌍 World Discovery Engine (WDE) — Issue Template Architecture

.github/ISSUE_TEMPLATE/ARCHITECTURE.md

⸻

Purpose

This file defines the architecture and intent of GitHub issue templates for the WDE project.
It ensures consistency, modularity, reproducibility, and scientific rigor in how contributors file and manage issues.

Templates support bug fixes, config changes, performance optimizations, security reports, tasks, and feature requests, aligned with the WDE pipeline and Kaggle challenge requirements.

⸻

Design Principles
	1.	Modularity — Each template is independent and maps to a pipeline component (ingest → detect → evaluate → verify → report) ￼ ￼.
	2.	Reproducibility — Issues must document seeds, configs, datasets, and runs for deterministic replay ￼ ￼.
	3.	Scientific Validity — Changes must reference archaeological significance, ADE fingerprints, and multi-proof validation ￼ ￼ ￼.
	4.	Ethical Compliance — All issues respect CARE principles and local/Indigenous sovereignty ￼.
	5.	CI/CD Integration — Issues trigger GitHub Actions checks (lint, tests, reproducibility) ￼ ￼.
	6.	Transparency — Markdown forms guide users to provide rationale, benchmarks, rollback plans, and dataset provenance.

⸻

Templates Overview

🐛 Bug Report — bug_report.yml
	•	Purpose: Document reproducible errors or malfunctions in ingestion, anomaly detection, verification, or reporting stages.
	•	Requires: Minimal failing config, logs, expected vs observed behavior, rollback confirmation.

⚙️ Config Update — config_update.yml
	•	Purpose: Changes to configs/*.yaml (tile sizes, thresholds, seeds, dataset paths).
	•	Requires: Rationale, reproducibility checks, rollback plan ￼.

🚀 Performance Update — performance_update.yml
	•	Purpose: Runtime, memory, or throughput optimizations.
	•	Requires: Before/after benchmarks, reproducibility checks ￼.

🔒 Security Report — security_report.yml
	•	Purpose: Dependency CVEs, secrets leakage, compliance concerns.
	•	Requires: Description of vulnerability, affected modules, recommended remediation.

📌 Task Tracking — task_tracking.yml
	•	Purpose: Small, shippable tasks with CLI-first, Hydra-configured, CI-tested criteria ￼.

🌱 Feature Request — feature_request.yml
	•	Purpose: New functionality beyond configs/performance (e.g., new ADE detector, novel proxy integration, causal module).
	•	Requires: Clear description, expected archaeological/scientific value, reproducibility plan.

🧪 Config/Performance Combo
	•	Guidance: If both configs and performance are affected, file two linked issues so metrics can be tracked independently.

⸻

Workflow Integration
	•	Labels: Each template auto-assigns labels (bug, config, performance, security, task, feature).
	•	CI/CD: Every issue links to GitHub Actions workflows (lint.yml, ci.yml, security.yml, submission.yml) ￼.
	•	Reproducibility Logs: Issues must reference configs/*.yaml, Kaggle notebook runs, and output artifacts (GeoTIFFs, GeoJSON, dossiers) ￼ ￼.
	•	Ethics Hooks: Config updates and feature requests prompt contributors to confirm CARE/sovereignty compliance ￼.

⸻

Example Issue Flow
	1.	Researcher detects anomaly → files Bug Report (bad NDVI time-series alignment).
	2.	Engineer adjusts config → files Config Update (new tile size, seeds).
	3.	Optimizer tunes runtime → files Performance Update (batch size).
	4.	Security audit flags dependency → files Security Report (CVE in rasterio).
	5.	Contributor proposes new ADE fingerprint detector → files Feature Request.
	6.	Project lead schedules small CLI upgrade → files Task Tracking.

⸻

References
	•	Best Practices in AI Development ￼
	•	Kaggle Technical Guide ￼
	•	WDE Enrichment & Ethics ￼
	•	WDE Architecture Specification ￼
	•	WDE Repository Structure ￼
	•	ADE Discovery Pipeline Notebook ￼

⸻

✅ This ARCHITECTURE.md is the backbone for your .github/ISSUE_TEMPLATE/ directory.
It aligns issue templates with scientific rigor, reproducibility, ethics, and CI/CD automation.

⸻
