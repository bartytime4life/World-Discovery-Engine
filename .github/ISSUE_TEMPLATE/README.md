📝 Issue Templates — World Discovery Engine (WDE)

This repository uses GitHub Issue Forms to streamline collaboration and ensure every change request is reproducible, testable, and ethical.

Available Templates

Template	File	When to Use
🐛 Bug Report	bug_report.yml	For reproducible errors or pipeline malfunctions (ingestion, anomaly detection, reports, etc.).
⚙️ Config Update	config_update.yml	For changes to configuration files (configs/*.yaml): tile size, thresholds, seeds, dataset paths. Requires reproducibility + rollback plan ￼.
🚀 Performance Update	performance_update.yml	For runtime, memory, or throughput optimizations. Must include before/after benchmarks and reproducibility checks ￼.
🔒 Security Report	security_report.yml	For dependency CVEs, secrets leakage, or compliance concerns.
📌 Task Tracking	task_tracking.yml	For small, shippable tasks with clear acceptance criteria (CLI-first, Hydra-configured, CI-tested).
🌱 Feature Request	feature_request.yml	For new functionality beyond configs/performance (e.g., new anomaly detector, new ADE proxy).
🧪 Config/Performance Combo	Use both above	If your proposal affects both configs and performance, file two linked issues so reviewers can track metrics separately.


⸻

Guidelines
	•	Minimal, Reviewable Changes → keep PRs and issues small enough to test in isolation.
	•	Reproducibility First → every change must log seeds, config files, and dataset versions ￼.
	•	Ethics Guardrails → confirm sovereignty banners, coordinate masking, and CARE principle defaults remain intact ￼.
	•	Rollback Ready → every issue must describe how to revert safely if metrics regress.
	•	CI/CD Alignment → expect GitHub Actions (lint.yml, ci.yml, submission.yml) to validate your changes ￼.

⸻

References
	•	📖 ARCHITECTURE.md — CI/CD workflow design.
	•	📖 WDE Architecture Specification — full pipeline overview ￼.
	•	📖 ETHICS.md — CARE principles, sovereignty defaults ￼.

⸻

✅ Use the right template for the change.
✅ Keep it reproducible and ethical.
✅ Link issues to PRs for traceability.

⸻