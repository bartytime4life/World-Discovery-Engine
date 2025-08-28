Here’s a hardened upgraded SECURITY.md tailored for the World Discovery Engine (WDE) project. It aligns with best practices from NASA-grade reproducibility ￼, CLI/DevOps safety ￼, and archaeological data ethics ￼.

⸻

Security Policy — World Discovery Engine (WDE)

🔒 Reporting a Vulnerability
	•	Private channels only. Do not open public GitHub issues for vulnerabilities.
	•	Send a detailed report (steps to reproduce, impact, suggested fix) to the maintainers:
	•	📧 security@world-discovery-engine.org (example placeholder)
	•	You will receive an acknowledgment within 72 hours and a resolution plan within 7 business days.
	•	Critical issues may trigger emergency CI/CD blocks until resolved.

⸻

🔑 Secrets & Credentials
	•	Never commit secrets (API keys, tokens, passwords).
	•	Use:
	•	.env files (ignored in Git) for local dev.
	•	GitHub Actions secrets store for CI/CD.
	•	Vaults (e.g., HashiCorp Vault, AWS/GCP Secret Manager) for production deployments.
	•	Any accidentally exposed secret must be revoked immediately and rotated.

⸻

📦 Dependencies & Supply Chain
	•	WDE uses Poetry with pinned versions (poetry.lock) and DVC for artifact versioning ￼.
	•	Automated checks:
	•	pre-commit hooks (ruff, black, isort, bandit).
	•	CI workflows with pip-audit or safety scans.
	•	If you spot a dependency with a CVE:
	1.	Open a private security advisory or PR with a patched version.
	2.	CI will verify reproducibility and test coverage before merge.
	•	External models & datasets must be open-licensed and hash-verified before ingestion.

⸻

🌍 Geospatial & PII Protections
	•	Archaeological and Indigenous heritage data require special handling ￼:
	•	Do not expose exact site coordinates in public outputs; round to ~0.01° by default.
	•	Candidate site dossiers are for expert review only, not public release ￼.
	•	Follow CARE Principles (Collective Benefit, Authority, Responsibility, Ethics) when working with Indigenous or local data.
	•	Redact or anonymize any PII (names, addresses, sensitive notes) before sharing artifacts or logs.

⸻

🛡️ Runtime & CI/CD Safety
	•	All pipeline runs are CLI-first and reproducible ￼:
	•	Use wde selftest before running discovery pipelines.
	•	CI enforces dry-run checks, guardrails, and reproducibility hashes ￼.
	•	Kaggle & GitHub CI runners are sandboxed; no privileged ops or raw system calls allowed.
	•	Always assume outputs may be inspected publicly; treat them as read-only, non-trusted input until validated.

⸻

📝 Disclosure & Ethics
	•	We follow responsible disclosure: vulnerabilities will be patched before details are shared.
	•	Security fixes will be tagged in release notes under [security].
	•	For archaeological datasets, security is also ethics: misuse of site data can cause looting or cultural harm.
	•	See ETHICS.md for complementary safeguards.

⸻