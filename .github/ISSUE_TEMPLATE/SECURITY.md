SECURITY.md — World Discovery Engine (WDE)

0) Purpose & Scope

This document describes how to report security issues in the World Discovery Engine (WDE), our coordinated disclosure process, and the hardening standards we follow across code, configs, CI/CD, Docker, and Kaggle notebooks. It complements our repo architecture and CI/CD practices (modularity, reproducibility, ethics) ￼ ￼ ￼ ￼ ￼.

In scope:
	•	Source code under world_engine/, notebooks in notebooks/, configs in configs/, CI/CD in .github/workflows/, and container build files (e.g., Dockerfile) ￼.
	•	Reproducibility and ethics guardrails (coordinate masking, CARE/sovereignty banners) in outputs and reports ￼ ￼.
	•	Performance/security implications that affect our Kaggle notebook runs and end-to-end pipeline integrity ￼ ￼.

Out of scope:
	•	Third-party platforms’ infrastructure (Kaggle, GitHub, cloud providers).
	•	Social engineering, physical security, or policy-only issues unrelated to the codebase.

⸻

1) How to Report a Vulnerability (Private First)

Preferred (private) channels:
	1.	GitHub → Security → “Report a vulnerability” (private security advisory to maintainers).
	2.	If advisory is not available, open a private Security Advisory draft and add maintainers, or contact us via the repository owners’ preferred private channel (if published).

As a fallback (non-public issue):
	•	Use the Security Report issue form:
.github/ISSUE_TEMPLATE/security_report.yml (auto-labels security) — include evidence, reproduction, impact, mitigation, and rollback (the template enforces this) ￼.

Please include:
	•	A minimal PoC and exact reproduction steps (versions, OS, Docker image digest, notebook image).
	•	Affected paths (Dockerfile, .github/workflows/*.yml, requirements.txt, notebooks, etc.).
	•	References (CVE, GHSA, osv.dev), and SBOM snippet if relevant (we publish SBOMs with releases/artifacts where applicable).

We will:
	•	Acknowledge within 3 business days, triage within 7 days, and aim to ship a fix within 90 days (or provide a timeline if complex). We follow coordinated disclosure and request you keep reports private until a patch and advisory are available ￼.

⸻

2) Coordinated Disclosure & Safe Harbor
	•	Please give us reasonable time to investigate and remediate before public disclosure.
	•	We credit reporters (unless you request anonymity) in the security advisory and release notes.
	•	Safe Harbor: Good-faith research using non-destructive methods, honoring rate limits and no data exfiltration, will not be pursued legally. Do not access data you’re not entitled to, and avoid impacting other users or services.

⸻

3) Severity, Impact, and Embargo Guidelines
	•	We use standard severities (Critical/High/Medium/Low) and consider data integrity, credential/secret exposure, RCE, supply-chain tampering, and ethics guardrails impact (e.g., disabling coordinate masking or sovereignty banners) ￼ ￼.
	•	Where possible, provide a CVSS vector or a rationale for severity.

Embargo ends when:
	•	A fix is merged, CI/CD passes, SBOMs updated, and patched release/tag is published with an advisory.

⸻

4) Supported Branches / Versions
	•	main: actively developed and supported.
	•	Tagged releases: latest minor/patch lines receive fixes when security-relevant.
Older tags may require upgrading to the latest release.

⸻

5) Hardening & Reproducibility Standards

Our engineering standards emphasize modularity, testing, CI/CD, version/seed pinning, and containerized reproducibility ￼ ￼:
	•	CI/CD (GitHub Actions): Lint, unit/integration tests, submission pipeline checks; we treat CI as the enforcement layer for deterministic runs ￼ ￼.
	•	Reproducibility: seeds, configs, dependency versions, and container images are pinned, logged, and tracked to enable deterministic reruns in notebooks and CLI ￼ ￼ ￼.
	•	Kaggle Notebooks: we ensure the notebook runs end-to-end with fixed environments, proper “Save & Run All” semantics, and pinned environments when possible ￼.
	•	Ethics guardrails: reports and outputs mask precise coordinates by default and include CARE/sovereignty notices; security fixes must not weaken these defaults ￼ ￼.

⸻

6) Supply-Chain & Dependency Policy

To reduce supply-chain risk, we follow:
	1.	Pin everything:
	•	Python deps pinned in requirements.txt (or lockfiles) and audited in CI ￼.
	•	GitHub Actions pinned by SHA (not floating tags) where feasible.
	•	Docker images referred to by immutable digests for release builds.
	2.	Minimal base images and no root runtime (where possible).
	3.	SBOMs: We publish SBOMs with releases/artifacts when applicable and keep them updated after security releases (see advisory notes and artifacts referenced in issues).
	4.	Third-Party Models & Data:
	•	Only open datasets and CC-0 compatible inputs, per challenge rules and architecture ￼.
	•	Any model artifacts must be verifiable and openly licensed or locally built, with hashes recorded.
	5.	Secrets Management:
	•	Never commit tokens; use GitHub encrypted secrets and Kaggle Secrets for notebooks (internet-off by default unless required) ￼.
	•	CI logs must not print secrets; scrub/ mask logs and artifacts by default.

⸻

7) CI/CD Security & Integrity
	•	Workflows: lint → unit/integration tests → (optionally) packaging/submission routines, mirroring the architecture’s funnel to prevent regressions ￼.
	•	Isolation: PRs from forks have minimal privileges; actions run with least privilege and read-only tokens where feasible.
	•	Artifacts: Test artifacts and dashboards must not contain secrets or precise site coordinates by default; masking enforced in reporting code ￼ ￼.
	•	Kaggle Integration: The notebook uses pinned or pinned-compatible environments; users should confirm accelerator and image pinning in the Settings panel when possible ￼.

⸻

8) Special Considerations for Kaggle
	•	Runtime pinning: Prefer pinning to the original environment image when determinism matters; track additional pip installs in the notebook so versions are visible in “Save & Run All” logs ￼.
	•	Internet access: Off by default; if required for open data, clearly indicated and restricted to public sources noted in Connecting to Remote Sensing and Environmental Data Sources ￼.
	•	Data ethics: Public notebooks must keep sovereignty banners and coordinate masking enabled by default ￼.

⸻

9) Ethics, Privacy, and Sovereignty

Security is not only technical. We treat ethical misuse (e.g., turning off masking/sovereignty notices) as a high-impact regression. Candidate dossiers are intended for expert review with community engagement, not public pinpointing without consent ￼ ￼.

⸻

10) Credits

WDE’s security posture builds on our architecture spec, repository structure, AI engineering best practices, Kaggle notebook guidance, and ethics governance:
	•	Architecture & CI/CD funnel ￼; repo structure & testing layout ￼
	•	Modularity/testing/reproducibility/versioning standards ￼
	•	Kaggle runtime, environment pinning, and versioned notebook runs ￼
	•	CARE principles, sovereignty banners, and coordinate masking defaults ￼

⸻

11) Questions

If unsure whether something is a security issue, report privately anyway and we’ll guide triage. Thank you for helping keep WDE safe, reproducible, and ethically sound.