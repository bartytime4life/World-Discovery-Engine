
⸻

Contributing to WDE

Thank you for helping build the World Discovery Engine (WDE) — an open, reproducible, ethically-aware AI pipeline for archaeology and earth systems.
This document sets expectations and provides practical steps for contributing.

⸻

1. Ground Rules
	•	✅ Small, focused PRs with clear purpose.
	•	✅ Deterministic pipelines — control seeds (numpy, torch, random) and log all configs.
	•	✅ No large binaries in Git. Use DVC for data/artifacts, or link to open repositories (see datasets.md).
	•	✅ Always run make lint test before pushing.
	•	✅ Follow ethical guardrails in ETHICS.md — e.g., coarsen coordinates, flag Indigenous sovereignty.
	•	✅ Respect Kaggle reproducibility: notebook + CC-0 licensed data only ￼.

⸻

2. Developer Setup
	1.	Install Poetry:

pipx install poetry


	2.	Install dependencies (with all extras):

poetry install --with dev,geo,ml,viz,notebook


	3.	Install hooks:

poetry run pre-commit install


	4.	Initialize DVC:

poetry run dvc init



Optional: use Docker (Dockerfile) for a reproducible environment ￼.

⸻

3. Branching & Commits
	•	Branch naming:
	•	feature/<slug> for new features
	•	fix/<slug> for bugfixes
	•	docs/<slug> for documentation
	•	Conventional commits preferred:
	•	feat:, fix:, docs:, ci:, test:, refactor:, chore:.
	•	Update configs under configs/ when changing pipeline behavior.
	•	Document new datasets or sources in datasets.md.

⸻

4. Testing & CI/CD
	•	Run all tests:

poetry run pytest


	•	Coverage reports:

poetry run pytest --cov=wde


	•	Unit tests for every module (tests/unit/) and integration tests for each pipeline stage (tests/integration/).
	•	CI (GitHub Actions) runs linting, tests, and refutation checks (causal falsification) ￼.
	•	Avoid network-dependent tests unless mocked.

⸻

5. Data & Secrets
	•	❌ Do not commit credentials or .env files. Use .env.example for templates.
	•	Document every dataset in datasets.md with:
	•	Source URL / API
	•	License
	•	CRS, resolution, coverage
	•	Transform steps (linked to configs/preprocess.yaml) ￼.
	•	Large datasets → use DVC remotes (S3, GCS, SSH).

⸻

6. Code Style & Structure
	•	Python ≥ 3.11
	•	Formatters: ruff + black + isort (enforced via pre-commit).
	•	Modular design: follow world_engine/ pipeline stages (ingest, detect, evaluate, verify, report) ￼.
	•	Keep functions pure where possible; document side effects.
	•	Log everything: configs, seeds, data versions, anomalies, and refutations.

⸻

7. PR Checklist
	•	Code linted (make lint)
	•	Tests pass locally (make test)
	•	Added/updated configs (configs/*.yaml)
	•	Updated docs (README.md, docs/, or datasets.md)
	•	No large binaries in Git
	•	Ethical guardrails respected (ETHICS.md)
	•	Kaggle notebook reproducibility intact

⸻

8. Governance
	•	Code of Conduct: see CODE_OF_CONDUCT.md.
	•	Maintainers may request refactor/split of large PRs.
	•	Ethics overrides: if a dataset, method, or output violates community/Indigenous norms, maintainers may block its merge ￼.
	•	Decisions prioritize:
	1.	Archaeological significance
	2.	Reproducibility
	3.	Ethical defensibility

⸻

9. References
	•	[Best Practices in AI Development (2025)] ￼
	•	[Kaggle Platform Guide] ￼
	•	[WDE Architecture Specification] ￼
	•	[WDE Repository Structure] ￼
	•	[Remote Sensing Data Guide] ￼

⸻

🔍 Reminder: Every PR should enhance clarity, reproducibility, and ethics.
The World Discovery Engine is not just a pipeline — it’s a shared scientific instrument.

⸻