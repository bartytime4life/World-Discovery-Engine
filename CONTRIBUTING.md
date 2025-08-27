# FILE: CONTRIBUTING.md
# -------------------------------------------------------------------------------------------------
# Contributing to WDE

Thanks for helping build the World Discovery Engine! To keep the project healthy and reproducible:

## 1) Ground Rules
- Prefer small, focused PRs with tests.
- Keep pipeline deterministic. Control all randomness (seeds) and record configs.
- No large binaries in Git. Use DVC for data/artifacts or link to stable sources.
- Run `pre-commit` and `pytest` locally before pushing.

## 2) Dev Setup
- Install Poetry: `pipx install poetry`
- Install deps: `poetry install --with dev,geo,ml,viz,notebook`
- Install hooks: `poetry run pre-commit install`
- DVC init: `poetry run dvc init`

## 3) Branching & Commits
- Branch: `feature/<slug>`, `fix/<slug>`, or `docs/<slug>`.
- Conventional commits appreciated (feat:, fix:, docs:, chore:, test:, ci:).
- Add/Update `configs/*.yaml` when changing behavior. Document in README or docs/.

## 4) Tests
- `poetry run pytest`
- Add unit tests for new modules (fast), and integration tests for pipeline steps.
- Avoid network-dependent tests unless mocked.

## 5) Data & Secrets
- Don’t commit secrets. Use `.env` (see .env.example).
- Datasets: document access + license in `datasets.md`.
- For large files, use DVC remote storage (S3, GCS, SSH, etc.).

## 6) Code Style
- Python ≥ 3.11
- Ruff + Black + isort via pre-commit
- Keep functions small, pure when possible; document side effects.

## 7) PR Checklist
- [ ] Linted (`make lint`)
- [ ] Tests pass (`make test`)
- [ ] Updated configs/docs as needed
- [ ] No large binaries in Git
- [ ] Added notes to `datasets.md` if new data sources

## 8) Governance
- See CODE_OF_CONDUCT.md. Maintainers may request rework/split of large PRs.