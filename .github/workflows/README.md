🚦 GitHub Actions — Quick Guide

This repo uses GitHub Actions to keep the World Discovery Engine (WDE) reproducible, tested, secure, and Kaggle-ready.

What runs when?

flowchart LR
  A[Push / PR] --> B[lint.yml]
  A --> C[ci.yml]
  A --> D[security.yml]

  B --> E{All checks pass?}
  C --> E
  D --> E

  E -- no --> X[Block merge]
  E -- yes --> F[submission.yml (manual)]

  F --> G{main + secrets + submit=yes?}
  G -- no --> H[Bundle only<br/>submission.csv + manifest.json]
  G -- yes --> I[Kaggle Submission]

  H --> J[release.yml on tag<br/>Docker + dossiers]
  I --> J

Workflows at a glance
	•	lint.yml – Style & static checks (ruff/black/isort/mypy/nbqa/yamllint/prettier).
	•	ci.yml – Unit + integration tests across the pipeline (ingest → detect → evaluate → verify → report).
	•	security.yml – Dependency/CVE scan & code scanning (CodeQL).
	•	submission.yml – Manual dispatch to build submission.csv, validate, and (optionally) submit via Kaggle CLI.
	•	release.yml – On tag: build Docker image, archive dossier outputs, publish artifacts.

Workflows use standard GitHub Actions syntax and triggers; see the official references for events, expressions, and contexts.  ￼

⸻

Quick start (common commands)

1) Lint & type-check locally (matches CI)

make lint   # or: ruff, black --check, mypy, nbqa, yamllint

2) Run tests locally

make test   # pytest -q

3) Kick off the full CI locally (optional)

act -W .github/workflows/ci.yml

4) Manually run the Kaggle submission workflow

In GitHub → Actions → Submission → Run workflow → set inputs:

	•	submit: yes to actually upload to Kaggle; no to only bundle.

The workflow uses the Kaggle API under the hood; the same CLI can be used on your machine:

pip install kaggle
kaggle competitions submit -c <slug> -f outputs/submission.csv -m "CI submission"

See Kaggle’s API docs for auth & usage.  ￼ ￼ ￼

⸻

Caching & speed tips (already wired in CI)
	•	Python setup & dependency caching
We use actions/setup-python with built-in caching (cache: pip) to speed installs.
Example pattern:

- uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: pip
    cache-dependency-path: |
      requirements.txt
      requirements-dev.txt
- run: pip install -r requirements.txt

Docs & rationale: setup-python caching + dependency caching reference.  ￼ ￼ ￼

⸻

Security scanning & hardening
	•	CodeQL is enabled for code scanning (scheduled + on PRs).
Tune queries or schedules in the codeql workflow block.  ￼
	•	The CodeQL action we use is the official GitHub maintained action.  ￼

⸻

Env, secrets, and inputs
	•	Kaggle: set KAGGLE_USERNAME and KAGGLE_KEY as repo secrets to authenticate the CLI in submission.yml. See Kaggle docs for generating credentials.  ￼
	•	Other tokens (optional): if you enable data fetchers that require API keys, add them as encrypted secrets and map via env: in the workflow.
	•	Expressions/contexts (e.g., github.ref_name, needs.*.result) follow standard GitHub Actions expression syntax.  ￼

⸻

How the matrix works (CI)

ci.yml may run a small matrix (e.g., Python 3.10–3.12, Ubuntu-latest) for portability. Caching is keyed per Python version to avoid cross-contamination. For dependency caching behavior and restore keys, see the official docs.  ￼

⸻

Where to look if something fails
	•	YAML/syntax – reference & examples for workflow syntax and commands.  ￼
	•	Python setup/caches – verify cache: pip and the cache-dependency-path match your files.  ￼
	•	CodeQL – check the “Code scanning alerts” tab and confirm only one CodeQL workflow is active.  ￼
	•	Kaggle – ensure valid KAGGLE_USERNAME/KAGGLE_KEY secrets and that the competition slug matches.  ￼

⸻

Why this layout?
	•	Follows GitHub Actions best-practice: clear workflow responsibilities and explicit triggers.  ￼
	•	Fast runs with setup-python caching & dependency keys.  ￼ ￼
	•	Built-in security scanning via CodeQL on cron + PRs.  ￼
	•	Seamless Kaggle packaging/submission via the Kaggle CLI.  ￼

⸻

Need details on the full architecture? See ARCHITECTURE.md in this folder for the deep dive.