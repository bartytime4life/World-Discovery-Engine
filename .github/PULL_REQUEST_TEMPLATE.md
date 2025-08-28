# 🛰️ WDE Pull Request

> Please complete **every** section. If something doesn’t apply, write “N/A” and briefly explain why.

---

## 1) Title & Linked Issues
**PR Title (imperative):**
- Closes: #ISSUE_ID
- Related: #ISSUE_ID(s)

---

## 2) Summary — What & Why
**What changed (short):**

**Why (scientific/engineering rationale):**
- Expected impact on pipeline quality (anomaly recall/precision, dossier clarity, runtime, reproducibility)
- Tradeoffs and alternatives considered

---

## 3) Scope & Surface Area
- [ ] Code (Python modules)
- [ ] Configs (under `configs/`)
- [ ] Data plumbing (DVC, data registry)
- [ ] Notebooks (Kaggle / local demos)
- [ ] Docs (`README.md`, `docs/`, `datasets.md`, `ETHICS.md`)
- [ ] CI/CD (GitHub Actions, Docker)
- [ ] CLI / Make targets
- [ ] Other:

**Modules touched (paths):**

---

## 4) Reproducibility & Determinism
- [ ] All randomness is seeded (`numpy`, `random`, `torch`, etc.)
- [ ] Config changes logged & versioned; defaults avoid “magic numbers”
- [ ] `Dockerfile` or Poetry env builds cleanly
- [ ] DVC stages updated as needed; `dvc repro` passes locally
- [ ] Notebook(s): “Save & Run All” passes; outputs match code commit
- [ ] Output artifacts use stable filenames & contain run metadata (config hash, seed, timestamp)

**Notes:**

---

## 5) Ethics & Safety (MANDATORY)
- [ ] Coordinate masking in **public outputs** (≥ 0.01°); exact coords only in restricted artifacts
- [ ] Indigenous sovereignty check integrated; **banner** included if overlap detected
- [ ] Dataset licenses verified; non-commercial/CC terms respected; citations added
- [ ] No sensitive locations exposed in PR description, logs, or screenshots
- [ ] `ethics_guardrails` invoked in `report` stage (coordinate masking + sovereignty banner)
- [ ] If Brazil/region-specific controls apply (e.g., IPHAN), compliance toggles enforced

**Risk assessment (brief):**

---

## 6) Multi-Proof, Uncertainty & Refutation
- [ ] Multi-proof rule satisfied (≥ 2 modalities) for any new detection logic
- [ ] Uncertainty evaluated/calibrated (hist/quantiles); failure modes noted
- [ ] Counterfactual/ablation (e.g., SSIM what-if) implemented for new signals
- [ ] Benchmarks/plots included (if applicable)

**Evidence links (plots/artifacts):**

---

## 7) Data & Datasets Registry
- [ ] New or updated datasets documented in `datasets.md`:
  - Source URL/API, license, CRS/resolution, coverage, transforms, DVC path
- [ ] Large files **not** committed to Git; DVC pointers or Kaggle Datasets used
- [ ] Sample/minimal test fixtures created for unit & integration tests (small, open)

**Entries added/updated:**

---

## 8) Tests & CI
- [ ] Unit tests added/updated; fast and deterministic
- [ ] Integration tests per pipeline stage added/updated
- [ ] `make lint` passes (ruff/black/isort)
- [ ] `make test` passes locally
- [ ] CI (GitHub Actions) green: lint, tests, build, (optional) refutation suite

**Test notes (coverage, critical paths):**

---

## 9) Notebooks (if applicable)
- [ ] Reproducible on Kaggle (CPU baseline; optional GPU path)
- [ ] Inputs via Kaggle Datasets or open APIs; **no** hidden credentials
- [ ] Narrative cells explain steps, caveats, and ethical notes
- [ ] Output path: `/outputs/` with figures (PNG), GeoJSON, JSON logs

**Notebook links:**

---

## 10) CLI & Make Targets
- [ ] New/updated subcommands documented (`wde ...`) and tested
- [ ] `Makefile` targets added/updated (run, dvc, export-locks, kaggle-zip)
- [ ] Help text and error messages clear; dry-run where destructive

**Commands:**

---

## 11) Performance & Resource Use
- [ ] Runtime and memory within budget (document deltas vs. main)
- [ ] External dependencies reasonable; wheels available (esp. geo stack)

**Benchmarks (before → after):**

---

## 12) Docs & Examples
- [ ] `README.md` / `docs/` reflect changes (usage, flags, outputs)
- [ ] Minimal example or smoke script added (optional) for quick verification

**Doc links/sections:**

---

## 13) Reviewer Guide
- Suggested review order (files/commits):
- Key questions to validate:
  1. Config & determinism sound?
  2. Ethics guardrails enforced?
  3. Multi-proof + uncertainty + refutation evidence present?
  4. CI green; tests meaningful?

---

## 14) Checklist (One-Line Summary)
- [ ] Lint
- [ ] Tests
- [ ] Reproducibility
- [ ] Ethics
- [ ] Datasets
- [ ] Docs
- [ ] CI Green

---

## 15) Post-Merge Tasks (if any)
- [ ] Backfill docs/tutorials
- [ ] Release notes / changelog
- [ ] Dataset version bump
- [ ] CI schedule update