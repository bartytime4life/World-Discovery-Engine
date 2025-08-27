# FILE: SECURITY.md
# -------------------------------------------------------------------------------------------------
# Security Policy

## Reporting a Vulnerability
Please email the maintainers with details (steps to reproduce, impact, suggested fixes).
Do **not** open a public issue for sensitive reports.

## Secrets
Never commit credentials. Use `.env` and secret managers/CI vaults.

## Dependencies
We rely on Poetry pins and pre-commit checks. If you spot a vulnerable dependency, open a PR or a private report.

## Geo/PII
If data contains sensitive locations or PII, follow ETHICS.md and redact before sharing artifacts.