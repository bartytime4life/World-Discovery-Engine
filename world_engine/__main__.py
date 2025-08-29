# FILE: world_engine/__main__.py
# =============================================================================
# 🌍 World Discovery Engine (WDE)
# Package Entrypoint — enables `python -m world_engine` to launch the CLI.
#
# How it works
# ------------
# - Imports the Typer app entrypoint from `world_engine.cli`
# - Calls `main()` so all CLI subcommands are available:
#     python -m world_engine --help
#     python -m world_engine full-run -c configs/pipeline.yaml
#     python -m world_engine ingest -c configs/pipeline.yaml
#
# Notes
# -----
# - We intentionally keep this thin; all stage wiring, logging, and config
#   resolution lives in `world_engine.cli`.
# - If heavy optional deps are missing, the CLI modules will surface
#   a helpful warning at runtime (not here).
# =============================================================================

from __future__ import annotations

import sys

def _run() -> int:
    """
    Import and invoke the Typer CLI entrypoint.

    Returns
    -------
    int
        Process exit code (0 on success).
    """
    # Import here so that any CLI-only dependencies (Typer, YAML, etc.)
    # are not required just to import the package elsewhere.
    from world_engine.cli import main as _cli_main

    # Delegate to the CLI's main() which sets up logging/config and runs commands.
    _cli_main()
    return 0


if __name__ == "__main__":
    # Execute the CLI and propagate a non-zero exit on failure.
    sys.exit(_run())
