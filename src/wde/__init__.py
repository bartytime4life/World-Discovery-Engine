# FILE: src/wde/__init__.py
# =================================================================================================
# World Discovery Engine (WDE) — Python package initializer
# - Exposes package version and lightweight metadata
# - Keep this file minimal; heavy imports go in submodules to prevent CLI startup latency
# =================================================================================================

from __future__ import annotations

__all__ = ["__version__", "get_version"]

# NOTE: Synchronize with pyproject.toml version when you cut releases
__version__: str = "0.1.0"


def get_version() -> str:
    """
    Return the WDE package version as a string.

    This is used by the CLI (`wde --version`) and may be imported by other modules
    without importing the full Typer app or heavy geo/ML dependencies.
    """
    return __version__