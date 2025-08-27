# FILE: src/wde/utils/log.py
# =================================================================================================
# Logging / verbosity helpers
# - Simple global switches to keep dependencies minimal
# - Extend to use `logging` + RichHandler if you need structured logs
# =================================================================================================
from __future__ import annotations

import os
from typing import Optional

_VERBOSE_LEVEL = 0
_QUIET = False


def set_global_verbosity(verbose: int = 0, quiet: bool = False) -> None:
    """
    Set global verbosity for the CLI process.

    Parameters
    ----------
    verbose : int
        Verbosity level, where 0 is default, 1 is -v, 2 is -vv, etc.
    quiet : bool
        If True, minimize output (overrides verbose).
    """
    global _VERBOSE_LEVEL, _QUIET
    _VERBOSE_LEVEL = int(verbose)
    _QUIET = bool(quiet)
    os.environ["WDE_VERBOSE"] = str(_VERBOSE_LEVEL)
    os.environ["WDE_QUIET"] = "1" if _QUIET else "0"


def is_quiet() -> bool:
    """Return True if quiet mode is active."""
    return _QUIET


def verbose_level() -> int:
    """Return current verbosity level."""
    return _VERBOSE_LEVEL


def vprint(msg: str, level: int = 1) -> None:
    """
    Print message if verbosity >= level and not in quiet mode.
    """
    if not _QUIET and _VERBOSE_LEVEL >= level:
        print(msg)