# FILE: src/wde/__main__.py
# =================================================================================================
# Allows `python -m wde` to invoke the Typer CLI defined in wde/cli.py
# =================================================================================================
from __future__ import annotations

from .cli import app

if __name__ == "__main__":
    app()