# FILE: src/wde/cli.py
# =================================================================================================
# World Discovery Engine (WDE) — Unified Typer CLI
#
# Design goals
# ------------
# - Single entrypoint (`wde`) with subcommands matching DVC stages and notebook workflow:
#     wde data fetch
#     wde data preprocess
#     wde detect run
#     wde dossier build
#     wde export kaggle-notebook
# - Deterministic, reproducible behavior: reads from ./configs/*.yaml and .env (dotenv)
# - Lightweight: imports heavy geo/ML stacks only within the subcommand body
#
# Usage
# -----
#   poetry run python -m wde --help
#   poetry run wde --help                   # via [tool.poetry.scripts] if configured
#   poetry run wde data fetch --help
#
# Notes
# -----
# - Each subcommand writes logs to stdout; integrate a richer logger if desired.
# - This is a minimal, working skeleton that you can extend stage-by-stage.
# =================================================================================================
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import get_version
from .utils.io import load_yaml_safe, ensure_dir, write_json
from .utils.log import set_global_verbosity

# Initialize Typer app and console
app = typer.Typer(help="World Discovery Engine (WDE) — Multi-modal AI for Archaeology & Earth Systems")
console = Console()

# Load environment variables from .env if present (no error if missing)
load_dotenv(override=False)


# -------------------------------------------------------------------------------------------------
# Helper: project root discovery (simple heuristic)
# -------------------------------------------------------------------------------------------------
def project_root() -> Path:
    """
    Return the project root Path, assuming the CLI is executed from within the repo.
    Heuristic:
      - Use current working directory
      - If pyproject.toml is found in a parent, prefer that directory
    """
    cwd = Path.cwd()
    for p in [cwd, *cwd.parents]:
        if (p / "pyproject.toml").exists() or (p / ".git").exists():
            return p
    return cwd


# -------------------------------------------------------------------------------------------------
# Root commands
# -------------------------------------------------------------------------------------------------
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="Increase verbosity (-v, -vv)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """
    WDE unified CLI. Run with a subcommand (see --help).

    If invoked without a subcommand, prints a short banner and exits.
    """
    set_global_verbosity(verbose=verbose, quiet=quiet)
    if ctx.invoked_subcommand is not None:
        return

    banner = Panel.fit(
        f"[bold cyan]WDE — World Discovery Engine[/bold cyan]\n"
        f"[white]Multi-modal AI for Archaeology & Earth Systems[/white]\n"
        f"[dim]version {get_version()}[/dim]",
        border_style="cyan",
    )
    console.print(banner)
    console.print("Use [bold]wde --help[/bold] or a subcommand, e.g., [bold]wde data fetch[/bold].")


@app.command("version")
def version() -> None:
    """
    Print WDE version and selected environment details.
    """
    info = {
        "wde_version": get_version(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "WDE_DATA_ROOT": os.environ.get("WDE_DATA_ROOT", "./data"),
        "WDE_ARTIFACTS_ROOT": os.environ.get("WDE_ARTIFACTS_ROOT", "./artifacts"),
    }
    rprint(info)


@app.command("env")
def env() -> None:
    """
    Print environment variables relevant to WDE (sanitized).
    """
    keys = [
        "WDE_DATA_ROOT",
        "WDE_ARTIFACTS_ROOT",
        "KAGGLE_USERNAME",
        "PLANET_API_KEY",
        "SENTINELHUB_CLIENT_ID",
        "SENTINELHUB_CLIENT_SECRET",
        "MAPBOX_TOKEN",
    ]
    table = Table(title="WDE Environment", show_lines=True)
    table.add_column("Key", justify="right", style="bold")
    table.add_column("Value (sanitized)")

    for k in keys:
        v = os.environ.get(k, "")
        if not v:
            sval = "[dim]<unset>[/dim]"
        elif len(v) <= 8:
            sval = f"{v[:2]}…"
        else:
            sval = f"{v[:4]}…{v[-2:]}"
        table.add_row(k, sval)
    console.print(table)


# -------------------------------------------------------------------------------------------------
# Sub-APP: data
# -------------------------------------------------------------------------------------------------
data_app = typer.Typer(help="Data acquisition and preprocessing")
app.add_typer(data_app, name="data")


@data_app.command("fetch")
def data_fetch(
    config: Path = typer.Option(Path("configs/data.yaml"), "--config", "-c", exists=False, help="Data config YAML"),
    out_dir: Path = typer.Option(Path(os.environ.get("WDE_DATA_ROOT", "./data")) / "raw", "--out", "-o"),
) -> None:
    """
    Fetch raw data per configs/data.yaml.

    This skeleton:
      - Creates data/raw/
      - Writes a tiny GeoJSON placeholder (empty FeatureCollection)
      - In real use: implement providers (Sentinel, Landsat, NICFI, DEM, LiDAR, etc.)
    """
    ensure_dir(out_dir)
    cfg = load_yaml_safe(config) if config.exists() else {"sources": []}
    from .data.fetch import run_fetch  # lazy import

    console.print(Panel.fit(f"Running data fetch → [bold]{out_dir}[/bold]", border_style="green"))
    artifacts = run_fetch(cfg, out_dir)
    write_json(out_dir / "fetch_manifest.json", artifacts)
    rprint({"fetched": len(artifacts.get("files", [])), "out_dir": str(out_dir)})


@data_app.command("preprocess")
def data_preprocess(
    config: Path = typer.Option(
        Path("configs/preprocess.yaml"), "--config", "-c", exists=False, help="Preprocess config YAML"
    ),
    in_dir: Path = typer.Option(Path(os.environ.get("WDE_DATA_ROOT", "./data")) / "raw", "--in", "-i"),
    out_dir: Path = typer.Option(Path(os.environ.get("WDE_DATA_ROOT", "./data")) / "processed", "--out", "-o"),
) -> None:
    """
    Preprocess raw data into analysis-ready layers.
    """
    ensure_dir(out_dir)
    cfg = load_yaml_safe(config) if config.exists() else {"steps": []}
    from .data.preprocess import run_preprocess  # lazy import

    console.print(Panel.fit(f"Preprocessing data {in_dir} → {out_dir}", border_style="green"))
    manifest = run_preprocess(cfg, in_dir, out_dir)
    write_json(out_dir / "preprocess_manifest.json", manifest)
    rprint({"processed_layers": len(manifest.get("layers", [])), "out_dir": str(out_dir)})


# -------------------------------------------------------------------------------------------------
# Sub-APP: detect
# -------------------------------------------------------------------------------------------------
detect_app = typer.Typer(help="Candidate detection")
app.add_typer(detect_app, name="detect")


@detect_app.command("run")
def detect_run(
    config: Path = typer.Option(Path("configs/detect.yaml"), "--config", "-c", exists=False, help="Detect config YAML"),
    in_dir: Path = typer.Option(Path(os.environ.get("WDE_DATA_ROOT", "./data")) / "processed", "--in", "-i"),
    out_dir: Path = typer.Option(Path(os.environ.get("WDE_ARTIFACTS_ROOT", "./artifacts")) / "candidates", "--out", "-o"),
) -> None:
    """
    Execute the detection pipeline on processed layers.
    """
    ensure_dir(out_dir)
    cfg = load_yaml_safe(config) if config.exists() else {"model": {"type": "baseline"}}
    from .detect.run import run_detection  # lazy import

    console.print(Panel.fit(f"Detecting candidates from {in_dir} → {out_dir}", border_style="magenta"))
    result = run_detection(cfg, in_dir, out_dir)
    write_json(out_dir / "candidates.json", result)
    rprint({"candidates": len(result.get("candidates", [])), "out_dir": str(out_dir)})


# -------------------------------------------------------------------------------------------------
# Sub-APP: dossier
# -------------------------------------------------------------------------------------------------
dossier_app = typer.Typer(help="Dossier building")
app.add_typer(dossier_app, name="dossier")


@dossier_app.command("build")
def dossier_build(
    config: Path = typer.Option(Path("configs/dossier.yaml"), "--config", "-c", exists=False, help="Dossier config YAML"),
    in_dir: Path = typer.Option(Path(os.environ.get("WDE_ARTIFACTS_ROOT", "./artifacts")) / "candidates", "--in", "-i"),
    out_dir: Path = typer.Option(Path(os.environ.get("WDE_ARTIFACTS_ROOT", "./artifacts")) / "dossiers", "--out", "-o"),
) -> None:
    """
    Build candidate dossiers (maps, overlays, references, uncertainty).
    """
    ensure_dir(out_dir)
    cfg = load_yaml_safe(config) if config.exists() else {"layout": "minimal"}
    from .dossier.build import run_dossier_build  # lazy import

    console.print(Panel.fit(f"Building dossiers from {in_dir} → {out_dir}", border_style="blue"))
    result = run_dossier_build(cfg, in_dir, out_dir)
    write_json(out_dir / "dossiers_manifest.json", result)
    rprint({"dossiers": len(result.get("dossiers", [])), "out_dir": str(out_dir)})


# -------------------------------------------------------------------------------------------------
# Sub-APP: export
# -------------------------------------------------------------------------------------------------
export_app = typer.Typer(help="Export utilities")
app.add_typer(export_app, name="export")


@export_app.command("kaggle-notebook")
def export_kaggle_notebook(
    out_path: Path = typer.Option(Path("notebooks/wde_kaggle.ipynb"), "--out", "-o", help="Notebook output path"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite if exists"),
) -> None:
    """
    Write a ready-to-upload Kaggle notebook (minimal, runnable locally).
    The generated notebook:
      - Shows the WDE pipeline stages
      - Demonstrates data directories and config reads
      - Runs tiny no-op / placeholder logic so it executes quickly
    """
    from .utils.io import write_ipynb_minimal  # lazy import

    if out_path.exists() and not overwrite:
        raise typer.Exit(
            code=1,
        )

    ensure_dir(out_path.parent)
    nb = _build_minimal_kaggle_notebook_json()
    write_ipynb_minimal(out_path, nb)
    console.print(Panel.fit(f"Wrote Kaggle notebook → [bold]{out_path}[/bold]", border_style="cyan"))


def _build_minimal_kaggle_notebook_json() -> dict:
    """
    Return a minimal, runnable Jupyter notebook JSON compatible with Kaggle.
    The cells import WDE, print versions, and outline the CLI-equivalent calls.
    """
    # Note: We keep this hand-crafted to avoid importing nbformat at runtime.
    md1 = (
        "# World Discovery Engine (WDE) — Kaggle Notebook\n"
        "This minimal notebook demonstrates the WDE pipeline stages (data→preprocess→detect→dossier) "
        "with placeholder logic. Replace with real data connections and models as you develop."
    )
    code1 = (
        "import os, json, sys, pathlib\n"
        "print('Python:', sys.version)\n"
        "print('CWD:', pathlib.Path.cwd())\n"
        "os.environ.setdefault('WDE_DATA_ROOT', './data')\n"
        "os.environ.setdefault('WDE_ARTIFACTS_ROOT', './artifacts')\n"
        "print('WDE_DATA_ROOT=', os.environ['WDE_DATA_ROOT'])\n"
        "print('WDE_ARTIFACTS_ROOT=', os.environ['WDE_ARTIFACTS_ROOT'])\n"
    )
    code2 = (
        "from wde import get_version\n"
        "print('WDE version:', get_version())\n"
        "from wde.utils.io import ensure_dir\n"
        "ensure_dir('data/raw'); ensure_dir('data/processed'); ensure_dir('artifacts/candidates'); ensure_dir('artifacts/dossiers')\n"
        "print('Verified minimal directory structure.')\n"
    )
    code3 = (
        "# Simulate pipeline manifests\n"
        "from wde.utils.io import write_json\n"
        "write_json('data/raw/fetch_manifest.json', {'files': []})\n"
        "write_json('data/processed/preprocess_manifest.json', {'layers': []})\n"
        "write_json('artifacts/candidates/candidates.json', {'candidates': []})\n"
        "write_json('artifacts/dossiers/dossiers_manifest.json', {'dossiers': []})\n"
        "print('Wrote placeholder manifests.')\n"
    )
    md2 = (
        "## Next Steps\n"
        "1. Implement real data fetchers in `src/wde/data/fetch.py` using your provider APIs.\n"
        "2. Flesh out `src/wde/data/preprocess.py` to tile/reproject/normalize layers.\n"
        "3. Replace `src/wde/detect/run.py` with your model inference and heuristics.\n"
        "4. Enrich `src/wde/dossier/build.py` to render maps, overlays, and uncertainty.\n"
        "5. Keep configs in `configs/*.yaml` and version data with DVC."
    )
    def nb_cell(source: str, ctype: str) -> dict:
        if ctype == "markdown":
            return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}

    return {
        "cells": [
            nb_cell(md1, "markdown"),
            nb_cell(code1, "code"),
            nb_cell(code2, "code"),
            nb_cell(code3, "code"),
            nb_cell(md2, "markdown"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": sys.version.split()[0]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }