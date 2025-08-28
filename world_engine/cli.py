"""
WDE Typer CLI

Usage examples
--------------
# Full pipeline
python -m world_engine.cli full-run --config configs/pipeline.yaml

# Stage-by-stage
python -m world_engine.cli ingest  --config configs/pipeline.yaml
python -m world_engine.cli scan    --config configs/pipeline.yaml
python -m world_engine.cli evaluate --config configs/pipeline.yaml
python -m world_engine.cli verify  --config configs/pipeline.yaml
python -m world_engine.cli report  --config configs/pipeline.yaml
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .utils.config_loader import load_yaml, resolve_config
from .utils.logging_utils import get_logger, init_logging
from .ingest import run_ingest
from .detect import run_detect
from .evaluate import run_evaluate
from .verify import run_verify
from .report import run_report

app = typer.Typer(add_completion=False, help="World Discovery Engine (WDE) CLI")

@app.command("full-run")
def full_run(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
    overrides: Optional[str] = typer.Option(
        None,
        "--override",
        "-o",
        help='JSON string of key=value overrides, e.g. \'{"aoi":{"bbox":[-70,-13,-62,-6]}}\'',
    ),
):
    """Run the full discovery funnel: ingest → detect → evaluate → verify → report."""
    cfg = resolve_config(config, overrides_json=overrides)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    init_logging(cfg)
    log = get_logger("wde.cli")

    log.info("=== WDE :: FULL RUN ===")
    artifacts = {}

    if cfg["pipeline"]["ingest"]["enabled"]:
        artifacts["ingest"] = run_ingest(cfg)
    else:
        log.warning("Ingest stage disabled by config.")

    if cfg["pipeline"]["detect"]["enabled"]:
        artifacts["detect"] = run_detect(cfg, prev=artifacts.get("ingest"))
    else:
        log.warning("Detect stage disabled by config.")

    if cfg["pipeline"]["evaluate"]["enabled"]:
        artifacts["evaluate"] = run_evaluate(cfg, prev=artifacts.get("detect"))
    else:
        log.warning("Evaluate stage disabled by config.")

    if cfg["pipeline"]["verify"]["enabled"]:
        artifacts["verify"] = run_verify(cfg, prev=artifacts.get("evaluate"))
    else:
        log.warning("Verify stage disabled by config.")

    if cfg["pipeline"]["report"]["enabled"]:
        artifacts["report"] = run_report(cfg, prev=artifacts.get("verify"))
    else:
        log.warning("Report stage disabled by config.")

    # Save a simple run manifest to outputs/
    (out_dir / "run_manifest.json").write_text(json.dumps(artifacts, indent=2))
    log.info("Full run completed. Artifacts manifest written to outputs/run_manifest.json")


@app.command("ingest")
def cli_ingest(config: str = typer.Option(..., "--config", "-c")):
    """Run only the ingest stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_ingest(cfg)


@app.command("scan")
def cli_scan(config: str = typer.Option(..., "--config", "-c")):
    """Run only the detect stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_detect(cfg)


@app.command("evaluate")
def cli_evaluate(config: str = typer.Option(..., "--config", "-c")):
    """Run only the evaluate stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_evaluate(cfg)


@app.command("verify")
def cli_verify(config: str = typer.Option(..., "--config", "-c")):
    """Run only the verify stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_verify(cfg)


@app.command("report")
def cli_report(config: str = typer.Option(..., "--config", "-c")):
    """Run only the report stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_report(cfg)


if __name__ == "__main__":
    app()
