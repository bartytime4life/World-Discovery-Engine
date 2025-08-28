# FILE: world_engine/cli.py
"""
World Discovery Engine (WDE) — Typer CLI

Usage examples
--------------
# Full pipeline
python -m world_engine.cli full-run --config configs/pipeline.yaml

# Stage-by-stage
python -m world_engine.cli ingest   --config configs/pipeline.yaml
python -m world_engine.cli scan     --config configs/pipeline.yaml
python -m world_engine.cli evaluate --config configs/pipeline.yaml
python -m world_engine.cli verify   --config configs/pipeline.yaml
python -m world_engine.cli report   --config configs/pipeline.yaml

# Show version & run hash
python -m world_engine.cli version --config configs/pipeline.yaml
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import typer

from .utils.config_loader import load_yaml, resolve_config
from .utils.logging_utils import get_logger, init_logging
from .ingest import run_ingest
from .detect import run_detect
from .evaluate import run_evaluate
from .verify import run_verify
from .report import run_report  # must exist
# Optional import: per-site manifest util (if present in report.py)
try:
    from .report import dump_site_manifest as _dump_site_manifest  # type: ignore
except Exception:  # pragma: no cover
    _dump_site_manifest = None  # fallback to CLI-based manifest builder

app = typer.Typer(add_completion=False, help="World Discovery Engine (WDE) CLI")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hash_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _load_verify_geojson(out_dir: Path) -> Dict[str, Any]:
    gj = out_dir / "verify_candidates.geojson"
    if gj.exists():
        return _load_json(gj)
    return {"type": "FeatureCollection", "features": []}


def _derive_site_id(idx: int, props: Dict[str, Any]) -> str:
    # Prefer explicit IDs if available; otherwise stable index
    return str(props.get("site_id") or props.get("tile_id") or f"{idx:03d}")


def _manifest_from_feature(
    feature: Dict[str, Any],
    site_id: str,
    reports_dir: Path,
    geojson_rel_root: Path,
) -> Dict[str, Any]:
    geom = feature.get("geometry") or {}
    coords = None
    if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), (list, tuple)):
        # GeoJSON Point is [lon, lat]
        lon, lat = geom["coordinates"][:2]
        coords = {"lat": lat, "lon": lon}
    props = feature.get("properties") or {}
    manifest = {
        "site_id": site_id,
        "generated_at": _utc_now_iso(),
        "coordinates": coords,
        "properties": props,
        "reports": {
            "markdown": str((reports_dir / f"candidate_{site_id}.md").as_posix()),
            "html": str((reports_dir / f"candidate_{site_id}.html").as_posix()),
        },
        "source": {
            "verify_candidates_geojson": str((geojson_rel_root / "verify_candidates.geojson").as_posix()),
        },
    }
    return manifest


def _write_per_site_manifests_from_geojson(
    out_dir: Path, reports_dir: Path, log, strict: bool = False
) -> Tuple[int, Dict[str, str]]:
    """
    Fallback path for generating per-site manifests using verify_candidates.geojson
    when report.dump_site_manifest is not available.
    """
    gj = _load_verify_geojson(out_dir)
    feats = gj.get("features", []) or []
    count = 0
    index: Dict[str, str] = {}
    for i, feat in enumerate(feats, 1):
        props = feat.get("properties") or {}
        site_id = _derive_site_id(i, props)
        manifest = _manifest_from_feature(
            feature=feat,
            site_id=site_id,
            reports_dir=reports_dir,
            geojson_rel_root=out_dir,
        )
        path = reports_dir / f"candidate_{site_id}_manifest.json"
        try:
            _write_json(path, manifest)
            index[site_id] = str(path.relative_to(out_dir))
            count += 1
        except Exception as e:  # pragma: no cover
            msg = f"[WDE] Failed writing site manifest for {site_id}: {e}"
            if strict:
                raise RuntimeError(msg) from e
            log.warning(msg)
    return count, index


def _record_run_manifest(
    out_dir: Path,
    artifacts: Dict[str, Any],
    cfg_path: Optional[Path],
    cfg_obj: Dict[str, Any],
) -> Path:
    run_info = {
        "pipeline_version": _safe_get(cfg_obj, "run", "version", default="v0.0.0"),
        "timestamp_utc": _utc_now_iso(),
        "config_path": str(cfg_path.as_posix()) if cfg_path else None,
        "config_hash": _hash_file(cfg_path) if cfg_path and cfg_path.exists() else None,
        "artifacts": artifacts,
    }
    path = out_dir / "run_manifest.json"
    _write_json(path, run_info)
    return path


# --------------------------------------------------------------------------------------
# Global callback: common options (verbosity later if desired)
# --------------------------------------------------------------------------------------


@app.callback(invoke_without_command=False)
def _root() -> None:
    """World Discovery Engine (WDE) — CLI entrypoint."""
    # No-op; reserved for shared/global flags in future (e.g., --verbose)
    return


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


@app.command("version")
def cli_version(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to pipeline config YAML.")
):
    """
    Show CLI version and (if config provided) its hash for reproducibility.
    """
    # Version strategy: derive from environment variable or fallback constant
    cli_version = os.environ.get("WDE_CLI_VERSION", "1.0.0")
    cfg_hash = None
    if config:
        p = Path(config)
        if p.exists():
            cfg_hash = _hash_file(p)
    typer.echo(json.dumps({"wde_cli_version": cli_version, "config_hash": cfg_hash}, indent=2))


@app.command("selftest")
def cli_selftest(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
):
    """
    Quick environment & config sanity checks.
    """
    cfg = resolve_config(config)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging(cfg)
    log = get_logger("wde.cli")

    # Smoke checks
    required_sections = ["run", "pipeline", "ethics", "output"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise typer.Exit(code=2)

    # Hash config and write a tiny report
    report = {
        "timestamp_utc": _utc_now_iso(),
        "config_path": str(Path(config).resolve().as_posix()),
        "config_hash": _hash_file(Path(config)),
        "output_dir": str(out_dir.as_posix()),
        "status": "ok" if not missing else "missing",
        "missing_sections": missing,
    }
    _write_json(out_dir / "selftest_report.json", report)
    log.info("Selftest passed. Wrote outputs/selftest_report.json")


@app.command("full-run")
def full_run(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
    overrides: Optional[str] = typer.Option(
        None,
        "--override",
        "-o",
        help='JSON string of key=value overrides, e.g. \'{"aoi":{"bbox":[-70,-13,-62,-6]}}\'',
    ),
    skip_manifests: bool = typer.Option(
        False, "--skip-manifests", help="Skip writing per-site manifest.json files in report stage."
    ),
):
    """
    Run the full discovery funnel: ingest → detect → evaluate → verify → report.
    Also writes outputs/run_manifest.json and per-site manifests (unless skipped).
    """
    cfg = resolve_config(config, overrides_json=overrides)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    init_logging(cfg)
    log = get_logger("wde.cli")

    log.info("=== WDE :: FULL RUN ===")
    artifacts: Dict[str, Any] = {}

    if _safe_get(cfg, "pipeline", "ingest", "enabled", default=True):
        artifacts["ingest"] = run_ingest(cfg)
    else:
        log.warning("Ingest stage disabled by config.")

    if _safe_get(cfg, "pipeline", "detect", "enabled", default=True):
        artifacts["detect"] = run_detect(cfg, prev=artifacts.get("ingest"))
    else:
        log.warning("Detect stage disabled by config.")

    if _safe_get(cfg, "pipeline", "evaluate", "enabled", default=True):
        artifacts["evaluate"] = run_evaluate(cfg, prev=artifacts.get("detect"))
    else:
        log.warning("Evaluate stage disabled by config.")

    if _safe_get(cfg, "pipeline", "verify", "enabled", default=True):
        artifacts["verify"] = run_verify(cfg, prev=artifacts.get("evaluate"))
    else:
        log.warning("Verify stage disabled by config.")

    report_enabled = _safe_get(cfg, "pipeline", "report", "enabled", default=True)
    if report_enabled:
        summary = run_report(cfg, prev=artifacts.get("verify"))
        artifacts["report"] = summary

        # Always write per-site manifests unless user opts out
        if not skip_manifests:
            reports_dir = Path(summary["reports_dir"]).resolve()
            written, idx = _ensure_site_manifests(cfg, reports_dir, log=log)
            artifacts["report"]["per_site_manifests"] = idx
            log.info(f"[WDE] Wrote {written} per-site manifest.json files")
        else:
            log.warning("[WDE] Skipping per-site manifests by user request (--skip-manifests)")
    else:
        log.warning("Report stage disabled by config.")

    # Write run manifest (includes config hash & artifact pointers)
    run_manifest_path = _record_run_manifest(
        out_dir=out_dir, artifacts=artifacts, cfg_path=Path(config), cfg_obj=cfg
    )
    log.info(f"Full run completed. Artifacts manifest → {run_manifest_path.as_posix()}")


@app.command("ingest")
def cli_ingest(config: str = typer.Option(..., "--config", "-c")):
    """Run only the ingest stage."""
    cfg = resolve_config(config)
    init_logging(cfg)
    run_ingest(cfg)


@app.command("scan")
def cli_scan(config: str = typer.Option(..., "--config", "-c")):
    """Run only the detect (scan) stage."""
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
def cli_report(
    config: str = typer.Option(..., "--config", "-c"),
    skip_manifests: bool = typer.Option(
        False, "--skip-manifests", help="Skip writing per-site manifest.json files."
    ),
):
    """
    Run only the report stage.
    Also writes per-site manifest.json files (unless --skip-manifests).
    """
    cfg = resolve_config(config)
    init_logging(cfg)
    summary = run_report(cfg)
    log = get_logger("wde.cli")
    if skip_manifests:
        log.warning("[WDE] Skipping per-site manifests by user request (--skip-manifests)")
        return
    reports_dir = Path(summary["reports_dir"]).resolve()
    written, _ = _ensure_site_manifests(cfg, reports_dir, log=log)
    log.info(f"[WDE] Wrote {written} per-site manifest.json files")


# --------------------------------------------------------------------------------------
# Internal: ensure per-site manifests are created (prefer report's helper, else fallback)
# --------------------------------------------------------------------------------------


def _ensure_site_manifests(cfg: Dict[str, Any], reports_dir: Path, log) -> Tuple[int, Dict[str, str]]:
    """
    Generate per-site manifest.json for each candidate in the report.
    Prefer the report.dump_site_manifest helper if available; otherwise build from verify_candidates.geojson.
    Returns (count_written, site_id->manifest_relpath_index).
    """
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, str] = {}
    written = 0

    # Fast path: if report exposed dump_site_manifest and report generated an index we can use
    if _dump_site_manifest is not None:
        # Attempt to derive features from verify_candidates.geojson for site list
        gj = _load_verify_geojson(out_dir)
        feats = gj.get("features", []) or []
        for i, feat in enumerate(feats, 1):
            props = feat.get("properties") or {}
            site_id = _derive_site_id(i, props)
            lat = None
            lon = None
            geom = feat.get("geometry") or {}
            if geom.get("type") == "Point" and isinstance(geom.get("coordinates"), (list, tuple)):
                lon, lat = geom["coordinates"][:2]
            candidate_stub = {
                "site_id": site_id,
                "tile_id": props.get("tile_id", site_id),
                "center": (lat, lon) if (lat is not None and lon is not None) else None,
                "score": props.get("score"),
                "ade_modalities": props.get("ade_modalities", []),
                "uncertainty": props.get("uncertainty"),
            }
            try:
                path = _dump_site_manifest(candidate_stub, reports_dir)  # type: ignore
                idx[site_id] = str(Path(path).relative_to(out_dir))
                written += 1
            except Exception as e:  # pragma: no cover
                log.warning(f"[WDE] dump_site_manifest failed for {site_id}: {e}")

        if written:
            _write_json(out_dir / "manifest_index.json", {"sites": idx})
            return written, idx

    # Fallback: build manifests directly from GeoJSON and our CLI helper
    w, idx2 = _write_per_site_manifests_from_geojson(out_dir=out_dir, reports_dir=reports_dir, log=log)
    if w:
        _write_json(out_dir / "manifest_index.json", {"sites": idx2})
    return w, idx2


if __name__ == "__main__":
    app()
