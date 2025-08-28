# FILE: world_engine/cli.py
# =============================================================================
# 🌍 World Discovery Engine (WDE) — Typer CLI (Upgraded)
#
# Usage examples
# --------------
# Full pipeline
#   python -m world_engine.cli full-run --config configs/pipeline.yaml
#
# Stage-by-stage
#   python -m world_engine.cli ingest   --config configs/pipeline.yaml
#   python -m world_engine.cli scan     --config configs/pipeline.yaml
#   python -m world_engine.cli evaluate --config configs/pipeline.yaml
#   python -m world_engine.cli verify   --config configs/pipeline.yaml
#   python -m world_engine.cli report   --config configs/pipeline.yaml
#
# Show version & run hash
#   python -m world_engine.cli version --config configs/pipeline.yaml
#
# New conveniences
# ----------------
#   - --override/-o '{"aoi":{"bbox":[-3.5,-60.5,-3.4,-60.4]}}'  (JSON overrides)
#   - --dry-run (preview commands)
#   - selftest            (quick env/config smoke-test)
#   - code-hash           (hash tracked code/config set for provenance)
#   - env                 (print environment snapshot for reproducibility)
#   - full-run --resume-from <stage>  (resume mid-pipeline)
#   - per-site manifest generation fallback if report helper missing
# =============================================================================
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import typer

# --- Project utils & stages (must exist in repo) --------------------------------------
from .utils.config_loader import load_yaml, resolve_config
from .utils.logging_utils import get_logger, init_logging
from .ingest import run_ingest
from .detect import run_detect
from .evaluate import run_evaluate
from .verify import run_verify
from .report import run_report  # must exist

# Optional import: per-site manifest util (if present in report.py)
try:
    from .report import dump_site_manifest as _dump_site_manifest  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _dump_site_manifest = None  # fallback to CLI-based manifest builder

app = typer.Typer(add_completion=False, help="World Discovery Engine (WDE) — Pipeline CLI")

# --------------------------------------------------------------------------------------
# Helpers — time, hashing, IO
# --------------------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(d: Dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _repo_files_for_hash(
    roots: Iterable[Path],
    include_ext: Tuple[str, ...] = (".py", ".yaml", ".yml", ".toml", ".json", ".md"),
    exclude_dirs: Tuple[str, ...] = (".git", ".cache", "__pycache__", ".venv", "venv", "artifacts", "outputs"),
) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir():
                # skip excluded directories
                if p.name in exclude_dirs:
                    continue
            else:
                if p.suffix.lower() in include_ext:
                    # exclude common generated / heavy dirs by path substring
                    parts = set(p.parts)
                    if parts.intersection(exclude_dirs):
                        continue
                    files.append(p)
    files.sort()
    return files


def _hash_tree(files: List[Path]) -> Dict[str, Any]:
    entries = []
    h = hashlib.sha256()
    for p in files:
        digest = _sha256_file(p)
        entries.append({"path": str(p.as_posix()), "sha256": digest.split(":", 1)[1]})
        h.update(digest.encode("utf-8"))
    return {"tree_hash": "sha256:" + h.hexdigest(), "files": entries, "count": len(entries)}


def _record_run_manifest(
    out_dir: Path,
    artifacts: Dict[str, Any],
    cfg_path: Optional[Path],
    cfg_obj: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    run_info = {
        "pipeline_version": _safe_get(cfg_obj, "run", "version", default="v0.0.0"),
        "timestamp_utc": _utc_now_iso(),
        "config_path": str(cfg_path.as_posix()) if cfg_path else None,
        "config_hash": _sha256_file(cfg_path) if cfg_path and cfg_path.exists() else None,
        "artifacts": artifacts,
        "environment": _env_snapshot_dict(),
    }
    if extra:
        run_info.update(extra)
    path = out_dir / "run_manifest.json"
    _write_json(path, run_info)
    return path


def _env_snapshot_dict() -> Dict[str, Any]:
    py = sys.version.replace("\n", " ")
    snap = {
        "python": py,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
    }
    try:
        import torch  # type: ignore

        snap["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        }
    except Exception:
        snap["torch"] = None
    return snap


# --------------------------------------------------------------------------------------
# GeoJSON helpers for per-site manifests
# --------------------------------------------------------------------------------------


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
            "pdf": str((reports_dir / f"candidate_{site_id}.pdf").as_posix()),
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

    # Preferred path: report.dump_site_manifest exists
    if _dump_site_manifest is not None:
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
                path = _dump_site_manifest(candidate_stub, reports_dir)  # type: ignore[attr-defined]
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


# --------------------------------------------------------------------------------------
# CLI Commands
# --------------------------------------------------------------------------------------


@app.command("version")
def cli_version(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to pipeline config YAML.")
):
    """
    Show CLI version and (if config provided) its hash for reproducibility.
    """
    cli_version = os.environ.get("WDE_CLI_VERSION", "1.1.0")
    cfg_hash = None
    if config:
        p = Path(config)
        if p.exists():
            cfg_hash = _sha256_file(p)
    payload = {
        "wde_cli_version": cli_version,
        "config_hash": cfg_hash,
        "timestamp_utc": _utc_now_iso(),
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command("env")
def cli_env():
    """
    Print environment snapshot (python/platform/torch) for reproducibility logs.
    """
    typer.echo(json.dumps(_env_snapshot_dict(), indent=2))


@app.command("selftest")
def cli_selftest(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML.")
):
    """
    Quick environment & config sanity checks.
    """
    cfg = resolve_config(config)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    init_logging(cfg)
    log = get_logger("wde.cli")

    required_sections = ["run", "pipeline", "ethics", "output"]
    missing = [s for s in required_sections if s not in cfg]

    # tiny IO + hash write
    report = {
        "timestamp_utc": _utc_now_iso(),
        "config_path": str(Path(config).resolve().as_posix()),
        "config_hash": _sha256_file(Path(config)),
        "output_dir": str(out_dir.as_posix()),
        "status": "ok" if not missing else "missing",
        "missing_sections": missing,
        "environment": _env_snapshot_dict(),
    }
    _write_json(out_dir / "selftest_report.json", report)
    if missing:
        log.error("Selftest found missing sections: %s", missing)
        raise typer.Exit(code=2)
    log.info("Selftest passed. → outputs/selftest_report.json")


@app.command("code-hash")
def cli_code_hash(
    roots: List[str] = typer.Argument(
        ["world_engine", "configs"], help="Directories to hash for provenance."
    ),
    out: Optional[str] = typer.Option(None, "--out", help="Write JSON to this path."),
):
    """
    Hash the code/config tree to capture a compact provenance fingerprint.
    """
    paths = [Path(r) for r in roots]
    files = _repo_files_for_hash(paths)
    tree = _hash_tree(files)
    if out:
        _write_json(Path(out), tree)
        typer.echo(f"Wrote code hash JSON → {out}")
    else:
        typer.echo(json.dumps(tree, indent=2))


# ------------------------------- STAGES ---------------------------------------


@app.command("full-run")
def full_run(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
    overrides: Optional[str] = typer.Option(
        None,
        "--override",
        "-o",
        help='JSON string of key=value overrides, e.g. \'{"aoi":{"bbox":[-70,-13,-62,-6]}}\'',
    ),
    resume_from: Optional[str] = typer.Option(
        None,
        "--resume-from",
        help="Resume from stage: ingest|scan|evaluate|verify|report",
    ),
    skip_manifests: bool = typer.Option(
        False, "--skip-manifests", help="Skip writing per-site manifest.json files in report stage."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print planned stages but do not execute."),
):
    """
    Run the full discovery funnel: ingest → scan → evaluate → verify → report.
    Writes outputs/run_manifest.json and per-site manifests (unless skipped).
    """
    cfg = resolve_config(config, overrides_json=overrides)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    init_logging(cfg)
    log = get_logger("wde.cli")
    log.info("=== WDE :: FULL RUN ===")
    artifacts: Dict[str, Any] = {}

    order = ["ingest", "scan", "evaluate", "verify", "report"]
    if resume_from:
        resume_from = resume_from.strip().lower()
        if resume_from not in order:
            raise typer.BadParameter("resume-from must be one of: ingest|scan|evaluate|verify|report")
        # trim pipeline to resume point
        start_idx = order.index(resume_from)
        order = order[start_idx:]

    def _would_run(stage: str) -> bool:
        enabled = _safe_get(cfg, "pipeline", stage, "enabled", default=True)
        if not enabled:
            log.warning("%s stage disabled by config.", stage.capitalize())
            return False
        return True

    # Preview mode
    if dry_run:
        plan = [s for s in order if _would_run(s)]
        typer.echo(json.dumps({"plan": plan, "resume_from": resume_from, "dry_run": True}, indent=2))
        return

    # Execute in order (with dependency passing in memory via return values)
    prev = None
    if "ingest" in order and _would_run("ingest"):
        artifacts["ingest"] = run_ingest(cfg)
        prev = artifacts["ingest"]

    if "scan" in order and _would_run("scan"):
        artifacts["detect"] = run_detect(cfg, prev=prev)  # keep artifact key "detect"
        prev = artifacts["detect"]

    if "evaluate" in order and _would_run("evaluate"):
        artifacts["evaluate"] = run_evaluate(cfg, prev=prev)
        prev = artifacts["evaluate"]

    if "verify" in order and _would_run("verify"):
        artifacts["verify"] = run_verify(cfg, prev=prev)
        prev = artifacts["verify"]

    if "report" in order and _would_run("report"):
        summary = run_report(cfg, prev=prev)
        artifacts["report"] = summary

        if not skip_manifests:
            reports_dir = Path(summary["reports_dir"]).resolve()
            written, idx = _ensure_site_manifests(cfg, reports_dir, log=log)
            artifacts["report"]["per_site_manifests"] = idx
            log.info(f"[WDE] Wrote {written} per-site manifest.json files")
        else:
            log.warning("[WDE] Skipping per-site manifests by user request (--skip-manifests)")

    # Provenance: code/config hash tree
    code_tree = _hash_tree(_repo_files_for_hash([Path("world_engine"), Path("configs")]))

    # Write run manifest
    run_manifest_path = _record_run_manifest(
        out_dir=out_dir,
        artifacts=artifacts,
        cfg_path=Path(config),
        cfg_obj=cfg,
        extra={"code_tree": code_tree},
    )
    log.info("Full run completed. Artifacts manifest → %s", run_manifest_path.as_posix())


@app.command("ingest")
def cli_ingest(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
):
    """Run only the ingest stage."""
    cfg = resolve_config(config, overrides_json=overrides)
    init_logging(cfg)
    if dry_run:
        typer.echo(json.dumps({"stage": "ingest", "dry_run": True}, indent=2))
        return
    run_ingest(cfg)


@app.command("scan")
def cli_scan(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
):
    """Run only the detect (scan) stage."""
    cfg = resolve_config(config, overrides_json=overrides)
    init_logging(cfg)
    if dry_run:
        typer.echo(json.dumps({"stage": "scan", "dry_run": True}, indent=2))
        return
    run_detect(cfg)


@app.command("evaluate")
def cli_evaluate(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
):
    """Run only the evaluate stage."""
    cfg = resolve_config(config, overrides_json=overrides)
    init_logging(cfg)
    if dry_run:
        typer.echo(json.dumps({"stage": "evaluate", "dry_run": True}, indent=2))
        return
    run_evaluate(cfg)


@app.command("verify")
def cli_verify(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
):
    """Run only the verify stage."""
    cfg = resolve_config(config, overrides_json=overrides)
    init_logging(cfg)
    if dry_run:
        typer.echo(json.dumps({"stage": "verify", "dry_run": True}, indent=2))
        return
    run_verify(cfg)


@app.command("report")
def cli_report(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    skip_manifests: bool = typer.Option(
        False, "--skip-manifests", help="Skip writing per-site manifest.json files."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
):
    """
    Run only the report stage.
    Also writes per-site manifest.json files (unless --skip-manifests).
    """
    cfg = resolve_config(config, overrides_json=overrides)
    init_logging(cfg)
    if dry_run:
        typer.echo(json.dumps({"stage": "report", "dry_run": True}, indent=2))
        return
    summary = run_report(cfg)
    log = get_logger("wde.cli")
    if skip_manifests:
        log.warning("[WDE] Skipping per-site manifests by user request (--skip-manifests)")
        return
    reports_dir = Path(summary["reports_dir"]).resolve()
    written, _ = _ensure_site_manifests(cfg, reports_dir, log=log)
    log.info(f"[WDE] Wrote {written} per-site manifest.json files")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------


@app.callback(invoke_without_command=False)
def _root() -> None:
    """World Discovery Engine (WDE) — CLI entrypoint."""
    return


if __name__ == "__main__":
    app()
