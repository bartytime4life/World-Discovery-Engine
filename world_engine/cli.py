# FILE: world_engine/cli.py
# =============================================================================
# 🌍 World Discovery Engine (WDE) — Typer CLI (Ultimate Upgrade + Run Logging)
#
# What's new in this upgrade
# --------------------------
# 1) Logging controls on every command:
#    --run-id auto|<str>   → stamps file/JSON logs with a stable run id (auto = ts + git short hash)
#    --log-level LEVEL     → overrides config.logging.level (INFO|DEBUG|...)
#    --log-file/--no-log-file, --log-json/--no-log-json → force on/off regardless of config
#
# 2) Config + provenance helpers:
#    - effective-config     Emit fully resolved config (after overrides) to JSON or YAML
#    - code-hash            Tree hash of repo (provenance for manifests)
#
# 3) Docs & artifacts quality-of-life:
#    - docs                 View key project docs from CLI (architecture/datasets/ethics/repo)
#    - validate-artifacts   Run artifacts validator (scripts/validate_artifacts.py) on outputs/
#    - report-open          Open reports/index.html (cross-platform best-effort)
#
# 4) Script wrappers (thin shims to existing scripts/):
#    - fetch-data           scripts/fetch_datasets.sh (bbox/flags passthrough)
#    - export-kaggle        scripts/export_kaggle.sh (owner/slug/etc.)
#    - generate-reports     scripts/generate_reports.sh (zip/mask/etc.)
#    - profile              scripts/profiling_tools.py (cprofile|memory|timers|pyspy)
#
# Existing features kept and improved:
#  - full-run, ingest, scan, evaluate, verify, report
#  - --override JSON patch; --dry-run preview plan
#  - selftest, env, version
#  - per-site manifest fallback if dump helper absent
#
# Usage examples
# --------------
#   # Full pipeline with explicit run logging (file + JSONL + auto run-id)
#   python -m world_engine.cli full-run -c configs/pipeline.yaml --run-id auto --log-file --log-json
#
#   # Docs & config
#   python -m world_engine.cli docs --section architecture
#   python -m world_engine.cli effective-config -c configs/pipeline.yaml -o resolved.yaml
#
#   # Validation & artifacts
#   python -m world_engine.cli validate-artifacts --root outputs --strict
#   python -m world_engine.cli report-open --root outputs
#
#   # Scripts wrappers (assumes scripts/* exist and are executable)
#   python -m world_engine.cli fetch-data --aoi-bbox "-60.50,-3.50,-60.40,-3.40" --dem --hydro --s2
#   python -m world_engine.cli export-kaggle --owner you --name wde-ade --notebook notebooks/ade_discovery_pipeline.ipynb --make-zip
#   python -m world_engine.cli generate-reports --root outputs --zip --mask
#
#   # Profiling
#   python -m world_engine.cli profile cprofile --cmd "python -m world_engine.cli full-run -c configs/pipeline.yaml"
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import click  # pager for docs
import typer

# --- Project utils & stages ---------------------------------------------------
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

# Optional import: yaml for effective-config YAML output
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # we will fallback to JSON if PyYAML is absent

app = typer.Typer(add_completion=False, help="World Discovery Engine (WDE) — Pipeline CLI")

# =============================================================================
# Helpers — time, hashing, IO
# =============================================================================


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
                if p.name in exclude_dirs:
                    continue
            else:
                if p.suffix.lower() in include_ext:
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


# =============================================================================
# Run-id & Logging overrides
# =============================================================================


def _git_short_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _auto_run_id() -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    g = _git_short_hash()
    return f"{ts}_{g}" if g else ts


def _merge_logging_overrides(cfg: Dict[str, Any],
                             level: Optional[str],
                             to_file: Optional[bool],
                             to_json: Optional[bool]) -> Dict[str, Any]:
    c = dict(cfg or {})
    lc = dict(c.get("logging", {}) or {})
    if level:
        lc["level"] = level
    if to_file is not None:
        lc["to_file"] = bool(to_file)
    if to_json is not None:
        lc["to_json"] = bool(to_json)
    # default logs dir if user didn't set
    lc.setdefault("dir", "logs")
    c["logging"] = lc
    return c


def _bootstrap_logging(cfg: Dict[str, Any],
                       run_id: Optional[str],
                       level: Optional[str],
                       to_file: Optional[bool],
                       to_json: Optional[bool]) -> Tuple[Dict[str, Any], str]:
    """
    Apply CLI logging overrides, compute run_id (auto|str), and initialize logging.
    Returns (merged_cfg, resolved_run_id).
    """
    merged = _merge_logging_overrides(cfg, level, to_file, to_json)
    rid = _auto_run_id() if (run_id == "auto" or not run_id) else run_id
    # init_logging from utils accepts (cfg, run_id) in upgraded version
    try:
        init_logging(merged, run_id=rid)  # upgraded utils.logging_utils supports run_id
    except TypeError:
        # fallback to older signature init_logging(cfg)
        init_logging(merged)  # type: ignore[arg-type]
    log = get_logger("wde.cli")
    log.info("[RunMeta] run_id=%s cfg_hash=%s", rid, _sha256_bytes(json.dumps(merged, sort_keys=True).encode("utf-8")))
    return merged, rid


# =============================================================================
# GeoJSON helpers for per-site manifests
# =============================================================================


def _load_verify_geojson(out_dir: Path) -> Dict[str, Any]:
    gj = out_dir / "verify_candidates.geojson"
    if gj.exists():
        return _load_json(gj)
    return {"type": "FeatureCollection", "features": []}


def _derive_site_id(idx: int, props: Dict[str, Any]) -> str:
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
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    idx: Dict[str, str] = {}
    written = 0

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

    w, idx2 = _write_per_site_manifests_from_geojson(out_dir=out_dir, reports_dir=reports_dir, log=log)
    if w:
        _write_json(out_dir / "manifest_index.json", {"sites": idx2})
    return w, idx2


# =============================================================================
# Core CLI Commands
# =============================================================================


@app.command("version")
def cli_version(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to pipeline config YAML.")
):
    cli_version = os.environ.get("WDE_CLI_VERSION", "1.3.0")
    cfg_hash = None
    if config:
        p = Path(config)
        if p.exists():
            cfg_hash = _sha256_file(p)
    payload = {"wde_cli_version": cli_version, "config_hash": cfg_hash, "timestamp_utc": _utc_now_iso()}
    typer.echo(json.dumps(payload, indent=2))


@app.command("env")
def cli_env():
    typer.echo(json.dumps(_env_snapshot_dict(), indent=2))


@app.command("selftest")
def cli_selftest(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
    run_id: str = typer.Option("auto", "--run-id", help='Run identifier ("auto" => timestamp+git).'),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override logging level (e.g. INFO, DEBUG)."),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file", help="Enable/disable file logging."),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json", help="Enable/disable JSONL logging."),
):
    cfg = resolve_config(config)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg, rid = _bootstrap_logging(cfg, run_id, log_level, log_file, log_json)
    log = get_logger("wde.cli")

    required_sections = ["run", "pipeline", "ethics", "output"]
    missing = [s for s in required_sections if s not in cfg]

    report = {
        "timestamp_utc": _utc_now_iso(),
        "run_id": rid,
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
    roots: List[str] = typer.Argument(["world_engine", "configs"], help="Directories to hash for provenance."),
    out: Optional[str] = typer.Option(None, "--out", help="Write JSON to this path."),
):
    paths = [Path(r) for r in roots]
    files = _repo_files_for_hash(paths)
    tree = _hash_tree(files)
    if out:
        _write_json(Path(out), tree)
        typer.echo(f"Wrote code hash JSON → {out}")
    else:
        typer.echo(json.dumps(tree, indent=2))


@app.command("effective-config")
def cli_effective_config(
    config: str = typer.Option(..., "--config", "-c", help="Path to YAML config."),
    overrides: Optional[str] = typer.Option(None, "--override", "-o", help="JSON string of overrides."),
    out: Optional[str] = typer.Option(None, "--out", help="Write resolved config to this path (json|yaml)."),
):
    """
    Render the fully-resolved config (after JSON overrides). Handy for debugging & provenance.
    """
    cfg = resolve_config(config, overrides_json=overrides)
    if out:
        outp = Path(out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        if outp.suffix.lower() in (".yml", ".yaml") and yaml is not None:
            outp.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")  # type: ignore[arg-type]
        else:
            _write_json(outp, cfg)
        typer.echo(f"Wrote resolved config → {outp.as_posix()}")
    else:
        typer.echo(json.dumps(cfg, indent=2))


# ------------------------------- STAGES ---------------------------------------


@app.command("full-run")
def full_run(
    config: str = typer.Option(..., "--config", "-c", help="Path to pipeline config YAML."),
    overrides: Optional[str] = typer.Option(None, "--override", "-o", help='JSON string of overrides.'),
    resume_from: Optional[str] = typer.Option(None, "--resume-from", help="ingest|scan|evaluate|verify|report"),
    skip_manifests: bool = typer.Option(False, "--skip-manifests", help="Skip per-site manifests."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
    run_id: str = typer.Option("auto", "--run-id", help='Run identifier ("auto" => timestamp+git).'),
    log_level: Optional[str] = typer.Option(None, "--log-level", help="Override logging level (e.g. INFO, DEBUG)."),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file", help="Enable/disable file logging."),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json", help="Enable/disable JSONL logging."),
):
    cfg = resolve_config(config, overrides_json=overrides)
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, rid = _bootstrap_logging(cfg, run_id, log_level, log_file, log_json)
    log = get_logger("wde.cli")
    log.info("=== WDE :: FULL RUN :: run_id=%s ===", rid)
    artifacts: Dict[str, Any] = {"run_id": rid}

    order = ["ingest", "scan", "evaluate", "verify", "report"]
    if resume_from:
        resume_from = resume_from.strip().lower()
        if resume_from not in order:
            raise typer.BadParameter("resume-from must be one of: ingest|scan|evaluate|verify|report")
        order = order[order.index(resume_from) :]

    def _would_run(stage: str) -> bool:
        enabled = _safe_get(cfg, "pipeline", stage, "enabled", default=True)
        if not enabled:
            log.warning("%s stage disabled by config.", stage.capitalize())
            return False
        return True

    if dry_run:
        plan = [s for s in order if _would_run(s)]
        typer.echo(json.dumps({"plan": plan, "resume_from": resume_from, "dry_run": True, "run_id": rid}, indent=2))
        return

    prev = None
    if "ingest" in order and _would_run("ingest"):
        artifacts["ingest"] = run_ingest(cfg)
        prev = artifacts["ingest"]

    if "scan" in order and _would_run("scan"):
        artifacts["detect"] = run_detect(cfg, prev=prev)
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

    code_tree = _hash_tree(_repo_files_for_hash([Path("world_engine"), Path("configs")]))
    run_manifest_path = _record_run_manifest(
        out_dir=out_dir, artifacts=artifacts, cfg_path=Path(config), cfg_obj=cfg, extra={"code_tree": code_tree}
    )
    log.info("Full run completed. Artifacts manifest → %s", run_manifest_path.as_posix())


def _stage_entry(
    stage_name: str,
    runner,
    config: str,
    overrides: Optional[str],
    dry_run: bool,
    run_id: str,
    log_level: Optional[str],
    log_file: Optional[bool],
    log_json: Optional[bool],
):
    cfg = resolve_config(config, overrides_json=overrides)
    cfg, rid = _bootstrap_logging(cfg, run_id, log_level, log_file, log_json)
    if dry_run:
        typer.echo(json.dumps({"stage": stage_name, "dry_run": True, "run_id": rid}, indent=2))
        return
    return runner(cfg)


@app.command("ingest")
def cli_ingest(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
    run_id: str = typer.Option("auto", "--run-id", help='Run identifier ("auto" => timestamp+git).'),
    log_level: Optional[str] = typer.Option(None, "--log-level"),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file"),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json"),
):
    _stage_entry("ingest", run_ingest, config, overrides, dry_run, run_id, log_level, log_file, log_json)


@app.command("scan")
def cli_scan(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
    run_id: str = typer.Option("auto", "--run-id"),
    log_level: Optional[str] = typer.Option(None, "--log-level"),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file"),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json"),
):
    _stage_entry("scan", run_detect, config, overrides, dry_run, run_id, log_level, log_file, log_json)


@app.command("evaluate")
def cli_evaluate(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
    run_id: str = typer.Option("auto", "--run-id"),
    log_level: Optional[str] = typer.Option(None, "--log-level"),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file"),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json"),
):
    _stage_entry("evaluate", run_evaluate, config, overrides, dry_run, run_id, log_level, log_file, log_json)


@app.command("verify")
def cli_verify(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
    run_id: str = typer.Option("auto", "--run-id"),
    log_level: Optional[str] = typer.Option(None, "--log-level"),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file"),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json"),
):
    _stage_entry("verify", run_verify, config, overrides, dry_run, run_id, log_level, log_file, log_json)


@app.command("report")
def cli_report(
    config: str = typer.Option(..., "--config", "-c"),
    overrides: Optional[str] = typer.Option(None, "--override", "-o"),
    skip_manifests: bool = typer.Option(False, "--skip-manifests", help="Skip per-site manifest.json files."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without executing."),
    run_id: str = typer.Option("auto", "--run-id"),
    log_level: Optional[str] = typer.Option(None, "--log-level"),
    log_file: Optional[bool] = typer.Option(None, "--log-file/--no-log-file"),
    log_json: Optional[bool] = typer.Option(None, "--log-json/--no-log-json"),
):
    cfg = resolve_config(config, overrides_json=overrides)
    cfg, rid = _bootstrap_logging(cfg, run_id, log_level, log_file, log_json)
    if dry_run:
        typer.echo(json.dumps({"stage": "report", "dry_run": True, "run_id": rid}, indent=2))
        return
    summary = run_report(cfg)
    log = get_logger("wde.cli")
    if skip_manifests:
        log.warning("[WDE] Skipping per-site manifests by user request (--skip-manifests)")
        return
    reports_dir = Path(summary["reports_dir"]).resolve()
    written, _ = _ensure_site_manifests(cfg, reports_dir, log=log)
    log.info(f"[WDE] Wrote {written} per-site manifest.json files")


# =============================================================================
# Docs Viewer (architecture / datasets / ethics / repo)
# =============================================================================


@app.command("docs")
def cli_docs(
    section: str = typer.Option(
        "architecture", "--section", "-s", help="architecture | datasets | ethics | repo"
    ),
    pager: bool = typer.Option(True, "--pager/--no-pager", help="Use a scrollable pager."),
    path_only: bool = typer.Option(False, "--path-only", help="Print absolute file path only."),
):
    candidates = {
        "architecture": Path("scripts/ARCHITECTURE.md"),
        "datasets": Path("docs/datasets.md"),
        "ethics": Path("docs/ETHICS_and_GOVERNANCE.md"),
        "repo": Path("docs/REPOSITORY_STRUCTURE.md"),
    }
    key = section.strip().lower()
    if key not in candidates:
        valid = ", ".join(sorted(candidates.keys()))
        typer.secho(f"Unknown docs section: {section!r}. Valid: {valid}", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    doc_path = candidates[key].resolve()
    if not doc_path.exists():
        msg = f"Document not found: {doc_path.as_posix()}"
        if key == "repo":
            msg += " (Create docs/REPOSITORY_STRUCTURE.md or update this mapping.)"
        typer.secho(msg, fg=typer.colors.RED)
        raise typer.Exit(code=2)

    if path_only:
        typer.echo(str(doc_path))
        raise typer.Exit(code=0)

    text = doc_path.read_text(encoding="utf-8", errors="replace")
    if pager:
        click.echo_via_pager(text)
    else:
        typer.echo(text)


# =============================================================================
# Validation / Artifacts Helpers
# =============================================================================


@app.command("validate-artifacts")
def cli_validate_artifacts(
    root: str = typer.Option("outputs", "--root", help="Artifacts root directory."),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as failures."),
    quiet: bool = typer.Option(False, "--quiet", help="Less stdout; still writes JSON."),
    python_bin: str = typer.Option(sys.executable, "--python", help="Python interpreter to use."),
):
    """
    Run the artifacts validator (scripts/validate_artifacts.py).
    """
    script = Path("scripts/validate_artifacts.py")
    if not script.exists():
        typer.secho("scripts/validate_artifacts.py not found.", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    cmd = [python_bin, str(script), "--root", root]
    if strict:
        cmd.append("--strict")
    if quiet:
        cmd.append("--quiet")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("report-open")
def cli_report_open(
    root: str = typer.Option("outputs", "--root", help="Artifacts root directory (with reports/).")
):
    """
    Best-effort open the generated reports index.html in your system browser.
    """
    idx = Path(root) / "reports" / "index.html"
    if not idx.exists():
        typer.secho(f"Not found: {idx.as_posix()}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    try:
        webbrowser.open_new_tab(idx.as_uri())
        typer.echo(f"Opened {idx.as_posix()} in browser.")
    except Exception:
        typer.echo(idx.as_posix())


# =============================================================================
# Script Wrappers (fetch-data, export-kaggle, generate-reports, profiling)
# =============================================================================


def _ensure_executable(path: Path):
    if not path.exists():
        typer.secho(f"Missing script: {path.as_posix()}", fg=typer.colors.RED)
        raise typer.Exit(code=2)
    if os.name != "nt":
        # Ensure executable bit (best effort)
        try:
            mode = os.stat(path).st_mode
            os.chmod(path, mode | 0o111)
        except Exception:
            pass


@app.command("fetch-data")
def cli_fetch_data(
    aoi_bbox: str = typer.Option(..., "--aoi-bbox", help='"minLon,minLat,maxLon,maxLat"'),
    out_dir: str = typer.Option("data/raw", "--out", help="Output directory"),
    max_scenes: int = typer.Option(1, "--max-scenes", help="Scene limit for S2/Landsat"),
    dem: bool = typer.Option(False, "--dem", help="Fetch DEM (SRTM via OpenTopography)"),
    hydro: bool = typer.Option(False, "--hydro", help="Fetch HydroRIVERS"),
    s2: bool = typer.Option(False, "--s2", help="Fetch Sentinel-2 sample COGs"),
    s1: bool = typer.Option(False, "--s1", help="Fetch Sentinel-1 (needs EARTHDATA creds)"),
    landsat: bool = typer.Option(False, "--landsat", help="Fetch Landsat L2 scene"),
    soilgrids: bool = typer.Option(False, "--soilgrids", help="Fetch SoilGrids WCS clip"),
    nicfi: bool = typer.Option(False, "--nicfi", help="Fetch NICFI Planet mosaic quad"),
    all_flags: bool = typer.Option(False, "--all", help="Attempt to fetch all sources"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print intended actions only"),
):
    """
    Wrapper for scripts/fetch_datasets.sh
    """
    script = Path("scripts/fetch_datasets.sh")
    _ensure_executable(script)
    cmd = [str(script), "--aoi-bbox", aoi_bbox, "--out", out_dir, "--max-scenes", str(max_scenes)]
    if dry_run:
        cmd.append("--dry-run")
    if all_flags:
        cmd.append("--all")
    else:
        if dem:
            cmd.append("--dem")
        if hydro:
            cmd.append("--hydro")
        if s2:
            cmd.append("--s2")
        if s1:
            cmd.append("--s1")
        if landsat:
            cmd.append("--landsat")
        if soilgrids:
            cmd.append("--soilgrids")
        if nicfi:
            cmd.append("--nicfi")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("export-kaggle")
def cli_export_kaggle(
    owner: str = typer.Option(..., "--owner", help="Kaggle username/organization"),
    name: str = typer.Option(..., "--name", help="Dataset/notebook slug (URL-safe)"),
    title: str = typer.Option("", "--title", help="Human-friendly title"),
    notebook: str = typer.Option(
        "notebooks/ade_discovery_pipeline.ipynb", "--notebook", help="Notebook path"
    ),
    readme: str = typer.Option("", "--readme", help="README.md path (optional)"),
    license_path: str = typer.Option("", "--license", help="LICENSE path (optional)"),
    include_extra: str = typer.Option("", "--include", help="Comma-separated extra paths"),
    datasets: str = typer.Option("", "--datasets", help="Comma-separated Kaggle dataset refs"),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU for kernel"),
    internet: bool = typer.Option(False, "--internet", help="Enable internet for kernel"),
    private_kernel: bool = typer.Option(False, "--private", help="Make kernel private"),
    make_zip: bool = typer.Option(False, "--make-zip", help="Create zip bundle"),
    make_ds: bool = typer.Option(False, "--make-dataset", help="Create/version Kaggle dataset"),
    make_kernel: bool = typer.Option(False, "--make-kernel", help="Push Kaggle kernel"),
    outdir: str = typer.Option("artifacts/kaggle", "--outdir", help="Base export directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions only"),
):
    """
    Wrapper for scripts/export_kaggle.sh
    """
    script = Path("scripts/export_kaggle.sh")
    _ensure_executable(script)
    cmd = [
        str(script),
        "--owner",
        owner,
        "--name",
        name,
        "--notebook",
        notebook,
        "--outdir",
        outdir,
    ]
    if title:
        cmd += ["--title", title]
    if readme:
        cmd += ["--readme", readme]
    if license_path:
        cmd += ["--license", license_path]
    if include_extra:
        cmd += ["--include", include_extra]
    if datasets:
        cmd += ["--datasets", datasets]
    if gpu:
        cmd.append("--gpu")
    if internet:
        cmd.append("--internet")
    if private_kernel:
        cmd.append("--private")
    if make_zip:
        cmd.append("--make-zip")
    if make_ds:
        cmd.append("--make-dataset")
    if make_kernel:
        cmd.append("--make-kernel")
    if dry_run:
        cmd.append("--dry-run")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("generate-reports")
def cli_generate_reports(
    root: str = typer.Option("outputs", "--root", help="Artifacts root"),
    reports_dir: str = typer.Option("", "--reports-dir", help="Override reports dir"),
    verify_json: str = typer.Option("", "--verify-json", help="Override verify_candidates.geojson"),
    manifest_index: str = typer.Option("", "--manifest-index", help="Override manifest_index.json"),
    config: str = typer.Option("", "--config", help="Pipeline config if --run-report used"),
    run_report: bool = typer.Option(False, "--run-report", help="Call pipeline report stage first"),
    no_render: bool = typer.Option(False, "--no-render", help="Disable MD->HTML/PDF conversion"),
    html: bool = typer.Option(True, "--html/--no-html", help="Render HTML when possible"),
    pdf: bool = typer.Option(True, "--pdf/--no-pdf", help="Render PDF when possible"),
    mask: bool = typer.Option(False, "--mask", help="Mask/round coords in public index"),
    round_dec: int = typer.Option(2, "--round-decimals", help="Coordinate rounding decimals"),
    zip_bundle: bool = typer.Option(False, "--zip", help="Create reports_bundle.zip"),
    open_index: bool = typer.Option(False, "--open", help="Open index.html when done"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print actions only"),
):
    """
    Wrapper for scripts/generate_reports.sh
    """
    script = Path("scripts/generate_reports.sh")
    _ensure_executable(script)
    cmd = [str(script), "--root", root]
    if reports_dir:
        cmd += ["--reports-dir", reports_dir]
    if verify_json:
        cmd += ["--verify-json", verify_json]
    if manifest_index:
        cmd += ["--manifest-index", manifest_index]
    if config:
        cmd += ["--config", config]
    if run_report:
        cmd.append("--run-report")
    if no_render:
        cmd.append("--no-render")
    if html:
        cmd.append("--html")
    else:
        cmd.append("--no-html")
    if pdf:
        cmd.append("--pdf")
    else:
        cmd.append("--no-pdf")
    if mask:
        cmd.append("--mask")
    if round_dec is not None:
        cmd += ["--round-decimals", str(round_dec)]
    if zip_bundle:
        cmd.append("--zip")
    if open_index:
        cmd.append("--open")
    if dry_run:
        cmd.append("--dry-run")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise typer.Exit(code=rc)


@app.command("profile")
def cli_profile(
    tool: str = typer.Argument(..., help="cprofile | memory | timers | pyspy"),
    cmd: str = typer.Option("", "--cmd", help="Command to profile (quoted)"),
    label: str = typer.Option("command", "--label", help="Profile label"),
    out_dir: str = typer.Option("artifacts/profiles", "--out", help="Output directory"),
    topn: int = typer.Option(40, "--topn", help="Top-N functions in cProfile summary"),
    repeat: int = typer.Option(3, "--repeat", help="Repeat count for timers"),
    python_bin: str = typer.Option(sys.executable, "--python", help="Python interpreter to use."),
):
    """
    Wrapper for scripts/profiling_tools.py
    """
    script = Path("scripts/profiling_tools.py")
    if not script.exists():
        typer.secho("scripts/profiling_tools.py not found.", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    # Build command
    if tool == "cprofile":
        base = [python_bin, str(script), "cprofile", "--label", label, "--out", out_dir]
        if cmd:
            base += ["--cmd", cmd]
        base += ["--topn", str(topn)]
    elif tool == "memory":
        if not cmd:
            typer.secho("profile memory requires --cmd", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        base = [python_bin, str(script), "memory", "--label", label, "--out", out_dir, "--cmd", cmd]
    elif tool == "timers":
        if not cmd:
            typer.secho("profile timers requires --cmd", fg=typer.colors.RED)
            raise typer.Exit(code=2)
        base = [
            python_bin,
            str(script),
            "timers",
            "--label",
            label,
            "--out",
            out_dir,
            "--cmd",
            cmd,
            "--repeat",
            str(repeat),
        ]
    elif tool == "pyspy":
        base = [python_bin, str(script), "pyspy"]
    else:
        typer.secho("Unknown profiling tool. Use: cprofile | memory | timers | pyspy", fg=typer.colors.RED)
        raise typer.Exit(code=2)

    rc = subprocess.call(base)
    if rc != 0:
        raise typer.Exit(code=rc)


# =============================================================================
# Entrypoint
# =============================================================================


@app.callback(invoke_without_command=False)
def _root() -> None:
    """World Discovery Engine (WDE) — CLI entrypoint."""
    return


if __name__ == "__main__":
    app()
