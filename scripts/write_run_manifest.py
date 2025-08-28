# FILE: scripts/write_run_manifest.py
# -------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — Run Manifest Writer
#
# Purpose
# -------
# Generate a machine-readable provenance record for the current WDE run. This captures:
#   - UTC timestamp and host/system info
#   - Git commit / branch (if available)
#   - Config path and SHA-256 hash
#   - Deterministic seeds pulled from environment
#   - Python/package/runtime versions (Python, pip, Poetry if present)
#   - Optional DVC and CUDA/Torch environment hints (non-fatal if missing)
#   - A content-addressed run_hash derived from key fields (timestamp+git+config)
#   - Optional extra metadata passed via CLI (--extra)
#
# Output
# ------
# Writes JSON to artifacts/metrics/run_manifest.json (default) or --out path.
#
# Usage
# -----
#   poetry run python scripts/write_run_manifest.py \
#       --config configs/default.yaml \
#       --out artifacts/metrics/run_manifest.json
#
# Notes
# -----
# - This script is intentionally dependency-light (stdlib only) with graceful fallbacks.
# - It is safe to run multiple times; it overwrites the same file.
# - If a field cannot be detected, it is set to None.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------
# Small helpers (no deps)
# ---------------------------

def _utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 'YYYY-MM-DDTHH:MM:SSZ'."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_file(path: Path) -> Optional[str]:
    """Compute sha256 for a file; return None if missing."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()
    except Exception:
        return None


def _sha256_text(text: str) -> str:
    """Return sha256 for a string."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_cmd(cmd: str, timeout: float = 3.0) -> Optional[str]:
    """Run a shell command and return stripped stdout, or None on failure."""
    try:
        out = subprocess.check_output(shlex.split(cmd), stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _git_info() -> Dict[str, Optional[str]]:
    """Collect basic git info if repository present."""
    info = {
        "git_sha": None,
        "git_branch": None,
        "git_tag": None,
        "git_describe": None,
        "is_dirty": None,
    }
    sha = _run_cmd("git rev-parse --verify HEAD")
    if sha:
        info["git_sha"] = sha
        branch = _run_cmd("git rev-parse --abbrev-ref HEAD")
        info["git_branch"] = branch
        describe = _run_cmd("git describe --tags --always --dirty")
        info["git_describe"] = describe
        # Heuristic for dirty: describe contains '-dirty' or status shows changes
        dirty = False
        if describe and "dirty" in describe:
            dirty = True
        else:
            status = _run_cmd("git status --porcelain")
            dirty = bool(status)
        info["is_dirty"] = dirty
        # Best-effort to get an exact tag if HEAD is on a tag
        tag = _run_cmd("git describe --exact-match --tags") or None
        info["git_tag"] = tag
    return info


def _dvc_version() -> Optional[str]:
    """Detect dvc version (non-fatal)."""
    ver = _run_cmd("dvc --version")
    if ver:
        # normalize like "2.58.2"
        m = re.search(r"(\d+\.\d+\.\d+)", ver)
        return m.group(1) if m else ver
    return None


def _poetry_version() -> Optional[str]:
    """Detect Poetry version (non-fatal)."""
    ver = _run_cmd("poetry --version")
    if ver:
        # Example: "Poetry (version 1.8.3)"
        m = re.search(r"version\s+([\d\.]+)", ver, re.IGNORECASE)
        return m.group(1) if m else ver
    return None


def _python_packages_versions(names: list[str]) -> Dict[str, Optional[str]]:
    """Try to inspect installed versions using importlib.metadata (Py3.8+)."""
    out: Dict[str, Optional[str]] = {n: None for n in names}
    try:
        import importlib.metadata as imd  # py3.8+
    except Exception:
        return out
    for n in names:
        try:
            out[n] = imd.version(n)
        except Exception:
            out[n] = None
    return out


def _torch_env() -> Dict[str, Optional[Any]]:
    """Capture torch/CUDA hints if torch is installed; otherwise return Nones."""
    env: Dict[str, Optional[Any]] = {
        "torch_version": None,
        "cuda_available": None,
        "cudnn_available": None,
        "cuda_device_count": None,
    }
    try:
        import torch  # type: ignore

        env["torch_version"] = getattr(torch, "__version__", None)
        env["cuda_available"] = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        env["cuda_device_count"] = torch.cuda.device_count() if hasattr(torch.cuda, "device_count") else None
        # cuDNN
        cudnn = None
        try:
            import torch.backends.cudnn as cudnn_mod  # type: ignore
            cudnn = bool(getattr(cudnn_mod, "is_available", lambda: False)())
        except Exception:
            cudnn = None
        env["cudnn_available"] = cudnn
    except Exception:
        pass
    return env


def _ensure_parent(path: Path) -> None:
    """Create parent directories for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------
# Data models
# ---------------------------

@dataclass
class Seeds:
    numpy: int = field(default_factory=lambda: int(os.environ.get("SEED_NUMPY", "42")))
    random: int = field(default_factory=lambda: int(os.environ.get("SEED_RANDOM", "42")))
    torch: int = field(default_factory=lambda: int(os.environ.get("SEED_TORCH", "42")))


@dataclass
class SystemInfo:
    timestamp_utc: str
    python_version: str
    platform: str
    platform_release: str
    platform_version: str
    machine: str
    processor: str


@dataclass
class ToolVersions:
    pip: Optional[str] = None
    poetry: Optional[str] = None
    dvc: Optional[str] = None
    packages: Dict[str, Optional[str]] = field(default_factory=dict)


@dataclass
class GitInfo:
    git_sha: Optional[str] = None
    git_branch: Optional[str] = None
    git_tag: Optional[str] = None
    git_describe: Optional[str] = None
    is_dirty: Optional[bool] = None


@dataclass
class TorchInfo:
    torch_version: Optional[str] = None
    cuda_available: Optional[bool] = None
    cudnn_available: Optional[bool] = None
    cuda_device_count: Optional[int] = None


@dataclass
class RunManifest:
    pipeline_version: Optional[str]
    run_hash: str
    system: SystemInfo
    seeds: Seeds
    git: GitInfo
    tools: ToolVersions
    torch: TorchInfo
    config_path: Optional[str]
    config_hash: Optional[str]
    aoi_bbox: Optional[list[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Core builder
# ---------------------------

def build_manifest(
    config_path: Optional[Path],
    out_path: Path,
    aoi_bbox: Optional[list[float]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> RunManifest:
    # System info
    sysinfo = SystemInfo(
        timestamp_utc=_utc_now_iso(),
        python_version=sys.version.split()[0],
        platform=platform.system(),
        platform_release=platform.release(),
        platform_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),
    )

    # Seeds
    seeds = Seeds()

    # Config hashing
    cfg_hash = _sha256_file(config_path) if (config_path and config_path.exists()) else None

    # Git info
    gitinfo = GitInfo(**_git_info())

    # Tool versions
    pkg_versions = _python_packages_versions(
        names=[
            "numpy",
            "pandas",
            "rasterio",
            "geopandas",
            "opencv-python-headless",
            "scikit-image",
            "scikit-learn",
            "pyyaml",
        ]
    )
    tools = ToolVersions(
        pip=_run_cmd("pip --version"),
        poetry=_poetry_version(),
        dvc=_dvc_version(),
        packages=pkg_versions,
    )

    # Torch / CUDA hints
    torchinfo = TorchInfo(**_torch_env())

    # Pipeline version (prefer env override)
    pipeline_version = os.environ.get("WDE_VERSION", None)

    # Compose run hash from timestamp + git sha + config hash (content-addressed)
    components = [
        sysinfo.timestamp_utc or "",
        gitinfo.git_sha or "",
        cfg_hash or "",
        pipeline_version or "",
    ]
    run_hash = _sha256_text("|".join(components))

    # Merge extra metadata
    extra_meta = extra.copy() if extra else {}
    # Also include the git branch/tag in extras for easier filtering in dashboards
    if gitinfo.git_branch and "git_branch" not in extra_meta:
        extra_meta["git_branch"] = gitinfo.git_branch
    if gitinfo.git_tag and "git_tag" not in extra_meta:
        extra_meta["git_tag"] = gitinfo.git_tag

    manifest = RunManifest(
        pipeline_version=pipeline_version,
        run_hash=run_hash,
        system=sysinfo,
        seeds=seeds,
        git=gitinfo,
        tools=tools,
        torch=torchinfo,
        config_path=str(config_path) if config_path else None,
        config_hash=cfg_hash,
        aoi_bbox=aoi_bbox,
        extra=extra_meta,
    )

    # Ensure output directory and write JSON
    _ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_as_clean_dict(manifest), f, indent=2, ensure_ascii=False)

    return manifest


def _as_clean_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dict (recursively), suitable for JSON serialization."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _as_clean_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _as_clean_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_clean_dict(v) for v in obj]
    return obj


# ---------------------------
# CLI
# ---------------------------

def _parse_extra_json(s: Optional[str]) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception as e:
        sys.stderr.write(f"[write_run_manifest] Warning: failed to parse --extra JSON: {e}\n")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write WDE run manifest JSON for provenance.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("WDE_CONFIG", "configs/default.yaml"),
        help="Path to pipeline config (YAML/JSON). Defaults to env WDE_CONFIG or configs/default.yaml",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.environ.get("WDE_MANIFEST_OUT", "artifacts/metrics/run_manifest.json"),
        help="Output JSON path. Defaults to env WDE_MANIFEST_OUT or artifacts/metrics/run_manifest.json",
    )
    parser.add_argument(
        "--aoi-bbox",
        type=str,
        default=os.environ.get("WDE_AOI_BBOX", None),
        help="Optional AOI bbox as 'min_lat,min_lon,max_lat,max_lon' (comma-separated).",
    )
    parser.add_argument(
        "--extra",
        type=str,
        default=None,
        help="Optional JSON string of extra metadata to include (e.g. '{\"runner\":\"CI\"}').",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    out_path = Path(args.out)

    aoi_bbox = None
    if args.aoi_bbox:
        try:
            parts = [float(x.strip()) for x in args.aoi_bbox.split(",")]
            if len(parts) == 4:
                aoi_bbox = parts
            else:
                raise ValueError("Expected 4 comma-separated floats.")
        except Exception as e:
            sys.stderr.write(f"[write_run_manifest] Warning: invalid --aoi-bbox: {e}\n")

    extra = _parse_extra_json(args.extra)

    manifest = build_manifest(
        config_path=cfg_path if (cfg_path and cfg_path.exists()) else None,
        out_path=out_path,
        aoi_bbox=aoi_bbox,
        extra=extra,
    )

    # Also print a brief one-line summary to stdout for logs
    print(
        f"[write_run_manifest] Wrote {out_path} | run_hash={manifest.run_hash} | "
        f"git_sha={manifest.git.git_sha or 'NA'} | config_hash={manifest.config_hash or 'NA'}"
    )


if __name__ == "__main__":
    main()
