# FILE: world_engine/utils/env.py
# -------------------------------------------------------------------------------------------------
# World Discovery Engine (WDE) — Environment Loader & Validator
#
# Purpose
# -------
# Centralize loading of environment variables (.env) into structured configs with:
#   - type conversion (str → bool/int)
#   - presence checks (required/optional)
#   - light validation (URLs, non-empty strings)
#   - warnings with actionable hints
#   - export helpers for pipeline configs (nested or flat dict)
#
# Usage
# -----
# from world_engine.utils.env import load_env, to_nested_dict, to_flat_dict
# env = load_env()
# cfg_nested = to_nested_dict(env)
# cfg_flat   = to_flat_dict(env, prefix="WDE")
#
# You can also write a quick YAML for inspection:
#   write_yaml("artifacts/metrics/env_dump.yaml", cfg_nested)
#
# Notes
# -----
# - This module has no external dependency (no pydantic/yaml) to keep Kaggle-friendly.
# - If you prefer rich validation, wire in 'pydantic' in your own fork.
# - Reads from os.environ only; use python-dotenv or your CI to populate variables.
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
logger = logging.getLogger("wde.env")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
_TRUE = {"1", "true", "yes", "on", "y", "t"}
_FALSE = {"0", "false", "no", "off", "n", "f"}


def _get(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch env var or default (strip whitespace)."""
    val = os.environ.get(key, default)
    if val is None:
        return None
    s = str(val).strip()
    return s if s != "" else None


def _require(key: str) -> str:
    """Require a non-empty env var, raise if missing."""
    val = _get(key, None)
    if val is None:
        raise KeyError(f"Missing required environment variable: {key}")
    return val


def _as_bool(key: str, default: bool = False) -> bool:
    raw = _get(key, None)
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    logger.warning(f"[ENV] {key} has non-boolean value '{raw}', using default={default}")
    return default


def _as_int(key: str, default: int) -> int:
    raw = _get(key, None)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        logger.warning(f"[ENV] {key} has non-integer value '{raw}', using default={default}")
        return default


_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def _looks_url(s: Optional[str]) -> bool:
    return bool(s and _URL_RE.match(s))


def _warn_opt_missing(name: str, howto: str = "") -> None:
    msg = f"[ENV] Optional variable '{name}' not set."
    if howto:
        msg += f" {howto}"
    logger.info(msg)


def _mask(s: Optional[str]) -> Optional[str]:
    """Return masked version of secrets in logs."""
    if not s:
        return s
    if len(s) <= 8:
        return "*" * len(s)
    return s[:4] + "*" * (len(s) - 8) + s[-4:]


# --------------------------------------------------------------------------------------
# Data classes: grouped config sections
# --------------------------------------------------------------------------------------
@dataclass
class KaggleConfig:
    username: Optional[str] = field(default_factory=lambda: _get("KAGGLE_USERNAME"))
    key: Optional[str] = field(default_factory=lambda: _get("KAGGLE_KEY"))
    competition: Optional[str] = field(default_factory=lambda: _get("KAGGLE_COMPETITION"))
    dataset_owner: Optional[str] = field(default_factory=lambda: _get("KAGGLE_DATASET_OWNER"))
    notebook_slug: Optional[str] = field(default_factory=lambda: _get("KAGGLE_NOTEBOOK_SLUG"))

    def validate(self) -> None:
        if (self.username and not self.key) or (self.key and not self.username):
            logger.warning("[ENV] KAGGLE_USERNAME/KAGGLE_KEY incomplete; Kaggle API may fail.")
        if self.key:
            logger.info(f"[ENV] Kaggle API detected user={self.username!r}, key={_mask(self.key)}")


@dataclass
class RemoteSensingConfig:
    sentinelhub_client_id: Optional[str] = field(default_factory=lambda: _get("SENTINELHUB_CLIENT_ID"))
    sentinelhub_client_secret: Optional[str] = field(default_factory=lambda: _get("SENTINELHUB_CLIENT_SECRET"))
    earthdata_username: Optional[str] = field(default_factory=lambda: _get("EARTHDATA_USERNAME"))
    earthdata_password: Optional[str] = field(default_factory=lambda: _get("EARTHDATA_PASSWORD"))
    planet_api_key: Optional[str] = field(default_factory=lambda: _get("PLANET_API_KEY"))
    opentopography_api_key: Optional[str] = field(default_factory=lambda: _get("OPENTOPOGRAPHY_API_KEY"))

    def validate(self) -> None:
        if self.sentinelhub_client_id and not self.sentinelhub_client_secret:
            logger.warning("[ENV] SENTINELHUB_CLIENT_SECRET missing; SentinelHub may fail.")
        if self.sentinelhub_client_secret and not self.sentinelhub_client_id:
            logger.warning("[ENV] SENTINELHUB_CLIENT_ID missing; SentinelHub may fail.")
        if not (self.earthdata_username and self.earthdata_password):
            _warn_opt_missing("EARTHDATA_USERNAME/EARTHDATA_PASSWORD",
                              "Needed for ASF/LP DAAC downloads (Sentinel-1/Landsat/SRTM).")
        if not self.planet_api_key:
            _warn_opt_missing("PLANET_API_KEY", "Required for NICFI mosaics if enabled.")
        if not self.opentopography_api_key:
            _warn_opt_missing("OPENTOPOGRAPHY_API_KEY", "Required for OpenTopography LiDAR/DEM API.")


@dataclass
class SoilCoreConfig:
    soilgrids_api: Optional[str] = field(default_factory=lambda: _get("SOILGRIDS_API"))
    wosis_api: Optional[str] = field(default_factory=lambda: _get("WOSIS_API"))
    noaa_paleo_api: Optional[str] = field(default_factory=lambda: _get("NOAA_PALEO_API"))
    pangaea_api: Optional[str] = field(default_factory=lambda: _get("PANGAEA_API"))
    neotoma_api: Optional[str] = field(default_factory=lambda: _get("NEOTOMA_API"))
    iodp_portal: Optional[str] = field(default_factory=lambda: _get("IODP_PORTAL"))

    def validate(self) -> None:
        for name in ("soilgrids_api", "wosis_api", "noaa_paleo_api", "pangaea_api", "neotoma_api", "iodp_portal"):
            val = getattr(self, name)
            if val and not _looks_url(val):
                logger.warning(f"[ENV] {name} does not look like a URL: {val!r}")


@dataclass
class MapConfig:
    mapbox_token: Optional[str] = field(default_factory=lambda: _get("MAPBOX_TOKEN"))
    google_maps_api_key: Optional[str] = field(default_factory=lambda: _get("GOOGLE_MAPS_API_KEY"))
    carto_api_key: Optional[str] = field(default_factory=lambda: _get("CARTO_API_KEY"))

    def validate(self) -> None:
        if not self.mapbox_token and not self.google_maps_api_key and not self.carto_api_key:
            _warn_opt_missing("MAPBOX_TOKEN/GOOGLE_MAPS_API_KEY/CARTO_API_KEY",
                              "Optional; used for nicer basemaps in HTML reports.")


@dataclass
class EthicsConfig:
    mask_coords: bool = field(default_factory=lambda: _as_bool("WDE_MASK_COORDS", True))
    flag_indigenous_overlap: bool = field(default_factory=lambda: _as_bool("WDE_FLAG_INDIGENOUS_OVERLAP", True))

    def validate(self) -> None:
        if not self.mask_coords:
            logger.warning("[ENV] WDE_MASK_COORDS=false — public outputs may reveal exact coordinates.")
        if not self.flag_indigenous_overlap:
            logger.warning("[ENV] WDE_FLAG_INDIGENOUS_OVERLAP=false — sovereignty alerts disabled.")


@dataclass
class RuntimeConfig:
    data_root: str = field(default_factory=lambda: _get("WDE_DATA_ROOT", "./data") or "./data")
    artifacts_root: str = field(default_factory=lambda: _get("WDE_ARTIFACTS_ROOT", "./artifacts") or "./artifacts")
    output_dir: str = field(default_factory=lambda: _get("WDE_OUTPUT_DIR", "./outputs") or "./outputs")
    random_seed: int = field(default_factory=lambda: _as_int("WDE_RANDOM_SEED", 42))
    use_gpu: bool = field(default_factory=lambda: _as_bool("WDE_USE_GPU", False))

    def validate(self) -> None:
        # Very light checks; full path validation happens where used.
        for name in ("data_root", "artifacts_root", "output_dir"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"[ENV] {name} must be a non-empty string; got {val!r}")
        if self.random_seed < 0:
            logger.warning(f"[ENV] WDE_RANDOM_SEED={self.random_seed} < 0; consider a non-negative seed.")


@dataclass
class CIConfig:
    github_token: Optional[str] = field(default_factory=lambda: _get("GITHUB_TOKEN"))
    hf_token: Optional[str] = field(default_factory=lambda: _get("HF_TOKEN"))
    mlflow_tracking_uri: Optional[str] = field(default_factory=lambda: _get("MLFLOW_TRACKING_URI"))
    mlflow_tracking_token: Optional[str] = field(default_factory=lambda: _get("MLFLOW_TRACKING_TOKEN"))
    dvc_remote: Optional[str] = field(default_factory=lambda: _get("DVC_REMOTE"))
    aws_access_key_id: Optional[str] = field(default_factory=lambda: _get("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key: Optional[str] = field(default_factory=lambda: _get("AWS_SECRET_ACCESS_KEY"))

    def validate(self) -> None:
        if self.mlflow_tracking_uri and not _looks_url(self.mlflow_tracking_uri):
            logger.warning(f"[ENV] MLFLOW_TRACKING_URI not a URL? {self.mlflow_tracking_uri!r}")
        if (self.aws_access_key_id and not self.aws_secret_access_key) or (
            self.aws_secret_access_key and not self.aws_access_key_id
        ):
            logger.warning("[ENV] AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY incomplete; S3/DVC ops may fail.")
        if self.github_token:
            logger.info(f"[ENV] GitHub token detected: {_mask(self.github_token)}")
        if self.hf_token:
            logger.info(f"[ENV] HF token detected: {_mask(self.hf_token)}")


# --------------------------------------------------------------------------------------
# Top-level Aggregate
# --------------------------------------------------------------------------------------
@dataclass
class EnvConfig:
    kaggle: KaggleConfig
    remote: RemoteSensingConfig
    soilcore: SoilCoreConfig
    maps: MapConfig
    ethics: EthicsConfig
    runtime: RuntimeConfig
    ci: CIConfig

    def validate(self) -> None:
        self.kaggle.validate()
        self.remote.validate()
        self.soilcore.validate()
        self.maps.validate()
        self.ethics.validate()
        self.runtime.validate()
        self.ci.validate()


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def load_env() -> EnvConfig:
    """
    Load environment into structured config with validation & friendly warnings.
    Will not raise if optional sections are missing; raises for egregious errors only.
    """
    cfg = EnvConfig(
        kaggle=KaggleConfig(),
        remote=RemoteSensingConfig(),
        soilcore=SoilCoreConfig(),
        maps=MapConfig(),
        ethics=EthicsConfig(),
        runtime=RuntimeConfig(),
        ci=CIConfig(),
    )
    cfg.validate()
    return cfg


def to_nested_dict(cfg: EnvConfig) -> Dict[str, Any]:
    """Serialize EnvConfig to a nested plain dict (JSON/YAML-friendly)."""
    return {
        "kaggle": asdict(cfg.kaggle),
        "remote": asdict(cfg.remote),
        "soilcore": asdict(cfg.soilcore),
        "maps": asdict(cfg.maps),
        "ethics": asdict(cfg.ethics),
        "runtime": asdict(cfg.runtime),
        "ci": asdict(cfg.ci),
    }


def to_flat_dict(cfg: EnvConfig, prefix: Optional[str] = None) -> Dict[str, Any]:
    """
    Serialize EnvConfig to a flat dict with optional prefix.
    Example:
        to_flat_dict(cfg, prefix="WDE") -> {"WDE.kaggle.username": "...", ...}
    """
    nested = to_nested_dict(cfg)
    flat: Dict[str, Any] = {}

    def _walk(d: Dict[str, Any], path: Tuple[str, ...]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _walk(v, path + (k,))
            else:
                key = ".".join(path + (k,))
                flat[key if not prefix else f"{prefix}.{key}"] = v

    _walk(nested, tuple())
    return flat


def write_json(path: str, obj: Dict[str, Any]) -> None:
    """Write a dict to JSON (UTF-8, pretty) without extra deps."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    logger.info(f"[ENV] Wrote JSON → {path}")


def _yaml_dump_lines(d: Any, indent: int = 0) -> str:
    """
    Minimal YAML emitter (handles dicts, lists, scalars) to avoid PyYAML dependency.
    This is simplistic but sufficient for inspection/export.
    """
    sp = "  " * indent
    if isinstance(d, dict):
        lines = []
        for k, v in d.items():
            key = str(k)
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}{key}:")
                lines.append(_yaml_dump_lines(v, indent + 1))
            else:
                sval = "null" if v is None else (str(v).lower() if isinstance(v, bool) else str(v))
                # Quote strings that look like reserved YAML tokens
                if isinstance(v, str) and (v.strip() == "" or v.lower() in {"null", "true", "false"}):
                    sval = f"'{v}'"
                lines.append(f"{sp}{key}: {sval}")
        return "\n".join(lines)
    elif isinstance(d, list):
        lines = []
        for item in d:
            if isinstance(item, (dict, list)):
                lines.append(f"{sp}-")
                lines.append(_yaml_dump_lines(item, indent + 1))
            else:
                sval = "null" if item is None else (str(item).lower() if isinstance(item, bool) else str(item))
                lines.append(f"{sp}- {sval}")
        return "\n".join(lines)
    else:
        sval = "null" if d is None else (str(d).lower() if isinstance(d, bool) else str(d))
        return f"{sp}{sval}"


def write_yaml(path: str, obj: Dict[str, Any]) -> None:
    """Write a dict to YAML (UTF-8) without PyYAML."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    txt = _yaml_dump_lines(obj)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt + ("\n" if not txt.endswith("\n") else ""))
    logger.info(f"[ENV] Wrote YAML → {path}")


# --------------------------------------------------------------------------------------
# CLI convenience (optional)
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick inspection mode:
    cfg = load_env()
    nested = to_nested_dict(cfg)
    flat = to_flat_dict(cfg, prefix="WDE")

    # Print a brief summary to stdout
    print(json.dumps({**nested, "__flat__": flat}, indent=2, ensure_ascii=False))

    # Optionally write artifacts for CI/debug
    out_json = _get("WDE_ENV_OUT_JSON", "./artifacts/metrics/env_config.json")
    out_yaml = _get("WDE_ENV_OUT_YAML", "./artifacts/metrics/env_config.yaml")
    try:
        write_json(out_json, nested)
        write_yaml(out_yaml, nested)
    except Exception as e:
        logger.warning(f"[ENV] Could not write env artifacts: {e}")
