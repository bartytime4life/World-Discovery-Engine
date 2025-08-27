# FILE: src/wde/data/fetch.py
# =================================================================================================
# Data fetching stage (placeholder implementation)
#
# Responsibilities
# ---------------
# - Read provider configuration (configs/data.yaml)
# - Authenticate via environment variables (.env)
# - Download / sync raw layers into data/raw/ with provenance metadata
#
# Current skeleton
# ----------------
# - Creates an empty FeatureCollection GeoJSON and a provenance.json
# - Returns a manifest dict
# =================================================================================================
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Any

from ..utils.io import ensure_dir, write_json
from ..utils.log import vprint


def run_fetch(config: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    """
    Execute data fetch according to `config` and write outputs under `out_dir`.

    Parameters
    ----------
    config : dict
        Parsed YAML from configs/data.yaml (e.g., provider sources).
    out_dir : Path
        Target directory for raw data.

    Returns
    -------
    dict
        Manifest with basic info (files, created_at, providers).
    """
    ensure_dir(out_dir)

    # Placeholder: write an empty FeatureCollection GeoJSON
    geojson_path = out_dir / "placeholder.geojson"
    geojson = {"type": "FeatureCollection", "features": []}
    write_json(geojson_path, geojson)  # reuse JSON writer; it's valid JSON for GeoJSON

    # Minimal provenance
    provenance = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "providers": [s.get("name", "unknown") for s in config.get("sources", [])],
        "notes": "Placeholder fetch; replace with real downloads (Sentinel/Landsat/NICFI/DEM/LiDAR).",
        "env": {
            "PLANET_API_KEY_set": bool(os.environ.get("PLANET_API_KEY")),
            "SENTINELHUB_CLIENT_ID_set": bool(os.environ.get("SENTINELHUB_CLIENT_ID")),
        },
    }
    write_json(out_dir / "provenance.json", provenance)

    vprint(f"[fetch] wrote {geojson_path}", level=1)
    vprint(f"[fetch] wrote provenance.json", level=1)

    return {"files": [str(geojson_path)], "provenance": provenance}