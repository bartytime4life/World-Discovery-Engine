# FILE: src/wde/data/preprocess.py
# =================================================================================================
# Data preprocessing stage (placeholder)
#
# Responsibilities
# ---------------
# - Ingest raw layers, reproject into common CRS, tile/chunk, normalize/standardize
# - Produce analysis-ready raster/vector products under data/processed/
#
# Current skeleton
# ----------------
# - Copies/derives a trivial "layers.json" describing zero layers
# - Replace with actual geoprocessing using rasterio/geopandas when ready
# =================================================================================================
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Any

from ..utils.io import write_json
from ..utils.log import vprint


def run_preprocess(config: Dict[str, Any], in_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Execute preprocessing from raw → processed.

    Parameters
    ----------
    config : dict
        Parsed YAML from configs/preprocess.yaml
    in_dir : Path
        Directory containing raw data
    out_dir : Path
        Output directory for processed layers

    Returns
    -------
    dict
        Manifest describing produced layers.
    """
    manifest = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "steps": config.get("steps", []),
        "layers": [],
        "notes": "Placeholder preprocess; implement tiling/reprojection/normalization.",
    }
    write_json(out_dir / "layers.json", {"layers": manifest["layers"]})
    vprint(f"[preprocess] wrote layers.json", level=1)
    return manifest