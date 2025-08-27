# FILE: src/wde/dossier/build.py
# =================================================================================================
# Dossier building stage (placeholder)
#
# Responsibilities
# ---------------
# - For each candidate, assemble a dossier with maps/overlays/references
# - Apply ethics: coarsen coordinates, include uncertainty, provenance, and refutation attempts
# - Save under artifacts/dossiers/
#
# Current skeleton
# ----------------
# - Creates an empty dossiers_manifest.json
# - Replace with real map rendering (e.g., Folium/Leaflet, Raster overlays) and report generation
# =================================================================================================
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

from ..utils.io import write_json
from ..utils.log import vprint


def run_dossier_build(config: Dict[str, Any], in_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Build dossiers from detected candidates.

    Parameters
    ----------
    config : dict
        Parsed YAML from configs/dossier.yaml
    in_dir : Path
        Directory with candidates.json or similar
    out_dir : Path
        Output directory for dossiers

    Returns
    -------
    dict
        Manifest describing produced dossiers.
    """
    manifest: Dict[str, Any] = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "layout": config.get("layout", "minimal"),
        "dossiers": [],
        "notes": "Placeholder dossier builder; integrate map rendering and reports.",
    }
    write_json(out_dir / "dossiers_manifest.json", manifest)
    vprint(f"[dossier] wrote dossiers_manifest.json", level=1)
    return manifest