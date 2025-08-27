# FILE: src/wde/detect/run.py
# =================================================================================================
# Candidate detection stage (placeholder)
#
# Responsibilities
# ---------------
# - Load processed layers and model configuration
# - Run ML/heuristic pipeline to generate candidate site proposals with scores/uncertainties
# - Save outputs under artifacts/candidates/
#
# Current skeleton
# ----------------
# - Emits an empty list of candidates and a basic schema
# - Replace with your detector (e.g., random forest, CNN, transformer, or rule-based)
# =================================================================================================
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Any, List

from ..utils.io import write_json
from ..utils.log import vprint


def run_detection(config: Dict[str, Any], in_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """
    Run candidate detection.

    Parameters
    ----------
    config : dict
        Parsed YAML from configs/detect.yaml
    in_dir : Path
        Directory of processed layers
    out_dir : Path
        Output directory for candidate artifacts

    Returns
    -------
    dict
        Result object containing candidate list and metadata.
    """
    result: Dict[str, Any] = {
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "model": config.get("model", {"type": "baseline"}),
        "candidates": [],
        "schema": {"id": "str", "geometry": "GeoJSON", "score": "float", "uncertainty": "float"},
        "notes": "Placeholder detection stage; replace with your ML/heuristics.",
    }
    write_json(out_dir / "candidates.json", result)
    vprint(f"[detect] wrote candidates.json", level=1)
    return result