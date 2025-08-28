"""
Stage 2 — Detect: Coarse anomaly scan (lightweight)

This stage simulates a coarse scan by:
- Loading tiles.json
- Generating a simple "edge_score" proxy per tile based on coordinates (deterministic toy)
- Producing detect_candidates.json sorted by score

In a real setup, you'd plug in OpenCV, scikit-image, or a VLM caption model here.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List

from .utils.logging_utils import get_logger


def _toy_edge_score(tile_center_lat: float, tile_center_lon: float, seed: int = 42) -> float:
    """
    A deterministic "toy" anomaly score using trigonometry so repeated runs match (for CI).
    This avoids heavy CV while preserving a comparable interface for downstream.
    """
    rnd = random.Random(seed + int((tile_center_lat + 90) * 1000) + int((tile_center_lon + 180) * 1000))
    base = 0.5 * (1.0 + math.sin(math.radians(tile_center_lat * 11.37)) * math.cos(math.radians(tile_center_lon * 7.53)))
    noise = 0.1 * rnd.random()
    return max(0.0, min(1.0, base + noise))


def run_detect(cfg: Dict, prev: Dict | None = None) -> Dict:
    log = get_logger("wde.detect")
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    tiles_file = out_dir / "tiles.json"
    if not tiles_file.exists():
        raise FileNotFoundError("Missing tiles.json. Run ingest first.")

    tiles: List[Dict] = json.loads(tiles_file.read_text())
    candidates: List[Dict] = []

    threshold = float(cfg["pipeline"]["detect"].get("anomaly_threshold", 0.7))
    for t in tiles:
        lat, lon = t["center"]
        score = _toy_edge_score(lat, lon, seed=cfg["run"]["random_seed"])
        if score >= threshold:
            candidates.append(
                {
                    "tile_id": t["id"],
                    "center": t["center"],
                    "bbox": t["bbox"],
                    "score": round(score, 4),
                    "reasons": ["toy_edge_score>=threshold"],  # placeholder for CV/VLM indicators
                }
            )

    # Sort by score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)

    out_path = out_dir / "detect_candidates.json"
    out_path.write_text(json.dumps(candidates, indent=2))
    log.info(f"Detect: kept {len(candidates)} candidates (threshold={threshold}) → {out_path}")

    return {
        "stage": "detect",
        "candidates_file": str(out_path),
        "threshold": threshold,
        "num_candidates": len(candidates),
    }
