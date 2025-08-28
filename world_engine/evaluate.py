"""
Stage 3 — Evaluate: Mid-scale evaluation

This stage "enriches" detected candidates with:
- toy NDVI/EVI stability flags (deterministic stand-ins)
- hydro-geomorph plausibility (distance to a notional river line)
- optional LiDAR-enabled flag (from config)

Outputs evaluate_candidates.json with additional fields.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

from .utils.logging_utils import get_logger


def _toy_ndvi_stability(lat: float, lon: float) -> bool:
    """Deterministic NDVI/EVI persistence proxy."""
    # Use a smooth function that occasionally dips to simulate instability.
    v = math.sin(math.radians(lat * 5.0)) * math.cos(math.radians(lon * 3.0))
    return v > -0.2  # most tiles stable, few unstable


def _toy_distance_to_river_km(center_lat: float, center_lon: float) -> float:
    """
    Fake distance-to-river: imagine a diagonal river: lon ≈ lat * k + b
    This yields a meaningful numeric that the plausibility rule can use.
    """
    # Define a diagonal “river” line in (lat,lon) space (coarse toy)
    k, b = 0.7, -5.0
    lon_on_line = center_lat * k + b
    deg_dist = abs(center_lon - lon_on_line)
    # Convert degrees lon at given latitude roughly to km (approx)
    km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))
    km = abs(deg_dist) * km_per_deg_lon
    return round(km, 2)


def run_evaluate(cfg: Dict, prev: Dict | None = None) -> Dict:
    log = get_logger("wde.evaluate")
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    cand_file = out_dir / "detect_candidates.json"
    if not cand_file.exists():
        raise FileNotFoundError("Missing detect_candidates.json. Run detect first.")

    candidates: List[Dict] = json.loads(cand_file.read_text())
    results: List[Dict] = []

    near_km = float(cfg["pipeline"]["evaluate"]["hydro_geomorph"].get("require_near_water_km", 6.0))
    lidar_enabled = bool(cfg["pipeline"]["evaluate"]["lidar"].get("enabled", False))

    for c in candidates:
        lat, lon = c["center"]
        ndvi_stable = _toy_ndvi_stability(lat, lon)
        dist_km = _toy_distance_to_river_km(lat, lon)
        plausible = (dist_km <= near_km) and ndvi_stable

        enriched = dict(c)
        enriched.update(
            {
                "ndvi_stable": ndvi_stable,
                "distance_to_river_km": dist_km,
                "hydro_geomorph_plausible": plausible,
                "lidar_available": lidar_enabled,  # purely config-driven label
            }
        )
        results.append(enriched)

    out_path = out_dir / "evaluate_candidates.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(
        f"Evaluate: enriched {len(results)} → {out_path} "
        f"(near_water_km<= {near_km}, lidar_enabled={lidar_enabled})"
    )

    return {
        "stage": "evaluate",
        "evaluate_file": str(out_path),
        "num_enriched": len(results),
        "near_water_km": near_km,
        "lidar_enabled": lidar_enabled,
    }
