"""
Stage 1 — Ingest: AOI tiling & dataset prep

This stage:
- Validates AOI bbox & tiling size from config.
- Produces tile descriptors in outputs/tiles.json (list of dicts).
- Creates a tiny "cache" structure (optional) to cache prepped assets.

No heavy geospatial libs are required here; we keep it light and file-driven.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .utils.logging_utils import get_logger


@dataclass
class Tile:
    """A simple tile descriptor."""
    id: int
    bbox: Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)
    center: Tuple[float, float]              # (lat, lon)


def _generate_tiles(bbox_ll: List[float], tile_size_deg: float) -> List[Tile]:
    """Generate tiles covering the AOI bbox at tile_size_deg increments.

    bbox_ll: list [min_lon, min_lat, max_lon, max_lat] or [min_lat, min_lon, max_lat, max_lon]
    The pipeline config uses aoi.bbox in (min_lat, min_lon, max_lat, max_lon). We'll handle both gracefully.
    """
    # Normalize to (min_lat, min_lon, max_lat, max_lon)
    if bbox_ll[0] < -90 or bbox_ll[0] > 90:
        # assume lon,lat,lon,lat => convert to lat,lon,lat,lon
        min_lon, min_lat, max_lon, max_lat = bbox_ll
        bbox = [min_lat, min_lon, max_lat, max_lon]
    else:
        # already lat,lon,lat,lon
        bbox = bbox_ll

    min_lat, min_lon, max_lat, max_lon = bbox
    tiles: List[Tile] = []
    tid = 0
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            tmin_lat = lat
            tmin_lon = lon
            tmax_lat = min(lat + tile_size_deg, max_lat)
            tmax_lon = min(lon + tile_size_deg, max_lon)
            center = ((tmin_lat + tmax_lat) * 0.5, (tmin_lon + tmax_lon) * 0.5)
            tiles.append(Tile(id=tid, bbox=(tmin_lat, tmin_lon, tmax_lat, tmax_lon), center=center))
            tid += 1
            lon += tile_size_deg
        lat += tile_size_deg
    return tiles


def run_ingest(cfg: Dict) -> Dict:
    """Run the ingest stage. Returns a simple artifact dict."""
    log = get_logger("wde.ingest")
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi = cfg["aoi"]
    bbox = aoi["bbox"]
    tile_sz = float(aoi.get("tile_size_deg", cfg.get("meta", {}).get("default_tile_size_deg", 0.05)))

    tiles = _generate_tiles(bbox, tile_sz)
    tiles_json_path = out_dir / "tiles.json"
    tiles_json_path.write_text(json.dumps([asdict(t) for t in tiles], indent=2))

    log.info(f"Ingest produced {len(tiles)} tiles → {tiles_json_path}")
    return {
        "stage": "ingest",
        "tiles_file": str(tiles_json_path),
        "num_tiles": len(tiles),
        "tile_size_deg": tile_sz,
    }
