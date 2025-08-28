"""
Lightweight geospatial helpers (no heavy GDAL/PROJ required).

If needed later, you can swap with proper libs; these helpers keep CI/Kaggle-friendly by default.
"""
from __future__ import annotations
from typing import Tuple


def bbox_center(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> Tuple[float, float]:
    """Return centroid (lat, lon) for bbox (lat/lon degrees)."""
    return ((min_lat + max_lat) * 0.5, (min_lon + max_lon) * 0.5)
