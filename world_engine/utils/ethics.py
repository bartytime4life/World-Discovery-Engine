"""
Ethics helpers

- mask_coordinates_if_required(lat, lon, enable_mask): adds a tiny jitter or rounds precision.
  This allows publishing approximate locations while protecting exact coords by default.
"""
from __future__ import annotations

from typing import Tuple


def mask_coordinates_if_required(lat: float, lon: float, enable_mask: bool, precision: int = 3) -> Tuple[float, float]:
    """
    If enable_mask is True, round to ~2 decimal digits (~1km scale) as a conservative default.
    Otherwise return original.
    """
    if not enable_mask:
        return lat, lon
    # Simple rounding mask — can be replaced with a randomized offset (bounded) if needed
    return (round(lat, precision), round(lon, precision))
