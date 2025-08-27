# tools/wde_features.py
# Feature computation stubs for WDE (Kaggle-safe, soft GIS deps).
from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd

def compute_demo_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Add a few deterministic demo features to the candidate table.

    Parameters
    ----------
    df : DataFrame with columns [lat, lon, score]
    cfg : Dict of feature switches (bins, normalization options, etc.)

    Returns
    -------
    DataFrame with added columns like 'lat_norm', 'lon_norm', 'quad', 'ring'.
    """
    df = df.copy()
    # Normalized latitude/longitude to [0,1] within AOI bounds (rough demo ranges)
    lat_min, lat_max = -15.0, -2.0
    lon_min, lon_max = -75.0, -45.0
    df["lat_norm"] = (df["lat"] - lat_min) / (lat_max - lat_min)
    df["lon_norm"] = (df["lon"] - lon_min) / (lon_max - lon_min)
    # Simple engineered features
    df["quad"] = (df["lat_norm"] > 0.5).astype(int) * 2 + (df["lon_norm"] > 0.5).astype(int)
    df["ring"] = np.sqrt((df["lat_norm"] - 0.5)**2 + (df["lon_norm"] - 0.5)**2)
    return df