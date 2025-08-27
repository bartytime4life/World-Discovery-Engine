# tools/wde_pipeline.py
# World Discovery Engine (WDE) — minimal, Kaggle-safe pipeline stubs.
# Replace these stubs with real fusion, features, and scoring modules.
from __future__ import annotations
import math
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting (safe on Kaggle)
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ----------------------------------------------------------------------------------
# Seeding & simple utilities
# ----------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Set the numpy RNG seed for deterministic demo behavior."""
    np.random.seed(seed)

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------------------------------------------------------------
# Demo pipeline (Kaggle-safe)
# ----------------------------------------------------------------------------------
def run_demo_candidate_generation(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Generate a toy candidate table to wire up the notebook flow.

    Columns
    -------
    lat, lon : float
        Pseudo-random coordinates bounded to a South American-ish AOI window.
    score : float
        Random score in [0,1) to mimic a site ranking output.
    """
    seed = int(cfg.get("random_seed", 42))
    n = int(cfg.get("demo_size", 1000))
    set_seed(seed)

    # Rough Amazon basin-ish bounding box for demo purposes
    lat = np.random.uniform(-15.0, -2.0, size=n)
    lon = np.random.uniform(-75.0, -45.0, size=n)
    score = np.random.random(size=n)
    df = pd.DataFrame({"lat": lat, "lon": lon, "score": score})
    df.sort_values("score", ascending=False, inplace=True)
    return df

def export_topk(df: pd.DataFrame, out_dir: str | Path, k: int = 50) -> Path:
    """Save the top-K rows by `score` to a CSV and return the path."""
    out = ensure_dir(out_dir) / "demo_candidates_top50.csv"
    df.head(k).to_csv(out, index=False)
    return out

def plot_candidates(df: pd.DataFrame, out_dir: str | Path) -> Optional[Path]:
    """Create a simple lon/lat scatter for quick visual signal. Returns PNG path if saved."""
    if plt is None:
        return None
    out_png = ensure_dir(out_dir) / "demo_candidates_scatter.png"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(df["lon"], df["lat"], s=4, alpha=0.35)
    ax.set_title("WDE demo candidate distribution")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png

def write_manifest(out_dir: str | Path) -> Path:
    """Write a tiny manifest of current files for reproducibility."""
    out_dir = ensure_dir(out_dir)
    files = [str(p) for p in Path(out_dir).glob("**/*") if p.is_file()]
    manifest = {"artifacts": files}
    out = Path(out_dir) / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out