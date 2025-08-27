# tools/wde_scoring.py
# Scoring stubs for WDE (Kaggle-safe). Replace with physics/causal/scientific scoring later.
from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

def score_demo(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Combine existing columns into a mock 'model' score, deterministic by seed.

    We demonstrate a few trivial schemes:
    - weighted sum of [score, 1 - ring] to prefer central AOI values
    - optional bonus for a specific quadrant
    """
    df = df.copy()
    w_score = float(cfg.get("w_random", 0.7))
    w_center = float(cfg.get("w_center", 0.3))
    bonus_q = int(cfg.get("bonus_quadrant", -1))
    bonus_val = float(cfg.get("bonus_value", 0.05))

    center_pref = 1.0 - df.get("ring", 0.5)
    base = w_score * df.get("score", 0.0) + w_center * center_pref

    if bonus_q in {0,1,2,3} and "quad" in df.columns:
        base = base + bonus_val * (df["quad"] == bonus_q).astype(float)

    df["score_model"] = base.clip(0.0, 1.0)
    return df