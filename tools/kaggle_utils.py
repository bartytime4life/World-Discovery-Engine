# tools/kaggle_utils.py
# Lightweight helpers for Kaggle notebooks (path listing, config, timers, safe CSV).
from __future__ import annotations
import os, sys, time, json, textwrap
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Dict, Any

try:
    import yaml
except Exception:
    yaml = None  # Kaggle images usually have PyYAML preinstalled

def list_input_tree(root: str | os.PathLike, max_depth: int = 2) -> None:
    root = str(root)
    base_depth = root.count(os.sep)
    for r, d, files in os.walk(root):
        depth = r.count(os.sep) - base_depth
        if depth > max_depth:
            d[:] = []
            continue
        print(f"{r}  (dirs={len(d)} files={len(files)})")
        for fn in files[:10]:
            print("   -", fn)
        if len(files) > 10:
            print("   ... (truncated)")

def read_config(path: str | os.PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    text = p.read_text()
    if p.suffix.lower() in {'.yaml', '.yml'} and yaml is not None:
        return yaml.safe_load(text)
    try:
        return json.loads(text)
    except Exception:
        # Fallback: simple KEY=VALUE lines
        cfg: Dict[str, Any] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                cfg[k.strip()] = v.strip()
        return cfg

def safe_read_csv(path: str | os.PathLike, nrows: Optional[int] = None):
    import pandas as pd
    try:
        return pd.read_csv(path, nrows=nrows)
    except Exception as e:
        print("CSV read failed, using Python engine:", e)
        return pd.read_csv(path, nrows=nrows, engine='python')

@contextmanager
def timer(msg: str):
    t0 = time.time()
    print(f"[START] {msg}")
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[END] {msg} â {dt:.2f}s")
