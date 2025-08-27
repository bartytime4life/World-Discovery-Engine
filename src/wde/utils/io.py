# FILE: src/wde/utils/io.py
# =================================================================================================
# I/O helpers: YAML/JSON load/save, directory ensure, and minimal notebook writer
# Keep this module lightweight and dependency-minimal.
# =================================================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists (mkdir -p) and return its Path.
    If `path` is a file path, ensure its parent exists.
    """
    p = Path(path)
    if p.suffix and not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml_safe(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a Python dict using safe loader.
    Returns {} if the file is empty.

    Parameters
    ----------
    path : str | Path
        Path to the YAML file.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.

    yaml.YAMLError
        If the YAML is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def write_json(path: str | Path, data: Dict[str, Any]) -> Path:
    """
    Write a JSON file with UTF-8 encoding and pretty formatting.

    Parameters
    ----------
    path : str | Path
        Output path.
    data : dict
        JSON-serializable mapping.
    """
    p = ensure_dir(path)
    with Path(p).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return Path(p)


def write_ipynb_minimal(path: str | Path, nb_json: Dict[str, Any]) -> Path:
    """
    Write a minimal Jupyter notebook (already-formed JSON dict) to `path`.

    This avoids importing nbformat to keep the CLI lightweight.
    """
    p = ensure_dir(path)
    with Path(p).open("w", encoding="utf-8") as f:
        json.dump(nb_json, f, ensure_ascii=False, indent=1)
    return Path(p)