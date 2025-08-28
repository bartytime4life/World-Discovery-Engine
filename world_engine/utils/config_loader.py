"""
Config loader & resolver

- load_yaml(path): loads YAML into dict (requires PyYAML)
- resolve_config(path, overrides_json): loads and applies optional JSON overrides

We avoid hard dependencies beyond PyYAML (very common) and standard library.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError as e:
    raise ImportError(
        "PyYAML is required to load configs. Install with: pip install pyyaml"
    ) from e


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    return yaml.safe_load(p.read_text())


def deep_merge(a: Dict, b: Dict) -> Dict:
    """Recursively merge dict b into a (returns a new dict)."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_config(path: str | Path, overrides_json: Optional[str] = None) -> Dict:
    cfg = load_yaml(path)
    if overrides_json:
        # Accept a JSON string (e.g. {"aoi":{"bbox":[-70,-13,-62,-6]}})
        overrides = json.loads(overrides_json)
        cfg = deep_merge(cfg, overrides)
    # Fill default run section if missing
    cfg.setdefault("run", {"output_dir": "outputs", "random_seed": 42, "use_gpu": False, "num_workers": 0})
    return cfg
