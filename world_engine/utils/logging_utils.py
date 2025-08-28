"""
Logging utils

- init_logging(cfg): configure root logger + file handler if enabled
- get_logger(name): module-level logger getter
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict


def init_logging(cfg: Dict):
    log_cfg = cfg.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if log_cfg.get("to_file", False):
        log_file = Path(log_cfg.get("file", "logs/pipeline.log"))
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
