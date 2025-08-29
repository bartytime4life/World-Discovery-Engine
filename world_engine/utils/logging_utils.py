# world_engine/utils/logging_utils.py
# ======================================================================================
# World Discovery Engine (WDE)
# Logging Utilities — Config-driven, reproducible, Kaggle/CI-friendly logging
# --------------------------------------------------------------------------------------
# Purpose
#   Provide a unified logging setup for the WDE pipeline, ensuring:
#     • Configurable levels, formats, and destinations (console, file, JSON).
#     • Kaggle/CI safety (defaults: INFO to stdout; file logs optional).
#     • Deterministic log file naming with timestamps + run IDs for reproducibility.
#     • JSON log export for structured downstream analysis if needed.
#
# Design
#   - init_logging(cfg): sets root logger with console + optional file/JSON handlers.
#   - get_logger(name): retrieve a namespaced logger.
#   - Utility: add_run_metadata(logger, run_id, cfg_hash) → auto-injects metadata fields.
#
# Dependencies: Python stdlib only (logging, json, datetime, pathlib).
#
# License
#   MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------------------
# Default formats
# --------------------------------------------------------------------------------------

_CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# JSON handler wrapper
class _JSONLogHandler(logging.Handler):
    """
    A logging handler that writes structured JSON logs to a file.
    Each record is stored as one JSON object per line (JSONL).
    """

    def __init__(self, path: Path, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_entry = {
                "time": datetime.utcnow().isoformat() + "Z",
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "module": record.module,
                "func": record.funcName,
                "line": record.lineno,
            }
            self._fh.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            if not self._fh.closed:
                self._fh.close()
        finally:
            super().close()


# --------------------------------------------------------------------------------------
# Init + Helpers
# --------------------------------------------------------------------------------------

def init_logging(cfg: Dict[str, Any], run_id: Optional[str] = None) -> None:
    """
    Configure logging system from config dictionary.

    Parameters
    ----------
    cfg : dict
        Config dictionary (expects key "logging" with options).
    run_id : str, optional
        Unique run identifier (e.g. timestamp, git hash). Used in file naming.
    """
    log_cfg = cfg.get("logging", {}) if cfg else {}
    level_str = str(log_cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    # Reset root logger
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level)

    # Console handler (always on)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(_CONSOLE_FORMAT))
    root.addHandler(ch)

    # Optional file handler
    if log_cfg.get("to_file", False):
        log_dir = Path(log_cfg.get("dir", "logs"))
        run_tag = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"wde_{run_tag}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(_FILE_FORMAT))
        root.addHandler(fh)

    # Optional JSON log handler
    if log_cfg.get("to_json", False):
        log_dir = Path(log_cfg.get("dir", "logs"))
        run_tag = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_file = log_dir / f"wde_{run_tag}.jsonl"
        root.addHandler(_JSONLogHandler(json_file, level=level))

    root.debug("Logging initialized", extra={"run_id": run_id})


def get_logger(name: str) -> logging.Logger:
    """
    Retrieve a module-specific logger.
    """
    return logging.getLogger(name)


def add_run_metadata(logger: logging.Logger, run_id: str, cfg_hash: str) -> None:
    """
    Inject metadata into logger's extra dict (used for dashboards/CI parsing).
    """
    logger.info(f"[RunMeta] run_id={run_id} cfg_hash={cfg_hash}")


# --------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    test_cfg = {
        "logging": {
            "level": "DEBUG",
            "to_file": True,
            "to_json": True,
            "dir": "_logs_test"
        }
    }
    init_logging(test_cfg, run_id="demo123")
    log = get_logger("wde.utils.logging")
    log.info("Hello logging world")
    log.warning("This is a warning with structured JSON too.")
    add_run_metadata(log, "demo123", "hash_abc")
