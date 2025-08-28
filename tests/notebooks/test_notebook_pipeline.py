# tests/notebooks/test_notebook_pipeline.py
# =============================================================================
# WDE — Notebook Execution Test (pytest)
# Runs the Kaggle-style pipeline notebook end-to-end with papermill, then
# validates that the key artifacts were produced in the configured outputs dir.
# =============================================================================

from __future__ import annotations

import json
import os
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pytest


def _require(module_name: str):
    """Dynamically import a module or skip the test with a helpful message."""
    try:
        return __import__(module_name)
    except Exception as e:
        pytest.skip(f"Optional dependency '{module_name}' is not available: {e}")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_candidates(out_dir: Path) -> Dict[str, List[Path]]:
    """Return a mapping of artifact labels to candidate path patterns."""
    return {
        "candidates.json": [out_dir / "candidates.json"],
        "candidates.geojson": [out_dir / "candidates.geojson"],
        "reports": [out_dir / "reports"],
        "pag_graphs": [out_dir / "pag"],
        "uncertainty": [out_dir / "uncertainty"],
        "ssim": [out_dir / "ssim"],
        "ndvi_timeseries": [out_dir / "ndvi_timeseries"],
    }


def _exists_any(paths: List[Path]) -> bool:
    for p in paths:
        if p.exists():
            # For directories, require they contain at least one file
            if p.is_dir():
                try:
                    next(p.iterdir())
                    return True
                except StopIteration:
                    continue
            return True
    return False


@pytest.mark.timeout(1800)  # 30 minutes budget; tune as needed
def test_ade_pipeline_notebook_executes_and_produces_artifacts(tmp_path: Path):
    """
    Execute notebooks/ade_discovery_pipeline.ipynb with papermill and check outputs.

    This test:
      1) Executes the notebook into a temporary working directory.
      2) Validates that required artifacts (JSON/GeoJSON/dossiers/etc.) exist.
      3) Optionally verifies deterministic hash for candidates.json (if present) by
         executing a second time and comparing the hash.
    """
    papermill = _require("papermill")
    nbformat = _require("nbformat")

    # Resolve notebook path (allow override via env for CI flexibility)
    repo_root = Path(__file__).resolve().parents[2]
    nb_path = Path(os.environ.get("WDE_NOTEBOOK_PATH", repo_root / "notebooks" / "ade_discovery_pipeline.ipynb"))
    assert nb_path.exists(), f"Notebook not found at {nb_path}"

    # Prepare a clean execution directory
    work_dir = tmp_path / "nb_run"
    out_dir = work_dir / "outputs"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Papermill output notebook path
    executed_nb = work_dir / "ade_discovery_pipeline_out.ipynb"

    # Parameters — the notebook should respect these (if defined).
    # We pass an outputs path so artifacts are written inside the temp dir.
    params: Dict[str, object] = {
        # Common patterns used in the scaffold:
        "OUTPUTS_DIR": str(out_dir),              # if the notebook accepts this param
        "RUN_PROFILE": "light",                   # e.g., "light"/"heavy" profiles
        "USE_LIDAR": False,                       # speed up CI by skipping heavy steps
        # Provide a tiny AOI if notebook supports AOI override (bbox or named AOI)
        # "AOI_bbox": (-3.50, -60.50, -3.49, -60.49),
    }

    # Execute the notebook
    papermill.execute_notebook(
        input_path=str(nb_path),
        output_path=str(executed_nb),
        parameters=params,
        cwd=str(work_dir),
        kernel_name=os.environ.get("WDE_KERNEL", "python3"),
        progress_bar=False,
        request_save_on_cell_execute=True,
    )

    # Basic sanity: executed notebook exists and is valid JSON/IPYNB
    assert executed_nb.exists(), "Executed notebook was not created"
    nb = nbformat.read(executed_nb, as_version=4)
    assert "cells" in nb and len(nb["cells"]) > 0, "Executed notebook appears empty"

    # Validate expected artifacts
    artifacts = _artifact_candidates(out_dir)
    missing: List[str] = []
    for label, paths in artifacts.items():
        if not _exists_any(paths):
            missing.append(label)

    # We tolerate partial outputs in "light" profile, but the core contract must hold:
    # require at least candidates.json, candidates.geojson, and reports/
    required_core = ["candidates.json", "candidates.geojson", "reports"]
    core_missing = [m for m in required_core if m in missing]
    assert not core_missing, (
        "Missing core artifacts: "
        + ", ".join(core_missing)
        + f"\nChecked output dir: {out_dir}"
        + f"\nAll missing (for reference): {missing}"
    )

    # If candidates.json exists, ensure it's valid JSON and contains plausible structure
    cand_json = out_dir / "candidates.json"
    if cand_json.exists():
        data = json.loads(cand_json.read_text())
        assert isinstance(data, (list, dict)), "candidates.json should be a list or dict"
        # Optional: if list, ensure elements are dict-like candidates
        if isinstance(data, list) and data:
            assert isinstance(data[0], dict), "candidates.json items should be objects"

    # Optional determinism check: rerun and compare hash of candidates.json (if present)
    if cand_json.exists():
        first_hash = _hash_file(cand_json)

        # Re-run quickly (ideally the notebook is deterministic given fixed seeds/config)
        executed_nb2 = work_dir / "ade_discovery_pipeline_out_rerun.ipynb"
        papermill.execute_notebook(
            input_path=str(nb_path),
            output_path=str(executed_nb2),
            parameters=params,
            cwd=str(work_dir),
            kernel_name=os.environ.get("WDE_KERNEL", "python3"),
            progress_bar=False,
            request_save_on_cell_execute=True,
        )
        assert cand_json.exists(), "candidates.json missing after rerun"
        second_hash = _hash_file(cand_json)

        assert first_hash == second_hash, (
            "Non-deterministic output detected for candidates.json.\n"
            f"First run hash:  {first_hash}\n"
            f"Second run hash: {second_hash}\n"
            "Ensure all RNG seeds are fixed and environment-dependent paths are stable."
        )


@pytest.mark.timeout(600)
def test_starter_notebook_runs(tmp_path: Path):
    """Smoke test for an optional starter/demo notebook (if present)."""
    papermill = _require("papermill")
    nbformat = _require("nbformat")

    repo_root = Path(__file__).resolve().parents[2]
    starter = repo_root / "notebooks" / "WDE_Kaggle_Starter.ipynb"
    if not starter.exists():
        pytest.skip("Starter notebook not found — skipping smoke test")

    executed = tmp_path / "starter_out.ipynb"
    papermill.execute_notebook(
        input_path=str(starter),
        output_path=str(executed),
        parameters={},
        cwd=str(tmp_path),
        kernel_name=os.environ.get("WDE_KERNEL", "python3"),
        progress_bar=False,
    )
    assert executed.exists(), "Starter notebook did not execute"
