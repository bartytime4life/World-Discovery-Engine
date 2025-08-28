"""
Shared pytest fixtures for WDE tests.

Creates a minimal pipeline config in a temp folder, points outputs to tmp_path,
and provides a loaded config dict via world_engine.utils.config_loader.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest
from world_engine.utils.config_loader import load_yaml


_MIN_PIPELINE_YAML = """\
run:
  profile: "test"
  random_seed: 42
  use_gpu: false
  num_workers: 0
  output_dir: "{OUT}"

aoi:
  # very small bbox; order: [min_lat, min_lon, max_lat, max_lon]
  bbox: [-8.60, -60.10, -8.55, -60.05]
  tile_size_deg: 0.05

pipeline:
  ingest:   {enabled: true, cache_dir: "data/cache/", reprojection: "EPSG:4326", normalize: true}
  detect:
    enabled: true
    anomaly_threshold: 0.6
  evaluate:
    enabled: true
    ndvi_timeseries: {enabled: true}
    hydro_geomorph:  {enabled: true, require_near_water_km: 30.0}
    lidar:           {enabled: false}
  verify:
    enabled: true
    multiproof_required: 2
    ade_fingerprints:
      enabled: true
      soil_p_threshold: 700
      dry_season_ndvi_peaks: true
      floristic_signatures: true
      geomorph_mound: true
    causal_graph:     {enabled: true}
    uncertainty:      {enabled: true, method: "bayesian_gnn"}
    ssim_counterfactual: {enabled: true}
  report:
    enabled: true
    format: "md"
    include:
      maps: true
      ndvi_plots: true
      soil_maps: true
      landcover: true
      sar: true
      pag_graph: true
      uncertainty: true
      ssim: true
      narrative: true
    redact_sensitive_coords: true

logging:
  level: "INFO"
  to_file: false

ethics:
  enable_sovereignty_checks: true
  mask_coords: true
  show_warnings: true
"""


@pytest.fixture(scope="function")
def cfg_path(tmp_path: Path) -> Path:
    """Writes a minimal pipeline.yaml into tmp_path/configs/ and returns its path."""
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "outputs"
    text = _MIN_PIPELINE_YAML.replace("{OUT}", str(out_dir.as_posix()))
    p = cfg_dir / "pipeline.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture(scope="function")
def cfg(tmp_path: Path, cfg_path: Path) -> Dict:
    """Loads the YAML produced by cfg_path for convenience."""
    return load_yaml(cfg_path)
