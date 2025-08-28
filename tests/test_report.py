from __future__ import annotations

import json
from pathlib import Path

from world_engine.ingest import run_ingest
from world_engine.detect import run_detect
from world_engine.evaluate import run_evaluate
from world_engine.verify import run_verify
from world_engine.report import run_report


def test_report_outputs(tmp_path: Path, cfg):
    cfg["run"]["output_dir"] = str((tmp_path / "outputs").as_posix())
    run_ingest(cfg)
    run_detect(cfg)
    run_evaluate(cfg)
    run_verify(cfg)
    art = run_report(cfg)
    reports_dir = Path(art["reports_dir"])
    geojson = Path(art["geojson"])
    summary = Path(tmp_path / "outputs" / "report_summary.json")
    # Existence
    assert reports_dir.exists()
    assert geojson.exists()
    assert summary.exists()
    # Structure checks
    gj = json.loads(geojson.read_text())
    assert gj.get("type") == "FeatureCollection"
    feats = gj.get("features", [])
    for f in feats:
        assert f["geometry"]["type"] == "Point"
        assert "tile_id" in f["properties"]
