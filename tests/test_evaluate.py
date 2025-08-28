from __future__ import annotations

import json
from pathlib import Path

from world_engine.ingest import run_ingest
from world_engine.detect import run_detect
from world_engine.evaluate import run_evaluate


def test_evaluate_enrichment(tmp_path: Path, cfg):
    cfg["run"]["output_dir"] = str((tmp_path / "outputs").as_posix())
    run_ingest(cfg)
    run_detect(cfg)
    art = run_evaluate(cfg)
    eval_path = Path(art["evaluate_file"])
    assert eval_path.exists()
    rows = json.loads(eval_path.read_text())
    assert isinstance(rows, list)
    if rows:
        r0 = rows[0]
        # Expect enrichment fields from evaluate stage
        assert {"ndvi_stable", "distance_to_river_km", "hydro_geomorph_plausible"} <= set(r0.keys())
