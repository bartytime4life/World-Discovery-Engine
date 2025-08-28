from __future__ import annotations

import json
from pathlib import Path

from world_engine.ingest import run_ingest
from world_engine.detect import run_detect


def test_detect_candidates_exist(tmp_path: Path, cfg):
    cfg["run"]["output_dir"] = str((tmp_path / "outputs").as_posix())
    run_ingest(cfg)
    art = run_detect(cfg)
    cand_path = Path(art["candidates_file"])
    assert cand_path.exists(), "detect_candidates.json should exist"
    cands = json.loads(cand_path.read_text())
    assert isinstance(cands, list)
    # With threshold=0.6 in fixture, keep >=0 candidates (depends on toy scoring)
    assert all("tile_id" in c and "score" in c for c in cands)
