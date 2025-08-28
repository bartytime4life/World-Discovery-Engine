from __future__ import annotations

import json
from pathlib import Path

from world_engine.ingest import run_ingest


def test_ingest_produces_tiles(tmp_path: Path, cfg):
    cfg["run"]["output_dir"] = str((tmp_path / "outputs").as_posix())
    art = run_ingest(cfg)
    tiles_file = Path(art["tiles_file"])
    assert tiles_file.exists(), "tiles.json should exist"
    tiles = json.loads(tiles_file.read_text())
    assert isinstance(tiles, list) and len(tiles) >= 1, "Tiles list should be non-empty"
    assert {"id", "bbox", "center"} <= set(tiles[0].keys())
