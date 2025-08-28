from __future__ import annotations

import json
from pathlib import Path

from world_engine.ingest import run_ingest
from world_engine.detect import run_detect
from world_engine.evaluate import run_evaluate
from world_engine.verify import run_verify


def test_verify_filters_and_scores(tmp_path: Path, cfg):
    cfg["run"]["output_dir"] = str((tmp_path / "outputs").as_posix())
    run_ingest(cfg)
    run_detect(cfg)
    run_evaluate(cfg)
    art = run_verify(cfg)
    vf = Path(art["verify_file"])
    assert vf.exists()
    finals = json.loads(vf.read_text())
    assert isinstance(finals, list)
    # finals may be empty or not depending on the toy rules; check schema if present
    for row in finals:
        assert row.get("passes_verification", False) is True
        assert "ade_modalities_count" in row
        assert "uncertainty" in row
        assert "verify_score" in row
