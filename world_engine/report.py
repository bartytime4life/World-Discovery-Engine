"""
Stage 5 — Report: Candidate dossiers (lightweight)

Generates:
- JSON summary (report_summary.json)
- A Markdown dossier per candidate in outputs/reports/
- A GeoJSON with points for quick GIS inspection

This remains light: no heavy plotting or PDF engines to keep CI/Kaggle-friendly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .utils.ethics import mask_coordinates_if_required
from .utils.logging_utils import get_logger


def _as_feature(candidate: Dict, mask: bool, precision: int = 4) -> Dict:
    """Convert a candidate to a minimal GeoJSON Feature (Point)."""
    lat, lon = candidate["center"]
    lat_m, lon_m = mask_coordinates_if_required(lat, lon, enable_mask=mask)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(lon_m, precision), round(lat_m, precision)]},
        "properties": {
            "tile_id": candidate["tile_id"],
            "score": candidate.get("score"),
            "ade_modalities": candidate.get("ade_modalities", []),
            "uncertainty": candidate.get("uncertainty", "unknown"),
        },
    }


def run_report(cfg: Dict, prev: Dict | None = None) -> Dict:
    log = get_logger("wde.report")
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    verify_file = out_dir / "verify_candidates.json"
    if not verify_file.exists():
        raise FileNotFoundError("Missing verify_candidates.json. Run verify first.")

    candidates: List[Dict] = json.loads(verify_file.read_text())

    # Prepare reporting dirs
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Ethics settings
    ethics_cfg = cfg.get("ethics", {})
    mask_coords = bool(ethics_cfg.get("mask_coords", True))
    redact_coords = bool(cfg["pipeline"]["report"].get("redact_sensitive_coords", True))
    do_mask = mask_coords or redact_coords

    # Write per-candidate dossier (Markdown)
    for i, c in enumerate(candidates, 1):
        lat, lon = c["center"]
        lat_m, lon_m = mask_coordinates_if_required(lat, lon, enable_mask=do_mask)

        md = [
            f"# WDE Candidate Dossier — {i}",
            "",
            "## Location",
            f"- Center (masked): lat={lat_m:.4f}, lon={lon_m:.4f}" if do_mask else f"- Center: lat={lat:.4f}, lon={lon:.4f}",
            f"- Tile ID: {c['tile_id']}",
            "",
            "## Evidence Summary",
            f"- Detect score: {c.get('score', 'NA')}",
            f"- ADE Modalities: {', '.join(c.get('ade_modalities', []))} (count={c.get('ade_modalities_count', 0)})",
            f"- Soil P (ppm): {c.get('soil_p_ppm', 'NA')} (threshold={prev.get('soil_p_threshold', 'cfg') if prev else 'cfg'})",
            f"- NDVI stable: {c.get('ndvi_stable', 'NA')}",
            f"- Hydro-geomorph plausible: {c.get('hydro_geomorph_plausible', 'NA')} (distance_to_river_km={c.get('distance_to_river_km','NA')})",
            "",
            "## Verification & Robustness",
            f"- Required modalities: {c.get('required_modalities', 'NA')}",
            f"- SSIM remove-NDVI still passes: {c.get('ssim_removed_ndvi_still_pass', 'NA')}",
            f"- Uncertainty: {c.get('uncertainty', 'NA')}",
            f"- Causal chain: {c.get('causal_chain', 'NA')}",
            f"- Verify score: {c.get('verify_score', 'NA')}",
            "",
            "## Notes",
            "- Coordinates may be masked for ethics/sovereignty compliance.",
        ]
        (rep_dir / f"candidate_{i:03d}.md").write_text("\n".join(md), encoding="utf-8")

    # GeoJSON export
    features = [_as_feature(c, mask=do_mask) for c in candidates]
    geojson = {"type": "FeatureCollection", "features": features}
    (out_dir / "verify_candidates.geojson").write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    # Summary JSON
    summary = {
        "num_candidates": len(candidates),
        "reports_dir": str(rep_dir),
        "geojson": str(out_dir / "verify_candidates.geojson"),
        "coordinates_masked": bool(do_mask),
    }
    (out_dir / "report_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"Report: wrote {len(candidates)} dossiers → {rep_dir}")

    return {"stage": "report", **summary}
