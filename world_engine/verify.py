"""
Stage 4 — Verify: Evidence fusion & gates

Implements:
- Multi-proof rule: require >= N modalities
- ADE fingerprints (toy proxies): soil-P threshold (config), NDVI peak & geomorph flag from evaluate
- Causal plausibility (stub): tag a simple cause chain if plausible
- Uncertainty (stub): derive a trivial uncertainty bucket
- SSIM what-if (stub): remove NDVI and see if candidate drops below plausibility

Outputs verify_candidates.json (final filtered set + scores) for reporting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .utils.logging_utils import get_logger


def _toy_soil_p_ppm(center_lat: float, center_lon: float) -> int:
    """Deterministic 'soil phosphorus' ppm proxy based on coords."""
    # Keep simple & repeatable: vary across 600..1200 ppm in a lat/lon-based pattern
    v = 900 + int((center_lat * 13 + center_lon * 7) % 300) - 150  # 750..1050
    return max(600, min(1200, v))


def run_verify(cfg: Dict, prev: Dict | None = None) -> Dict:
    log = get_logger("wde.verify")
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    eval_file = out_dir / "evaluate_candidates.json"
    if not eval_file.exists():
        raise FileNotFoundError("Missing evaluate_candidates.json. Run evaluate first.")

    required_modalities = int(cfg["pipeline"]["verify"].get("multiproof_required", 2))
    soil_p_threshold = int(cfg["pipeline"]["verify"]["ade_fingerprints"].get("soil_p_threshold", 800))
    use_ndvi = bool(cfg["pipeline"]["verify"]["ade_fingerprints"].get("dry_season_ndvi_peaks", True))
    use_geom = bool(cfg["pipeline"]["verify"]["ade_fingerprints"].get("geomorph_mound", True))
    do_ssim = bool(cfg["pipeline"]["verify"]["ssim_counterfactual"].get("enabled", True))

    candidates: List[Dict] = json.loads(eval_file.read_text())
    final: List[Dict] = []

    for c in candidates:
        lat, lon = c["center"]
        soil_p = _toy_soil_p_ppm(lat, lon)

        # Evaluate ADE fingerprints
        ade_hits = 0
        if soil_p >= soil_p_threshold:
            ade_hits += 1
        if use_ndvi and c.get("ndvi_stable", False):
            ade_hits += 1
        if use_geom and c.get("hydro_geomorph_plausible", False):
            ade_hits += 1

        # Build a trivial modality list
        modalities = []
        if soil_p >= soil_p_threshold:
            modalities.append("soil_p_high")
        if c.get("ndvi_stable", False):
            modalities.append("ndvi_stable")
        if c.get("hydro_geomorph_plausible", False):
            modalities.append("hydro_geomorph")

        # SSIM what-if (toy): if we "remove" NDVI evidence, does it still pass?
        still_passes = True
        if do_ssim and "ndvi_stable" in modalities:
            mods_no_ndvi = [m for m in modalities if m != "ndvi_stable"]
            still_passes = len(mods_no_ndvi) >= required_modalities

        # Uncertainty proxy: fewer modalities → higher uncertainty
        uncertainty = "low" if len(modalities) >= (required_modalities + 1) else ("medium" if len(modalities) >= required_modalities else "high")

        # Multi-proof gate
        passes = len(modalities) >= required_modalities and still_passes

        enriched = dict(c)
        enriched.update(
            {
                "soil_p_ppm": soil_p,
                "ade_modalities": modalities,
                "ade_modalities_count": len(modalities),
                "required_modalities": required_modalities,
                "uncertainty": uncertainty,
                "passes_verification": bool(passes),
                "ssim_removed_ndvi_still_pass": bool(still_passes),
                "causal_chain": "elevation→soil_moisture→vegetation" if c.get("hydro_geomorph_plausible") else "unclear",
                "verify_score": round(min(1.0, 0.25 * len(modalities)), 3),  # a simple score
            }
        )
        if passes:
            final.append(enriched)

    out_path = out_dir / "verify_candidates.json"
    out_path.write_text(json.dumps(final, indent=2))
    log.info(f"Verify: kept {len(final)} final candidates → {out_path}")

    return {
        "stage": "verify",
        "verify_file": str(out_path),
        "num_final": len(final),
        "soil_p_threshold": soil_p_threshold,
        "required_modalities": required_modalities,
    }
