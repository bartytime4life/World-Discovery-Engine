# FILE: world_engine/report.py
# -------------------------------------------------------------------------------------------------
# Stage 5 — Report: Candidate dossiers (lightweight, CI/Kaggle-friendly)
#
# Responsibilities
# ----------------
# 1) Read verified candidates (verify_candidates.json) produced by the "verify" stage.
# 2) Generate:
#    - Per-candidate Markdown dossiers (outputs/reports/candidate_XXX.md)
#    - Per-candidate manifest JSON (outputs/reports/candidate_XXX_manifest.json)
#    - A lightweight HTML mirror for each dossier (no extra deps) (outputs/reports/candidate_XXX.html)
#    - GeoJSON of candidate points (outputs/verify_candidates.geojson)
#    - CSV + Markdown index table of all candidates (outputs/report_index.csv / outputs/report_index.md)
#    - Summary JSON (outputs/report_summary.json)
#
# Design Notes
# ------------
# - No heavy plotting or PDF engines to keep CI & Kaggle runtime fast and reproducible.
# - Coordinates are masked/obfuscated if configured via ethics settings.
# - Each candidate gets a small, machine-readable manifest capturing evidence paths, provenance, and flags.
# - Strictly uses stdlib + internal utils (no external converters).
# - Defensive I/O: creates directories, checks existence, logs everything.
#
# Expected Inputs
# ---------------
# - cfg: dict-like pipeline config with at least:
#     cfg["run"]["output_dir"] : str  -> path to the run outputs root
#     cfg["pipeline"]["report"] : dict (optional) -> additional report settings
#     cfg["ethics"] : dict (optional) -> ethics/masking switches
# - verify_candidates.json at {output_dir}/verify_candidates.json
#
# Produced Outputs
# ----------------
# - {output_dir}/reports/candidate_XXX.md
# - {output_dir}/reports/candidate_XXX.html
# - {output_dir}/reports/candidate_XXX_manifest.json
# - {output_dir}/verify_candidates.geojson
# - {output_dir}/report_index.csv
# - {output_dir}/report_index.md
# - {output_dir}/report_summary.json
#
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils.ethics import mask_coordinates_if_required
from .utils.logging_utils import get_logger


# ------------------------------- Helpers: GeoJSON / Coordinates ---------------------------------


def _as_feature(candidate: Dict[str, Any], mask: bool, precision: int = 4) -> Dict[str, Any]:
    """
    Convert a candidate dict to a minimal GeoJSON Feature (Point).

    Required candidate keys (set in verify stage):
      - "center" : (lat, lon)
      - "tile_id" : any
    Optional:
      - "score", "verify_score", "uncertainty", "ade_modalities", ...

    Masking is performed if enabled.
    """
    lat, lon = candidate["center"]
    lat_m, lon_m = mask_coordinates_if_required(lat, lon, enable_mask=mask)
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [round(lon_m, precision), round(lat_m, precision)],
        },
        "properties": {
            "tile_id": candidate.get("tile_id"),
            "score": candidate.get("score"),
            "verify_score": candidate.get("verify_score"),
            "uncertainty": candidate.get("uncertainty"),
            "ade_modalities": candidate.get("ade_modalities", []),
            "coordinates_masked": bool(mask),
        },
    }


def _masked_center(center: Tuple[float, float], do_mask: bool, precision: int = 6) -> Tuple[float, float]:
    lat, lon = center
    lat_m, lon_m = mask_coordinates_if_required(lat, lon, enable_mask=do_mask)
    return round(lat_m, precision), round(lon_m, precision)


# --------------------------------------- Helpers: Files ------------------------------------------


def _safe_id(index: int) -> str:
    """Create a stable 3-digit candidate id string."""
    return f"{index:03d}"


def _utc_now() -> str:
    """ISO 8601 UTC timestamp."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ------------------------------------ Manifest per candidate -------------------------------------


def _dump_site_manifest(
    site_id: str,
    out_dir: Path,
    bbox: List[float],
    evidence: Dict[str, Any],
    confidence: Optional[float],
    uncertainty: Optional[float],
    provenance: Dict[str, Any],
    ethical_flags: Dict[str, Any],
) -> Path:
    """
    Write per-site manifest JSON capturing minimal machine-readable dossier info.

    Returns Path to written manifest.
    """
    manifest = {
        "site_id": site_id,
        "generated_at": _utc_now(),
        "bbox": bbox,
        "evidence": evidence,  # e.g. {"markdown": "...md", "html": "...html", "geojson_index": "..."}
        "confidence": confidence,
        "uncertainty": uncertainty,
        "provenance": provenance,  # e.g. {"datasets": [...], "config_path": "...", "stage": "verify→report"}
        "ethical_flags": ethical_flags,
    }
    manifest_path = out_dir / f"{site_id}_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path


# --------------------------------- Dossier: Markdown / HTML --------------------------------------


def _md_lines_for_candidate(
    idx: int,
    cand: Dict[str, Any],
    latlon_masked: Tuple[float, float],
    do_mask: bool,
    prev: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Construct the dossier markdown body for a candidate.

    We keep it lightweight and structured for fast inspection.
    """
    lat_m, lon_m = latlon_masked
    lat, lon = cand["center"]

    lines: List[str] = []
    lines.append(f"# WDE Candidate Dossier — {idx}")
    lines.append("")
    lines.append("## Location")
    if do_mask:
        lines.append(f"- Center (masked): lat={lat_m:.6f}, lon={lon_m:.6f}")
    else:
        lines.append(f"- Center: lat={lat:.6f}, lon={lon:.6f}")
    lines.append(f"- Tile ID: {cand.get('tile_id', 'NA')}")
    lines.append("")
    lines.append("## Evidence Summary")
    lines.append(f"- Detect score: {cand.get('score', 'NA')}")
    lines.append(f"- Verify score: {cand.get('verify_score', 'NA')}")
    lines.append(f"- ADE Modalities: {', '.join(cand.get('ade_modalities', []))} (count={cand.get('ade_modalities_count', 0)})")
    lines.append(f"- Soil P (ppm): {cand.get('soil_p_ppm', 'NA')} (threshold={prev.get('soil_p_threshold', 'cfg') if prev else 'cfg'})")
    lines.append(f"- NDVI stable: {cand.get('ndvi_stable', 'NA')}")
    lines.append(f"- Hydro-geomorph plausible: {cand.get('hydro_geomorph_plausible', 'NA')} (distance_to_river_km={cand.get('distance_to_river_km', 'NA')})")
    lines.append("")
    lines.append("## Verification & Robustness")
    lines.append(f"- Required modalities: {cand.get('required_modalities', 'NA')}")
    lines.append(f"- SSIM remove-NDVI still passes: {cand.get('ssim_removed_ndvi_still_pass', 'NA')}")
    lines.append(f"- Uncertainty: {cand.get('uncertainty', 'NA')}")
    lines.append(f"- Causal chain: {cand.get('causal_chain', 'NA')}")
    lines.append("")
    lines.append("## Ethics & Notes")
    lines.append("- Coordinates may be masked for ethics/sovereignty compliance.")
    if cand.get("indigenous_territory", False):
        lines.append("- ⚠ Candidate overlaps Indigenous territory: consult communities/authorities before action.")
    lines.append("")
    return lines


def _as_html_from_md(md_lines: List[str], title: str) -> str:
    """
    Very small Markdown→HTML shim (limited) to avoid extra dependencies.
    We only transform headers and bullet points for a clean HTML mirror.
    """
    html: List[str] = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append("<style>body{font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:1rem; max-width:760px; margin:auto;} h1{font-size:1.6rem} h2{font-size:1.2rem;margin-top:1.2rem} ul{line-height:1.5} code,pre{background:#f6f8fa;padding:.2rem .4rem;border-radius:.2rem}</style>")
    html.append("</head><body>")

    in_list = False
    for line in md_lines:
        if line.startswith("# "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>")
                in_list = True
            html.append(f"<li>{line[2:].strip()}</li>")
        elif line.strip() == "":
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append("<p></p>")
        else:
            # paragraph
            if in_list:
                html.append("</ul>")
                in_list = False
            html.append(f"<p>{line}</p>")
    if in_list:
        html.append("</ul>")

    html.append("</body></html>")
    return "\n".join(html)


# --------------------------------------- Public API ----------------------------------------------


def run_report(cfg: Dict[str, Any], prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate dossiers, per-site manifests, and indexes for verified candidates.

    Returns a small dict summary of produced artifacts.
    """
    log = get_logger("wde.report")

    # --------------------------- resolve paths & read verified input ------------------------------
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    verify_file = out_dir / "verify_candidates.json"
    if not verify_file.exists():
        raise FileNotFoundError("Missing verify_candidates.json. Run 'verify' stage first.")

    try:
        candidates: List[Dict[str, Any]] = json.loads(verify_file.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read/parse {verify_file}: {e}")

    # --------------------------- prepare directories & settings ----------------------------------
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Ethics settings
    ethics_cfg = cfg.get("ethics", {})
    mask_coords_default = bool(ethics_cfg.get("mask_coords", True))
    redact_coords = bool(cfg.get("pipeline", {}).get("report", {}).get("redact_sensitive_coords", True))
    do_mask = mask_coords_default or redact_coords

    # Optional provenance
    datasets_used = cfg.get("data", {}).get("datasets_used", [])  # optional; up to upstream stages to fill
    config_path = cfg.get("run", {}).get("config_path", "configs/default.yaml")

    # ------------------------------ per-candidate artifacts --------------------------------------

    md_index_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []

    for i, cand in enumerate(candidates, 1):
        cid = _safe_id(i)
        site_id = f"candidate_{cid}"
        md_path = rep_dir / f"{site_id}.md"
        html_path = rep_dir / f"{site_id}.html"

        # Coordinates / bbox
        lat_m, lon_m = _masked_center(tuple(cand["center"]), do_mask=do_mask, precision=6)
        bbox = cand.get("bbox") or [lat_m, lon_m, lat_m, lon_m]  # if not present, use point bbox

        # Build dossier (Markdown)
        md_lines = _md_lines_for_candidate(i, cand, (lat_m, lon_m), do_mask, prev)
        _write_text(md_path, "\n".join(md_lines))

        # Lightweight HTML mirror (for quick viewing in browsers)
        html = _as_html_from_md(md_lines, title=f"WDE Dossier — {site_id}")
        _write_text(html_path, html)

        # Minimal evidence bag (paths to artifacts we produced here)
        evidence = {
            "markdown": str(md_path.relative_to(out_dir)),
            "html": str(html_path.relative_to(out_dir)),
        }

        # Manifest per site (machine-readable)
        manifest_path = _dump_site_manifest(
            site_id=site_id,
            out_dir=rep_dir,
            bbox=bbox,
            evidence=evidence,
            confidence=cand.get("verify_score"),
            uncertainty=cand.get("uncertainty"),
            provenance={
                "datasets": datasets_used,
                "config_path": config_path,
                "stage": "verify→report",
                "source": str(verify_file.relative_to(out_dir)),
            },
            ethical_flags={
                "coordinates_masked": do_mask,
                "indigenous_territory": bool(cand.get("indigenous_territory", False)),
            },
        )

        # Collect for indexes
        md_index_rows.append(
            {
                "id": site_id,
                "lat": lat_m if do_mask else cand["center"][0],
                "lon": lon_m if do_mask else cand["center"][1],
                "score": cand.get("score", ""),
                "verify_score": cand.get("verify_score", ""),
                "uncertainty": cand.get("uncertainty", ""),
                "report_md": evidence["markdown"],
                "report_html": evidence["html"],
                "manifest": str(manifest_path.relative_to(out_dir)),
            }
        )
        csv_rows.append(
            {
                "id": site_id,
                "lat": lat_m if do_mask else cand["center"][0],
                "lon": lon_m if do_mask else cand["center"][1],
                "score": cand.get("score", ""),
                "verify_score": cand.get("verify_score", ""),
                "uncertainty": cand.get("uncertainty", ""),
            }
        )

        # GeoJSON Feature
        features.append(_as_feature(cand, mask=do_mask))

    # ------------------------------ collection-level exports -------------------------------------

    # GeoJSON export (all candidates)
    geojson_path = out_dir / "verify_candidates.geojson"
    _write_json(geojson_path, {"type": "FeatureCollection", "features": features})

    # CSV index
    csv_path = out_dir / "report_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "lat", "lon", "score", "verify_score", "uncertainty", "report_md", "report_html", "manifest"],
        )
        writer.writeheader()
        for row in md_index_rows:
            writer.writerow(row)

    # Markdown index (human-friendly)
    md_index_lines: List[str] = []
    md_index_lines.append("# WDE Report Index")
    md_index_lines.append("")
    md_index_lines.append(f"- Generated: {_utc_now()}")
    md_index_lines.append(f"- Coordinates masked: **{do_mask}**")
    md_index_lines.append("")
    md_index_lines.append("| # | Site ID | Lat | Lon | Score | Verify | Unc. | MD | HTML | Manifest |")
    md_index_lines.append("|:-:|:-------:|:---:|:---:|:-----:|:-----:|:----:|:--:|:----:|:--------:|")
    for k, row in enumerate(md_index_rows, 1):
        md_index_lines.append(
            f"| {k} | {row['id']} | {row['lat']} | {row['lon']} | {row['score']} | {row['verify_score']} | "
            f"{row['uncertainty']} | "
            f"[md]({row['report_md']}) | [html]({row['report_html']}) | [json]({row['manifest']}) |"
        )
    md_index_path = out_dir / "report_index.md"
    _write_text(md_index_path, "\n".join(md_index_lines))

    # Summary JSON
    summary = {
        "stage": "report",
        "generated_at": _utc_now(),
        "num_candidates": len(candidates),
        "reports_dir": str(rep_dir),
        "index_csv": str(csv_path),
        "index_md": str(md_index_path),
        "geojson": str(geojson_path),
        "coordinates_masked": bool(do_mask),
    }
    summary_path = out_dir / "report_summary.json"
    _write_json(summary_path, summary)

    log.info(
        f"Report complete: {len(candidates)} dossiers → {rep_dir} | "
        f"index: {csv_path.name}, {md_index_path.name} | geojson: {geojson_path.name}"
    )

    return summary
