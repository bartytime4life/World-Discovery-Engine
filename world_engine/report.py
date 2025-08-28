# FILE: world_engine/report.py
# -------------------------------------------------------------------------------------------------
# Stage 5 — Report: Candidate dossiers (lightweight, CI/Kaggle-friendly, ethics-aware, reproducible)
#
# Responsibilities
# ----------------
# 1) Load verified candidates (verify_candidates.json OR verify_candidates.geojson) produced by "verify".
# 2) Generate for each candidate:
#    - Markdown dossier .................. outputs/reports/candidate_XXX.md
#    - HTML mirror ........................ outputs/reports/candidate_XXX.html (no external deps)
#    - Per-site manifest .................. outputs/reports/candidate_XXX_manifest.json
# 3) Generate collection artifacts:
#    - GeoJSON index of candidates ........ outputs/verify_candidates.geojson
#    - CSV index .......................... outputs/report_index.csv
#    - Markdown index ..................... outputs/report_index.md
#    - JSON index (machine) ............... outputs/dossiers_index.json
#    - Summary JSON ....................... outputs/report_summary.json
#    - Manifest index ..................... outputs/manifest_index.json
#
# Design & Engineering Goals
# --------------------------
# - No heavy plotting/PDF deps → fast in CI & Kaggle.
# - Default-on per-site manifests (reproducibility, provenance).
# - Ethics guardrails: coordinate masking by default, optional Indigenous overlap flagging.
# - Hardened I/O: create dirs, validate inputs, graceful fallbacks, exhaustive logging hooks.
# - Stdlib-only (plus project utils) to avoid extra runtime requirements.
#
# Expected Config (subset)
# ------------------------
# cfg["run"]["output_dir"] : str
# cfg["run"]["version"] : str (optional)           -> included in summary/provenance
# cfg["run"]["config_path"] : str (optional)      -> included in provenance
#
# cfg["pipeline"]["report"]["redact_sensitive_coords"] : bool (default True)
# cfg["pipeline"]["report"]["mask_precision"] : int (default 6)
#
# cfg["ethics"]["mask_coords"] : bool (default True)
# cfg["ethics"]["indigenous_bounds_geojson"] : str (optional path to polygons GeoJSON)
#
# Input Files (produced by previous stages)
# -----------------------------------------
# {output_dir}/verify_candidates.json      : list[dict] of candidates  (preferred)
# OR
# {output_dir}/verify_candidates.geojson   : FeatureCollection          (fallback or overwritten here)
#
# -------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .utils.ethics import mask_coordinates_if_required
from .utils.logging_utils import get_logger


# =================================================================================================
# Small, local utilities (stdlib-only)
# =================================================================================================


def _utc_iso() -> str:
    """Return current UTC timestamp as ISO-8601 (Z) string."""
    return datetime.utcnow().replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _safe_id(i: int) -> str:
    """Zero-padded three-digit site id."""
    return f"{i:03d}"


def _precision(cfg: Dict[str, Any], default: int = 6) -> int:
    return int(cfg.get("pipeline", {}).get("report", {}).get("mask_precision", default))


# =================================================================================================
# Indigenous overlap (optional): tiny GeoJSON reader + ray casting Point-In-Polygon
# =================================================================================================


def _load_indigenous_polygons(path: Optional[Union[str, Path]]) -> List[List[Tuple[float, float]]]:
    """
    Load a very small subset of GeoJSON: return list of exterior rings for Polygon/MultiPolygon.
    Each ring is a list of (lon, lat) tuples. Stdlib only (no shapely/geopandas).
    """
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        return []
    try:
        gj = _read_json(p)
    except Exception:
        return []

    rings: List[List[Tuple[float, float]]] = []

    def _coerce_ring(coords: Iterable) -> List[Tuple[float, float]]:
        ring: List[Tuple[float, float]] = []
        for c in coords:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                ring.append((float(c[0]), float(c[1])))  # (lon, lat)
        return ring

    def _add_polygon(poly: Dict[str, Any]):
        # GeoJSON polygon coords: [ [ring0], [hole1], ... ] ; we use exterior ring only
        coords = poly.get("coordinates")
        if isinstance(coords, list) and coords:
            rings.append(_coerce_ring(coords[0]))

    gtype = (gj.get("type") or "").lower()
    if gtype == "featurecollection":
        feats = gj.get("features", [])
        for feat in feats:
            geom = feat.get("geometry") or {}
            t = (geom.get("type") or "").lower()
            if t == "polygon":
                _add_polygon(geom)
            elif t == "multipolygon":
                mpc = geom.get("coordinates", [])
                for poly_coords in mpc:
                    # normalize to polygon object
                    _add_polygon({"coordinates": poly_coords})
    elif gtype == "polygon":
        _add_polygon(gj)
    elif gtype == "multipolygon":
        for poly_coords in gj.get("coordinates", []):
            _add_polygon({"coordinates": poly_coords})

    # filter empty rings
    return [r for r in rings if len(r) >= 3]


def _point_in_ring(lon: float, lat: float, ring: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm for point-in-polygon (works for simple rings).
    Coordinates expected as lon/lat pairs. Returns True if inside or on edge.
    """
    inside = False
    n = len(ring)
    if n < 3:
        return False
    x, y = lon, lat
    x0, y0 = ring[0]
    for i in range(1, n + 1):
        x1, y1 = ring[i % n]
        # Check if the point is exactly on a segment (edge)
        # (rough check to avoid false negatives on border)
        if (min(y0, y1) <= y <= max(y0, y1)) and (min(x0, x1) <= x <= max(x0, x1)):
            # Compute cross product to see if collinear
            dx = x1 - x0
            dy = y1 - y0
            if abs(dy * (x - x0) - dx * (y - y0)) < 1e-12:
                return True
        # Ray casting: edges crossing the horizontal ray to the right of point
        intersect = ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-300) + x0)
        if intersect:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def _flag_indigenous_overlap(lon: float, lat: float, rings: List[List[Tuple[float, float]]]) -> bool:
    if not rings:
        return False
    for ring in rings:
        if _point_in_ring(lon, lat, ring):
            return True
    return False


# =================================================================================================
# GeoJSON feature conversion & masking
# =================================================================================================


def _as_feature(candidate: Dict[str, Any], mask: bool, precision: int = 4) -> Dict[str, Any]:
    """
    Convert a candidate dict to a minimal GeoJSON Feature (Point).

    Required candidate keys:
      - "center": (lat, lon)
    Optional:
      - "tile_id", "score", "verify_score", "uncertainty", "ade_modalities", ...
    """
    lat, lon = candidate["center"]
    ml, mlon = mask_coordinates_if_required(lat, lon, enable_mask=mask)
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [round(mlon, precision), round(ml, precision)]},
        "properties": {
            "tile_id": candidate.get("tile_id"),
            "score": candidate.get("score"),
            "verify_score": candidate.get("verify_score"),
            "uncertainty": candidate.get("uncertainty"),
            "ade_modalities": candidate.get("ade_modalities", []),
            "coordinates_masked": bool(mask),
        },
    }


def _masked_center(center: Tuple[float, float], do_mask: bool, precision: int) -> Tuple[float, float]:
    lat, lon = center
    ml, mlon = mask_coordinates_if_required(lat, lon, enable_mask=do_mask)
    return round(ml, precision), round(mlon, precision)


# =================================================================================================
# Dossier content (Markdown → HTML mirror)
# =================================================================================================


def _md_lines_for_candidate(
    idx: int,
    cand: Dict[str, Any],
    latlon_masked: Tuple[float, float],
    do_mask: bool,
    prev: Optional[Dict[str, Any]],
    indigenous_overlap: bool,
) -> List[str]:
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
    am = cand.get("ade_modalities", [])
    lines.append(f"- ADE Modalities: {', '.join(am) if am else 'NA'} (count={cand.get('ade_modalities_count', 0)})")
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
    if cand.get("indigenous_territory", False) or indigenous_overlap:
        lines.append("- ⚠ Candidate overlaps Indigenous territory: engage communities/authorities before action.")
    lines.append("")
    return lines


def _as_html_from_md(md_lines: List[str], title: str) -> str:
    html: List[str] = []
    html.append("<!doctype html><html><head><meta charset='utf-8'>")
    html.append(f"<title>{title}</title>")
    html.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:1rem;max-width:820px;margin:auto;}"
        "h1{font-size:1.6rem;margin:.2rem 0 .6rem} h2{font-size:1.2rem;margin:1rem 0 .4rem}"
        "ul{line-height:1.5} li{margin:.2rem 0} "
        "code,pre{background:#f6f8fa;padding:.2rem .4rem;border-radius:.2rem}"
        ".banner{padding:.5rem .75rem;border-radius:.5rem;margin:.5rem 0;background:#fff8e5;border:1px solid #f3d37a}"
        "</style>"
    )
    html.append("</head><body>")

    in_list = False
    for line in md_lines:
        if line.startswith("# "):
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("## "):
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("- "):
            if not in_list:
                html.append("<ul>"); in_list = True
            html.append(f"<li>{line[2:].strip()}</li>")
        elif not line.strip():
            if in_list:
                html.append("</ul>"); in_list = False
            html.append("<p></p>")
        else:
            if in_list:
                html.append("</ul>"); in_list = False
            html.append(f"<p>{line}</p>")

    if in_list:
        html.append("</ul>")

    html.append("</body></html>")
    return "\n".join(html)


# =================================================================================================
# Candidate / input loading
# =================================================================================================


def _load_candidates(out_dir: Path) -> List[Dict[str, Any]]:
    """
    Preferred: verify_candidates.json (list[dict]).
    Fallback : verify_candidates.geojson (FeatureCollection) → convert minimal fields.
    """
    jpath = out_dir / "verify_candidates.json"
    if jpath.exists():
        data = _read_json(jpath)
        if isinstance(data, list):
            return data

    gjpath = out_dir / "verify_candidates.geojson"
    if gjpath.exists():
        gj = _read_json(gjpath)
        if isinstance(gj, dict) and (gj.get("type") == "FeatureCollection"):
            cands: List[Dict[str, Any]] = []
            for i, feat in enumerate(gj.get("features", []) or [], 1):
                geom = feat.get("geometry") or {}
                props = feat.get("properties") or {}
                if (geom.get("type") == "Point" and isinstance(geom.get("coordinates"), (list, tuple))):
                    lon, lat = geom["coordinates"][:2]
                    cands.append(
                        {
                            "center": (float(lat), float(lon)),
                            "tile_id": props.get("tile_id", f"tile_{i:03d}"),
                            "score": props.get("score"),
                            "verify_score": props.get("verify_score"),
                            "uncertainty": props.get("uncertainty"),
                            "ade_modalities": props.get("ade_modalities", []),
                        }
                    )
            return cands

    raise FileNotFoundError("Missing verify_candidates.json or verify_candidates.geojson. Run 'verify' first.")


# =================================================================================================
# Per-site manifest (exposed for CLI; default-on)
# =================================================================================================


def dump_site_manifest(candidate: Dict[str, Any], out_dir: Path) -> Path:
    """
    Public helper for CLI: write a per-site manifest for one candidate.
    'candidate' should include:
        - site_id   : str (required)
        - tile_id   : str (optional)
        - center    : (lat,lon) or None
        - score / verify_score / uncertainty / ade_modalities ... (optional)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    site_id = str(candidate.get("site_id") or candidate.get("tile_id") or "site_unknown")
    path = out_dir / f"{site_id}_manifest.json"

    manifest = {
        "site_id": site_id,
        "generated_at": _utc_iso(),
        "center": candidate.get("center"),
        "score": candidate.get("score"),
        "verify_score": candidate.get("verify_score"),
        "uncertainty": candidate.get("uncertainty"),
        "ade_modalities": candidate.get("ade_modalities", []),
        "provenance": candidate.get("provenance"),
        "ethical_flags": {
            "coordinates_masked": bool(candidate.get("coordinates_masked", False)),
            "indigenous_territory": bool(candidate.get("indigenous_territory", False)),
        },
    }
    _write_json(path, manifest)
    return path


# =================================================================================================
# Public API: run_report
# =================================================================================================


def run_report(cfg: Dict[str, Any], prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate dossiers, per-site manifests, and indexes for verified candidates.

    Returns a dict summary of produced artifacts.
    """
    log = get_logger("wde.report")

    # ---------------------------------------------------------------------------------------------
    # Resolve paths, load inputs
    # ---------------------------------------------------------------------------------------------
    out_dir = Path(cfg["run"]["output_dir"]).resolve()
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load candidates (preferred JSON list; otherwise GeoJSON FeatureCollection)
    candidates = _load_candidates(out_dir)

    # Ethics & report settings
    ethics_cfg = cfg.get("ethics", {})
    mask_coords_default = bool(ethics_cfg.get("mask_coords", True))
    redact_coords = bool(cfg.get("pipeline", {}).get("report", {}).get("redact_sensitive_coords", True))
    do_mask = mask_coords_default or redact_coords
    prec = _precision(cfg, default=6)

    # Optional Indigenous territory overlap check (GeoJSON polygons)
    rings = _load_indigenous_polygons(ethics_cfg.get("indigenous_bounds_geojson"))

    # Provenance
    datasets_used = cfg.get("data", {}).get("datasets_used", [])
    config_path = Path(cfg.get("run", {}).get("config_path", "configs/default.yaml"))
    pipeline_version = cfg.get("run", {}).get("version", "v0.0.0")
    config_hash = _sha256_file(config_path) if config_path.exists() else None

    # ---------------------------------------------------------------------------------------------
    # Per-candidate artifacts
    # ---------------------------------------------------------------------------------------------
    features: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    md_index_rows: List[Dict[str, Any]] = []
    manifest_index: Dict[str, str] = []

    per_site_manifest_map: Dict[str, str] = {}

    for idx, cand in enumerate(candidates, 1):
        cid = _safe_id(idx)
        site_id = f"candidate_{cid}"
        md_path = reports_dir / f"{site_id}.md"
        html_path = reports_dir / f"{site_id}.html"

        # Coordinates / bbox / overlap
        lat, lon = cand["center"]
        lat_m, lon_m = _masked_center((lat, lon), do_mask=do_mask, precision=prec)
        bbox = cand.get("bbox") or [lat_m, lon_m, lat_m, lon_m]

        # Optional indigenous overlap using true (unmasked) coords
        overlap_flag = False
        if rings:
            try:
                overlap_flag = _flag_indigenous_overlap(float(lon), float(lat), rings)
            except Exception:
                overlap_flag = False

        # Build dossier (Markdown → HTML)
        md_lines = _md_lines_for_candidate(idx, cand, (lat_m, lon_m), do_mask, prev, overlap_flag)
        _write_text(md_path, "\n".join(md_lines))
        html = _as_html_from_md(md_lines, title=f"WDE Dossier — {site_id}")
        _write_text(html_path, html)

        # Evidence bag + checksums (provenance-hardened)
        evidence = {
            "markdown": str(md_path.relative_to(out_dir)),
            "html": str(html_path.relative_to(out_dir)),
            "markdown_hash": _sha256_file(md_path),
            "html_hash": _sha256_file(html_path),
        }

        # Per-site manifest (default-on)
        manifest = {
            "site_id": site_id,
            "generated_at": _utc_iso(),
            "bbox": bbox,
            "evidence": {"markdown": evidence["markdown"], "html": evidence["html"]},
            "confidence": cand.get("verify_score"),
            "uncertainty": cand.get("uncertainty"),
            "provenance": {
                "datasets": datasets_used,
                "config_path": str(config_path),
                "config_hash": config_hash,
                "pipeline_version": pipeline_version,
                "stage": "verify→report",
                "source": "verify_candidates.json or geojson",
            },
            "ethical_flags": {
                "coordinates_masked": do_mask,
                "indigenous_territory": bool(cand.get("indigenous_territory", False) or overlap_flag),
            },
        }
        manifest_path = reports_dir / f"{site_id}_manifest.json"
        _write_json(manifest_path, manifest)
        per_site_manifest_map[site_id] = str(manifest_path.relative_to(out_dir))

        # Index rows
        md_index_rows.append(
            {
                "id": site_id,
                "lat": lat_m if do_mask else lat,
                "lon": lon_m if do_mask else lon,
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
                "lat": lat_m if do_mask else lat,
                "lon": lon_m if do_mask else lon,
                "score": cand.get("score", ""),
                "verify_score": cand.get("verify_score", ""),
                "uncertainty": cand.get("uncertainty", ""),
            }
        )
        # GeoJSON feature
        features.append(_as_feature(cand, mask=do_mask, precision=4))

    # ---------------------------------------------------------------------------------------------
    # Collection-level artifacts
    # ---------------------------------------------------------------------------------------------
    # GeoJSON export
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

    # Markdown index
    md_index = [
        "# WDE Report Index",
        "",
        f"- Generated: {_utc_iso()}",
        f"- Coordinates masked: **{do_mask}**",
        "",
        "| # | Site ID | Lat | Lon | Score | Verify | Unc. | MD | HTML | Manifest |",
        "|:-:|:-------:|:---:|:---:|:-----:|:-----:|:----:|:--:|:----:|:--------:|",
    ]
    for k, row in enumerate(md_index_rows, 1):
        md_index.append(
            f"| {k} | {row['id']} | {row['lat']} | {row['lon']} | {row['score']} | {row['verify_score']} | "
            f"{row['uncertainty']} | [md]({row['report_md']}) | [html]({row['report_html']}) | [json]({row['manifest']}) |"
        )
    md_index_path = out_dir / "report_index.md"
    _write_text(md_index_path, "\n".join(md_index))

    # JSON index (machine-consumable) — list of dossiers + manifests
    dossiers_index = {
        "generated_at": _utc_iso(),
        "reports_dir": str(reports_dir.relative_to(out_dir)),
        "dossiers": md_index_rows,
    }
    dossiers_index_path = out_dir / "dossiers_index.json"
    _write_json(dossiers_index_path, dossiers_index)

    # Manifest index (site_id → manifest path)
    manifest_index_path = out_dir / "manifest_index.json"
    _write_json(manifest_index_path, {"sites": per_site_manifest_map})

    # Summary JSON
    summary = {
        "stage": "report",
        "generated_at": _utc_iso(),
        "pipeline_version": pipeline_version,
        "config_path": str(config_path),
        "config_hash": config_hash,
        "num_candidates": len(candidates),
        "reports_dir": str(reports_dir),
        "index_csv": str(csv_path),
        "index_md": str(md_index_path),
        "dossiers_index": str(dossiers_index_path),
        "manifest_index": str(manifest_index_path),
        "geojson": str(geojson_path),
        "coordinates_masked": bool(do_mask),
        "files_hashes": {
            "report_index.csv": _sha256_file(csv_path),
            "report_index.md": _sha256_file(md_index_path),
            "verify_candidates.geojson": _sha256_file(geojson_path),
            "dossiers_index.json": _sha256_file(dossiers_index_path),
            "manifest_index.json": _sha256_file(manifest_index_path),
        },
    }
    summary_path = out_dir / "report_summary.json"
    _write_json(summary_path, summary)

    # Log & return
    log.info(
        f"Report complete: {len(candidates)} dossiers → {reports_dir} | "
        f"indexes: {csv_path.name}, {md_index_path.name}, {dossiers_index_path.name} | geojson: {geojson_path.name}"
    )
    return summary
