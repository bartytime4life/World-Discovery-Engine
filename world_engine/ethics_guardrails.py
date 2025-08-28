# FILE: world_engine/ethics_guardrails.py
# -------------------------------------------------------------------------------------------------
"""
World Discovery Engine (WDE) — Ethics Guardrails

Executable safeguards aligned with ETHICS.md:
- Coordinate masking (rounding) in public artifacts (default: 2 decimals ≈ ~1 km)
- Indigenous land sovereignty checks (requires a boundary dataset; optional)
- Sovereignty banners for dossiers when overlaps are detected
- Dataset license validation (simple allowlist)
- Enforced "ethical mode" (default ON; cannot be silently disabled)

This module is imported and invoked by world_engine/report.py and exposed via the CLI (wde report).

"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import geopandas as gpd  # Optional; used for sovereignty overlap checks
except ImportError:
    gpd = None  # Sovereignty checks will be skipped if geopandas is unavailable

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("wde.ethics")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Global configuration (runtime-tunable via set_ethical_mode / set_mask_precision)
# -----------------------------------------------------------------------------
ETHICAL_MODE = True     # Default ON
MASK_PRECISION = 2      # Coordinates rounded to 2 decimals by default (~0.01°)

VALID_LICENSES = {
    "esa copernicus": "open",
    "usgs landsat": "public domain",
    "planet nicfi": "non-commercial",
    "isric soilgrids": "cc-by",
    "hydrosheds": "open",
    "modis/nasa": "public domain",
}


def set_ethical_mode(enabled: bool) -> None:
    """
    Explicitly set ethical mode at runtime.
    NOTE: Disabling must be explicit via CLI; never silently disable within code paths.
    """
    global ETHICAL_MODE
    ETHICAL_MODE = bool(enabled)
    state = "ENABLED" if ETHICAL_MODE else "DISABLED"
    logger.warning(f"[WDE][ETHICS] ETHICAL_MODE {state}")


def set_mask_precision(decimals: int) -> None:
    """Adjust coordinate masking precision (>= 0)."""
    global MASK_PRECISION
    MASK_PRECISION = max(0, int(decimals))
    logger.info(f"[WDE][ETHICS] MASK_PRECISION set to {MASK_PRECISION} decimals")


# -----------------------------------------------------------------------------
# Coordinate Masking
# -----------------------------------------------------------------------------
def mask_coordinates(coords: List[Tuple[float, float]], precision: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Round coordinates to coarse precision for public release.

    Args:
        coords: list of (lat, lon) tuples
        precision: override global MASK_PRECISION

    Returns:
        Masked coordinates list
    """
    if not coords:
        return []
    prec = MASK_PRECISION if precision is None else precision
    return [(round(float(lat), prec), round(float(lon), prec)) for (lat, lon) in coords]


# -----------------------------------------------------------------------------
# Sovereignty Checks
# -----------------------------------------------------------------------------
def check_indigenous_overlap(
    candidate_coords: Tuple[float, float],
    indigenous_bounds_path: Optional[Path] = None,
) -> bool:
    """
    Return True if candidate site lies within Indigenous territory polygons.

    Args:
        candidate_coords: (lat, lon) in EPSG:4326
        indigenous_bounds_path: path to GeoJSON/GeoPackage/Shapefile

    Returns:
        bool
    """
    if gpd is None:
        logger.warning("[WDE][ETHICS] geopandas not available — skipping Indigenous overlap check.")
        return False
    if indigenous_bounds_path is None or not Path(indigenous_bounds_path).exists():
        logger.warning("[WDE][ETHICS] Indigenous territory dataset not provided — skipping overlap check.")
        return False

    lat, lon = float(candidate_coords[0]), float(candidate_coords[1])
    pt = gpd.GeoSeries.from_xy([lon], [lat], crs="EPSG:4326")

    territories = gpd.read_file(indigenous_bounds_path)
    if territories.crs is None:
        territories.set_crs("EPSG:4326", inplace=True)
    else:
        territories = territories.to_crs("EPSG:4326")

    contains = territories.contains(pt[0])
    return bool(contains.any())


def sovereignty_banner(overlap: bool) -> str:
    """Return a sovereignty banner string when overlap is detected; empty string otherwise."""
    if overlap:
        return (
            "⚠ Candidate overlaps Indigenous territory. "
            "Engage communities and competent authorities before further action."
        )
    return ""


# -----------------------------------------------------------------------------
# License Validation
# -----------------------------------------------------------------------------
def validate_dataset_license(dataset_meta: Dict[str, Any]) -> bool:
    """
    Minimal allowlist check for dataset license compliance.

    Args:
        dataset_meta: dict with at least fields {"name": str, "license": str}

    Returns:
        True if dataset appears compliant, False otherwise.
    """
    name = (dataset_meta.get("name") or "").strip().lower()
    lic = (dataset_meta.get("license") or "").strip().lower()

    for k, v in VALID_LICENSES.items():
        if k in name and v in lic:
            return True

    logger.error(f"[WDE][ETHICS] Non-compliant or unknown dataset license: {dataset_meta}")
    return False


# -----------------------------------------------------------------------------
# Dossier Injection
# -----------------------------------------------------------------------------
def inject_ethics_into_dossier(
    dossier: Dict[str, Any],
    candidate_coords: Tuple[float, float],
    indigenous_bounds_path: Optional[Path] = None,
    public: bool = True,
) -> Dict[str, Any]:
    """
    Enforce ethics guardrails directly in a candidate dossier dictionary.

    Args:
        dossier: mutable dict representing the candidate report
        candidate_coords: (lat, lon)
        indigenous_bounds_path: optional path for overlap checks
        public: whether the dossier is intended for public output

    Returns:
        The updated dossier dict.
    """
    # Always record exact coordinates internally for audit (not necessarily emitted)
    dossier["coordinates_exact"] = (float(candidate_coords[0]), float(candidate_coords[1]))

    if ETHICAL_MODE:
        # Masked/public coordinates
        if public:
            dossier["coordinates_public"] = mask_coordinates([candidate_coords])[0]

        # Sovereignty overlap check
        overlap = check_indigenous_overlap(candidate_coords, indigenous_bounds_path)
        banner = sovereignty_banner(overlap)
        if banner:
            notes = dossier.setdefault("ethics_notes", [])
            notes.append(banner)

        # Ensure uncertainty + provenance scaffolds exist
        dossier.setdefault("uncertainty", "Provide model probability + interval; include SSIM counterfactuals.")
        dossier.setdefault("provenance", []).append("All sources & transforms logged per ETHICS.md.")
    else:
        logger.warning("[WDE][ETHICS] ETHICAL_MODE is DISABLED. Skipping masking/sovereignty checks.")

    return dossier


# -----------------------------------------------------------------------------
# Simple CLI test
# -----------------------------------------------------------------------------
def _demo() -> None:
    c = (-3.4567, -60.4321)
    d = {"site_id": "demo001"}
    out = inject_ethics_into_dossier(d, c, indigenous_bounds_path=None, public=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _demo()


# FILE: world_engine/report.py
# -------------------------------------------------------------------------------------------------
"""
World Discovery Engine (WDE) — Report Generation

Reads verified candidates and emits per-site dossiers with ethics guardrails applied by default.

Inputs:
  - verified candidates file (GeoJSON/JSON) from 'verify' stage
  - optional Indigenous territory dataset (GeoJSON/GPKG/SHP) for sovereignty checks

Outputs:
  - one JSON dossier per candidate under the output directory
  - an index file 'dossiers_index.json' summarizing all generated dossiers

This module is wired to the CLI (wde report) and integrates ethics_guardrails.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional: geopandas for GeoJSON parsing convenience (fallback to stdlib if missing)
try:
    import geopandas as gpd  # type: ignore
except ImportError:
    gpd = None  # We'll handle pure-JSON fallback

from . import ethics_guardrails as eg

logger = logging.getLogger("wde.report")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_verified(verified_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load verified candidates from a GeoJSON FeatureCollection or JSON list/dict.

    Expected schemas:
      - GeoJSON FeatureCollection: features[*].properties + geometry.coordinates
      - JSON list of dicts: each has 'site_id' and 'coordinates' or similar

    Returns:
      List of candidate dicts with minimal fields: {"site_id": str, "coordinates": (lat, lon), ...}
    """
    path = Path(verified_path)
    if not path.exists():
        raise FileNotFoundError(f"Verified candidates not found: {path}")

    # Try GeoJSON path via geopandas first
    if gpd is not None and path.suffix.lower() in (".geojson", ".gpkg", ".shp", ".json"):
        try:
            gdf = gpd.read_file(path)
            if gdf.empty:
                return []
            out: List[Dict[str, Any]] = []
            for i, row in gdf.iterrows():
                props = dict(row)
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                # Use centroid for non-point geometries
                pt = geom if geom.geom_type == "Point" else geom.centroid
                lat, lon = float(pt.y), float(pt.x)
                site_id = str(props.get("site_id", f"site_{i:04d}"))
                out.append({"site_id": site_id, "coordinates": (lat, lon), "properties": props})
            return out
        except Exception as e:
            logger.warning(f"[WDE][REPORT] geopandas read failed on {path}: {e}; falling back to raw JSON")

    # Fallback raw JSON parsing
    data = json.loads(path.read_text(encoding="utf-8"))
    candidates: List[Dict[str, Any]] = []

    # If dict with 'features' (GeoJSON-like)
    if isinstance(data, dict) and "features" in data:
        for i, feat in enumerate(data.get("features", []) or []):
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            coords = geom.get("coordinates")
            site_id = str(props.get("site_id", f"site_{i:04d}"))
            if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                # Assume [lon, lat] for GeoJSON
                lon, lat = float(coords[0]), float(coords[1])
                candidates.append({"site_id": site_id, "coordinates": (lat, lon), "properties": props})
    # If already a list of dicts
    elif isinstance(data, list):
        for i, rec in enumerate(data):
            latlon = rec.get("coordinates")
            if latlon and isinstance(latlon, (list, tuple)) and len(latlon) == 2:
                lat, lon = float(latlon[0]), float(latlon[1])
            else:
                lat, lon = float(rec.get("lat", 0.0)), float(rec.get("lon", 0.0))
            site_id = str(rec.get("site_id", f"site_{i:04d}"))
            candidates.append({"site_id": site_id, "coordinates": (lat, lon), "properties": rec})
    else:
        logger.error(f"[WDE][REPORT] Unsupported verified file format: {type(data)}")

    return candidates


def _build_base_dossier(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct a minimal dossier scaffold from a verified record.
    Real implementations should attach plots/figures/metrics here.
    """
    site_id = rec.get("site_id", "site_unk")
    lat, lon = rec.get("coordinates", (None, None))
    props = rec.get("properties", {})

    dossier: Dict[str, Any] = {
        "site_id": site_id,
        "coordinates_exact": (lat, lon),
        "evidence": {
            # Placeholders; upstream stages should populate these
            "modalities": props.get("modalities", ["optical", "radar", "dem"]),
            "summary": props.get("summary", "Multi-modal evidence fused in verification stage."),
        },
        "uncertainty": props.get("uncertainty", None),  # Will be filled by ethics guardrails if missing
        "provenance": props.get("provenance", []),
    }
    return dossier


def generate_dossiers(
    verified_path: Union[str, Path],
    out_dir: Union[str, Path],
    *,
    indigenous_bounds: Optional[Union[str, Path]] = None,
    public: bool = True,
    ethical_mode: bool = True,
    mask_precision: Optional[int] = None,
) -> Path:
    """
    Generate per-candidate dossiers with ethics guardrails applied.

    Args:
        verified_path: path to verified candidates (GeoJSON/JSON)
        out_dir: output directory for dossiers
        indigenous_bounds: path to Indigenous territory boundaries dataset (optional)
        public: whether outputs are for public release (controls coordinate masking)
        ethical_mode: enforce ethics guardrails (default True)
        mask_precision: override coordinate rounding decimals

    Returns:
        Path to the dossiers index file (dossiers_index.json)
    """
    eg.set_ethical_mode(ethical_mode)
    if mask_precision is not None:
        eg.set_mask_precision(mask_precision)

    out = _ensure_dir(out_dir)
    candidates = _load_verified(verified_path)
    if not candidates:
        logger.warning("[WDE][REPORT] No verified candidates found; exiting with empty index.")
        index_path = out / "dossiers_index.json"
        index_path.write_text(json.dumps({"dossiers": []}, indent=2), encoding="utf-8")
        return index_path

    bounds_path = Path(indigenous_bounds) if indigenous_bounds else None

    dossiers_index: List[Dict[str, Any]] = []

    for rec in candidates:
        dossier = _build_base_dossier(rec)

        lat, lon = rec["coordinates"]
        dossier = eg.inject_ethics_into_dossier(
            dossier,
            (lat, lon),
            indigenous_bounds_path=bounds_path,
            public=public,
        )

        # Save per-site dossier
        site_id = dossier.get("site_id", "site_unk")
        dossier_path = out / f"{site_id}.dossier.json"
        dossier_path.write_text(json.dumps(dossier, indent=2), encoding="utf-8")

        dossiers_index.append(
            {
                "site_id": site_id,
                "path": str(dossier_path),
                "coordinates_public": dossier.get("coordinates_public"),
                "sovereignty_flag": "ethics_notes" in dossier and any("overlaps Indigenous" in n for n in dossier.get("ethics_notes", [])),
            }
        )

    index_path = out / "dossiers_index.json"
    index_path.write_text(json.dumps({"dossiers": dossiers_index}, indent=2), encoding="utf-8")
    logger.info(f"[WDE][REPORT] Wrote {len(dossiers_index)} dossiers → {index_path}")
    return index_path


# FILE: world_engine/cli.py
# -------------------------------------------------------------------------------------------------
"""
World Discovery Engine (WDE) — Typer CLI

Subcommands (skeleton):
  selftest    : quick sanity checks
  ingest      : tiling & ingestion
  scan        : coarse anomaly detection
  evaluate    : mid-scale evaluation
  verify      : multi-modal fusion & verification
  report      : dossier generation (includes ethics guardrails)

This file wires ethics enforcement into 'report' by default.
"""

from __future__ import annotations
import sys
from pathlib import Path
import typer

from . import ethics_guardrails as eg
from . import report as wde_report

app = typer.Typer(add_completion=False, help="World Discovery Engine (WDE) CLI")


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _echo(msg: str) -> None:
    typer.echo(f"[WDE] {msg}")


# -----------------------------------------------------------------------------
# Stubs for other stages (optional placeholders)
# -----------------------------------------------------------------------------
@app.command()
def selftest() -> None:
    """Run basic environment and path checks."""
    _echo("Selftest OK (placeholder).")


@app.command()
def ingest(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config (YAML/JSON)"),
    out: Path = typer.Option("data/raw", "--out", help="Output directory"),
) -> None:
    _echo(f"Ingest with {config} → {out} (stub).")


@app.command()
def scan(
    config: Path = typer.Option(..., "--config", "-c"),
    _in: Path = typer.Option("data/raw", "--in"),
    out: Path = typer.Option("artifacts/candidates", "--out"),
) -> None:
    _echo(f"Scan {_in} → {out} (stub).")


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c"),
    _in: Path = typer.Option("artifacts/candidates", "--in"),
    out: Path = typer.Option("artifacts/evaluated", "--out"),
) -> None:
    _echo(f"Evaluate {_in} → {out} (stub).")


@app.command()
def verify(
    config: Path = typer.Option(..., "--config", "-c"),
    _in: Path = typer.Option("artifacts/evaluated", "--in"),
    out: Path = typer.Option("artifacts/verified", "--out"),
) -> None:
    _echo(f"Verify {_in} → {out} (stub).")


# -----------------------------------------------------------------------------
# REPORT (wired with ethics)
# -----------------------------------------------------------------------------
@app.command()
def report(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config (unused here; for parity)"),
    _in: Path = typer.Option("artifacts/verified/verified_candidates.geojson", "--in", help="Verified candidates file"),
    out: Path = typer.Option("artifacts/dossiers", "--out", help="Output directory for dossiers"),
    indigenous_bounds: Path = typer.Option(
        None,
        "--indigenous-bounds",
        help="Path to Indigenous territory dataset (GeoJSON/GPKG/SHP) for sovereignty checks",
    ),
    public: bool = typer.Option(True, "--public/--private", help="Apply coordinate masking for public outputs"),
    ethical_mode: bool = typer.Option(True, "--ethical-mode/--no-ethical-mode", help="Enforce ethics guardrails"),
    mask_precision: int = typer.Option(2, "--mask-precision", help="Coordinate rounding decimals for public outputs"),
) -> None:
    """
    Generate candidate dossiers with ethics guardrails applied by default.
    This command is the endpoint used by the DVC 'report' stage (make run / dvc repro).
    """
    _echo("Generating dossiers with ethics guardrails...")
    eg.set_ethical_mode(ethical_mode)
    eg.set_mask_precision(mask_precision)

    index_path = wde_report.generate_dossiers(
        verified_path=_in,
        out_dir=out,
        indigenous_bounds=indigenous_bounds,
        public=public,
        ethical_mode=ethical_mode,
        mask_precision=mask_precision,
    )
    _echo(f"Dossiers index: {index_path}")


def main() -> None:
    try:
        app()
    except Exception as e:
        _echo(f"ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()