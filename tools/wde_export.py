#!/usr/bin/env python3
"""
WDE Export Utilities

This module provides small, dependency-light helpers to:
- generate a Kaggle-friendly submission CSV from a candidates GeoJSON/JSON,
- write a run manifest (listing outputs),
- create a zipped bundle of dossiers and artifacts.

Usage:
  python tools/wde_export.py submission --candidates outputs/verified_candidates.geojson --out outputs/submission.csv
  python tools/wde_export.py manifest   --outputs outputs/ --out outputs/run_manifest.json
  python tools/wde_export.py bundle     --outputs outputs/ --zip outputs/bundle.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import zipfile


def _load_candidates(path: Path) -> List[Dict[str, Any]]:
    """
    Load candidate features from GeoJSON/JSON.
    Expected schema:
      - GeoJSON FeatureCollection with Point/Polygon geometry and properties
        OR
      - List[dict] where dicts have at least: id, lat, lon (or geometry), score/confidence/net_evidence

    Returns a normalized list of dicts with fields used by export.
    """
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    feats: List[Dict[str, Any]] = []
    # GeoJSON FeatureCollection
    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        for feat in data.get("features", []):
            props = feat.get("properties", {}) or {}
            geom = feat.get("geometry", {}) or {}
            lat, lon = None, None
            if geom.get("type") == "Point":
                coords = geom.get("coordinates", [None, None])
                lon, lat = coords[0], coords[1]
            # basic polygon centroid if needed
            if (lat is None or lon is None) and geom.get("type") == "Polygon":
                try:
                    ring = geom.get("coordinates", [])[0]
                    xs = [p[0] for p in ring]
                    ys = [p[1] for p in ring]
                    lon = sum(xs) / len(xs)
                    lat = sum(ys) / len(ys)
                except Exception:
                    pass
            feats.append({
                "id": props.get("id") or props.get("site_id") or f"cand_{len(feats)+1}",
                "lat": lat,
                "lon": lon,
                "score": props.get("score"),
                "confidence": props.get("confidence"),
                "notes": props.get("notes") or props.get("narrative") or "",
            })
    elif isinstance(data, list):
        for i, row in enumerate(data):
            feats.append({
                "id": row.get("id") or f"cand_{i+1}",
                "lat": row.get("lat"),
                "lon": row.get("lon"),
                "score": row.get("score"),
                "confidence": row.get("confidence"),
                "notes": row.get("notes") or "",
            })
    else:
        raise ValueError("Unsupported candidates format; provide GeoJSON FeatureCollection or a JSON list")

    # sanitize
    out = []
    for f in feats:
        lat = None if f["lat"] is None else round(float(f["lat"]), 6)
        lon = None if f["lon"] is None else round(float(f["lon"]), 6)
        out.append({**f, "lat": lat, "lon": lon})
    return out


def export_submission_csv(candidates_path: Path, out_csv: Path) -> None:
    feats = _load_candidates(candidates_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "lat", "lon", "score", "confidence", "notes"])
        for row in feats:
            writer.writerow([row["id"], row["lat"], row["lon"], row["score"], row["confidence"], row["notes"]])
    print(f"[export] Wrote submission CSV â {out_csv}")


def write_manifest(outputs_dir: Path, out_json: Path) -> None:
    outputs_dir = outputs_dir.resolve()
    artifacts = []
    for p in outputs_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(outputs_dir).as_posix()
            stat = p.stat()
            artifacts.append({
                "path": rel,
                "size_bytes": stat.st_size,
                "modified": stat.st_mtime,
            })
    manifest = {
        "generated": time.time(),
        "outputs_dir": str(outputs_dir),
        "artifacts": artifacts,
        "tool": "wde_export.py",
        "version": "1.0.0",
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[export] Wrote manifest JSON â {out_json}")


def zip_outputs(outputs_dir: Path, zip_path: Path) -> None:
    outputs_dir = outputs_dir.resolve()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in outputs_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(outputs_dir))
    print(f"[export] Wrote bundle ZIP â {zip_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WDE Export utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sub = sub.add_parser("submission", help="Create submission.csv from a candidates file")
    p_sub.add_argument("--candidates", required=True, help="Path to candidates GeoJSON/JSON")
    p_sub.add_argument("--out", required=True, help="Output CSV path")

    p_man = sub.add_parser("manifest", help="Write a run manifest for an outputs directory")
    p_man.add_argument("--outputs", required=True, help="Outputs directory path")
    p_man.add_argument("--out", required=True, help="Manifest JSON path")

    p_zip = sub.add_parser("bundle", help="Zip the outputs directory into a single bundle")
    p_zip.add_argument("--outputs", required=True, help="Outputs directory path")
    p_zip.add_argument("--zip", required=True, help="Zip file path")

    args = parser.parse_args(argv)

    if args.cmd == "submission":
        export_submission_csv(Path(args.candidates), Path(args.out))
    elif args.cmd == "manifest":
        write_manifest(Path(args.outputs), Path(args.out))
    elif args.cmd == "bundle":
        zip_outputs(Path(args.outputs), Path(args.zip))
    else:
        parser.print_help()
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())