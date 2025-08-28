# FILE: scripts/validate_artifacts.py
# =============================================================================
# 🌍 World Discovery Engine (WDE) — Artifact Validator
#
# Purpose
# -------
# Validate a WDE pipeline run’s artifacts for structure, consistency, and
# minimal completeness. Produces a machine-readable validation report and
# non-zero exit code on hard failures (useful for CI).
#
# What it checks (minimum viable rules)
# -------------------------------------
# 1) outputs/run_manifest.json exists and has key fields
# 2) Stage directories referenced in run_manifest artifacts exist
# 3) verify_candidates.geojson (if present) is a valid FeatureCollection
# 4) Reports directory exists (if present) and per-site manifests are sane
# 5) manifest_index.json consistency (if present)
# 6) Basic size/count stats & summary
#
# Output
# ------
# - Writes JSON report: <root>/validation_report.json
# - Prints a human summary to stdout
# - Exit codes:
#     0  = OK (no errors)
#     1  = OK with warnings (no errors but had warnings)
#     2  = FAILED (at least one error)
#
# Usage
# -----
#   python scripts/validate_artifacts.py \
#       --root outputs \
#       --strict
#
# Tips
# ----
# - Keep it stdlib-only for portability (no extra deps).
# - This does not enforce domain scoring or scientific quality — only structure.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


# ----------------------------- Data structures -------------------------------

@dataclass
class Issue:
    level: str       # "ERROR" | "WARN" | "INFO"
    message: str
    path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Summary:
    ok: bool
    errors: int
    warnings: int
    infos: int
    files_count: int
    total_size_bytes: int


@dataclass
class ValidationReport:
    root: str
    generated_at: str
    summary: Summary
    issues: List[Issue]
    snapshots: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "generated_at": self.generated_at,
            "summary": asdict(self.summary),
            "issues": [i.to_dict() for i in self.issues],
            "snapshots": self.snapshots,
        }


# ----------------------------- Utility helpers -------------------------------

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_feature_collection(obj: Dict[str, Any]) -> bool:
    return isinstance(obj, dict) and obj.get("type") == "FeatureCollection" and isinstance(obj.get("features"), list)


def sizeof_path(path: Path) -> int:
    if path.is_file():
        try:
            return path.stat().st_size
        except Exception:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return total


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(1 for _ in path.rglob("*") if _.is_file())


def record_issue(issues: List[Issue], level: str, msg: str, path: Optional[Path] = None) -> None:
    issues.append(Issue(level=level, message=msg, path=str(path) if path else None))


def norm_rel(root: Path, p: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(p)


# ------------------------------ Core validator -------------------------------

class ArtifactValidator:
    def __init__(self, root: Path, strict: bool = False) -> None:
        self.root = root
        self.strict = strict
        self.issues: List[Issue] = []
        self.snapshots: Dict[str, Any] = {}

    # ---- Rule 1: run_manifest.json exists & basic fields
    def check_run_manifest(self) -> Dict[str, Any]:
        rm = self.root / "run_manifest.json"
        if not rm.exists():
            record_issue(self.issues, "ERROR", "Missing run_manifest.json", rm)
            return {}
        data = read_json(rm)
        if data is None:
            record_issue(self.issues, "ERROR", "Invalid JSON in run_manifest.json", rm)
            return {}
        required_top = ["pipeline_version", "timestamp_utc", "artifacts"]
        for k in required_top:
            if k not in data:
                record_issue(self.issues, "ERROR", f"run_manifest.json missing '{k}'", rm)
        # Snapshot
        self.snapshots["run_manifest"] = {
            "pipeline_version": data.get("pipeline_version"),
            "timestamp_utc": data.get("timestamp_utc"),
            "config_hash": data.get("config_hash"),
            "artifacts_keys": list((data.get("artifacts") or {}).keys()),
        }
        return data

    # ---- Rule 2: referenced stage dirs exist
    def check_stage_dirs(self, manifest: Dict[str, Any]) -> Tuple[Optional[Path], Dict[str, Path]]:
        artifacts = manifest.get("artifacts") or {}
        report_dir: Optional[Path] = None
        stage_dirs: Dict[str, Path] = {}
        # Ingest
        if "ingest" in artifacts:
            ing_root = (artifacts["ingest"] or {}).get("ingest_dir")
            if ing_root:
                p = Path(ing_root)
                if not p.is_absolute():
                    p = self.root / p
                stage_dirs["ingest_dir"] = p
                if not p.exists():
                    record_issue(self.issues, "ERROR", "Ingest directory missing", p)
        # Detect
        if "detect" in artifacts:
            det_root = (artifacts["detect"] or {}).get("detect_dir")
            if det_root:
                p = Path(det_root)
                if not p.is_absolute():
                    p = self.root / p
                stage_dirs["detect_dir"] = p
                if not p.exists():
                    record_issue(self.issues, "ERROR", "Detect directory missing", p)
        # Evaluate
        if "evaluate" in artifacts:
            ev_root = (artifacts["evaluate"] or {}).get("evaluate_dir")
            if ev_root:
                p = Path(ev_root)
                if not p.is_absolute():
                    p = self.root / p
                stage_dirs["evaluate_dir"] = p
                if not p.exists():
                    record_issue(self.issues, "ERROR", "Evaluate directory missing", p)
        # Verify
        if "verify" in artifacts:
            ver_root = (artifacts["verify"] or {}).get("verify_dir")
            if ver_root:
                p = Path(ver_root)
                if not p.is_absolute():
                    p = self.root / p
                stage_dirs["verify_dir"] = p
                if not p.exists():
                    record_issue(self.issues, "ERROR", "Verify directory missing", p)
        # Report
        if "report" in artifacts:
            rep_root = (artifacts["report"] or {}).get("reports_dir")
            if rep_root:
                p = Path(rep_root)
                if not p.is_absolute():
                    p = self.root / p
                report_dir = p
                stage_dirs["reports_dir"] = p
                if not p.exists():
                    record_issue(self.issues, "ERROR", "Reports directory missing", p)

        # Snapshot
        self.snapshots["stage_dirs"] = {k: norm_rel(self.root, v) for k, v in stage_dirs.items()}
        return report_dir, stage_dirs

    # ---- Rule 3: verify_candidates.geojson validity
    def check_verify_candidates(self) -> Dict[str, Any]:
        path = self.root / "verify_candidates.geojson"
        if not path.exists():
            record_issue(self.issues, "WARN", "verify_candidates.geojson not found (no candidates?)", path)
            return {}
        data = read_json(path)
        if data is None or not is_feature_collection(data):
            record_issue(self.issues, "ERROR", "Invalid or malformed verify_candidates.geojson", path)
            return {}
        feats = data.get("features", [])
        # minimal geometry sanity
        for idx, feat in enumerate(feats, 1):
            if not isinstance(feat, dict) or feat.get("type") != "Feature":
                record_issue(self.issues, "ERROR", f"Feature[{idx}] not a valid GeoJSON Feature", path)
                continue
            geom = feat.get("geometry")
            if not isinstance(geom, dict) or "type" not in geom:
                record_issue(self.issues, "ERROR", f"Feature[{idx}] missing geometry", path)
            props = feat.get("properties")
            if not isinstance(props, dict):
                record_issue(self.issues, "WARN", f"Feature[{idx}] missing properties object", path)
        # Snapshot
        self.snapshots["verify_candidates"] = {"count": len(feats)}
        return data

    # ---- Rule 4: reports + per-site manifests
    def check_reports_and_manifests(
        self,
        report_dir: Optional[Path],
        verify_fc: Dict[str, Any],
        manifest: Dict[str, Any],
    ) -> None:
        if report_dir is None:
            record_issue(self.issues, "WARN", "reports_dir not referenced in run_manifest artifacts")
            return
        # manifest_index.json (optional)
        idx_path = self.root / "manifest_index.json"
        if idx_path.exists():
            idx = read_json(idx_path)
            if not isinstance(idx, dict) or "sites" not in idx:
                record_issue(self.issues, "ERROR", "manifest_index.json malformed", idx_path)
            else:
                # quick pass
                sites = idx.get("sites") or {}
                if not isinstance(sites, dict):
                    record_issue(self.issues, "ERROR", "manifest_index.sites must be an object map", idx_path)
                self.snapshots["manifest_index"] = {"site_count": len(sites)}

        # from run_manifest
        per_site_map = ((manifest.get("artifacts") or {}).get("report") or {}).get("per_site_manifests")

        # If we have verify candidates, attempt to validate expected report files
        feats = verify_fc.get("features", []) if is_feature_collection(verify_fc) else []
        if feats and report_dir.exists():
            missing = 0
            checked = 0
            for i, feat in enumerate(feats, 1):
                props = feat.get("properties") or {}
                site_id = props.get("site_id") or props.get("tile_id") or f"{i:03d}"
                # Expect at least one of the report payloads; be lenient:
                exp = [
                    report_dir / f"candidate_{site_id}.html",
                    report_dir / f"candidate_{site_id}.md",
                    report_dir / f"candidate_{site_id}.pdf",
                    report_dir / f"candidate_{site_id}_manifest.json",
                ]
                if not any(p.exists() for p in exp):
                    record_issue(
                        self.issues,
                        "WARN",
                        f"Missing rendered report and manifest for site_id={site_id}",
                        report_dir,
                    )
                    missing += 1
                checked += 1
            self.snapshots["reports_check"] = {"sites_checked": checked, "missing": missing}

        # If per_site_manifests is present, sanity check referenced files
        if isinstance(per_site_map, dict):
            bad = 0
            for sid, relp in per_site_map.items():
                p = (self.root / relp) if not os.path.isabs(relp) else Path(relp)
                if not p.exists():
                    record_issue(self.issues, "WARN", f"per_site manifest missing for site_id={sid}", p)
                    bad += 1
            self.snapshots["per_site_manifests_map"] = {"sites": len(per_site_map), "missing": bad}

    # ---- Rule 5: size/count stats
    def stats_snapshot(self, stage_dirs: Dict[str, Path]) -> None:
        files = count_files(self.root)
        size = sizeof_path(self.root)
        self.snapshots["root_stats"] = {"files": files, "total_bytes": size}
        # Per stage
        per = {}
        for k, p in stage_dirs.items():
            per[k] = {"files": count_files(p), "total_bytes": sizeof_path(p)}
        self.snapshots["stage_stats"] = per

    # ---- Run all checks
    def run(self) -> ValidationReport:
        manifest = self.check_run_manifest()
        report_dir, stage_dirs = self.check_stage_dirs(manifest)
        verify_fc = self.check_verify_candidates()
        self.check_reports_and_manifests(report_dir, verify_fc, manifest)
        self.stats_snapshot(stage_dirs)

        # Summarize
        errs = sum(1 for i in self.issues if i.level == "ERROR")
        warns = sum(1 for i in self.issues if i.level == "WARN")
        infos = sum(1 for i in self.issues if i.level == "INFO")

        total_files = self.snapshots.get("root_stats", {}).get("files", 0)
        total_size = self.snapshots.get("root_stats", {}).get("total_bytes", 0)

        ok = errs == 0
        if self.strict and warns > 0:
            # In strict mode, treat warnings as failures.
            ok = False
            # escalate all warnings to errors for output clarity
            for i in self.issues:
                if i.level == "WARN":
                    i.level = "ERROR"

        report = ValidationReport(
            root=str(self.root),
            generated_at=datetime.utcnow().isoformat() + "Z",
            summary=Summary(
                ok=ok,
                errors=sum(1 for i in self.issues if i.level == "ERROR"),
                warnings=sum(1 for i in self.issues if i.level == "WARN"),
                infos=sum(1 for i in self.issues if i.level == "INFO"),
                files_count=int(total_files),
                total_size_bytes=int(total_size),
            ),
            issues=self.issues,
            snapshots=self.snapshots,
        )
        return report


# --------------------------------- CLI entry ----------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate WDE artifacts under a given root (default: outputs)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--root", type=str, default="outputs", help="Artifacts root directory")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures (exit 2)")
    ap.add_argument("--quiet", action="store_true", help="Reduce stdout (still writes JSON report)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    validator = ArtifactValidator(root=root, strict=bool(args.strict))
    report = validator.run()
    out_path = root / "validation_report.json"
    out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    # Human summary
    if not args.quiet:
        print("────────────────────────────────────────────────────────")
        print(f" WDE Artifact Validation  •  root: {root.as_posix()}")
        print("────────────────────────────────────────────────────────")
        print(f" OK ............: {report.summary.ok}")
        print(f" Errors ........: {report.summary.errors}")
        print(f" Warnings ......: {report.summary.warnings}")
        print(f" Files .........: {report.summary.files_count}")
        print(f" Total bytes ...: {report.summary.total_size_bytes}")
        print("────────────────────────────────────────────────────────")
        if report.issues:
            for i in report.issues:
                loc = f" [{i.path}]" if i.path else ""
                print(f" {i.level:5s} • {i.message}{loc}")
        else:
            print(" No issues found.")
        print(f"\nReport → {out_path.as_posix()}")

    # Exit code
    if report.summary.errors > 0:
        raise SystemExit(2)
    # In non-strict mode, warnings are acceptable (exit 0)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
