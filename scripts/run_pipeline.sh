#!/usr/bin/env bash
# ==============================================================================
# 🌍 World Discovery Engine (WDE) — Full Pipeline Runner
# scripts/run_pipeline.sh
#
# Purpose:
#   Orchestrate the WDE pipeline end-to-end in a **config-driven** and
#   **reproducible** way with clear logs and an output manifest:
#     1) ingest   → tile & fetch/prepare inputs
#     2) detect   → coarse anomaly scan (CV/VLM/DEM)
#     3) evaluate → NDVI/terrain/historical overlays
#     4) verify   → ADE fingerprints, causal graph, uncertainty
#     5) report   → candidate dossiers (MD/HTML/PDF if configured)
#   Optional:
#     - post-validate artifacts
#     - build Kaggle bundle or reports bundle
#
# Key features:
#   - Works locally or on Kaggle; auto-picks output dir for Kaggle kernels
#   - Deterministic seeds; logs to artifacts/<run>/logs
#   - Writes a fallback run_manifest.json (if CLI didn’t)
#   - Pass-through of extra args to `wde` after `--`
#
# Examples:
#   # Default run with config
#   scripts/run_pipeline.sh --config ./configs/default.yaml
#
#   # Skip report stage, run validation, custom OUT_DIR
#   OUT_DIR=./artifacts/demo scripts/run_pipeline.sh --config ./configs/default.yaml --no-report --validate
#
#   # Only run detect→verify (assume ingest done), pass extra CLI args
#   scripts/run_pipeline.sh --config ./configs/default.yaml --stages detect,evaluate,verify -- --aoi data/aoi/brazil.geojson
#
# Notes:
#   - Requires `wde` CLI on PATH (Typer-based, usually via `pip install -e .`).
#   - This script is conservative: it fails fast on hard errors; warnings do not halt.
# ==============================================================================

set -Eeuo pipefail

# ------------------------ helpers ------------------------
ts()  { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "${LOG_DIR}/pipeline.log"; }
die() { log "ERROR: $*"; exit 1; }
ensure() { mkdir -p "$1"; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# ------------------------ defaults ------------------------
CONFIG="${WDE_CONFIG:-./configs/default.yaml}"
RUN_ID="$(date -u +'%Y%m%dT%H%M%SZ')"
# If on Kaggle, prefer working dir for persistence
if [[ -d "/kaggle/working" ]]; then
  OUT_DIR="${OUT_DIR:-/kaggle/working/outputs/${RUN_ID}}"
else
  OUT_DIR="${OUT_DIR:-./artifacts/${RUN_ID}}"
fi
LOG_DIR="${OUT_DIR}/logs"
SEED="${WDE_SEED:-42}"

# Control which stages run
STAGES="ingest,detect,evaluate,verify,report"
DO_VALIDATE=0
DO_REPORT_BUNDLE=0
DO_KAGGLE_BUNDLE=0
DRY_RUN=0

# Extra args passed to `wde` after --
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- <extra args passed to wde>]

Options:
  --config PATH            Pipeline config YAML (default: ${CONFIG})
  --out DIR                Output directory (default: ${OUT_DIR})
  --seed N                 Random seed (default: ${SEED})
  --stages s1,s2,...       Subset of stages to run (default: ${STAGES})
                           Allowed: ingest,detect,evaluate,verify,report
  --no-report              Shortcut for --stages ingest,detect,evaluate,verify
  --validate               Run scripts/validate_artifacts.py on OUT_DIR
  --bundle-reports         Zip reports into <OUT_DIR>/reports_bundle.zip
  --bundle-kaggle          Build Kaggle bundle zip in ./artifacts/kaggle_bundle.zip
  --dry-run                Print actions only

Examples:
  $0 --config ./configs/kaggle.yaml
  OUT_DIR=./artifacts/demo $0 --no-report
  $0 --stages detect,verify -- --aoi data/aoi/brazil.geojson

Notes:
  - Ensure 'wde' CLI is available (install project in editable mode).
  - This script writes a fallback run_manifest.json if the CLI didn’t.
EOF
}

# ------------------------ parse args ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --no-report) STAGES="ingest,detect,evaluate,verify"; shift;;
    --validate) DO_VALIDATE=1; shift;;
    --bundle-reports) DO_REPORT_BUNDLE=1; shift;;
    --bundle-kaggle) DO_KAGGLE_BUNDLE=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA_ARGS+=("$@"); break;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

# ------------------------ preflight ------------------------
need wde
[[ -f "$CONFIG" ]] || die "Config not found: $CONFIG"

ensure "$OUT_DIR"
ensure "$LOG_DIR"

export WDE_CONFIG="$CONFIG"
export OUT_DIR
export WDE_SEED="$SEED"
export RUN_STARTED="$(ts)"

log "WDE Pipeline Runner"
log "Config     : $CONFIG"
log "OUT_DIR    : $OUT_DIR"
log "Seed       : $SEED"
log "Stages     : $STAGES"
log "Extra args : ${EXTRA_ARGS[*]:-<none>}"

_run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY-RUN $*"
  else
    eval "$*"
  fi
}

# Split stages into array
IFS=',' read -r -a STAGE_ARR <<< "$STAGES"

# ------------------------ stage runner ------------------------
run_stage() {
  local name="$1"
  shift || true
  log "▶ Stage: ${name}"
  set -x
  _run "wde ${name} --config '$CONFIG' --out '$OUT_DIR' ${EXTRA_ARGS[*]}" 2>&1 | tee -a "${LOG_DIR}/${name}.log"
  set +x
  log "✔ Stage complete: ${name}"
}

allowed="ingest detect evaluate verify report"
for s in "${STAGE_ARR[@]}"; do
  [[ " $allowed " == *" $s "* ]] || die "Unknown stage: '$s' (allowed: $allowed)"
done

# Execute stages in given order
for s in "${STAGE_ARR[@]}"; do
  run_stage "$s"
done

# ------------------------ post: write fallback manifest ------------------------
# If CLI already wrote a manifest we keep it; otherwise write a minimal one.
if ! ls "$OUT_DIR"/**/run_manifest.json >/dev/null 2>&1; then
  MANIFEST="${OUT_DIR}/run_manifest.json"
  python - <<'PY' > "$MANIFEST"
import os, sys, time, json, hashlib, glob
out=os.environ.get("OUT_DIR",".")
cfg=os.environ.get("WDE_CONFIG","")
seed=os.environ.get("WDE_SEED","42")
def sha256(p):
  h=hashlib.sha256()
  with open(p,'rb') as f:
    for ch in iter(lambda:f.read(8192), b''):
      h.update(ch)
  return h.hexdigest()
arts=[]
for p in glob.glob(os.path.join(out,"**","*"), recursive=True):
  if os.path.isfile(p):
    try:
      arts.append({"path":os.path.relpath(p,out),"bytes":os.path.getsize(p),"sha256":sha256(p)})
    except Exception:
      pass
manifest={
  "run_started_utc": os.environ.get("RUN_STARTED", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
  "run_finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "config_path": cfg,
  "seed": int(seed),
  "out_dir": out,
  "artifacts": arts,
}
json.dump(manifest, sys.stdout, indent=2)
PY
  log "Wrote fallback manifest: $MANIFEST"
fi

# ------------------------ optional: validate artifacts ------------------------
if [[ $DO_VALIDATE -eq 1 ]]; then
  if [[ -f "scripts/validate_artifacts.py" ]]; then
    log "Running artifact validation…"
    set -x
    _run "python scripts/validate_artifacts.py '${OUT_DIR}'" | tee -a "${LOG_DIR}/validate.log"
    set +x
  else
    log "Validator not found (scripts/validate_artifacts.py) — skipping."
  fi
fi

# ------------------------ optional: bundle reports ----------------------------
if [[ $DO_REPORT_BUNDLE -eq 1 ]]; then
  if [[ -f "scripts/generate_reports.sh" ]]; then
    log "Bundling reports…"
    set -x
    _run "scripts/generate_reports.sh --root '${OUT_DIR}' --zip"
    set +x
  else
    log "Report bundler not found (scripts/generate_reports.sh) — skipping."
  fi
fi

# ------------------------ optional: Kaggle bundle -----------------------------
if [[ $DO_KAGGLE_BUNDLE -eq 1 ]]; then
  if [[ -f "scripts/export_kaggle.sh" ]]; then
    log "Building Kaggle bundle…"
    set -x
    _run "scripts/export_kaggle.sh"
    set +x
  else
    log "Kaggle bundler not found (scripts/export_kaggle.sh) — skipping."
  fi
fi

log "Run complete."
