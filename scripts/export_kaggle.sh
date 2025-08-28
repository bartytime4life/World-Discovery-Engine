#!/usr/bin/env bash
# ==============================================================================
# 🌍 World Discovery Engine (WDE) — Kaggle Exporter
# scripts/export_kaggle.sh
#
# Purpose:
#   Build a Kaggle-ready bundle for the WDE notebook and (optionally) publish it
#   as a Kaggle Dataset and/or push it as a Kaggle Notebook (Kernel).
#
# What this does:
#   1) Creates a clean export folder (./artifacts/kaggle/<slug>/)
#   2) Copies the main notebook + minimal files (README, LICENSE, configs)
#   3) Generates Kaggle metadata:
#        - dataset-metadata.json      (for kaggle datasets create/version)
#        - kernel-metadata.json       (for kaggle kernels push)
#   4) Optionally calls Kaggle CLI to create/version Dataset and push Kernel
#
# Requirements:
#   - bash, jq, zip
#   - Kaggle CLI installed and authenticated (one-time):
#       • pip install kaggle
#       • mkdir -p ~/.kaggle && echo '{"username":"<USER>","key":"<KEY>"}' > ~/.kaggle/kaggle.json
#       • chmod 600 ~/.kaggle/kaggle.json
#
# Usage:
#   scripts/export_kaggle.sh \
#     --owner your_kaggle_username \
#     --name wde-ade-discovery \
#     --title "World Discovery Engine — ADE Discovery Pipeline" \
#     --notebook notebooks/ade_discovery_pipeline.ipynb \
#     --datasets "owner/dataset1,other_owner/dataset2" \
#     --gpu --internet \
#     --make-zip --make-dataset --make-kernel
#
# Common flags:
#   --owner <user>         Kaggle owner (required for dataset id)
#   --name <slug>          URL-safe slug (required for export dir & dataset id)
#   --title <text>         Human title (defaults to slug)
#   --notebook <path>      Notebook to export (default: notebooks/ade_discovery_pipeline.ipynb)
#   --readme <path>        README.md to include (default: README.md if present)
#   --license <path>       LICENSE to include (default: LICENSE if present)
#   --include <paths>      Extra comma-separated paths to include (e.g., configs/default.yaml)
#   --datasets <csv>       Comma-separated list of Kaggle dataset sources to attach to kernel
#   --gpu                  Enable GPU in kernel-metadata.json
#   --internet             Enable Internet in kernel-metadata.json
#   --private              Make kernel private (dataset is private by default until published)
#   --make-zip             Create a zip bundle under artifacts/
#   --make-dataset         Run `kaggle datasets create` or `version` for the export folder
#   --make-kernel          Run `kaggle kernels push` for the export folder
#   --outdir <dir>         Base export dir (default: artifacts/kaggle)
#   --dry-run              Print actions only
#
# Notes:
#   - Keep bundle small: the kernel pulls data via Kaggle "Add data" or via datasets listed.
#   - For reproducibility, prefer pinning requirements or using a lightweight notebook environment.
# ==============================================================================

set -euo pipefail

# --------- helpers ---------
log()  { echo -e "[$(date +%H:%M:%S)] $*"; }
warn() { echo -e "[$(date +%H:%M:%S)] \033[33mWARN:\033[0m $*"; }
die()  { echo -e "[$(date +%H:%M:%S)] \033[31mERROR:\033[0m $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "DRY-RUN $*"; else eval "$*"; fi; }

# --------- defaults ---------
OWNER=""
SLUG=""
TITLE=""
NB_PATH="notebooks/ade_discovery_pipeline.ipynb"
README_PATH=""
LICENSE_PATH=""
INCLUDE_EXTRA=""
KAGGLE_DATASETS=""
ENABLE_GPU=0
ENABLE_INTERNET=0
PRIVATE_KERNEL=0
MAKE_ZIP=0
MAKE_DS=0
MAKE_KERNEL=0
OUT_BASE="artifacts/kaggle"
DRY_RUN=0

# --------- parse args ---------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --owner)      OWNER="$2"; shift 2 ;;
    --name)       SLUG="$2"; shift 2 ;;
    --title)      TITLE="$2"; shift 2 ;;
    --notebook)   NB_PATH="$2"; shift 2 ;;
    --readme)     README_PATH="$2"; shift 2 ;;
    --license)    LICENSE_PATH="$2"; shift 2 ;;
    --include)    INCLUDE_EXTRA="$2"; shift 2 ;;
    --datasets)   KAGGLE_DATASETS="$2"; shift 2 ;;
    --gpu)        ENABLE_GPU=1; shift ;;
    --internet)   ENABLE_INTERNET=1; shift ;;
    --private)    PRIVATE_KERNEL=1; shift ;;
    --make-zip)   MAKE_ZIP=1; shift ;;
    --make-dataset|--make-ds) MAKE_DS=1; shift ;;
    --make-kernel) MAKE_KERNEL=1; shift ;;
    --outdir)     OUT_BASE="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

# --------- sanity checks ---------
need jq
need zip
[[ -f "$NB_PATH" ]] || die "Notebook not found: $NB_PATH"
[[ -n "$OWNER" ]] || die "--owner is required (Kaggle username)"
[[ -n "$SLUG"  ]] || die "--name is required (URL-safe slug)"

TITLE="${TITLE:-$SLUG}"
EXPORT_DIR="${OUT_BASE%/}/${SLUG}"
mkdir -p "$EXPORT_DIR"

# Defaults if present
[[ -z "$README_PATH"  && -f "README.md" ]] && README_PATH="README.md"
[[ -z "$LICENSE_PATH" && -f "LICENSE"   ]] && LICENSE_PATH="LICENSE"

# --------- collect files ---------
log "Preparing export folder: $EXPORT_DIR"
run "mkdir -p '$EXPORT_DIR'"

# Notebook
log "Copy → $NB_PATH"
run "cp -f '$NB_PATH' '$EXPORT_DIR/notebook.ipynb'"

# README / LICENSE if present
if [[ -n "$README_PATH" && -f "$README_PATH" ]]; then
  log "Copy → $README_PATH"
  run "cp -f '$README_PATH' '$EXPORT_DIR/README.md'"
fi
if [[ -n "$LICENSE_PATH" && -f "$LICENSE_PATH" ]]; then
  log "Copy → $LICENSE_PATH"
  run "cp -f '$LICENSE_PATH' '$EXPORT_DIR/LICENSE'"
fi

# Optional include list (comma-separated)
if [[ -n "$INCLUDE_EXTRA" ]]; then
  IFS=',' read -r -a EXTRA <<< "$INCLUDE_EXTRA"
  for p in "${EXTRA[@]}"; do
    if [[ -e "$p" ]]; then
      dst="$EXPORT_DIR/$(basename "$p")"
      log "Copy → $p"
      run "cp -rf '$p' '$dst'"
    else
      warn "Include path not found: $p"
    fi
  done
fi

# --------- dataset-metadata.json (for Kaggle Dataset) ---------
DS_META="$EXPORT_DIR/dataset-metadata.json"
log "Write dataset-metadata.json"
# Kaggle expects: { "title": "...", "id": "owner/slug", "licenses": [ {"name":"CC0-1.0"} ], ... }
run "jq -n \
  --arg title \"$TITLE\" \
  --arg id    \"$OWNER/$SLUG\" \
  '{
     title: \$title,
     id: \$id,
     licenses: [{name: \"CC0-1.0\"}],
     subtitle: \"World Discovery Engine export\",
     description: \"Bundle for the WDE ADE Discovery Pipeline notebook. Includes notebook and minimal config files.\"
   }' > '$DS_META'"

# --------- kernel-metadata.json (for Kaggle Kernel) ---------
KMETA="$EXPORT_DIR/kernel-metadata.json"
log "Write kernel-metadata.json"
# Fields:
#   id: "owner/slug", title, code_file, language, kernel_type, is_private,
#   enable_gpu, enable_internet, dataset_sources, competition_sources, kernel_sources
GPU=$( [[ $ENABLE_GPU -eq 1 ]] && echo true || echo false )
INET=$( [[ $ENABLE_INTERNET -eq 1 ]] && echo true || echo false )
PRIV=$( [[ $PRIVATE_KERNEL -eq 1 ]] && echo true || echo false )

# Parse dataset sources into JSON array
DATASET_ARRAY="[]"
if [[ -n "$KAGGLE_DATASETS" ]]; then
  # split by comma into array of "owner/dataset"
  IFS=',' read -r -a DS <<< "$KAGGLE_DATASETS"
  DATASET_ARRAY=$(printf '%s\n' "${DS[@]}" | jq -R . | jq -s .)
fi

run "jq -n \
  --arg id \"$OWNER/$SLUG\" \
  --arg title \"$TITLE\" \
  --argjson enable_gpu $GPU \
  --argjson enable_internet $INET \
  --argjson is_private $PRIV \
  --argjson dataset_sources '$DATASET_ARRAY' \
  '{
     id: \$id,
     title: \$title,
     code_file: \"notebook.ipynb\",
     language: \"python\",
     kernel_type: \"notebook\",
     is_private: \$is_private,
     enable_gpu: \$enable_gpu,
     enable_internet: \$enable_internet,
     dataset_sources: \$dataset_sources,
     competition_sources: [],
     kernel_sources: []
   }' > '$KMETA'"

# --------- zip bundle (optional) ---------
if [[ $MAKE_ZIP -eq 1 ]]; then
  ZIP_OUT="${EXPORT_DIR}.zip"
  log "Create zip → $ZIP_OUT"
  ( cd "$(dirname "$EXPORT_DIR")" && run "zip -r -q '$(basename "$ZIP_OUT")' '$(basename "$EXPORT_DIR")'" )
fi

# --------- Kaggle CLI actions (optional) ---------
if [[ $MAKE_DS -eq 1 || $MAKE_KERNEL -eq 1 ]]; then
  if ! command -v kaggle >/dev/null 2>&1; then
    die "Kaggle CLI not found. Install with: pip install kaggle"
  fi
fi

if [[ $MAKE_DS -eq 1 ]]; then
  log "Kaggle Dataset: owner=$OWNER id=$OWNER/$SLUG"
  # Try create; if exists, version instead.
  if kaggle datasets status "$OWNER/$SLUG" >/dev/null 2>&1; then
    log "Dataset exists → versioning"
    run "kaggle datasets version -p '$EXPORT_DIR' -m 'Update $(date -u +%Y-%m-%dT%H:%M:%SZ)'"
  else
    log "Creating dataset"
    run "kaggle datasets create -p '$EXPORT_DIR' --dir-mode zip"
  fi
fi

if [[ $MAKE_KERNEL -eq 1 ]]; then
  log "Kaggle Kernel push: id=$OWNER/$SLUG"
  # kernel-metadata.json must be in folder root; Kaggle will create or update
  run "kaggle kernels push -p '$EXPORT_DIR'"
fi

log "✅ Kaggle export prepared at: $EXPORT_DIR"
if [[ $MAKE_ZIP -eq 1 ]]; then
  log "📦 Zip bundle: ${EXPORT_DIR}.zip"
fi
