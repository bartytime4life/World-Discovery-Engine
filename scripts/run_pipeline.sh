#!/usr/bin/env bash
# ==============================================================================
# 🌍 World Discovery Engine (WDE) — Pipeline Runner
# scripts/run_pipeline.sh
#
# Purpose:
#   One-command entrypoint to execute the full WDE discovery pipeline:
#   1. Ingestion (AOI tiling + data fetch)
#   2. Coarse anomaly detection
#   3. Mid-scale evaluation
#   4. Verification & evidence fusion
#   5. Candidate dossier generation
#
# Design:
#   - Config-driven: defaults from configs/default.yaml
#   - CLI-first: uses world_engine/cli.py (Typer app)
#   - Reproducible: logs run info + seeds
#   - Kaggle/CI/CD friendly
#
# Usage:
#   ./scripts/run_pipeline.sh [OPTIONS]
#
# Options:
#   --config <path>     Path to config file (default: configs/default.yaml)
#   --aoi <file>        Override AOI GeoJSON
#   --out <dir>         Output directory (default: outputs/)
#   --dry-run           Print commands but don’t execute
#   --help              Show this help
#
# Example:
#   ./scripts/run_pipeline.sh --config configs/amazon.yaml
# ==============================================================================

set -euo pipefail

# Defaults
CONFIG="configs/default.yaml"
OUTPUT_DIR="outputs/"
AOI_OVERRIDE=""
DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --aoi) AOI_OVERRIDE="--aoi $2"; shift 2 ;;
    --out) OUTPUT_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help)
      grep '^#' "$0" | cut -c 3-
      exit 0
      ;;
    *) echo "❌ Unknown option: $1"; exit 1 ;;
  esac
done

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"

# Run commands
run_cmd() {
  echo "▶ $*"
  if [[ $DRY_RUN -eq 0 ]]; then
    eval "$@"
  fi
}

# Log start
echo "=================================================="
echo "🌍 WDE Pipeline Run"
echo " Config:   $CONFIG"
echo " AOI:      ${AOI_OVERRIDE:-default in config}"
echo " Output:   $OUTPUT_DIR"
echo " Dry-run:  $DRY_RUN"
echo "=================================================="

# Step 1. Ingest
run_cmd "python -m world_engine.cli ingest --config $CONFIG $AOI_OVERRIDE --out $OUTPUT_DIR/ingested"

# Step 2. Detect
run_cmd "python -m world_engine.cli detect --config $CONFIG --in $OUTPUT_DIR/ingested --out $OUTPUT_DIR/detected"

# Step 3. Evaluate
run_cmd "python -m world_engine.cli evaluate --config $CONFIG --in $OUTPUT_DIR/detected --out $OUTPUT_DIR/evaluated"

# Step 4. Verify
run_cmd "python -m world_engine.cli verify --config $CONFIG --in $OUTPUT_DIR/evaluated --out $OUTPUT_DIR/verified"

# Step 5. Report
run_cmd "python -m world_engine.cli report --config $CONFIG --in $OUTPUT_DIR/verified --out $OUTPUT_DIR/reports"

# Completion
echo "=================================================="
echo "✅ WDE pipeline complete. Candidate dossiers saved to $OUTPUT_DIR/reports"
echo "=================================================="
