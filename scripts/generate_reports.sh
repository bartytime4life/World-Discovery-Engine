#!/usr/bin/env bash
# ==============================================================================
# 🌍 World Discovery Engine (WDE) — Report Generator
# scripts/generate_reports.sh
#
# Purpose:
#   Turn verified candidates into polished, shareable reports by:
#     1) Calling the WDE "report" stage (optional if already run)
#     2) Rendering Markdown → HTML/PDF (if markdown exists and tools available)
#     3) Building an index page (HTML + JSON) linking all site dossiers
#     4) Masking/rounding sensitive coordinates for public bundles (optional)
#     5) Zipping a distributable "reports_bundle.zip"
#
# Inputs (defaults assume WDE outputs directory layout):
#   --root <dir>             Root artifacts dir (default: outputs)
#   --reports-dir <dir>      Reports dir (default: <root>/reports)
#   --verify-json <file>     Candidates GeoJSON (default: <root>/verify_candidates.geojson)
#   --manifest-index <file>  Per-site manifest index (default: <root>/manifest_index.json)
#   --config <file>          Pipeline config YAML (for running CLI report)
#
# Rendering flags:
#   --run-report             Invoke `python -m world_engine.cli report --config ...`
#   --no-render              Do not convert MD → HTML/PDF (skip rendering)
#   --html                   Render HTML (default: on if MD exists and pandoc present)
#   --pdf                    Render PDF  (default: on if MD exists and pandoc + wkhtmltopdf or LaTeX present)
#
# Ethics & masking (applied on index.html and generated JSON):
#   --mask                   Mask coords (coarse rounding) for the public index
#   --round-decimals <N>     Decimal places for coords when masking (default: 2)
#
# Packaging / utility:
#   --zip                    Create reports_bundle.zip in <root>
#   --open                   Try to open the generated index.html
#   --dry-run                Print actions only
#
# Requirements:
#   - bash, jq
#   - (optional) python + world_engine installed for --run-report
#   - (optional) pandoc (for MD → HTML/PDF), wkhtmltopdf or LaTeX for PDFs
#
# Examples:
#   # Basic: render existing reports, build index, zip bundle
#   scripts/generate_reports.sh --root outputs --zip
#
#   # Run report stage first, render HTML/PDF, mask coords for public share
#   scripts/generate_reports.sh --run-report --config configs/default.yaml --mask --zip
#
# Notes:
#   - This script is conservative: it won’t fail if a renderer is missing; it will skip with a warning.
#   - If report stage already produced candidate_*.html/pdf, conversion is skipped for that file.
# ==============================================================================

set -euo pipefail

# ------------------------ helpers ------------------------
log()  { echo -e "[$(date +%H:%M:%S)] $*"; }
warn() { echo -e "[$(date +%H:%M:%S)] \033[33mWARN:\033[0m $*"; }
die()  { echo -e "[$(date +%H:%M:%S)] \033[31mERROR:\033[0m $*"; exit 1; }
run()  { if [[ $DRY_RUN -eq 1 ]]; then echo "DRY-RUN $*"; else eval "$*"; fi; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }

# ------------------------ defaults ------------------------
ROOT="outputs"
REPORTS_DIR=""            # default computed as ROOT/reports
VERIFY_JSON=""            # default computed as ROOT/verify_candidates.geojson
MANIFEST_INDEX=""         # default computed as ROOT/manifest_index.json
CONFIG=""
RUN_REPORT=0
RENDER=1
RENDER_HTML=1
RENDER_PDF=1
MASK=0
ROUND_DEC=2
DO_ZIP=0
OPEN=0
DRY_RUN=0

# ------------------------ parse args ------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)            ROOT="$2"; shift 2 ;;
    --reports-dir)     REPORTS_DIR="$2"; shift 2 ;;
    --verify-json)     VERIFY_JSON="$2"; shift 2 ;;
    --manifest-index)  MANIFEST_INDEX="$2"; shift 2 ;;
    --config)          CONFIG="$2"; shift 2 ;;

    --run-report)      RUN_REPORT=1; shift ;;
    --no-render)       RENDER=0; shift ;;
    --html)            RENDER_HTML=1; shift ;;
    --pdf)             RENDER_PDF=1; shift ;;

    --mask)            MASK=1; shift ;;
    --round-decimals)  ROUND_DEC="$2"; shift 2 ;;

    --zip)             DO_ZIP=1; shift ;;
    --open)            OPEN=1; shift ;;
    --dry-run)         DRY_RUN=1; shift ;;

    -h|--help)
      sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

need jq

ROOT="${ROOT%/}"
[[ -n "$REPORTS_DIR"   ]] || REPORTS_DIR="$ROOT/reports"
[[ -n "$VERIFY_JSON"   ]] || VERIFY_JSON="$ROOT/verify_candidates.geojson"
[[ -n "$MANIFEST_INDEX" ]] || MANIFEST_INDEX="$ROOT/manifest_index.json"

mkdir -p "$REPORTS_DIR"

log "Artifacts root .......: $ROOT"
log "Reports directory ....: $REPORTS_DIR"
log "Verify GeoJSON .......: $VERIFY_JSON"
log "Manifest index ........: $MANIFEST_INDEX"
log "Run report stage ......: $RUN_REPORT"
log "Render MD→HTML/PDF ....: $RENDER (HTML=$RENDER_HTML, PDF=$RENDER_PDF)"
log "Mask coordinates ......: $MASK (round=$ROUND_DEC)"
log "Zip bundle ............: $DO_ZIP"

# ------------------------ 1) Run report stage (optional) ------------------------
if [[ $RUN_REPORT -eq 1 ]]; then
  [[ -n "$CONFIG" ]] || die "--config is required with --run-report"
  if ! command -v python >/dev/null 2>&1; then
    die "Python not found for running world_engine CLI"
  fi
  log "Running WDE CLI report stage via config: $CONFIG"
  run "python -m world_engine.cli report --config '$CONFIG'"
fi

# ------------------------ 2) Collect sites ------------------------
SITES_JSON="$REPORTS_DIR/_sites.json"

collect_sites() {
  # Preferred: manifest_index.json produced by CLI/report
  if [[ -f "$MANIFEST_INDEX" ]]; then
    jq -r '
      .sites as $m
      | [ keys[] as $k | {site_id:$k, manifest: $m[$k]} ]
    ' "$MANIFEST_INDEX" > "$SITES_JSON" || true
    if [[ -s "$SITES_JSON" ]]; then
      log "Collected sites from manifest_index.json"
      return 0
    fi
  fi

  # Fallback: derive from verify_candidates.geojson
  if [[ -f "$VERIFY_JSON" ]]; then
    jq -r '
      .features // [] | to_entries
      | [ .[] | {
            site_id: ( .value.properties.site_id // .value.properties.tile_id // ( ( .key + 1 | tostring ) | lpad(3; "0") ) ),
            center:  ( if (.value.geometry.type=="Point") then {lat:(.value.geometry.coordinates[1]), lon:(.value.geometry.coordinates[0])} else null end )
          }
        ]
    ' "$VERIFY_JSON" > "$SITES_JSON" || true
    if [[ -s "$SITES_JSON" ]]; then
      log "Collected sites from verify_candidates.geojson"
      return 0
    fi
  fi

  warn "No sites discovered (no manifest_index.json and no verify_candidates.geojson)."
  echo "[]" > "$SITES_JSON"
  return 0
}

collect_sites

SITE_COUNT=$(jq 'length' "$SITES_JSON")
log "Sites discovered: $SITE_COUNT"

# ------------------------ 3) Render MD → HTML/PDF ------------------------
have_pandoc=0
if command -v pandoc >/dev/null 2>&1; then have_pandoc=1; fi

render_md() {
  local md="$1"
  local base="${md%.md}"

  # HTML
  if [[ $RENDER_HTML -eq 1 && ! -f "${base}.html" ]]; then
    if [[ $have_pandoc -eq 1 && $RENDER -eq 1 ]]; then
      log "Render HTML: $(basename "$md")"
      run "pandoc '$md' -s -o '${base}.html'"
    else
      warn "Skipping HTML render (pandoc missing or --no-render)"
    fi
  fi

  # PDF
  if [[ $RENDER_PDF -eq 1 && ! -f "${base}.pdf" ]]; then
    if [[ $have_pandoc -eq 1 && $RENDER -eq 1 ]]; then
      # Try wkhtmltopdf or LaTeX-based PDF; pandoc will pick best available
      log "Render PDF: $(basename "$md")"
      run "pandoc '$md' -s -o '${base}.pdf'" || warn "PDF render failed for $md"
    else
      warn "Skipping PDF render (pandoc missing or --no-render)"
    fi
  fi
}

if [[ $RENDER -eq 1 ]]; then
  # For each site, if candidate_<id>.md exists, render to HTML and PDF if missing
  for row in $(jq -rc '.[]' "$SITES_JSON"); do
    SITE_ID=$(jq -r '.site_id' <<<"$row")
    MD="$REPORTS_DIR/candidate_${SITE_ID}.md"
    if [[ -f "$MD" ]]; then
      render_md "$MD"
    fi
  done
fi

# ------------------------ 4) Build index.html + public JSON ------------------------
INDEX_HTML="$REPORTS_DIR/index.html"
PUBLIC_JSON="$REPORTS_DIR/_public_index.json"

mask_coord() {
  local val="$1" dec="$2"
  # Round or return empty
  python - "$val" "$dec" <<'PY'
import sys, math
v = sys.argv[1]
d = int(sys.argv[2])
try:
    x = float(v)
    print(f"{round(x, d):.{d}f}")
except:
    print("")
PY
}

build_public_json() {
  local round_dec="$1"
  jq --argjson m "$MASK" --argjson d "$round_dec" -rc '
    . as $sites
    | [ $sites[] | {
        site_id,
        files: {
          html:  ("candidate_" + .site_id + ".html"),
          pdf:   ("candidate_" + .site_id + ".pdf"),
          md:    ("candidate_" + .site_id + ".md"),
          manifest: ("candidate_" + .site_id + "_manifest.json")
        }
      } ]
  ' "$SITES_JSON" > "$PUBLIC_JSON.tmp"

  # If masking requested and we have verify_candidates.geojson, enrich with rounded coords
  if [[ $MASK -eq 1 && -f "$VERIFY_JSON" ]]; then
    # Build a map site_id -> (lat,lon rounded)
    TMP_MAP="$REPORTS_DIR/_coords_map.json"
    jq -rc '
      .features // [] | to_entries
      | [ .[] | {
            site_id: ( .value.properties.site_id // .value.properties.tile_id // ( ( .key + 1 | tostring ) | lpad(3; "0") ) ),
            lat: (if .value.geometry.type=="Point" then .value.geometry.coordinates[1] else null end),
            lon: (if .value.geometry.type=="Point" then .value.geometry.coordinates[0] else null end)
        } ]
    ' "$VERIFY_JSON" > "$TMP_MAP"

    # Round numerically via jq+awk fallback (use shell function for portability)
    # Merge into PUBLIC_JSON.tmp
    python - "$PUBLIC_JSON.tmp" "$TMP_MAP" "$ROUND_DEC" "$PUBLIC_JSON" <<'PY'
import json, sys, math
src, coords, dec, outp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(src,'r') as f: items=json.load(f)
with open(coords,'r') as f: rows=json.load(f)
m={}
for r in rows:
    sid=r.get("site_id")
    if sid is None: continue
    lat=r.get("lat"); lon=r.get("lon")
    if isinstance(lat,(int,float)) and isinstance(lon,(int,float)):
        m[str(sid)]={"lat":round(lat,dec),"lon":round(lon,dec)}
for it in items:
    sid=it.get("site_id")
    if sid in m:
        it["coordinates"]=m[sid]
with open(outp,'w') as f: json.dump(items,f,indent=2)
PY
  else
    mv "$PUBLIC_JSON.tmp" "$PUBLIC_JSON"
  fi
}

build_index_html() {
  local title="WDE Candidate Dossiers"
  {
    echo "<!DOCTYPE html><html><head><meta charset='utf-8'/>"
    echo "<title>${title}</title>"
    echo "<style>body{font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 16px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;}th{background:#f5f5f5;text-align:left;}a{color:#0a58ca;text-decoration:none;}small{color:#555;}</style>"
    echo "</head><body>"
    echo "<h1>${title}</h1>"
    echo "<p>Generated: $(date -u +"%Y-%m-%d %H:%M:%SZ")</p>"
    if [[ $MASK -eq 1 ]]; then
      echo "<p><strong>Notice:</strong> Coordinates masked to ${ROUND_DEC} decimals for public sharing.</p>"
    fi
    echo "<table><thead><tr><th>Site</th><th>Coordinates</th><th>Artifacts</th></tr></thead><tbody>"
    # rows from PUBLIC_JSON
    jq -rc '.[]' "$PUBLIC_JSON" | while read -r row; do
      sid=$(jq -r '.site_id' <<<"$row")
      lat=$(jq -r '.coordinates.lat // empty' <<<"$row")
      lon=$(jq -r '.coordinates.lon // empty' <<<"$row")
      fhtml=$(jq -r '.files.html' <<<"$row")
      fpdf=$(jq -r '.files.pdf' <<<"$row")
      fmd=$(jq -r '.files.md' <<<"$row")
      fman=$(jq -r '.files.manifest' <<<"$row")
      echo "<tr><td><strong>${sid}</strong></td>"
      if [[ -n "$lat" && -n "$lon" ]]; then
        echo "<td>lat=${lat}, lon=${lon}</td>"
      else
        echo "<td><small>n/a</small></td>"
      fi
      echo "<td>"
      [[ -f "$REPORTS_DIR/$fhtml" ]] && echo "<a href='./${fhtml}'>HTML</a> "
      [[ -f "$REPORTS_DIR/$fpdf"  ]] && echo "<a href='./${fpdf}'>PDF</a> "
      [[ -f "$REPORTS_DIR/$fmd"   ]] && echo "<a href='./${fmd}'>MD</a> "
      [[ -f "$REPORTS_DIR/$fman"  ]] && echo "<a href='./${fman}'>manifest</a> "
      echo "</td></tr>"
    done
    echo "</tbody></table>"
    echo "</body></html>"
  } > "$INDEX_HTML"
}

build_public_json "$ROUND_DEC"
build_index_html
log "Index built → $INDEX_HTML"

# ------------------------ 5) Zip bundle (optional) ------------------------
if [[ $DO_ZIP -eq 1 ]]; then
  if command -v zip >/dev/null 2>&1; then
    ZIP_PATH="$ROOT/reports_bundle.zip"
    ( cd "$ROOT" && run "zip -r -q '$(basename "$ZIP_PATH")' '$(basename "$REPORTS_DIR")'" )
    log "Bundle created → $ZIP_PATH"
  else
    warn "zip not found — skipping bundle"
  fi
fi

# ------------------------ 6) Open index (optional) ------------------------
if [[ $OPEN -eq 1 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    run "xdg-open '$INDEX_HTML'"
  elif command -v open >/dev/null 2>&1; then
    run "open '$INDEX_HTML'"
  else
    warn "No system opener found (xdg-open/open). Open manually: $INDEX_HTML"
  fi
fi

log "✅ Report generation complete."
