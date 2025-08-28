#!/usr/bin/env bash
# ==============================================================================
# 🌍 World Discovery Engine (WDE) — Dataset Fetcher
# scripts/fetch_datasets.sh
#
# Purpose:
#   One-command helper to pull small, AOI-scoped, OPEN datasets into ./data/raw
#   so the pipeline can run reproducibly (even offline later).
#
# What it can fetch (modular, opt-in flags):
#   - DEM (SRTM 30 m) via OpenTopography Global DEM API
#   - Hydro datasets (HydroRIVERS South America, optional)
#   - Sentinel-2 L2A COGs (visual/B04/B08) via AWS Earth-Search STAC
#   - Sentinel-1 GRD (first hit) via NASA ASF (requires Earthdata credentials)
#   - Landsat Collection 2 L2 (first hit) via USGS LandsatLook STAC
#   - SoilGrids WCS raster clip (optional; small AOI only)
#   - NICFI Planet mosaic quad (optional; requires PLANET API key)
#
# Requirements:
#   - bash, curl, jq, unzip
#   - (optional) OPENTOPO_API_KEY for OpenTopography GlobalDEM (can work without)
#   - (optional) EARTHDATA_USER / EARTHDATA_PASS for ASF Sentinel-1 downloads
#   - (optional) PLANET_API_KEY for NICFI mosaics
#
# Usage:
#   scripts/fetch_datasets.sh --aoi-bbox "minLon,minLat,maxLon,maxLat" [flags]
#
# Examples:
#   # Minimum (DEM + HydroRIVERS + Sentinel-2 visual preview)
#   scripts/fetch_datasets.sh --aoi-bbox "-60.50,-3.50,-60.40,-3.40" --dem --hydro --s2
#
#   # Everything possible (will skip items lacking tokens)
#   scripts/fetch_datasets.sh --aoi-bbox "-60.50,-3.50,-60.40,-3.40" --all
#
# Flags:
#   --dem           Fetch DEM (SRTMGL1 via OpenTopography)
#   --hydro         Fetch HydroRIVERS SA shapefile
#   --s2            Fetch a tiny Sentinel-2 L2A sample (visual/B04/B08 COGs)
#   --s1            Fetch a Sentinel-1 GRD product (first hit; requires Earthdata)
#   --landsat       Fetch a Landsat L2 scene (first hit)
#   --soilgrids     Fetch a SoilGrids WCS clip (small AOI only)
#   --nicfi         Fetch a Planet NICFI mosaic quad (requires PLANET_API_KEY)
#   --all           Try all of the above (skips items lacking creds)
#
#   --aoi-bbox  "<minLon,minLat,maxLon,maxLat>"   REQUIRED for most sources
#   --out        <dir>      Default: data/raw
#   --max-scenes <N>        Limit S2/Landsat scenes (default: 1)
#   --dry-run               Print intended actions only
#
# Notes:
#   - Choose a SMALL AOI (≈ 0.1° or less) to avoid large downloads and timeouts.
#   - This script is careful & transparent: it prints each URL and skips politely if
#     a dependency or token is missing.
# ==============================================================================

set -euo pipefail

# ---------------------------- Defaults & CLI parse ----------------------------

OUT_DIR="data/raw"
BBOX=""                      # minLon,minLat,maxLon,maxLat
DO_DEM=0
DO_HYDRO=0
DO_S2=0
DO_S1=0
DO_LANDSAT=0
DO_SOIL=0
DO_NICFI=0
MAX_SCENES=1
DRY_RUN=0

log()   { echo -e "[$(date +%H:%M:%S)] $*"; }
warn()  { echo -e "[$(date +%H:%M:%S)] \033[33mWARN:\033[0m $*"; }
err()   { echo -e "[$(date +%H:%M:%S)] \033[31mERROR:\033[0m $*"; }
die()   { err "$*"; exit 1; }
run()   { if [[ $DRY_RUN -eq 1 ]]; then echo "DRY-RUN $*"; else eval "$*"; fi; }

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    die "Missing required command: $1"
  fi
}

usage() {
  sed -n '1,80p' "$0" | sed -n '1,80p' | sed 's/^# \{0,1\}//' | sed 's/^$//'
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --aoi-bbox)   BBOX="$2"; shift 2 ;;
    --out)        OUT_DIR="$2"; shift 2 ;;
    --max-scenes) MAX_SCENES="$2"; shift 2 ;;
    --dem)        DO_DEM=1; shift ;;
    --hydro)      DO_HYDRO=1; shift ;;
    --s2)         DO_S2=1; shift ;;
    --s1)         DO_S1=1; shift ;;
    --landsat)    DO_LANDSAT=1; shift ;;
    --soilgrids)  DO_SOIL=1; shift ;;
    --nicfi)      DO_NICFI=1; shift ;;
    --all)        DO_DEM=1; DO_HYDRO=1; DO_S2=1; DO_S1=1; DO_LANDSAT=1; DO_SOIL=1; DO_NICFI=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage ;;
    *)            err "Unknown arg: $1"; usage ;;
  esac
done

[[ -z "$BBOX" ]] && die "Provide --aoi-bbox \"minLon,minLat,maxLon,maxLat\""

# deps
need curl
need jq
need unzip

mkdir -p "$OUT_DIR"
RAW_DEM="$OUT_DIR/dem"
RAW_HYDRO="$OUT_DIR/hydro"
RAW_S2="$OUT_DIR/sentinel2"
RAW_S1="$OUT_DIR/sentinel1"
RAW_LS="$OUT_DIR/landsat"
RAW_SOIL="$OUT_DIR/soilgrids"
RAW_NICFI="$OUT_DIR/nicfi"

mkdir -p "$RAW_DEM" "$RAW_HYDRO" "$RAW_S2" "$RAW_S1" "$RAW_LS" "$RAW_SOIL" "$RAW_NICFI"

# Parse bbox
IFS=',' read -r MINLON MINLAT MAXLON MAXLAT <<< "$BBOX" || true

# ------------------------------ Fetch functions ------------------------------

fetch_dem_srtm() {
  local west="$MINLON" south="$MINLAT" east="$MAXLON" north="$MAXLAT"
  local base="https://portal.opentopography.org/API/globaldem"
  local demtype="SRTMGL1"
  local out="$RAW_DEM/srtm_${west}_${south}_${east}_${north}.tif"
  local url="${base}?demtype=${demtype}&west=${west}&south=${south}&east=${east}&north=${north}&outputFormat=GTiff"
  # Optional API key
  if [[ -n "${OPENTOPO_API_KEY:-}" ]]; then
    url="${url}&API_Key=${OPENTOPO_API_KEY}"
  else
    warn "OPENTOPO_API_KEY not set — attempting anonymous DEM request"
  fi
  log "DEM: $url"
  run "curl -fSL --retry 3 --output '$out' '$url'" || warn "DEM fetch failed"
}

fetch_hydrorivers_sa() {
  # HydroRIVERS South America shapefile (stable public link)
  local zip="$RAW_HYDRO/HydroRIVERS_v10_sa_shp.zip"
  local url="https://www.hydrosheds.org/images/inpages/HydroRIVERS/HydroRIVERS_v10_sa_shp.zip"
  log "HydroRIVERS SA: $url"
  run "curl -fSL --retry 3 --output '$zip' '$url'" || warn "HydroRIVERS download failed"
  if [[ -f "$zip" ]]; then
    run "unzip -o '$zip' -d '$RAW_HYDRO/HydroRIVERS_SA'"
  fi
}

fetch_s2_cogs() {
  # Earth-Search STAC (AWS Element84) for sentinel-s2-l2a-cogs
  # We’ll request a single recent cloud-masked item intersecting our bbox and pull 3 assets.
  local stac="https://earth-search.aws.element84.com/v0/search"
  local payload
  payload=$(jq -n --arg bbox "$MINLON,$MINLAT,$MAXLON,$MAXLAT" '
    {
      "collections":["sentinel-s2-l2a-cogs"],
      "bbox":[
        ($bbox|split(",")[0]|tonumber),
        ($bbox|split(",")[1]|tonumber),
        ($bbox|split(",")[2]|tonumber),
        ($bbox|split(",")[3]|tonumber)
      ],
      "limit": 1,
      "sort":[{"field":"properties.datetime","direction":"desc"}],
      "query": { "eo:cloud_cover": {"lt": 20} }
    }')
  log "S2 STAC query → $stac"
  local resp="$RAW_S2/_stac_s2.json"
  run "curl -fSL -H 'Content-Type: application/json' -d '$payload' '$stac' -o '$resp'" || { warn "S2 STAC query failed"; return; }
  local hrefs=()
  # Try to extract common assets: visual, B04, B08
  mapfile -t hrefs < <(jq -r '
    .features[0].assets as $a
    | [ ($a.visual.href? // empty), ($a.B04.href? // empty), ($a.B08.href? // empty) ]
    | .[] | select(length>0)
  ' "$resp" || true)
  if [[ "${#hrefs[@]}" -eq 0 ]]; then
    warn "No S2 assets found for bbox"
    return
  fi
  local idx=0
  for h in "${hrefs[@]}"; do
    idx=$((idx+1))
    local name=$(basename "$h")
    log "S2 COG[$idx]: $h"
    run "curl -fSL --retry 3 -o '$RAW_S2/$name' '$h'" || warn "Failed: $h"
  done
}

fetch_s1_asf() {
  # Minimal: search & download first GRD with ASF; requires Earthdata credentials
  if [[ -z "${EARTHDATA_USER:-}" || -z "${EARTHDATA_PASS:-}" ]]; then
    warn "EARTHDATA_USER/EARTHDATA_PASS not set — skipping S1"
    return
  fi
  local poly="POLYGON((${MINLON} ${MINLAT}, ${MINLON} ${MAXLAT}, ${MAXLON} ${MAXLAT}, ${MAXLON} ${MINLAT}, ${MINLON} ${MINLAT}))"
  local base="https://api.daac.asf.alaska.edu/services/search/param"
  local url="${base}?platform=Sentinel-1&processingLevel=GRD&start=2023-01-01T00:00:00Z&end=2025-12-31T23:59:59Z&intersectsWith=$(python - <<PY
from urllib.parse import quote; print(quote(\"$poly\"))
PY
)"
  log "ASF S1 param search → $url"
  local json="$RAW_S1/_asf_s1.json"
  run "curl -fSL --user '${EARTHDATA_USER}:${EARTHDATA_PASS}' '$url' -o '$json'" || { warn "ASF search failed"; return; }
  local dl
  dl=$(jq -r '.[0].url // empty' "$json" 2>/dev/null || true)
  if [[ -z "$dl" ]]; then
    warn "No S1 results"
    return
  fi
  log "ASF S1 download → $dl"
  local zip="$RAW_S1/$(basename "$dl")"
  run "curl -fSL --user '${EARTHDATA_USER}:${EARTHDATA_PASS}' '$dl' -o '$zip'" || warn "S1 download failed"
}

fetch_landsat_ll() {
  # LandsatLook STAC server
  local stac="https://landsatlook.usgs.gov/stac-server/search"
  local payload
  payload=$(jq -n --arg bbox "$MINLON,$MINLAT,$MAXLON,$MAXLAT" '
    {
      "collections":["landsat-c2l2-sr"],
      "bbox":[
        ($bbox|split(",")[0]|tonumber),
        ($bbox|split(",")[1]|tonumber),
        ($bbox|split(",")[2]|tonumber),
        ($bbox|split(",")[3]|tonumber)
      ],
      "limit": 1,
      "sort":[{"field":"properties.datetime","direction":"desc"}]
    }')
  log "Landsat STAC query → $stac"
  local resp="$RAW_LS/_stac_ls.json"
  run "curl -fSL -H 'Content-Type: application/json' -d '$payload' '$stac' -o '$resp'" || { warn "Landsat STAC query failed"; return; }
  # pull a couple of assets if public (some assets may require signing)
  local hrefs=()
  mapfile -t hrefs < <(jq -r '
    .features[0].assets as $a
    | [ ($a.SR_B4.href? // empty), ($a.SR_B5.href? // empty), ($a.thumbnail.href? // empty) ]
    | .[] | select(length>0)
  ' "$resp" || true)
  if [[ "${#hrefs[@]}" -eq 0 ]]; then
    warn "No Landsat assets found for bbox"
    return
  fi
  local idx=0
  for h in "${hrefs[@]}"; do
    idx=$((idx+1))
    local name=$(basename "$h")
    log "Landsat asset[$idx]: $h"
    run "curl -fSL --retry 3 -o '$RAW_LS/$name' '$h'" || warn "Failed: $h"
  done
}

fetch_soilgrids_wcs() {
  # Small WCS clip; note: SoilGrids WCS can be rate limited; keep AOI tiny
  local srv="https://maps.isric.org/mapserv?map=/map/soilgrids.map"
  local cov="phh2o_0-5cm_mean"
  local out="$RAW_SOIL/soilgrids_${cov}_${MINLON}_${MINLAT}_${MAXLON}_${MAXLAT}.tif"
  local url="${srv}&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID=${cov}&subset=long(${MINLON},${MAXLON})&subset=lat(${MINLAT},${MAXLAT})&FORMAT=image/geotiff"
  log "SoilGrids WCS: $url"
  run "curl -fSL --retry 2 -o '$out' '$url'" || warn "SoilGrids WCS failed (AOI too big or rate-limited)"
}

fetch_nicfi_planet() {
  # Requires PLANET_API_KEY. Downloads first quad of latest NICFI mosaic intersecting bbox.
  if [[ -z "${PLANET_API_KEY:-}" ]]; then
    warn "PLANET_API_KEY not set — skipping NICFI"
    return
  fi
  local mosaics="https://api.planet.com/basemaps/v1/mosaics?name__contains=NICFI"
  log "NICFI list → $mosaics"
  local mos="$RAW_NICFI/_nicfi_mosaics.json"
  run "curl -fSL -u '${PLANET_API_KEY}:' '$mosaics' -o '$mos'" || { warn "NICFI list failed"; return; }
  local mosaic_id
  mosaic_id=$(jq -r '.mosaics[0].id // empty' "$mos" || true)
  if [[ -z "$mosaic_id" ]]; then
    warn "No NICFI mosaics visible to this key"
    return
  fi
  local quads="https://api.planet.com/basemaps/v1/mosaics/${mosaic_id}/quads?bbox=${MINLON},${MINLAT},${MAXLON},${MAXLAT}"
  log "NICFI quads → $quads"
  local qjson="$RAW_NICFI/_nicfi_quads.json"
  run "curl -fSL -u '${PLANET_API_KEY}:' '$quads' -o '$qjson'" || { warn "NICFI quads failed"; return; }
  local dl
  dl=$(jq -r '.items[0]._links.download // empty' "$qjson" || true)
  if [[ -z "$dl" ]]; then
    warn "No NICFI quad in bbox"
    return
  fi
  local out="$RAW_NICFI/$(basename "$dl" | sed 's/\?.*$//')"
  log "NICFI quad → $dl"
  run "curl -fSL -u '${PLANET_API_KEY}:' '$dl' -o '$out'" || warn "NICFI download failed"
}

# --------------------------------- Orchestrate --------------------------------

log "AOI bbox: $BBOX"
log "Output root: $OUT_DIR"

if [[ $DO_DEM -eq 1 ]];      then fetch_dem_srtm;        fi
if [[ $DO_HYDRO -eq 1 ]];    then fetch_hydrorivers_sa;  fi
if [[ $DO_S2 -eq 1 ]];       then fetch_s2_cogs;         fi
if [[ $DO_S1 -eq 1 ]];       then fetch_s1_asf;          fi
if [[ $DO_LANDSAT -eq 1 ]];  then fetch_landsat_ll;      fi
if [[ $DO_SOIL -eq 1 ]];     then fetch_soilgrids_wcs;   fi
if [[ $DO_NICFI -eq 1 ]];    then fetch_nicfi_planet;    fi

log "Done."
