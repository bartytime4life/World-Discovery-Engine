# FILE: datasets.md
# -------------------------------------------------------------------------------------------------
# WDE Datasets — Sources & Access

> Keep this file as the **single source of truth** for data provenance, access, and licensing.

## Remote Sensing (examples; customize to your scope)
- **Sentinel-2 (optical)** — access via Copernicus/OpenHub APIs or Sentinel Hub; Level-2A surface reflectance.
- **Sentinel-1 (SAR)** — GRD backscatter; flood/soil moisture proxies; requires radiometric terrain correction.
- **Landsat 5/7/8/9** — historical archives for temporal analysis.
- **NICFI Planet Mosaics** — monthly basemaps (tropics) with non-commercial license.
- **DEM/DTM** — SRTM, Copernicus DEM, FABDEM; beware voids and vertical datums.
- **LiDAR** — GEDI shots or regional point clouds (OpenTopography); record CRS, vertical units, density.

## Environmental & Thematic
- **Soils/Vegetation** — e.g., SoilGrids, local soil surveys, MapBiomas; track resolutions and years.
- **Hydrology** — HydroSHEDS/HydroRIVERS, floodplains; link to flow accumulation/hand models.
- **Climate** — MODIS or reanalysis layers where justified.

## Historical & Archival
- Georeferenced maps, expeditions/diaries, gazetteers, and site databases. Track licenses & permissions.

## Core Sampling Resources
- Regional core catalogs and publications; document sampling depth, coordinates, dating methods.

## Provenance Checklist
- [ ] Source URL or API
- [ ] License
- [ ] Version/date
- [ ] Spatial/temporal coverage
- [ ] CRS/projection, resolution
- [ ] Transform steps (preprocess.yaml)
- [ ] DVC path (e.g., data/raw/sentinel2/…)