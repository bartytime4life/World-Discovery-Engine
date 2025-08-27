# 📊 datasets.md — World Discovery Engine (WDE)

**OpenAI → Z Challenge · Archaeology & Earth Systems**

---

## 🌍 Purpose

This document catalogs the **datasets used by the World Discovery Engine (WDE)**, including their **sources, access methods, licenses, and integration notes**.  
All datasets are **open access or CC-0** compatible, ensuring reproducibility on **Kaggle, local, or cloud environments**.

---

## 📡 Core Remote Sensing Datasets

### Sentinel-2 (Optical Imagery)
- **Source:** Copernicus / ESA  
- **Access:** Copernicus Data Space (STAC API, OData), AWS S3 (`s3://sentinel-s2-l1c`), Google Earth Engine  
- **Format:** SAFE package (JP2 bands) or GeoTIFF composites  
- **Usage:** NDVI/EVI, vegetation vigor, soil/landcover classification  
- **License:** Copernicus open data policy (free to use with attribution) [oai_citation:0‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

### Sentinel-1 (SAR Radar)
- **Source:** Copernicus / ESA, NASA ASF DAAC  
- **Access:** ASF Vertex API, Copernicus Data Space, AWS S3 (`s3://sentinel-s1-l1c`)  
- **Format:** SAFE (GRD, SLC), GeoTIFF after calibration  
- **Usage:** Under-canopy structures, moisture & roughness anomalies  
- **License:** Copernicus open data policy [oai_citation:1‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

### Landsat Archive (1972–Present)
- **Source:** USGS (EarthExplorer, GEE, AWS Open Data)  
- **Access:** USGS API, AWS S3 (`usgs-landsat`), Google Earth Engine  
- **Format:** GeoTIFF (Level-1 TOA reflectance, Level-2 surface reflectance)  
- **Usage:** Long-term landcover & vegetation change, site persistence checks  
- **License:** Public domain (US Government data) [oai_citation:2‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

### NICFI Planet Monthly Mosaics
- **Source:** Planet Labs (funded by Norway’s International Climate & Forest Initiative)  
- **Access:** Planet Basemaps API, NICFI portal (requires free registration)  
- **Format:** 4-band GeoTIFF (RGB + NIR) mosaics at 4.7 m resolution  
- **Usage:** High-resolution canopy detection in the tropics  
- **License:** NICFI special license (non-commercial, attribution required) [oai_citation:3‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

---

## 🏔️ Elevation & Terrain Data

### SRTM DEM
- **Source:** NASA / USGS  
- **Resolution:** 30 m (1 arc-second) global DEM  
- **Access:** NASA Earthdata, USGS EarthExplorer, OpenTopography API, AWS S3 (`opentopography`)  
- **Format:** HGT, GeoTIFF, Cloud Optimized GeoTIFF (COG)  
- **Usage:** Slope, aspect, hydrology modeling, LRM anomaly extraction  
- **License:** Public domain [oai_citation:4‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

### Copernicus DEM
- **Source:** ESA Copernicus  
- **Resolution:** 30 m (GLO-30), 90 m (GLO-90), 10 m (Europe only)  
- **Access:** AWS S3 (`copernicus-dem-30m`), Copernicus Data Space, OpenTopography API  
- **Format:** Cloud Optimized GeoTIFF  
- **Usage:** Topographic context, hydrological plausibility  
- **License:** Copernicus DEM License (free, attribution required) [oai_citation:5‡Connecting to Remote Sensing and Environmental Data Sources.pdf](file-service://file-EQDY4o4qS3syntDXF39LSp)  

### GEDI LiDAR
- **Source:** NASA GEDI (Global Ecosystem Dynamics Investigation)  
- **Access:** NASA LP DAAC, OpenTopography  
- **Format:** HDF5, CSV footprints  
- **Usage:** Canopy height, canopy structure, DEM correction  
- **License:** NASA open data [oai_citation:6‡Enriching the World Discovery Engine for Archaeology & Earth Systems.pdf](file-service://file-XaAoukBHFHF8vdPNUYGZhm)  

---

## 🌱 Soil & Environmental Data

### SoilGrids
- **Source:** ISRIC – World Soil Information  
- **Resolution:** 250 m global soil grids  
- **Variables:** pH, phosphorus, carbon, nitrogen, organic matter  
- **Access:** ISRIC API, AWS Open Data  
- **Format:** GeoTIFF  
- **Usage:** ADE proxy detection (high phosphorus, soil fertility anomalies)  
- **License:** Open data, attribution required [oai_citation:7‡Enriching the World Discovery Engine for Archaeology & Earth Systems.pdf](file-service://file-XaAoukBHFHF8vdPNUYGZhm)  

### MODIS
- **Source:** NASA  
- **Products:** NDVI/EVI vegetation indices, land surface temperature  
- **Resolution:** 250–1000 m  
- **Access:** NASA Earthdata, Google Earth Engine  
- **Format:** HDF, GeoTIFF  
- **Usage:** Seasonal vegetation patterns, anomaly persistence checks  
- **License:** NASA open data [oai_citation:8‡Enriching the World Discovery Engine for Archaeology & Earth Systems.pdf](file-service://file-XaAoukBHFHF8vdPNUYGZhm)  

### MapBiomas
- **Source:** Brazil’s MapBiomas Consortium  
- **Coverage:** Annual land cover/use (1985–present, South America)  
- **Access:** MapBiomas portal, GEE  
- **Format:** Raster (GeoTIFF)  
- **Usage:** Land use change detection, settlement impact mapping  
- **License:** Open for non-commercial research [oai_citation:9‡Enriching the World Discovery Engine for Archaeology & Earth Systems.pdf](file-service://file-XaAoukBHFHF8vdPNUYGZhm)  

---

## 💧 Hydrology Data

### HydroSHEDS
- **Source:** WWF / USGS  
- **Resolution:** Derived from SRTM DEM  
- **Access:** HydroSHEDS portal, AWS Open Data  
- **Format:** Raster (flow accumulation, direction), vector (rivers, basins)  
- **Usage:** Settlement proximity to rivers, terraces, floodplain modeling  
- **License:** Public domain [oai_citation:10‡Enriching the World Discovery Engine for Archaeology & Earth Systems.pdf](file-service://file-XaAoukBHFHF8vdPNUYGZhm)  

---

## 📜 Historical & Archival Data

- **Colonial-era maps** — digitized, georeferenced (via OCR + GIS alignment)  
- **Missionary diaries & reports** — ingested via OCR + NLP (entity/location extraction)  
- **Published archaeological site DBs** — e.g., Brazilian heritage inventories, academic datasets  
- **Integration:** Text snippets attached to tiles; mapped where georeference available [oai_citation:11‡World Discovery Engine (WDE) Architecture Specification.pdf](file-service://file-PiJpkFqNs2pDTFMtxexAuF)  

---

## 🧪 Core Sampling & Paleo Data

- **NOAA Paleoclimatology (NCEI Paleo)** — lake/marine/ice cores with pollen, isotope data [oai_citation:12‡Core Sampling Databases and Resources (Global and South America).pdf](file-service://file-UE6k5z89LhrGH16VjuxrRo)  
- **PANGAEA** — global georeferenced datasets (marine sediment, soils, rock cores) [oai_citation:13‡Core Sampling Databases and Resources (Global and South America).pdf](file-service://file-UE6k5z89LhrGH16VjuxrRo)  
- **Neotoma Paleoecology DB** — fossil/pollen/charcoal data across the Americas [oai_citation:14‡Core Sampling Databases and Resources (Global and South America).pdf](file-service://file-UE6k5z89LhrGH16VjuxrRo)  
- **IODP / ODP / DSDP** — marine drilling cores from Amazon Fan & global oceans [oai_citation:15‡Core Sampling Databases and Resources (Global and South America).pdf](file-service://file-UE6k5z89LhrGH16VjuxrRo)  
- **ICDP** — continental scientific drilling projects (e.g., Amazon/Andes basins) [oai_citation:16‡Core Sampling Databases and Resources (Global and South America).pdf](file-service://file-UE6k5z89LhrGH16VjuxrRo)  

**Usage in WDE:**  
Core records provide **ground-truth paleoenvironmental signals** (phosphorus levels, pollen shifts, charcoal evidence of anthropogenic activity) to validate remote sensing anomalies.

---

## ⚖️ Licensing Summary

- **Copernicus Sentinel (S1/S2, DEM):** Open access with attribution  
- **Landsat, SRTM, MODIS:** Public domain (USGS/NASA)  
- **SoilGrids, MapBiomas:** Open, attribution required  
- **NICFI Planet mosaics:** Non-commercial, attribution required  
- **Historical archives:** Attribution varies; OCR-extracted text flagged with original source  
- **Core samples (NOAA, PANGAEA, Neotoma, IODP, ICDP):** Open science licenses, DOI citation required  

---

## ✅ Integration Rules in WDE

- All datasets are **pre-processed into tile-based overlays** aligned to the AOI grid.  
- **Hashes + manifests** stored in `artifacts/manifests/` for reproducibility.  
- Kaggle Notebook checks for dataset availability; if offline, **fallback samples** are used.  
- Outputs cite data provenance automatically in dossier reports.  

---

✨ With this registry, the **World Discovery Engine** ensures that all discoveries are grounded in **open, traceable, reproducible data sources**, uniting **remote sensing, soil science, hydrology, and history** into a single discovery framework.