
⸻

WDE Datasets — Sources, Access & Provenance

This file is the single source of truth for data provenance, access methods, licensing, and reproducibility requirements of the World Discovery Engine (WDE).
All data used must be open, CC-0 compatible, and reproducible on Kaggle.

⸻

🌍 Remote Sensing & Geospatial
	•	Sentinel-2 (Optical Multispectral)
	•	Source: Copernicus Data Space, AWS/GCP open buckets ￼
	•	Product: Level-2A surface reflectance
	•	Bands: B02, B03, B04 (RGB), B08 (NIR), B11, B12 (SWIR)
	•	Usage: NDVI/EVI indices, anomaly overlays, ADE vegetation fingerprinting ￼
	•	License: ESA Copernicus open data ￼
	•	Sentinel-1 (SAR Radar)
	•	Source: ASF DAAC, Copernicus Data Space, AWS open bucket ￼
	•	Products: GRD (backscatter), SLC (complex)
	•	Usage: Structural anomalies under canopy, soil moisture proxies
	•	License: ESA Copernicus open data ￼
	•	Landsat (5–9)
	•	Source: USGS EarthExplorer, AWS/GCP public datasets ￼
	•	Products: Collection-2 Level-2 surface reflectance
	•	Usage: Historical change detection (since 1972), vegetation dynamics
	•	License: USGS Public Domain ￼
	•	NICFI Planet Mosaics (Tropics only)
	•	Source: Planet NICFI Basemaps API ￼
	•	Resolution: ~4.7 m, monthly mosaics
	•	Usage: High-res visual overlays for Amazonia, time-series anomaly check
	•	License: Non-commercial, attribution required (Planet + NICFI)
	•	Digital Elevation Models (DEM/DTM)
	•	Sources: SRTM (NASA/USGS), Copernicus DEM, FABDEM ￼
	•	Usage: Terrain indices, Local Relief Models (LRM), hillshades for micro-relief detection ￼
	•	License: Public domain (SRTM), ESA Copernicus (DEM)
	•	LiDAR
	•	Sources: GEDI spaceborne LiDAR (NASA), OpenTopography (regional surveys) ￼
	•	Usage: Canopy removal, micro-topography (mounds, ring ditches, terraces) ￼
	•	License: Open access; regional restrictions may apply (see ETHICS.md)

⸻

🌱 Environmental & Thematic
	•	Soils & Geochemistry
	•	Sources: SoilGrids (ISRIC), WoSIS profiles ￼
	•	Usage: ADE proxies — high phosphorus, carbon, pH anomalies ￼
	•	License: Open, CC-BY (ISRIC/SoilGrids)
	•	Vegetation & Floristics
	•	Sources: MapBiomas (annual landcover for Amazonia), Rainforest inventory plots ￼
	•	Usage: ADE indicators (Brazil nut, peach palm, cacao distributions) ￼
	•	License: CC-BY (MapBiomas); research networks vary
	•	Hydrology & Terrain
	•	Sources: HydroSHEDS / HydroRIVERS ￼
	•	Usage: Settlement plausibility (terraces, bluffs, floodplains, river adjacency) ￼
	•	License: Open data
	•	Climate Layers
	•	Sources: MODIS (NASA), reanalysis products
	•	Usage: Seasonal vegetation baselines, environmental controls
	•	License: NASA public domain

⸻

📜 Historical & Archival
	•	Maps & Expeditions
	•	Sources: Colonial/missionary maps (georeferenced), gazetteers, site databases ￼
	•	Methods: OCR → NLP → entity extraction for place names & features ￼
	•	License: Varies; ensure permission noted in dossier
	•	Ethnographic & Archival Texts
	•	Sources: Diaries, oral histories, published site records
	•	Usage: Contextual overlays for candidate site validation ￼

⸻

🧪 Core Sampling & Physical Archives
	•	Global Core Databases
	•	NOAA NCEI Paleoclimatology ￼
	•	PANGAEA Data Publisher ￼
	•	Neotoma Paleoecology Database ￼
	•	South America / Amazon Specific
	•	ICDP Trans-Amazon Drilling Project (TADP) ￼
	•	Regional core libraries (Brazil CPRM GeoSGB, University repositories) ￼
	•	Usage in WDE:
	•	Dating, soil/vegetation context, ADE validation, environmental baselines

⸻

📊 Provenance Checklist (Required in Dossiers)
	•	Source URL / API reference
	•	License (ESA, NASA, USGS, Planet, ISRIC, etc.)
	•	Version / acquisition date
	•	Spatial & temporal coverage
	•	CRS / projection, resolution
	•	Preprocessing steps (see configs/preprocess.yaml)
	•	DVC path (e.g., data/raw/sentinel2/...) ￼

⸻

⚖️ Ethical & Legal Notes
	•	Apply CARE Principles (Collective Benefit, Authority, Responsibility, Ethics) for Indigenous and local data ￼
	•	Respect national heritage laws (e.g., IPHAN in Brazil) — see ETHICS.md ￼
	•	Default outputs mask exact coordinates to prevent looting; dossiers are expert-review only
	•	Kaggle notebooks use only open datasets or CC-0 uploads — no restricted sources ￼

⸻