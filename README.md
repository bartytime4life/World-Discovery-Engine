
⸻

🌍 World Discovery Engine (WDE)

OpenAI → Z Challenge · Archaeology & Earth Systems

⸻

📌 Overview

The World Discovery Engine (WDE) is a multi-modal AI pipeline that detects archaeologically significant sites in the Amazon and beyond.
It fuses satellite imagery, radar, LiDAR, soil/vegetation maps, hydrology, historical archives, and core sampling data to generate candidate site dossiers.

Each dossier includes:
	•	📡 Multi-sensor overlays (Sentinel, Landsat, SAR, LiDAR)
	•	🌱 Soil & vegetation fingerprints (ADE indicators)
	•	📜 Historical & archival references (maps, diaries, site DBs)
	•	🔗 Causal plausibility graphs
	•	🎲 Uncertainty quantification & counterfactual tests
	•	📑 Confidence narratives

The deliverable is a single Kaggle Notebook (ade_discovery_pipeline.ipynb) that anyone can run end-to-end, fully reproducible and CC-0 licensed.

⸻

🛠️ Key Features
	•	Multi-source ingestion: Sentinel-1/2, Landsat, NICFI, DEM, GEDI LiDAR, SoilGrids, HydroSHEDS, MapBiomas ￼ ￼
	•	Anomaly detection: CV filters, texture metrics, terrain relief, VLM zero-shot captions ￼
	•	ADE fingerprinting: Seasonal NDVI peaks, floristic markers, ring ditches ￼
	•	Evidence fusion: Multi-proof rule (≥2 modalities), Bayesian GNN uncertainty, causal PAG graphs ￼
	•	Candidate dossiers: Site-level PDFs/Markdown reports with overlays, graphs, and confidence narratives ￼
	•	Reproducibility: DVC data tracking, Hydra configs, MLflow logging, Docker environments ￼
	•	Ethics: CARE principles, Indigenous sovereignty flags, legal compliance, anti-data-colonialism safeguards ￼

⸻

📂 Repository Structure

World-Discovery-Engine/
├─ notebooks/
│  └─ ade_discovery_pipeline.ipynb   # Kaggle-ready notebook
├─ src/wde/
│  ├─ ingest/                        # Sentinel, DEM, LiDAR, SoilGrids loaders
│  ├─ detect/                        # CV, VLM, anomaly filters
│  ├─ evaluate/                      # NDVI time-series, hydrology plausibility
│  ├─ verify/                        # ADE fingerprints, causal graphs, B-GNN
│  ├─ reports/                       # Candidate dossier generator
│  └─ utils/                         # Geo, I/O, seeds, logging
├─ configs/                          # Hydra YAMLs (data, model, pipeline)
├─ artifacts/                        # Manifests, logs, outputs
├─ tests/                            # Unit & integration tests
├─ docs/
│  ├─ architecture.md                # Full system architecture
│  └─ datasets.md                    # Data registry & access notes
├─ wde.py                            # Typer CLI entrypoint
├─ dvc.yaml                          # DVC pipeline stages
├─ Dockerfile                        # Reproducible runtime
└─ README.md                         # (this file)


⸻

🚀 Quickstart

Kaggle Notebook
	1.	Fork or open ade_discovery_pipeline.ipynb.
	2.	Attach required Kaggle Datasets (Sentinel-2, Sentinel-1, DEM, SoilGrids).
	3.	Run all cells → produces:
	•	submission.csv (competition submission)
	•	outputs/ (candidate dossiers: PNGs, JSON, GeoJSON, PDFs)

Local Repo

# Clone and install
git clone https://github.com/bartytime4life/World-Discovery-Engine.git
cd World-Discovery-Engine
poetry install   # or uv sync

# Run pipeline stages
python wde.py ingest --aoi amazon.geojson
python wde.py detect --tile-grid 0.05
python wde.py evaluate --with-lidar --with-hydro
python wde.py verify --with-ade --with-causal
python wde.py reports --top_k 50

# Bundle for Kaggle
python wde.py bundle-kaggle


⸻

🔬 Scientific & Technical Foundations
	•	Fractal & pattern analysis — distinguishes natural irregularity vs. anthropogenic geometry ￼
	•	Physics-informed models — terrain dynamics, vegetation stability, causal flow ￼
	•	Simulation & validation — NASA-grade V&V, counterfactual SSIM ablations ￼ ￼
	•	CausalOps lifecycle — Arrange → Create → Validate → Test → Publish → Operate → Monitor → Document ￼

⸻

⚖️ Ethics & Governance
	•	CARE Principles (Collective Benefit, Authority to Control, Responsibility, Ethics) ￼
	•	Indigenous Data Sovereignty — detections flagged when overlapping Indigenous lands ￼
	•	Legal Compliance — IPHAN (Brazil) and national heritage protections built into the pipeline ￼
	•	Anti-Data Colonialism — outputs designed for expert review, not open site publication ￼

⸻

✅ Success Criteria
	•	Archaeological impact: ADE proxies & geoglyphs surfaced.
	•	Evidence depth: ≥2 independent modalities per site.
	•	Clarity: Transparent overlays, interpretable causal graphs.
	•	Reproducibility: Fully rerunnable on Kaggle & Docker.
	•	Ethics: CARE-aligned, sovereignty-respecting.

⸻

📜 Citation

If you use WDE in research, please cite:

World Discovery Engine (WDE) — OpenAI → Z Challenge
https://github.com/bartytime4life/World-Discovery-Engine

Datasets must be cited according to their respective licenses (Copernicus Sentinel, NASA SRTM, USGS Landsat, NICFI Planet, SoilGrids, etc.).

⸻

🤝 Contributing

Contributions are welcome! Please see:
	•	CONTRIBUTING.md
	•	ETHICS.md
	•	datasets.md

⸻

📧 Contact

Maintained by Andy Barta & collaborators.
For issues or suggestions, open a GitHub Issue or start a Discussion.

⸻

✨ The World Discovery Engine bridges AI, archaeology, and ethics — surfacing hidden histories while respecting the communities tied to them.

⸻
