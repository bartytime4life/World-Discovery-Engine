🌍 WDE Ethics & Responsible Research

The World Discovery Engine (WDE) identifies potentially sensitive archaeological candidates (e.g., ADEs, geoglyphs, earthworks).
Responsible research is mandatory. Ethical guardrails are embedded into the pipeline and must be followed in all contexts.

⸻

✨ Guiding Principles
	1.	Do No Harm
	•	Never release data in a way that could enable looting, disturbance, or exploitation of heritage ￼.
	•	Fragile locations default to restricted access.
	2.	Community Respect
	•	Indigenous and local communities hold authority over cultural heritage data.
	•	All outputs must respect Free, Prior, and Informed Consent (FPIC).
	•	Follow the CARE Principles (Collective Benefit, Authority to Control, Responsibility, Ethics) ￼.
	3.	Data Minimization
	•	Coarsen coordinates (≥0.01°) in public outputs.
	•	Detailed site information is private dossier only, shared under controlled access ￼.
	4.	Provenance & Uncertainty
	•	Every dossier includes:
	•	Source dataset list
	•	Transformations applied
	•	Uncertainty quantification (Bayesian GNN distributions, SSIM counterfactuals) ￼ ￼
	5.	Legal Compliance
	•	Adhere to national regulations (e.g., Brazil’s IPHAN laws) ￼.
	•	Respect dataset licenses (Copernicus, USGS, Planet NICFI, SoilGrids, etc.).
	6.	Reciprocity
	•	Research must benefit source communities.
	•	Avoid “data colonialism”: discoveries are not for extraction, but for collaborative archaeology ￼.

⸻

📜 Publication Norms
	•	Public releases must exclude precise site coordinates unless explicitly authorized.
	•	Each release must contain:
	•	Multi-modal evidence (≥2 independent proofs) ￼
	•	Confidence narratives (plain-language uncertainty explanation) ￼
	•	Refutation attempts (counterfactual tests, ablation checks) ￼

⸻

🔐 Access Controls
	•	Default Masking: All public reports mask exact coordinates.
	•	Tiered Access: Private remotes store sensitive dossiers for expert review.
	•	Indigenous Land Check: If candidates fall within Indigenous territories, dossiers automatically flag sovereignty notes ￼.
	•	Ethical Mode (Default): CLI and notebooks always apply safety filters unless explicitly overridden (documented justification required).

⸻

🛰️ Scientific & Simulation Ethics
	•	Align with NASA-STD-7009 standards on model credibility ￼:
	•	Verification (implementation correct)
	•	Validation (outputs match real-world references)
	•	Uncertainty quantification (confidence intervals, ensemble robustness)
	•	Require simulation-based refutation: counterfactual SSIM tests to confirm robustness ￼.
	•	Never present single-source anomalies without causal plausibility and cross-modality validation ￼.

⸻

⚖ Oversight & Appeals
	•	If ethical concerns arise:
	1.	Pause dissemination.
	2.	Convene WDE maintainers, advisors, and community representatives.
	3.	Resolve via modification, restriction, or withdrawal.
	•	Stakeholders may appeal dossier inclusion/exclusion. WDE governance commits to listening first.

⸻

✅ Ethical Defaults in WDE
	•	Coordinates rounded in all public artifacts.
	•	Dossiers include a sovereignty banner when Indigenous boundaries overlap:
⚠ Candidate overlaps Indigenous territory. Ensure engagement with communities and local authorities before any further action. ￼
	•	All outputs include a confidence + caveat narrative.
	•	CI/CD includes ethics checks (coordinate masking, license validation, sovereignty overlap scan).

⸻

🌱 Closing Commitment

The WDE is not just an AI pipeline—it is a collaborative archaeology framework.
Its discoveries are only valid if they:
	•	Protect cultural heritage,
	•	Respect sovereignty and law,
	•	Quantify uncertainty honestly,
	•	Provide benefit to communities, not extraction.

Only under these conditions can the WDE’s outputs be considered responsible scientific contributions.

⸻