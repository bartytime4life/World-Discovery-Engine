# ⚖️ ETHICS.md — World Discovery Engine (WDE)

**OpenAI → Z Challenge · Archaeology & Earth Systems**

---

## 🌍 Purpose

The **World Discovery Engine (WDE)** is built not only to **discover archaeologically significant sites** but also to **do so responsibly, ethically, and collaboratively**.  
This document outlines the ethical foundations guiding WDE’s design, data use, and outputs.

---

## 📜 Guiding Principles

### 1. CARE Principles
Adopted from the **Global Indigenous Data Alliance (GIDA)**:
- **Collective Benefit** — Ensure discoveries benefit local and Indigenous communities.  
- **Authority to Control** — Respect communities’ rights to govern their data and heritage.  
- **Responsibility** — Prevent harm, misuse, or exploitation of cultural knowledge.  
- **Ethics** — Commit to fair, just, and transparent use of data.  

### 2. FAIR Principles
- **Findable, Accessible, Interoperable, Reusable** — Open science standards for datasets and outputs.  
- All inputs/outputs are **documented, versioned, and licensed under CC-0 or equivalent**, unless restricted by Indigenous sovereignty or legal frameworks.  

---

## 🚫 Data Colonialism Safeguards

WDE is designed to **avoid extractive or colonialist practices** in computational archaeology:

- ❌ No open publication of **precise site coordinates** by default.  
- ❌ No assumption that AI-discovered sites can bypass **legal permits** or **Indigenous approval**.  
- ❌ No commercial exploitation of heritage data.  

Instead:

- ✅ **Candidate dossiers** are intended for **expert and community review**, not for public exposure.  
- ✅ Outputs highlight **uncertainty** and **refutation pathways**, avoiding false claims of “discovery.”  
- ✅ Indigenous consultation is acknowledged as a **necessary prerequisite** to field validation.  

---

## 🛡️ Legal & Regulatory Compliance

- **Brazil (IPHAN)** — Any archaeological discoveries must be reported to IPHAN (Instituto do Patrimônio Histórico e Artístico Nacional).  
- **Other National Frameworks** — WDE includes a configurable compliance layer for region-specific laws.  
- **Default Behavior** — Mask coordinates to ~2 decimal places (≈1 km) unless explicit authorization is provided.  

---

## 🧭 Sovereignty & Community Engagement

- **Indigenous Territories** — If anomalies overlap with Indigenous land boundaries, WDE **flags detections** with sovereignty notices.  
- **Consent-First Model** — Any further publication or exploration requires **prior informed consent**.  
- **Community Participation** — WDE supports annotation of results with local knowledge (oral histories, cultural metadata).  

---

## 🔍 Transparency & Accountability

- **Audit Logs** — Every run produces logs of data, configs, and outputs.  
- **Provenance** — All datasets are cited, hashed, and versioned.  
- **Explainability** — Candidate dossiers include **evidence chains** (maps, overlays, causal graphs) so decisions are transparent.  
- **Refutability** — Each detection includes **counterfactual SSIM tests** to allow challenge and verification.  

---

## 🚦 Default Ethical Modes

The WDE pipeline enforces **safe defaults**:

- Outputs anonymized coordinates in Kaggle Notebook runs.  
- Warnings displayed when processing data overlapping **protected or sovereign zones**.  
- Banner messages:  
  > ⚠️ *“Discovery involves cultural heritage — ensure engagement with local communities and legal authorities before any further action.”*  

---

## 🤝 Responsible Collaboration

- Collaborations with archaeologists, Indigenous leaders, and local governments are **encouraged as first-class stakeholders**.  
- Candidate site dossiers are **tools for dialogue**, not unilateral declarations.  
- Contributions to this project must uphold the above principles; code or datasets violating them will not be accepted.  

---

## ✅ Ethical Commitments Checklist

- [x] Respect Indigenous sovereignty (CARE)  
- [x] Cite and license all datasets (FAIR)  
- [x] Mask or generalize sensitive outputs by default  
- [x] Provide transparent, reproducible evidence  
- [x] Avoid data colonialism and exploitative practices  
- [x] Promote collaborative archaeology over extractive approaches  

---

✨ By embedding ethics at the **core of the architecture**, the **World Discovery Engine** ensures that **AI-driven archaeology** respects communities, heritage, and humanity’s shared history.