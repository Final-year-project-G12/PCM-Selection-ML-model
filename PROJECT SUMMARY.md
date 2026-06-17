# Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating

**Group 12 — Project Review 2**  
Guide: **Dr. T. Deepika** (Assistant Professor, Sr. Gd.)

## Team

| S.No | Register Number | Student Name |
|---|---|---|
| 1 | CB.SC.U4CSE23430 | MANDUVA JASWITA |
| 2 | CB.SC.U4CSE23318 | DUNGI MANVITHA |
| 3 | CB.SC.U4CSE23412 | DUDDEKUNTA YUVA HASINI |
| 4 | CB.SC.U4CSE23558 | CHIRUVOLU VENKATA KHYATHI |
| 5 | CB.SC.U4CSE23034 | K P N L K MAHITHA |

![Amrita Logo](./logo.png)

---

## Table of Contents

1. Introduction
2. Motivation
3. Research Theme
4. Literature Review
5. Research Gaps
6. Objectives
7. Problem Statement
8. Technical Knowledge
9. Proposed Methodology Workflow
10. Datasets
11. References

---

## Introduction

- Solar contributes **133 GW** out of **254 GW** total renewable capacity (Solar : Total Renewable ≈ **1 : 1.9**).
- Solar thermal systems face intermittent sunlight (night/cloudy constraints).
- PCM-based storage captures excess heat and releases later for continuity.

![Introduction Figure](./image_1.jpeg)

## Comparison of Solar Water Heater Companies & PCM Usage

| Company | Country | PCM Used? | PCMs Used |
|---|---|---|---|
| Sunamp | UK | Yes | Plentigrade, P58, SU58 |
| PCM Products Ltd | UK | Yes | Positive temp (salt hydrate + organic + high-temp salts) |
| Vaillant | Germany | No | — |
| PLUSS Advanced Technologies | India | Yes | OM Series, FS Series |
| Chemtex Speciality Ltd | India | Yes | Positive Temperature PCM, Eutectic Salt PCM |
| Emmvee Group | India | No | — |

**Why PCM is used more in foreign countries:** energy-efficiency policies, carbon goals, and stronger investment support.  
**Why PCM is less integrated in India:** higher design complexity and deployment barriers, despite strong efficiency potential.

---

## Motivation

- Thermal energy is ~48–50% of global final energy demand.
- Residential water heating is ~6–8% of household energy demand.
- Conventional SWH annual efficiency: ~45–70%.
- PCM integration can improve thermal efficiency, retention time, and effective storage.
- Need adaptive strategies under varying irradiance and load.

---

## Research Theme Covered

### Solar Energy Materials and Solar Cells (Special Issue)
- AI for TES systems
- Intelligent latent heat storage control
- ML-assisted material selection
- AI-enhanced solar thermal performance

### Applied Thermal Engineering (Special Issue)
- Photothermal-driven PCM storage
- Building-level integration
- ML/DL-based adaptive control
- Efficiency-focused performance evaluation

---

## Literature Survey

### 1) PCM-Focused
- Hamza (2025): PCM storage review, environmental/performance impacts
- Rathore (2024): PCM integration for thermal efficiency
- Martinez (2025): industrial PCM suitability using DSC/TGA
- Abdellatif (2025): modeling limits and heat-transfer enhancement

### 2) AI/ML in Thermal Energy Systems
- Mohammed et al. (2025): nanotech + AI optimization
- Nems (2025): AI prediction/optimization in PCM systems
- Liu (2025): AI vs conventional thermal modeling
- Yan (2025): ML prediction of PCM melting time (XGBoost/SVR/RF)

### 3) AI/ML in SWH-Specific Systems
- Odoi-Yorke (2025): AI applications in SWH
- Eldokaishi (2022): ANN modeling for PCM-SWH
- Assareh (2023): ML-driven multi-objective optimization
- Muthanna (2025): ANN controller for PCM-SWH

### 4) Climate-Integrated PCM in SWH
- Singh (2025): PCM storage review for SWH
- Kou (2025): heat pipe + PCM integration
- Emami (2026): DRL control for solar-TES cycles
- Chen (2025): Taguchi optimization for PCM-SWH design

### 5) Broader Advanced Topics
- Rajamurugu (2025): MLP/RF/SVR for PCM performance
- Terfai (2025): ANN + MPC for solar pond control
- Barghi (2025): AI optimization in PCM-assisted solar drying
- Ghodusinejad (2025): irradiance forecasting methods

---

## Research Gaps

1. Limited real-time adaptive control in PCM-based systems.
2. Lack of integrated PCM-AI-hardware prototypes.
3. Weak alignment with household demand dynamics.
4. Limited long-term field validation.
5. Limited predictive optimization under climatic uncertainty.

---

## Objectives

- Collect/process climate + thermal data to classify/select suitable PCM under forecasted climate and demand conditions (RG2, RG5).
- Train DRL agent for adaptive charge/discharge/bypass control to maximize hot-water availability (RG1, RG3, RG5).
- Build thermal simulation environment and benchmark against passive/rule-based baselines (RG1, RG4).
- Deploy embedded closed-loop prototype and evaluate across Indian climate profiles (RG2, RG3, RG4).

---

## Problem Statement

**Novel Idea:** Design and implement an autonomous AI-driven PCM thermal storage system that dynamically optimizes charging/discharging under real-world conditions.

### SDG Alignment

![SDG 7](./7.png)
![SDG 9](./9.png)
![SDG 12](./12.png)
![SDG 13](./13.png)

---

## Proposed Architecture

### Overall Architecture
![Overall Architecture](./architecture.png)

### Objective 1 Architecture
![Objective 1 Architecture](./obj_1.png)

### Objective 2 Architecture
![Objective 2 Architecture](./obj_2.png)

### Objective 3 & 4 Architecture
![Objective 3 and 4 Architecture](./obj_34.png)

---

## Technical Knowledge

### Objective 1: PCM Selection Methodology
- Integrate climate + PCM properties via ML for performance prediction.
- Apply **Taguchi** DOE for parameter optimization.
- Apply **Grey Relational Analysis (GRA)** for multi-criteria ranking.

### PCM Selection Criteria and GRG
- Criteria: melting temperature, latent heat, conductivity, storage capacity.
- Grey Relational Grade (GRG):

\[
\gamma_i = \frac{1}{n} \sum \xi_i
\]

\[
\xi_i = \frac{\Delta_{\min} + \zeta \Delta_{\max}}{\Delta_i + \zeta \Delta_{\max}}
\]

- Highest GRG PCM selected.

### Objective 2: AI-Based Control
- DRL state: \(s_t = [T_w, T_p, f, GHI, T_{amb}, wind, time]\)
- Actions: Charge, Discharge, Bypass
- Environment: Grey-box thermal model
- Agent: Actor-Critic
- Algorithm: PPO
- Reward: maximize hot-water availability and reliability, penalize unmet demand and slow charging.

---

## Data Collection — PCM Properties

- Selection order: latent heat → conductivity → melting temperature → specific heat → density.
- Target SWH range: **35–65 °C**.
- Candidate families:
  - **Rubitherm RT:** RT35, RT38HC, RT42, RT44HC, RT45HC, RT47, RT50, RT54HC, RT55, RT64HC
  - **PLUSS OM/savE series:** HS36, OM35, OM37, OM39, OM42, OM46, OM48, OM50
- Climate-informed state vector includes GHI, DNI, Tamb, wind, RH, time features, PCM/water thermal states, demand.

## Data Collection — Solar & Environmental Inputs

- **ERA5 (ECMWF/Copernicus):** hourly reanalysis backbone.
- **NASA POWER:** 0.5° validated solar/weather data.
- **ISRO Solar Calculator + Global Solar Atlas:** cross-validation and climate suitability index.
- RRTDHS index:

\[
\text{RRTDHS} = \frac{\bar{Q}_{sol}}{T_{set} - \bar{T}_{out}}
\]

---

## Appendix: Data Preprocessing Visualizations

| | |
|---|---|
| ![Graph 1](./graph1.jpeg) | ![Graph 2](./graph2.jpeg) |
| ![Interactive Map](./map_interactive.png) | ![Irradiance](./irridiance.png) |

Top row: feature distributions and correlation matrix (post-scaling).  
Bottom row: spatial coverage map and irradiance time-series example.

---

## References

- BibTeX source: `references.bib` via `\bibliography{references}` in the original TeX.
- Citation keys are preserved in the TeX source for IEEE formatting.

---

## Thank You

**Thank You**
