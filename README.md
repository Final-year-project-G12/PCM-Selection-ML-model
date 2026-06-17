# Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating

**Type 1: Research Project | Group 12 | Project Review 1**

---

## Team Members

| S.No | Register Number    | Student Name                  |
|------|--------------------|-------------------------------|
| 1    | CB.SC.U4CSE23430   | Manduva Jaswita               |
| 2    | CB.SC.U4CSE23318   | Dungi Manvitha                |
| 3    | CB.SC.U4CSE23412   | Duddekunta Yuva Hasini        |
| 4    | CB.SC.U4CSE23558   | Chiruvolu Venkata Khyathi     |
| 5    | CB.SC.U4CSE23034   | K P N L K Mahitha             |

**Guide:** Dr. T. Deepika  
**Designation:** Assistant Professor (Sr. Gd.)

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

## 1. Introduction

- Solar energy contributes **133 GW** out of **254 GW** of total renewable capacity — Solar : Total Renewable ≈ 1 : 1.9, meaning solar accounts for nearly half of the renewable mix.
- Solar thermal systems are limited by intermittent sunlight, resulting in reduced heat availability during night time and cloudy conditions.
- **PCM-based thermal storage** captures excess solar heat and releases it later, improving continuous thermal energy availability.

---

## 2. Comparison of Solar Water Heater Companies & PCM Usage

| Company         | Country | PCM? | PCMs Used                          |
|-----------------|---------|------|-------------------------------------|
| Sunamp          | UK      | ✓    | Plentigrade PCM, P58, SU58         |
| PCM Products Ltd| UK      | ✓    | Salt hydrate, organics             |
| Vaillant        | Germany | ✗    | None                               |
| PLUSS (IN)      | India   | ✓    | OM Series                          |
| Chemtex         | India   | ✓    | Positive Temp PCM                  |
| Emmvee          | India   | ✗    | None                               |

**Why PCM is used more in foreign countries:**  
Foreign markets focus strongly on energy efficiency and carbon reduction. Government support and higher investment capacity make PCM integration commercially feasible.

**Why PCM is not widely integrated in India:**  
PCM systems increase design complexity. However, with proper optimization and field validation, PCM can significantly improve hot-water availability, sustainability, and efficiency.

---

## 3. Motivation

- Thermal energy accounts for **48–50%** of global final energy consumption. Residential water heating uses **6–8%** of household energy, underscoring SWH potential.
- Conventional solar water heating (SWH) systems achieve only **45–70%** annual efficiency, wasting 30–55% of incident solar energy due to inadequate storage.
- PCM-integrated systems can boost thermal efficiency, extend heat retention time, and store ~50% of total heat with <20% PCM volume.
- Global solar thermal capacity reaches ~544–560 GW_th (2024–2025); the global market for water heaters was valued at **$23.7 billion** as of 2023 and is expected to reach **$32.1 billion by 2029**.
- Existing PCM-SWH systems use mostly fixed/offline strategies, causing losses under variable irradiance/load — motivating this research.

---

## 4. Literature Survey — Comprehensive Research Map

*Key Contributions: PCM Materials, AI Methods, SWH Applications*

| Author (Year)         | Focus              | Method            | Key Insight               |
|-----------------------|--------------------|-------------------|---------------------------|
| Hamza (2025)          | PCM storage        | Review, simulation| AI-assisted selection     |
| Rathore (2024)        | PCM-SWH            | Thermal model     | Heat transfer             |
| Martinez (2025)       | Industrial PCM     | DSC, TGA          | Tunable formulations      |
| Mohammed et al. (2025)| Nano-AI thermal    | ANN, opt          | AI models                 |
| Nemś (2025)           | AI-PCM predict     | ML, GA            | Real-time monitor         |
| Liu (2025)            | ANN vs modeling    | ANN, opt          | Sensor-driven             |
| Yan (2025)            | Melting time       | XGBoost, SVR      | TES tools                 |
| Odoi-Yorke (2025)     | AI in SWH          | ANN, analytics    | AI–PCM scheme             |
| Eldo (2022)           | ANN-PCM-SWH        | ANN sim           | Supervisory               |
| Assareh (2023)        | ML collector       | Multi-obj opt     | Climate-adaptive          |
| Muthanna (2025)       | ML controller      | ANN dyn sim       | Retrofit                  |
| Singh (2025)          | PCM-SWH tech       | Modeling          | Climate-specific          |
| Kou (2025)            | Heat pipe-PCM      | Num model         | Cross-climate             |
| Emami et al. (2026)   | DRL control        | DRL sim           | Grid-interactive          |
| Chen (2025)           | SWH opt            | Taguchi DOE       | Fins/heat pipes           |
| Rajamurugu (2025)     | PCM-ML perf        | MLP, RF, SVR      | Performance estimation    |
| Terfai et al. (2025)  | Pond thermal       | MPC, ANN          | Scalable AI               |
| Barghi (2025)         | Solar drying       | ANN, SVM          | Smart systems             |
| Ghodusinejad (2025)   | Solar forecast     | AI, NWP           | Adaptive method           |

---

## 5. Research Gaps

1. **Limited real-time adaptive control** — Most PCM-based systems use passive or fixed-rule strategies, lacking dynamic response to varying solar input and demand.
2. **Lack of integrated PCM–AI–hardware systems** — Prior studies treat materials, control, and implementation separately without a complete working prototype.
3. **Poor alignment with real usage patterns** — Existing systems rarely adapt storage and release based on actual household demand.
4. **Limited real-world validation** — Most research relies on simulations or short-term tests rather than long-term experimental evaluation.
5. **Limited predictive optimization under climatic uncertainty** — Most PCM-based systems lack predictive models to anticipate variations in solar input and demand, leading to suboptimal thermal storage utilization.

---

## 6. Objectives

> **Overall Problem:** Most PCM-based solar water heating systems use passive or fixed control rules, leading to delayed heat availability, inefficient storage, and energy loss under variable solar and demand conditions.

- **Objective 1 (RG 2, RG 5):** Collect and process climate data and thermal parameters to classify and select the most suitable PCM for forecasted climate and demand conditions.
- **Objective 2 (RG 1, RG 3, RG 5):** Train a DRL agent to adaptively control PCM charge, discharge, and bypass modes to maximize hot-water availability.
- **Objective 3 (RG 1, RG 4):** Build a grey-box thermal simulation as the DRL training environment and benchmark the controller against rule-based and passive baselines.
- **Objective 4 (RG 2, RG 3, RG 4):** Deploy the AI controller on embedded hardware as a closed-loop prototype and evaluate performance across Indian climate profiles.

---

## 7. Problem Statement

> **Novel Idea:** To design and implement an autonomous, AI-driven PCM-based thermal storage system that dynamically optimizes heat charging and discharging under real-world operating conditions.

**Sustainable Development Goals (SDGs) addressed:**

- SDG 7 — Affordable and Clean Energy
- SDG 9 — Industry, Innovation and Infrastructure
- SDG 12 — Responsible Consumption and Production
- SDG 13 — Climate Action

---

## 8. Technical Knowledge

### 8.1 PCM Selection & Optimization (Objective 1)

**Mapped Objective:** Optimal PCM selection under climate conditions

**Taguchi Method:** Statistical Design of Experiments (DOE) using orthogonal arrays to reduce the number of experiments.

**Grey Relational Analysis (GRA):** Multi-objective method that converts multiple criteria into a single score.

**Grey Relational Grade (GRG):**

$$\gamma_i = \frac{1}{n} \sum \xi_i$$

$$\xi_i = \frac{\Delta_{\min} + \zeta \Delta_{\max}}{\Delta_i + \zeta \Delta_{\max}}$$

**Selection Criteria:** Thermal efficiency and heat retention time

**Outcome:** HPCM with the highest GRG is selected as the optimal material.

---

### 8.2 Thermal Modeling (Objective 1)

**Mapped Objective:** Grey-box thermal modeling of PCM system

**Grey-Box Model:** Combines physics-based equations with real system behavior.

**Key Variables:**
- *T_w* — Water temperature
- *T_p* — PCM temperature
- *f* — Melt fraction

**Latent Heat:**

$$Q_{\text{latent}} = mL$$

**Sensible Heat:**

$$Q_{\text{sensible}} = m C_p (T_f - T_i)$$

**Total Energy:**

$$Q_{\text{total}} = Q_{\text{latent}} + Q_{\text{sensible}}$$

**Melting Dynamics:**

$$M_p L \frac{df}{dt} = hA(T_w - T_m)$$

**Outcome:** Predicts dynamic heat storage and release behavior.

---

### 8.3 AI-Based Control (Objective 2)

**Mapped Objective:** Intelligent control using Deep Reinforcement Learning

**DRL:** Learns optimal control policy under dynamic solar conditions.

**State:**

$$s_t = [T_w,\ T_p,\ f,\ GHI]$$

**Actions:** Charge, Discharge, Bypass

**Transition:**

$$s_{t+1} = f(s_t, a_t)$$

**Objective:** Maximize hot-water availability under varying solar conditions.

**Algorithms:**
- PPO (Proximal Policy Optimization)
- DDPG (Deep Deterministic Policy Gradient)

---

### 8.4 Performance Evaluation (Objective 4)

**Mapped Objective:** System evaluation and optimization

**Solar Input:**

$$Q_{\text{solar}} = A \cdot GHI$$

**Thermal Efficiency:**

$$\eta = \frac{Q_{\text{total}}}{A \cdot E}$$

**Heat Retention Time:**

$$t_{\text{retention}} = t_{\text{final}} - t_{\text{sunset}}$$

**Benchmarking:** Compare DRL controller against:
- Rule-based control
- Passive systems

**Outcome:** Improved efficiency and longer heat availability.

---

## 9. Data Collection

### 9.1 PCM Properties

- Selection criteria follow the priority ranking: latent heat → thermal conductivity → melting temperature → specific heat → density.
- Target operating range for solar water heating: **T_m = 35–65 °C**, validated across multiple SWH studies.

**Rubitherm RT Series** — commercial paraffin-based PCMs:
- RT35, RT38 HC, RT42, RT44 HC, RT45 HC, RT47, RT50, RT54 HC, RT55, RT64 HC
- Pre-engineered for higher λ; rated safe for domestic applications.

**PLUSS savE® / OM Series** — bio-based organics suited to Indian supply chains:
- savE HS36, OM35, OM37, OM39, OM42, OM46, OM48, OM50

**Eutectic blends** (e.g., Lauric–Myristic 66/34, T_m = 34.2 °C) provide tunable T_m for climate-zone-specific recommendation.

**Climate data state vector:**

$$\mathbf{s} = [\text{GHI, DNI, } T_{\text{amb}}, v_{\text{wind}}, \text{RH, Hour, Month, } T_{\text{PCM}}, T_w, f_{\text{melt}}, \text{Demand}]$$

---

### 9.2 Solar and Environmental Data

**ERA5 Reanalysis (Copernicus / ECMWF)** — primary multi-year climate backbone:
- GHI (surface solar radiation downwards), DNI, DHI, total cloud cover, surface pressure, relative humidity
- Hourly resolution; used to compute clear-sky index: K_c = GHI / GHI_clear

**NASA POWER Data Access Viewer** — validated solar resource data at 0.5° resolution for Indian cities (e.g., Coimbatore, Jaisalmer, Kochi):
- GHI, DNI, DHI, T_amb, wind speed, humidity feature vector

**ISRO Solar Energy Calculator & Global Solar Atlas** — used to cross-validate GHI estimates and derive the climate suitability index:

$$\text{RRTDHS} = \frac{\bar{Q}_{\text{sol}}}{T_{\text{set}} - \bar{T}_{\text{out}}}$$

**Sensor resolution benchmark:**
- Temperature: ±0.2 °C
- Irradiance: ±10 W/m²
- Interval: 60 s (consistent with DS18B20 + pyranometer hardware)

---

## 10. Data Preprocessing Pipeline

### Part 1 — ERA5 Climate Data Acquisition & Initial Processing

**Input:** Raw ERA5 reanalysis (NetCDF) for Tamil Nadu / target cities

**Goal:** Production-ready dataset (`climate_all_cities_preprocessed.csv`) for:
- XGBoost PCM classifier
- Grey-box thermal simulation for DRL training
- RPi / TFLite edge deployment

**Pipeline Structure (city-wise):**

1. Copernicus CDS API → hourly NetCDF files
2. NetCDF → clean hourly CSV + first feature engineering (pvlib)
3. Full monolithic preprocessing

A modular version is available in `preprocessing/` (6 independent scripts).

---

### Part 2 — Feature Engineering, Validation & Scaling

**Key Processing Stages (`03_preprocess_validate.py`):**

1. **Stage A — Data Cleaning:** Physical bounds check, duplicate removal, per-city linear interpolation (max 6 h gaps)
2. **Stage A2 — Feature Engineering:**
   - Cyclical encodings (sin/cos hour, month, DOY)
   - Lag features (GHI & T_amb at 1–24 h)
   - Rolling statistics (3 h, 6 h, 24 h)
   - Derived: RRTDHS, CSI, DHI, T_pcm_delta, solar geometry (pvlib)
3. **Stage B — Validation:** Temporal gaps, spatial join check, QC report
4. **Stage C — Three-scaler normalisation (no data leakage):**
   - MinMaxScaler → bounded variables
   - StandardScaler → Gaussian-like
   - RobustScaler → skewed (precipitation, wind direction)
5. **Stage E — Early Fusion:** Assembles solar-sensor, weather-NWP, PCM-thermal, time, lag & rolling groups into single feature matrix X

**Output:**
- `climate_all_cities_preprocessed.csv` (final ML-ready dataset)
- 3 fitted scalers (`.pkl`) ready for RPi / TFLite inference
- Validation reports + plots (correlation matrix, distributions, temporal heatmaps)

> **Challenges Addressed:** Spatiotemporal resolution mismatch, data alignment, missing values, heterogeneous modality handling, and early fusion strategy.

---

## 11. References

1. NITI Aayog — Solar Irradiance Data
2. Duraivel et al. (2025) — Performance of SWH Systems
3. Al-Mamun (2023) — SWH Review
4. Chen (2025) — Taguchi PCM Optimization
5. Odoi-Yorke (2025) — AI in SWH
6. Hamza (2025) — PCM Review
7. Rathore (2024) — Thermal PCM
8. Martinez (2025) — Industrial PCM
9. Mohammed et al. (2025) — AI Thermal
10. Nemś (2025) — AI-PCM Review
11. Liu (2025) — AI PCM
12. Yan (2025) — ML PCM Melting
13. Eldokaishi (2022) — ANN PCM SWH
14. Assareh (2023) — ML Solar Collector
15. MJET (2025) — PCM Controller
16. Singh (2025) — PCM SWH
17. Kou (2025) — BIHP PCM
18. Emami et al. (2026) — DRL Solar
19. Rajamurugu (2025) — PCM ML
20. Terfai et al. (2025) — Solar Pond ANN MPC
21. Barghi (2026) — Solar Drying PCM
22. Ghodusinejad (2026) — Solar Forecast
23. Rubitherm GmbH — RT Series PCM Data
24. PLUSS Advanced Technologies — savE / OM Series
25. ERA5 — ECMWF Reanalysis Dataset
26. NASA POWER — Data Access Viewer
27. ISRO Solar Energy Calculator
28. Global Solar Atlas
29. Abdellatif (2025) — PCM Modeling
30. Majdi (2025) — Time Series Preprocessing

---

*Thank You*