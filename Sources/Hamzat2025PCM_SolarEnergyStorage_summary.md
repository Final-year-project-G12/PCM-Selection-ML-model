# Phase Change Materials in Solar Energy Storage: Recent Progress, Environmental Impact, Challenges, and Perspectives

**Authors:** Abdulhammed K. Hamzat, Adewale Hammed Pasanaje, Mayowa I. Omisanya, Ahmet Z. Sahin, Adesewa O. Maselugbo, Ibrahim A. Adediran, Lateef Owolabi Mudashiru, Eylem Asmatulu, Oluremilekun Ropo Oyetunji, Ramazan Asmatulu  
**Year:** 2025  
**Journal/Conference:** Journal of Energy Storage, Vol. 114, Article 115762  
**DOI:** https://doi.org/10.1016/j.est.2025.115762  
**IEEE Citation:** A. K. Hamzat et al., "Phase change materials in solar energy storage: Recent progress, environmental impact, challenges, and perspectives," J. Energy Storage, vol. 114, p. 115762, 2025, doi: 10.1016/j.est.2025.115762.

---

## 1. One-Line Summary
This review synthesizes recent PCM-based solar thermal storage research and shows that heat-transfer enhancement (especially nano-dispersion and hybrid design/ML optimization) can raise performance substantially, with reported gains up to **73%**, while also detailing economic, environmental, and deployment constraints.

---

## 2. Problem Being Solved
- Conventional TES often suffers from low PCM thermal conductivity, slow charging/discharging, leakage/supercooling, and inconsistent material performance reporting across studies.
- Solar-integrated PCM systems need better technical optimization across melting/solidification, exergy, and cost, especially under variable weather and operating conditions.
- Environmental and economic claims are fragmented; lifecycle, emissions, and payback evidence is not consistently standardized across PCM technologies.
- AI/ML methods are promising but still early-stage for PCM-TES design/control, with limited multi-objective and real-world robust implementations.

---

## 3. Key Contributions
1. Broad review of PCM enhancement pathways for solar TES: fins/extended surfaces, heat pipes, cascaded PCMs, encapsulation, porous media, and nanoparticle-doped PCMs.
2. Quantitative synthesis of nano-PCM effects with many reported conductivity/charging/discharging improvements and comparison across materials and concentrations.
3. Dedicated review of AI/ML for PCM-TES (ANN, SVM, GPR, ensemble learning, PINN, DRL), including concrete metrics (R², MSE, MAE, MAPE) from published studies.
4. Integrated techno-economic and environmental discussion (LCOE/LCOS/payback, LCA, CO2 mitigation, recyclability, sustainability trade-offs).
5. Challenges/future directions section covering test standardization, data reliability, ML limitations, and sustainability-focused material development.

---

## 4. Methodology
### 4a. System / Experiment Setup
N/A — this is a **review article** (45 pages), not a single experimental rig.  
It compiles results across solar collectors, SWH, PV/T-PCM, heat pump coupling, greenhouse heating, building envelopes, and industrial heat recovery systems, including both numerical and experimental literature.

### 4b. Mathematical Models & Equations
N/A — the paper is primarily a narrative/quantitative review and does not present one new, unified governing model with a consistent equation set of its own.  
It reports metrics/equations as used in cited studies (e.g., MAPE, MAE, RMSE, R², LCOE/LCOS, exergy and energy efficiencies).

### 4c. Algorithm / Control Method Steps
Review-level ML/control pipeline extracted from surveyed studies:
1. Build datasets from experiments/simulations (thermal conductivity, phase fraction, outlet temperature, exergy, load, weather, geometry variables).
2. Train model families: ANN/FFNN/LSTM, SVM, KNN, CART, MARS, GPR, ensemble frameworks, PINN, and DRL controllers.
3. Optimize hyperparameters (examples include Sobol sampling + ANN tuning, Bayesian tuning, and ensemble stacking).
4. Validate against measured/CFD data using R², MAE, MSE, RMSE, and MAPE.
5. Deploy predictions/control for PCM charging/discharging, outlet-temperature tracking, and cost/exergy optimization.

### 4d. Data Sources & Dataset Details
- Secondary data from published numerical and experimental literature on PCM-TES and solar systems.
- Includes studies on solar water heating, PV/T-PCM, greenhouse heating, building HVAC, industrial waste heat recovery, and thermal batteries.
- ML studies include datasets like:
  - ~**911** points from **25** studies (for thermal conductivity prediction in one cited ANN/CART/MARS study).
  - Time-series operational datasets for TES charging/discharging and outlet-temperature forecasting in other cited works.
- Geographic coverage is multi-country (reviewed literature includes Central Europe, China, Iran, UK, etc.).

### 4e. Validation Method
- Review compiles validation metrics from surveyed studies, including:
  - **R² = 0.9999** (ANN for hybrid solar TES prediction in one cited study).
  - **R² = 0.97951** (group method neural model for thermal efficiency prediction in one cited collector-PCM study).
  - Ensemble LHTES model: MAPE improvement up to **7.82%** (charging) and **16.43%** (initial discharging).
  - Industrial PCM heat recovery model: max relative error **5.47%**.
  - PINN-based DCEE-linked TES predictions: deviation within **±7.8%**.

---

## 5. PCM Details (if applicable)
- **Materials tested:** Review covers organic/inorganic/eutectic/composite PCMs; examples include paraffin wax (PW), RT-series paraffins (RT35, RT44HC, RT50, RT54HC), OM65, hydrated salts, sodium acetate trihydrate, erythritol composites, myristic acid systems, etc.
- **Melting temperature range:** Study coverage spans low/mid/high application bands; reported examples include **26.61–27.12 °C** (one methyl palmitate composite case), **29 °C** transition in building control context, and high-temperature applications up to **885 °C** in waste-heat TES context.
- **Latent heat:** Reported examples include **96.1–96.7 J/g** (one CNF composite case) and strong dependence on nanoparticle loading/composition.
- **Thermal conductivity:** Reported improvements range widely; examples include baseline-to-enhanced increases of **53.58%**, **59.5%**, **71.5%**, **72.2%**, **86.36%**, **87.39%**, **109.2%**, **112.5%**, and up to **165.56%** in reviewed cases.
- **Specific heat (solid/liquid):** Not one fixed value (review spans many PCMs); one cited nano-salt case reported **19–24%** specific heat increase.
- **Density:** Material-specific and varies by system; no single universal density reported for the review.
- **Performance metrics reported:** Melting time reduction, charging/discharging rate, thermal/exergy efficiency, COP, LCOE/LCOS/payback, compressor runtime, fuel savings, and CO2 reduction.

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** ANN, FFNN, LSTM-BP, SVM, KNN, CART, MARS, GPR, Huber regressor, SGD, ensemble learning, PINN, DRL.
- **Input features / state space:** Depending on study: PCM type/fraction/thickness, nanoparticle type/concentration, flow rate, geometry (fin/porosity/tube), heat flux, weather/temperature, operating schedule.
- **Output / action space:** Thermal conductivity, melting/solidification behavior, outlet temperature/enthalpy, exergy performance, charging/discharging dynamics, operational control actions (in DRL studies).
- **Model architecture:** Includes feedforward ANN, multilayer perceptron, NARX, ensemble stacking frameworks, PINN hybrids, and DRL policy models.
- **Hyperparameters:** Reported examples include Sobol-sampled ANN tuning and Bayesian-tuned deep models (specific full sets vary by cited paper).
- **Training data size:** Example: **>911** samples from **25** studies in one conductivity prediction study.
- **Hardware used for training:** Not standardized in the review (depends on cited papers).
- **Performance metrics:** R², RMSE, MAE, MSE, MAPE, confidence/error levels, and control-cost reductions.

*If not applicable: N/A — reason*

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** Review-level synthesis of many papers; includes studies using local meteorological conditions and seasonal contexts (e.g., Central Europe, arid cities, greenhouse operations).
- **Variables used:** Solar radiation, ambient temperature, seasonal temperatures, thermal load, system temperatures, and operational demand profiles (varies by cited study).
- **Geographic scope:** Multi-region/global literature (examples include China, Iran, UK, Central Europe, Canada).
- **Temporal resolution:** Varies by source paper (dynamic/transient and seasonal analyses are included).
- **Time period covered:** Up to 2024/2025 literature in this review.
- **Clear-sky index / derived metrics:** Not consistently reported as a unified metric in the review.

---

## 8. Key Results & Numbers
- The review reports system performance improvements **up to 73%** from PCM enhancement strategies (abstract claim).
- Nano-PCM development in reviewed studies shows **25.6%** charging and **23.9%** discharging improvement versus conventional PCM systems (abstract claim).
- One ANN-based hybrid solar TES study achieved **R² = 0.9999** after hyperparameter tuning.
- Another neural model for PCM-based collector performance reported **R² = 0.97951**.
- A reviewed conductivity-prediction study used **>911** data points from **25** studies and achieved top ANN **R² = 0.96** (vs MARS/CART at **0.93**).
- Ensemble LHTES ML modeling reported MAPE improvement up to **7.82%** (charging) and **16.43%** (initial discharging), with error spread reduction up to **25.6%**.
- A DRL-controlled seasonal sorption storage case reported operational cost reductions of **28%** (60 winter days) and **13%** (120 winter days) over rule-based control.
- A PINN-based coupled DCEE/TES study kept prediction discrepancy within **±7.8%**.
- Greenhouse HRS + PCM-HRS integration improved mean energy efficiency by **33%** and **40%**, and exergy efficiency by **127%** and **263%**, respectively.
- Same greenhouse case reduced fuel consumption by **19%** (plain HRS) and **48%** (PCM-HRS), with payback around **3 months** and **4 months**.
- In an arid-climate building-envelope study, PCM integration reduced HVAC energy by **55.47%**, **53.89%**, **58.86%**, and **53.57%** in one scenario (Dubai/Jeddah/Kuwait/Lahore).
- Another scenario from the same study showed smaller reductions: **2.6%**, **2.03%**, **1.99%**, **5.6%**.
- Reported CO2 emission reductions in that study reached **56.27%**, **44.81%**, **45.27%**, and **58.5%**.
- A heat-pump TES study cited **75%** PCM integration reducing required tank volume by about **3×** versus water-only storage.
- SAHP + TES optimization found minimum tank volume **1300 L**, optimal PCM filling ratio **85%**, and compressor energy reduction **27.2%**.
- A 1 MW PVT-PPCM analysis reported annual output **1920 MWh** and about **30 tons/year** CO2 reduction.

---

## 9. Baseline Comparison
- **Baseline method(s):** Conventional PCM systems, rule-based control, plain HRS (without PCM), water-only storage tanks, and non-PCM HVAC/solar benchmarks in cited studies.
- **Proposed method:** Review supports enhanced PCM strategies (nano-PCMs, cascaded/encapsulated/composite PCMs, AI-optimized operation).
- **Improvement margin:** Reported margins include up to **73%** performance gain, **25.6%/23.9%** charge/discharge improvement, **28%** DRL cost reduction, and **27.2%** compressor-energy reduction in SAHP+PCM case.
- **Conditions of comparison:** Results are from heterogeneous studies with different climates, PCMs, geometries, and objectives; not one single uniform benchmark dataset.

*If no baseline comparison: N/A — [paper is a review / purely experimental / etc.]*

---

## 10. Hardware / Experimental Setup (if applicable)
- **Physical components:** Across reviewed studies: solar collectors, SWH tanks, PV/T modules, heat pumps, greenhouse heat exchangers, industrial heat-recovery units, fins/foam/encapsulation modules.
- **Sensor specs:** Not unified in this review; instrumentation depends on each cited study.
- **Embedded/compute platform:** AI models implemented in cited works; no single hardware platform reported by this review itself.
- **Test environment:** Includes simulation, lab-scale experiments, pilot/demonstration systems, building and industrial contexts.
- **Test duration:** Varies from transient charging tests to seasonal and lifecycle assessments.

*If simulation only: N/A — this paper is purely simulation/CFD-based.*

---

## 11. Limitations Acknowledged by Authors
- Authors explicitly note possible **veracity issues** in some surveyed findings and potential **literature coverage omissions**.
- They highlight inconsistency in reported nano-PCM thermophysical data and call for **standardized testing methods** and a reliable database.
- ML deployment is described as still in an **embryonic stage** for PCM-TES with limited multi-objective/system-level studies.
- Multi-variable optimization under uncertainty and algorithm selection remain unresolved due to computational complexity and data limitations.
- The paper calls for stronger encapsulation methods, optimized manufacturing, and full lifecycle sustainability studies for feasibility.

---

## 12. Direct Relevance to My Project
- **RG1 (No real-time adaptive control):** **Relevant.** The review cites DRL/ANN/PINN control and prediction studies with quantified gains (e.g., **28%** cost reduction), directly supporting adaptive control direction, though mostly outside domestic PCM-SWH prototypes.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially Relevant.** It documents AI + PCM integration trends and pilot studies, but most are not end-to-end embedded domestic systems with your exact hardware stack (RPi/ESP32/DS18B20/valve).
- **RG3 (Poor alignment with household demand patterns):** **Partially Relevant.** Some building/HVAC and TES scheduling studies address load dynamics; explicit household DHW draw-profile coupling is still limited.
- **RG4 (Limited real-world experimental validation):** **Relevant.** The review includes both simulation and experimental/pilot evidence, but repeatedly notes scarcity of standardized, long-term real-world validation across conditions.
- **RG5 (No predictive optimization under climatic uncertainty):** **Relevant.** Multiple ML studies incorporate seasonal/weather-sensitive optimization and forecasting, yet robust multi-climate predictive optimization remains a stated challenge.

---

## 13. Equations to Reuse or Adapt
| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| $Q = mL + mC_p\\Delta T$ | Total latent+sensible stored heat | Grey-box PCM tank energy model |
| $\\eta_{th}=\\frac{Q_{useful}}{A\\,G}$ | Collector/TES thermal efficiency | Compare control policies under same irradiance |
| $\\eta_{ex}=\\frac{Ex_{out}}{Ex_{in}}$ | Exergy efficiency | Second-law KPI for PCM charging quality |
| $\\mathrm{MAPE}=\\frac{100}{n}\\sum\\left|\\frac{y-\\hat y}{y}\\right|$ | ML forecast/control model error | Evaluate XGBoost/ANN thermal predictor |
| $\\mathrm{RMSE}=\\sqrt{\\frac{1}{n}\\sum(y-\\hat y)^2}$ | Prediction error magnitude | Model selection and validation metric |
| $\\mathrm{LCOE}=\\frac{\\sum_t \\frac{C_t}{(1+r)^t}}{\\sum_t \\frac{E_t}{(1+r)^t}}$ | Lifecycle electricity cost | Techno-economic comparison of PCM control strategies |

*If no reusable equations: N/A — [reason]*

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **D. E. Douvi et al., "Phase change materials in solar domestic hot water systems: a review," Int J Thermofluids, 2021** — Relevant because it is directly about PCM integration in **solar DHW/SWH** systems.
2. **B. Kanimozhi et al., "Thermal energy storage system operating with phase change materials for solar water heating applications: DOE modelling," Appl. Therm. Eng., 2017** — Relevant because it targets **SWH + PCM modeling** with quantified control/design outputs.
3. **A. Crespo et al., "Optimal control of a solar-driven seasonal sorption storage system through deep reinforcement learning," Appl. Therm. Eng., 2024** — Relevant because it provides **DRL-based thermal storage control** evidence.
4. **L. Yang et al., "Thermophysical properties and applications of nano-enhanced PCMs: an update review," Energy Convers. Manag., 2020** — Relevant because it supports **nano-PCM property enhancement** decisions for better charging/discharging.
5. **R. Aridi and A. Yehya, "Review on the sustainability of phase-change materials used in buildings," Energy Convers. Manag. X, 2022** — Relevant because it supports **environmental and lifecycle** sections for PCM selection.

---

## 15. Suggested Use in My IEEE Paper
| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Motivation for advanced PCM-TES | "Recent review evidence reports PCM enhancement routes yielding performance gains up to **73%** in solar TES contexts." |
| II. Literature Review | AI in PCM-TES summary entry | Method: ANN/ensemble/DRL/PINN; Key insight: DRL control showed up to **28%** operational-cost reduction in seasonal TES case studies. |
| III. Methodology | Modeling + control metrics | Use RMSE/MAPE/R² and exergy-based KPIs, mirroring reported PCM-TES ML validation framework. |
| IV. Dataset & Setup | Climate/load sensitivity argument | "PCM selection and control are strongly dependent on ambient temperature, solar radiation, and location-specific conditions." |
| V. Results | Baseline-comparison anchor | Report your controller vs rule-based with same style as reviewed literature (e.g., cost reduction %, charging/discharging improvement %, exergy gain %). |

---
