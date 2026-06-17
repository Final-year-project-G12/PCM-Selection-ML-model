# Experimental Validation and Enhanced Thermal Prediction of a Shallow Solar Pond Using Artificial Neural Network–Based Model Predictive Control for Real-Time Optimization Under Multiple Heat Extraction Modes

**Authors:** Abdelkrim Terfai, Younes Chiba, Mounir Zirari, Mohamed Najib Bouaziz  
**Year:** 2025  
**Journal/Conference:** Unconventional Resources, Vol. 8, Article 100240  
**DOI:** https://doi.org/10.1016/j.uncres.2025.100240  
**IEEE Citation:** A. Terfai et al., "Experimental validation and enhanced thermal prediction of a shallow solar pond using artificial neural network–based model predictive control for real-time optimization under multiple heat extraction modes," Unconv. Resour., vol. 8, p. 100240, 2025, doi: 10.1016/j.uncres.2025.100240.

---

## 1. One-Line Summary
This paper experimentally compares direct, open-cycle, and closed-cycle heat extraction from a custom shallow solar pond in Algeria, trains a Bayesian-regularized ANN (\(R^2 = 0.99919\)) on Arduino/DS18B20 data, and integrates it with MPC on a QR30E pump to cut outlet-temperature tracking error by **52.2%** (MAE **1.42 °C** vs **2.97 °C**).

---

## 2. Problem Being Solved
- Shallow solar ponds (SSPs) can store solar heat for water heating and industrial use, but thermal performance depends strongly on **heat extraction mode** (direct drain, open circulation, closed loop with storage), which is rarely compared under **identical** clear-sky conditions.
- Nonlinear, transient SSP dynamics (double glazing, shallow water mass, heat exchanger coupling) are difficult to capture with fixed-parameter analytical models alone.
- Real-time regulation of fluid flow under variable solar irradiance is needed to stabilize outlet temperature and reduce pump energy use—beyond static ANN prediction without control.
- Lack of an integrated **experimental + data-driven model + predictive control** pipeline validated on instrumented hardware for closed-cycle SSP operation (identified as the most stable mode).

---

## 3. Key Contributions
1. Side-by-side **experimental campaign** (August 15–17, clear sky, Tablat/Medea, Algeria) of **direct**, **open-cycle**, and **closed-cycle** extraction on one **1 m²** SSP (**~60 L**, double-glazed, insulated, PVC serpentine exchanger).
2. Systematic evaluation of **14 ANN configurations**; optimal model: **trainbr**, **2 hidden layers**, **15 neurons** — \(R^2 = 0.99919\), RMSE **32.7580%**, MRE **0.62%** across \(T_{C1}\), \(T_{C2}\), \(T_{wp}\), \(T_p\), \(T_{fo}\), \(T_{wt}\).
3. Demonstration that **closed-cycle** mode achieves highest/stablest pond and outlet temperatures with minimal convective/evaporative losses vs open and direct modes.
4. Hybrid **ANN–MPC** framework adjusting **mass flow rate** \(\dot{m}\) on **QR30E** pump in closed cycle; MPC correction Eq. **(5)** with optimized \(\alpha\), \(\beta\).
5. Quantified control improvement: outlet MAE **1.42 °C** (MPC-corrected) vs **2.97 °C** (ANN-only), **>50%** error reduction; peak \(T_{fo}\) **52.1 °C** (MPC) vs **49.3 °C** (ANN).

---

## 4. Methodology
### 4a. System / Experiment Setup
- **SSP geometry:** Galvanized sheet pond **0.76 × 1.30 m**, depth **0.06 m**, capacity **~60 L**; black bottom; **0.04 m** polystyrene insulation (\(k \approx 0.03\) W/m·K).
- **Glazing:** Two glass panels (**0.003 m** each), **0.03 m** air gap (\(\tau_g = 0.90\), \(\varepsilon_g = 0.90\), \(\alpha_g = 0.05\)).
- **Heat extraction:** Transparent **PVC** tube exchanger (**10 m** length, **0.008 m** ID, **0.001 m** wall, \(\lambda_{PVC} \approx 0.19\) W/m·K); **QR30E** brushless pump (max **240 L/h**); **20 L** insulated storage tank (closed cycle).
- **Working fluid:** Tap water (\(k = 0.6\) W/m·K, \(C_p = 4180\) J/kg·K, \(\rho = 1000\) kg/m³).
- **Scenarios (one per day):** (1) **Direct** — drain heated pond water (Aug 15); (2) **Open cycle** — in-pond exchanger, continuous flow (Aug 16); (3) **Closed cycle** — sealed loop pond ↔ tank with second tank exchanger (Aug 17).
- **DAQ:** **Arduino UNO**; **DHT22** (\(T_a\), RH); **seven DS18B20** probes at glass (upper/lower), absorber, pond water, HX inlet/outlet, tank; **1 min** logging **07:00–19:00** (~**720 points/day**, **~2160** total).
- **Solar input:** **Bird & Hulstrom** clear-sky model for **Tablat, Medea, Algeria** coordinates.

### 4b. Mathematical Models & Equations
**ANN performance metrics:**

- \(SSE = \sum_{i=1}^{n}(e_i - p_i)^2\) — **(1)**  
  (\(e_i\) experimental, \(p_i\) predicted, \(n\) samples)

- \(RMSE\ (\%) = \sqrt{SSE/n} \times 100\) — **(2)**

- \(R^2 = 1 - SSE / \sum_{i=1}^{n} p_i^2\) — **(3)**

- \(MRE\ (\%) = \frac{1}{n}\sum_{i=1}^{n} \left| \frac{e_i - p_i}{e_i} \right| \times 100\) — **(4)**

**MPC temperature correction (closed cycle):**

- \(T_{fo,corr} = T_{fo,pred} + \alpha \cdot \tanh\left(\beta(\dot{m} - 0.01)\right)\) — **(5)**  
  (\(\dot{m}\) in kg/s; **0.01** ≈ minimum QR30E operational flow; \(\alpha\) = gain, \(\beta\) = sensitivity)

**Offline identification of \(\alpha\), \(\beta\):** minimize MSE between \(T_{fo,set}\) and \(T_{fo,pred}\) using experimental \(\dot{m}\) profile and **fminsearch** — reported MSE **0.98907** after tuning.

*No enthalpy-porosity or PCM phase-change equations—SSP stores sensible heat in water.*

### 4c. Algorithm / Control Method Steps
**ANN training:**
1. Preprocess inputs: normalize \(T_a\), Hum, \(I_T\), Time, \(A\), depth, insulation thickness, wind; add mode-specific \(T_{fi}\), \(\dot{m}\) for open/closed cycles.
2. Remove outliers via Z-score (threshold **±3σ**); none excluded.
3. Split data **70% train / 15% validation / 15% test** (~**2160** points total).
4. Train **14** MLP variants: algorithms **trainbr**, **trainlm**, **trainbfg**, **trainscg**; **1–3** hidden layers; **5–20** neurons per layer.
5. Select model with lowest RMSE, highest \(R^2\) → **trainbr**, **2×15** hidden neurons.

**ANN–MPC real-time loop (closed cycle, §7):**
1. FNN predicts \(T_{fo,pred}\) from current inputs and history.
2. MPC simulates candidate \(\dot{m}\) values; applies Eq. **(5)** to estimate \(T_{fo,corr}\).
3. Choose \(\dot{m}\) minimizing \(|T_{fo,corr} - T_{fo,set}|\) while respecting pump flow limits.
4. Apply adjusted \(\dot{m}\) to QR30E; repeat each control step under measured/estimated \(I_T\).
5. Compare MPC-controlled vs ANN-training \(\dot{m}\) profiles and outlet temperatures (Figs. 17–18).

**Identified MPC parameters:** \(\alpha = -22.93619\), \(\beta = 33.23245\).

### 4d. Data Sources & Dataset Details
| Source | Variables | Resolution | Scope | Period / size |
|--------|-----------|------------|-------|----------------|
| **On-site SSP experiment** | \(T_{C1}, T_{C2}, T_{wp}, T_p, T_{fi}, T_{fo}, T_{wt}, T_a\), RH | **1 min** | Tablat, Medea, **Algeria** | **Aug 15–17** (clear sky), **07:00–19:00** each day |
| **Bird & Hulstrom [22]** | Solar radiation \(I_T\) | Modeled clear-sky | Same coordinates | Training/reference days; Nov 17 reduced-irradiance MPC test |
| **ANN dataset** | All above + geometry (\(A\), depth, insulation), wind | 1 min | Three extraction modes | **~2160** points; **70/15/15** split |

### 4e. Validation Method
- **Experimental cross-mode comparison** under matched clear-sky days (Aug 15–17).
- **ANN vs measured temperatures:** training scatter \(R^2 = 0.99919\); mode-wise MRE on \(T_{C2}\) **0.84–0.91%**; on \(T_{C1}\) **0.42–0.48%**; on \(T_{wp}\) **0.46–0.54%**; on \(T_p\) **0.42–0.52%**; on \(T_{fo}\) **0.63–0.66%**; on \(T_{wt}\) **0.68%** (closed cycle).
- **Sensor uncertainty (Table 2):** DHT22 **±0.5 °C**, **±2% RH**; DS18B20 **±0.5 °C** (−10 to +85 °C operating range cited).
- **MPC validation:** MAE vs \(T_{fo,set}\) — **1.42 °C** (MPC) vs **2.97 °C** (ANN-only), **52.2%** reduction; parameter fit MSE **0.98907** for \(\alpha,\beta\) optimization (Fig. 15).
- **No CFD benchmark;** future work cites need for **hardware-in-the-loop** validation beyond current bench setup.

---

## 5. PCM Details (if applicable)
N/A — this paper does not study phase change materials. The shallow solar pond stores **sensible heat in water** (~60 L); thermal buffering in closed cycle uses an **insulated water storage tank**, not PCM (Rubitherm/PLUSS-type latent storage is out of scope).

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** **MLP ANN** (Bayesian Regularization **trainbr** best); **Model Predictive Control (MPC)** with ANN as plant predictor; offline **fminsearch** for \(\alpha,\beta\) in Eq. **(5)**.
- **Input features / state space:** \(T_a\), Hum, \(I_T\), Time, SSP area \(A\), depth, insulation thickness, wind speed; plus \(T_{fi}\), \(\dot{m}\) for open/closed modes.
- **Output / action space:** **Outputs:** \(T_{C1}, T_{C2}, T_{wp}, T_p, T_{fo}, T_{wt}\). **Control action:** mass flow rate \(\dot{m}\) (kg/s), MPC-adjusted around **~9×10⁻³ kg/s** morning baseline.
- **Model architecture:** **2 hidden layers**, **15 neurons** each; feedforward MLP (Fig. 5).
- **Hyperparameters:** **trainbr** selected over trainlm, trainbfg, trainscg; best epoch **870** (Table 3 row: 2 layers, 15 neurons); data split **70/15/15**; Z-score outlier threshold **±3σ** (no removals).
- **Training data size:** **~2160** samples (3 days × 12 h × 60 min).
- **Hardware used for training:** N/A — MATLAB ANN toolbox implied; acquisition on **Arduino UNO**.
- **Performance metrics:** \(R^2 = 0.99919\); RMSE **32.7580%**; MRE **0.62%** (best ANN); MPC outlet MAE **1.42 °C** vs **2.97 °C** without MPC (**52.2%** lower).

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **Bird & Hulstrom** simplified clear-sky model [22] for **direct and diffuse insolation** on horizontal surface; **measured** \(T_a\), RH via DHT22; irradiance cross-checked against experimental thermal response (peak **909.82 W/m²** at **12:00 h** on test day in Fig. 13).
- **Variables used:** \(I_T\) (W/m²), \(T_a\), Hum, wind speed \(W_{speed}\), time of day.
- **Geographic scope:** **Tablat, Medea, Algeria** (University of Medea experimental site).
- **Temporal resolution:** **1 min** measurements; **07:00–19:00** daily window.
- **Time period covered:** Primary campaign **August 15–17** (clear sky); additional MPC comparison under **reduced irradiance** (**November 17**, dips below **500 W/m²** between 09:30–16:00) vs clear-sky reference from **August 17** (~**950 W/m²** peak in setpoint curve).
- **Clear-sky index / derived metrics:** Clear-sky model used for \(I_T\) estimation; explicit clearness index \(k_t\) **not** reported.

---

## 8. Key Results & Numbers
- Optimal ANN: **trainbr**, **2** hidden layers, **15** neurons — \(R^2 = 0.99919\), RMSE **32.7580%**, MRE **0.62%**, SSE **463.5746**, training time **93 s**, best epoch **870** (Table 3).
- **14** ANN variants tested; worst trainscg (3 layers, 20 neurons): RMSE **58.2566%**, MRE **1.05%**, \(R^2 = 0.9998\).
- **Solar peak (Fig. 13):** **909.82 W/m²** at **12:00 h**; thermal lag between radiation peak and fluid temperature peak due to water inertia.
- **Upper glass \(T_{C2}\) peaks:** Open **48.63 °C** (13:00–14:00); closed **47.94 °C**; direct **42.63 °C** (noon) — direct mode lowest peak.
- **Lower glass \(T_{C1}\) peaks:** Open **58.38 °C**; closed **63.00 °C** (~15:20); direct **66.87 °C** at **16:00 h**.
- **Pond water \(T_{wp}\) peaks:** Open plateau **53.94 °C** (13:00–15:00); closed **62.88 °C** (~16:00); direct **66.44 °C** at **16:00 h**.
- **Absorber \(T_p\) peaks:** Open **53.94 °C**; closed **62.02 °C**; direct **66.36 °C** at **16:00 h**.
- **Outlet fluid \(T_{fo}\):** Open stabilizes **~51.5 °C** (13:00–16:00); closed peak **58.88 °C** at **16:00 h**; max ANN deviation **~0.29 °C** (open), MRE **0.63–0.66%**.
- **Storage tank \(T_{wt}\) (closed):** **25 °C** → max **53.56 °C** ~**17:00 h**; MRE **0.68%**.
- **MPC vs ANN-only outlet:** max **52.1 °C** vs **49.3 °C** (**+2.8 °C**); min **24.9 °C** vs **24.0 °C**; MAE **1.42 °C** vs **2.97 °C** (**52.2%** error reduction).
- **MPC \(\dot{m}\):** starts **~9×10⁻³ kg/s**, lowers at midday, increases late afternoon (Fig. 17).
- **MPC tuning:** \(\alpha = -22.93619\), \(\beta = 33.23245\); correction fit MSE **0.98907**.

---

## 9. Baseline Comparison
- **Baseline method(s):** **Direct** and **open-cycle** heat extraction vs **closed-cycle**; **ANN prediction without MPC** vs **ANN–MPC** for outlet tracking; **14 ANN training algorithms/architectures** with non-optimal models as ML baselines.
- **Proposed method:** **Closed-cycle SSP** + **trainbr ANN (2×15)** + **MPC flow-rate optimization** (Eq. **(5)**).
- **Improvement margin:** Closed cycle — higher \(T_{wp}\), \(T_{fo}\), \(T_{wt}\) and longer evening retention vs open/direct (qualitative + peak deltas up to **~12 °C** on \(T_{wp}\) vs open); MPC — **52.2%** lower MAE, **+2.8 °C** higher peak \(T_{fo}\) vs ANN-only under same reduced-irradiance day.
- **Conditions of comparison:** Same SSP hardware; clear-sky days for mode comparison (Aug 15–17); MPC test uses **Nov 17** low-irradiance day vs **Aug 17** clear-sky setpoint profile.

---

## 10. Hardware / Experimental Setup (if applicable)
- **Physical components:** Custom **SSP** (0.76×1.30×0.06 m, **~60 L**); double glass cover; polystyrene insulation; black galvanized absorber (\(\alpha_p = 0.95\)); **PVC** serpentine HX (**10 m**, **8 mm** ID); **20 L** insulated tank; **QR30E** brushless circulation pump.
- **Sensor specs:** **DHT22** — \(T_a\): **−40 to +80 °C**, **±0.5 °C**; RH: **0–100%**, **±2%**; **DS18B20** (×7 waterproof) — **−55 to +125 °C** spec, **±0.5 °C** (−10 to +85 °C in table); **1 min** logging.
- **Embedded/compute platform:** **Arduino UNO** for DAQ; ANN/MPC in **MATLAB** (implied for training and MPC design).
- **Test environment:** Outdoor experimental setup, **Tablat, Medea, Algeria**; **clear-sky** conditions Aug 15–17.
- **Test duration:** **12 h/day** (07:00–19:00) × **3 days** for mode comparison; additional **Nov 17** MPC validation under variable/cloud-affected irradiance.

---

## 11. Limitations Acknowledged by Authors
- Future work will extend the strategy to **multi-objective optimization** and **real-world hardware-in-the-loop validation** — implying current MPC integration is not yet fully validated on embedded HIL hardware.
- Heat exchanger effectiveness \(\varepsilon\) was **not directly measured**; performance inferred from material properties and temperature rise only.
- The three extraction modes were run on **different consecutive days** (not simultaneous), though authors state clear-sky consistency across Aug 15–17.
- MPC comparison (Fig. 18) uses **November 17** reduced irradiance against an **August 17** clear-sky setpoint — a deliberate stress test but not identical weather ensembles.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Relevant.** ANN–MPC continuously adjusts **\(\dot{m}\)** to track \(T_{fo,set}\), cutting MAE by **52.2%** vs open-loop ANN—direct precedent for your **PPO/DDPG** pump/valve control, though on SSP water rather than PCM latent storage.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Demonstrates **Arduino + DS18B20 + pump** experimental loop with ML control—the same sensor/actuator class as your FYP—but **no PCM**, no Raspberry Pi/ESP32, and ANN/MPC runs off-board in MATLAB.
- **RG3 (Poor alignment with household demand patterns):** **Not Relevant.** No residential hot-water draw schedule; objectives are outlet temperature tracking and pump energy, not morning/evening demand peaks (Coimbatore/Jaisalmer/Kochi profiles).
- **RG4 (Limited real-world experimental validation):** **Highly relevant (positive example).** Full **bench-scale outdoor experiment** with **7** temperature nodes and **3** operating modes—supports your claim that PCM-SWH literature needs more work like this; paper itself is **SSP water**, not PCM-SWH.
- **RG5 (No predictive optimization under climatic uncertainty):** **Partially relevant.** MPC uses ANN forward prediction and responds to **irradiance dips** (Nov 17, **<500 W/m²**); solar input from **Bird–Hulstrom** model, not **ERA5/NASA POWER** forecasting—your XGBoost irradiance + PPO stack goes further.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(RMSE = \sqrt{SSE/n}\times 100\), \(R^2 = 1 - SSE/\sum p_i^2\) **(2)–(3)** | ANN regression accuracy | Benchmark XGBoost/Grey-box vs DS18B20 labels |
| \(T_{fo,corr} = T_{fo,pred} + \alpha\tanh(\beta(\dot{m}-0.01))\) **(5)** | Flow-dependent outlet temperature correction | Simplified surrogate for valve/pump action before full RL env |
| MAE vs setpoint (reported **1.42 °C** vs **2.97 °C**) | Control tracking quality | RL reward: minimize \(\|T_w - T_{set}\|\) under forecast GHI |
| Sensible heat dynamics (implicit \(T_{wp}\), lag vs \(I_T\) peak) | Thermal inertia of storage medium | Analogous PCM melting/charging lag vs pyranometer peak |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **A.A. El-Sebaii et al., "Thermal performance of shallow solar pond under open and closed cycle modes of heat extraction," *Sol. Energy*, 2013 [27]** — Relevant because: Foundational **closed vs open cycle SSP** performance data aligned with this paper’s best-mode conclusion.
2. **M. Mahfuz et al., "Performance investigation of TES with PCM for solar water heating," *Int. Commun. Heat Mass Transf.*, 2014 [26]** — Relevant because: Bridges **PCM storage** with **solar water heating**—the latent-storage layer this SSP paper lacks.
3. **H.K. Ghritlahre & R.K. Prasad, "Application of ANN to predict solar collector systems — A review," *Renew. Sustain. Energy Rev.*, 2018 [14]** — Relevant because: Review of **ANN for solar thermal** prediction cited in their methodology framing.
4. **A.H. Elsheikh et al., "Modeling of solar energy systems using ANN: comprehensive review," *Sol. Energy*, 2019 [20]** — Relevant because: Broad **ANN + solar system** reference for your literature review ML subsection.
5. **P.K. Bansal et al., "Effect of heat exchanger on the performance of a shallow solar pond water heater," *Energy Convers. Manag.*, 1984 [6]** — Relevant because: Classic **HX integration in SSP-SWH** geometry precedent for collector–storage coupling.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Real-time ML control on solar thermal hardware | "Terfai et al. cut outlet temperature MAE by **52.2%** (**1.42 °C** vs **2.97 °C**) using ANN–MPC on a closed-loop shallow solar pond with **DS18B20** sensing." |
| II. Literature Review | Experimental ML-control benchmark (non-PCM) | Method: **trainbr ANN + MPC** on \(\dot{m}\); Key insight: closed-cycle **62.88 °C** peak \(T_{wp}\) vs open **53.94 °C** |
| III. Methodology | Sensor stack + metrics | **7× DS18B20**, **±0.5 °C**; Eqs. **(2)–(4)** for model validation; Eq. **(5)** as lightweight control surrogate |
| IV. Dataset & Setup | Embedded DAQ pattern | **Arduino UNO**, **1 min** sampling, **07:00–19:00** — parallel to your RPi/ESP32 logging design |
| V. Results | Control improvement target | Exceed **52%** MAE reduction or beat **\(R^2 = 0.99919\)** thermal prediction when adding **PCM latent state** to the plant model |
