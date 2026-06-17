# Modeling of Water-PCM Solar Thermal Storage System for Domestic Hot Water Application Using Artificial Neural Networks

**Authors:** A.O. Eldokaishi, M.Y. Abdelsalam, M.M. Kamal, H.A. Abotaleb  
**Year:** 2022  
**Journal/Conference:** Applied Thermal Engineering, Vol. 204, Article 118009  
**DOI:** https://doi.org/10.1016/j.applthermaleng.2021.118009  
**IEEE Citation:** A. O. Eldokaishi, M. Y. Abdelsalam, M. M. Kamal, and H. A. Abotaleb, "Modeling of water-PCM solar thermal storage system for domestic hot water application using artificial neural networks," Appl. Therm. Eng., vol. 204, p. 118009, 2022, doi: 10.1016/j.applthermaleng.2021.118009.

---

## 1. One-Line Summary
This paper trains a Keras/TensorFlow feed-forward ANN surrogate (R² up to **0.9999**, **~10⁵×** faster than physics-based simulation) on an experimentally validated water–PCM SDHW model to predict **solar fraction** and generate design maps for tank volume, PCM volume fraction, and melting temperature.

---

## 2. Problem Being Solved
- Transient, nonlinear PCM phase-change makes **annual or large-parameter-sweep** numerical SDHW simulation **computationally prohibitive** (e.g., **>120 h** for **84,480** cases), limiting comprehensive design guidelines.
- Literature lacks systematic study of **ANN applicability** specifically to **PCM-integrated solar thermal storage** with tuned sampling and hyperparameters.
- Optimal **PCM volume fraction**, **melting temperature**, and **tank volume** for hybrid sensible–latent tanks remain unclear without fast performance predictors.
- Engineers need **visual design maps** (solar fraction contours) relating collector area, tank size, PCM fraction, and \(T_p\) without running full transient models for every point.

---

## 3. Key Contributions
1. End-to-end framework: **experimentally validated** Abdelsalam et al. hybrid water–PCM tank model → training data → **Sobol / LHS / Monte Carlo** sampling comparison (**39 ANN models**) → hyperparameter optimization → **design maps**.
2. Demonstrates **Sobol sequence sampling** outperforms LHS and MC at low sample counts (**44%** and **16%** lower testing MAE vs LHS and MC, respectively).
3. Optimized **multi-ANN ensemble** (3 best models, outlier rejection, average of two): testing **MAE = 0.00114**, **RMSE = 0.001745**, **R² = 0.99990**, **max absolute error = 0.0203** on solar fraction.
4. **Polynomial regression** surrogate (Eq. **14**) for SF with **R² = 0.98481** — shown inferior to ANN (**MAE 0.02010** vs **0.00114**).
5. PCM–SDHW design insights: e.g., **+13%** solar fraction when \(V_f\) increases **0 → 0.5** in **90 L** tank at \(T_p = 28\,°\mathrm{C}\); **90 L** PCM tank matches **210 L** water-only tank SF; up to **57%** tank volume reduction possible with proper PCM selection.

---

## 4. Methodology
### 4a. System / Experiment Setup
**Underlying physics model (not run live in this paper):** Abdelsalam et al. [4,17] **hybrid thermal storage** — flat-plate solar collector charging loop + domestic load discharging loop.

**Tank / PCM geometry:**
- Stratified storage tank with **immersed coil HX**: bottom coil (collector charging), top coil (load discharge).
- **Cylindrical PCM modules**, **20 mm** diameter, installed **vertically** inside tank.
- **ON/OFF** circulation pump controlled by collector-to-tank-bottom temperature difference (\(\Delta T_{on}\), \(\Delta T_{off}\)); supply limited to **90 °C** to avoid boiling.
- Load side: **auxiliary heater** + **tempering valve** to maintain setpoint \(T_l\).

**Parameter ranges (Table 1):**
| Parameter | Range |
|-----------|--------|
| Collector area \(A_c\) | **1–8 m²** |
| Tank volume \(V_{st}\) | **50–240 L** |
| Load temperature \(T_l\) | **55–60 °C** |
| PCM melting \(T_p\) | **25–35 °C** |
| PCM volume fraction \(V_f\) | **0.0–0.7** |

**Boundary / operating data:**
- **Weather:** hourly solar irradiance and dry-bulb temperature — **typical spring day, Toronto, Canada** (`weather.gc.ca`).
- **Demand:** dispersed hot-water draw profile from Edwards et al. [19]; **8 L/min** draw rate; **189 L/day** total.

**ANN software:** Python; **Keras** + **TensorFlow**; feed-forward multilayer perceptron.

### 4b. Mathematical Models & Equations
**Collector pump control (ON/OFF hysteresis):**

- \(\Delta T_{off} \leq \dfrac{A_c \times F_R \times U_L}{\dot{m} \times C_w \times \Delta T_{on}}\) — **(1)**

**Collector heat removal factor:**

- \(F_R = \dfrac{\dot{m} C_w \left(1 - e^{-A_c U_L F' / (\dot{m} C_w)}\right)}{A_c U_L}\) — **(2)**

**Input normalization:**

- \(X' = \dfrac{X - \bar{X}}{\sigma}\) — **(3)**

**Loss functions:**

- \(\mathrm{MAE} = \dfrac{1}{n}\sum_{i=1}^{n}|y_i - y'_i|\) — **(4)**
- \(\mathrm{MSE} = \dfrac{1}{n}\sum_{i=1}^{n}(y_i - y'_i)^2\) — **(5)**
- \(\mathrm{RMSE} = \sqrt{\dfrac{1}{n}\sum_{i=1}^{n}(y_i - y'_i)^2}\) — **(6)**

**Solar fraction (ANN target output):**

- \(\mathrm{SF} = \dfrac{\text{Thermal energy delivered to load}}{\text{Total thermal demand}}\) — **(7)**

**Activation functions tested (examples):**

- ReLU: \(f(x) = \max(0,x)\) — **(8)**
- Sigmoid: \(f(x) = 1/(1+e^{-x})\) — **(9)** *(selected)*
- Softplus, tanh, SELU, ELU — **(10)–(13)**

**Regression surrogate for SF (polynomial in \(A_c, V_{st}, T_l, T_p, V_f\)):**

- \(\mathrm{SF} = -0.009248 A_c^2 + 0.000055 A_c V_{st} - 0.001532 A_c T_l + 0.000589 A_c T_p + 0.001477 A_c V_f - 0.000003 V_{st}^2 + 0.000004 V_{st} T_p - 0.000242 V_{st} V_f - 0.000068 T_p^2 + 0.002308 T_p V_f - 0.04611 V_f^2 + 0.21993 A_c + 0.000779 V_{st} + 0.000112 T_l + 0.00086 T_p - 0.01 V_f - 0.1908\) — **(14)**

*PCM phase-change inside modules is handled by the cited Abdelsalam et al. [17] immersed-coil + PCM model (enthalpy-based), not re-derived in this paper.*

### 4c. Algorithm / Control Method Steps
**ANN training workflow:**
1. Run validated numerical model over design space; each sample = **5 inputs** (\(A_c, V_{st}, T_l, T_p, V_f\)) + **1 output** (SF).
2. **Normalize** inputs with Eq. **(3)**.
3. **Sample** training set via **Monte Carlo**, **Latin hypercube (LHS)**, or **Sobol** sequences (up to **10,000** samples; **13** sample-count levels: 250, 500, …, 10000 → **39** models).
4. Build **feed-forward** ANN; initialize synaptic weights; train for multiple **epochs** minimizing **MAE/MSE/RMSE** with **Adam** optimizer.
5. Evaluate on **84,480** held-out test points (full factorial-style coverage, never seen in training).
6. **Hyperparameter tuning** (Sobol, **3,000** training samples): learning rate, hidden layers (**1–4**), neurons/layer (**30, 40, 50**), activation function.
7. **Multi-ANN prediction:** select 3 best models; drop outlier per point; average remaining two → **~10%** MAE and **~21%** max-error reduction vs single model.
8. Deploy optimized ANN to generate **SF contour maps** vs \(V_{st}\), \(V_f\), \(T_p\) at fixed \(A_c = 4\,\mathrm{m}^2\), \(T_l = 55\,°\mathrm{C}\).

**Optimized hyperparameters (Table 7):**
| Hyperparameter | Value |
|----------------|-------|
| ANN type | Feed-forward multi-layer |
| Input neurons | **5** |
| Output neurons | **1** |
| Hidden layers | **3** |
| Neurons per hidden layer | **50** |
| Optimizer | **Adam** |
| Learning rate | **0.005** |
| Activation | **Sigmoid** |

### 4d. Data Sources & Dataset Details
| Source | Variables | Resolution / scope | Period |
|--------|-----------|-------------------|--------|
| Abdelsalam et al. [4,17] numerical model | Tank temps, PCM state, SF, collector operation | Transient simulation per design point | Single-day Toronto spring + daily demand |
| Environment Canada weather (`weather.gc.ca`) | Solar irradiance, ambient \(T_a\) | **Hourly** | **One spring day**, **Toronto** |
| Edwards et al. [19] draw profile | Hot water flow rate | High-resolution demand; scaled to **189 L/day** | **24 h** cycle |
| ANN training sets | \(A_c, V_{st}, T_l, T_p, V_f\) → SF | Up to **10,000** training points (Sobol/LHS/MC) | Design-space sweep |
| ANN test set | Same 5 inputs → SF | **84,480** samples | Full studied range |

No ERA5, NASA POWER, or Indian city data used.

### 4e. Validation Method
- **Training data generator:** physics model **experimentally validated** in prior work [4] (direct vs indirect HX SDHW with PCM).
- **ANN validation:** held-out **84,480** test samples; metrics **MAE**, **RMSE**, **R²**, **max absolute error**.
- **Best multi-ANN:** **MAE = 0.00114**, **RMSE = 0.001745**, **R² = 0.99990**, **max error = 0.0203** (solar fraction scale 0–1).
- **Abstract peak claim:** **R² = 0.9999** after proper configuration.
- **Speed benchmark:** full test set — numerical model **>120 h** vs ANN **~5 s** (~**5 orders of magnitude** reduction).
- **80%** of ANN test points have **MAE < 2×10⁻³**; regression model **80%** with **MAE < 3.2×10⁻²**.

---

## 5. PCM Details (if applicable)
- **Materials tested:** **Capric-acid-like** organic PCM (properties from Abhat [33]); not Rubitherm RT or PLUSS OM grades.
- **Melting temperature range:** **25–35 °C** (\(T_p\) design sweep).
- **Latent heat:** **182 kJ/kg**
- **Thermal conductivity:** Not varied in this study (fixed in underlying [17] model); water conductivity **0.63 W/m·K** listed for tank water.
- **Specific heat (solid/liquid):** Water **\(C_w = 4.18\)** kJ/kg·K (collector fluid); PCM \(c_p\) embedded in referenced model.
- **Density:** PCM **\(\rho_p = 870\)** kg/m³; water **\(\rho_w = 993\)** kg/m³.
- **Performance metrics reported:** **Solar fraction (SF)**; SF improvements with PCM (**+13%**, **+5%** cases); **57%** tank volume reduction potential; literature cites **40%** tank volume reduction with PCM modules [4], **20–40%** storage density gain [3].

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** **Feed-forward artificial neural network** (Keras/TensorFlow); **Adam** optimizer; compared sampling: **Sobol**, **LHS**, **MC**; optional **multi-ANN ensemble**; polynomial **regression** baseline Eq. **(14)**.
- **Input features / state space:** \(A_c\) [m²], \(V_{st}\) [L], \(T_l\) [°C], \(T_p\) [°C], \(V_f\) [–] (PCM volume / tank volume).
- **Output / action space:** **Solar fraction** SF [dimensionless, 0–1] — **prediction only**, not control actions.
- **Model architecture:** **3** hidden layers × **50** neurons each; **sigmoid** activation; **5** inputs, **1** output (Table 7, Fig. 10).
- **Hyperparameters:** Learning rate **0.005** (best among 0.1, 0.01, 0.005, 0.001); epochs varied during training (overfitting monitored); loss = **MAE/MSE/RMSE**.
- **Training data size:** Up to **10,000** per sampling study; **3,000** Sobol samples for final hyperparameter optimization; **39** models in sampling comparison.
- **Hardware used for training:** Not stated (Python on PC implied).
- **Performance metrics:**
  - Multi-ANN test: **MAE = 0.00114**, **RMSE = 0.001745**, **R² = 0.99990**, **max MAE = 0.0203**
  - Single ANN (best): **MAE = 0.00126**, **R² = 0.99987**
  - Regression: **MAE = 0.02010**, **RMSE = 0.024866**, **R² = 0.98481**
  - Sobol vs LHS/MC: **44%** / **16%** MAE reduction at low sample counts
  - Multi vs single ANN: **~10%** MAE, **~21%** max-error reduction

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **Environment Canada** (`https://weather.gc.ca`) — not ERA5, NASA POWER, ISRO, or Global Solar Atlas.
- **Variables used:** **Solar incident radiation** (hourly), **dry-bulb ambient temperature** \(T_a\); tank surroundings fixed at **20 °C** in Table 1.
- **Geographic scope:** **Toronto, Canada** — single **spring** day profile (Fig. 2).
- **Temporal resolution:** **Hourly** weather; demand profile at high temporal resolution from [19].
- **Time period covered:** **One representative day** (not annual, not multi-year).
- **Clear-sky index / derived metrics:** Not computed.

---

## 8. Key Results & Numbers
- Sobol sampling reduces testing **MAE by 44%** vs **LHS** and **16%** vs **Monte Carlo** at low training-sample counts.
- **39** ANN models trained (3 sampling methods × 13 sample sizes from **250** to **10,000**).
- Beyond **3,000** training samples, further sample increase gives **diminishing MAE** improvement.
- Best learning rate **0.005**: test **MAE = 0.00190**, **R² = 0.99976** (Table 2); inferior rates: LR **0.001** → **MAE = 0.00395**.
- Best topology: **3** hidden layers, **50** neurons/layer → **MAE = 0.00142**, **R² = 0.99986** (Table 3).
- Sigmoid activation: **MAE = 0.00126** vs ReLU **0.00165** (Table 4).
- **Multi-ANN** vs single: **MAE 0.00114 vs 0.00126**; max error **0.0203 vs 0.0256**; **R² 0.99990 vs 0.99987**.
- **Regression** vs multi-ANN: **MAE 0.02010 vs 0.00114** (~**17.6×** higher error).
- **80%** of ANN predictions: **MAE < 2×10⁻³**; regression: **80%** with **MAE < 3.2×10⁻²**.
- Computational time for **84,480** test simulations: numerical **>120 h** vs ANN **~5 s**.
- Design case (\(A_c = 4\,\mathrm{m}^2\), \(T_l = 55\,°\mathrm{C}\), \(T_p = 28\,°\mathrm{C}\)): **90 L** tank, \(V_f\) **0 → 0.5** → SF increase **~13%**; **150 L** tank same change → **~5%**.
- **90 L** tank with PCM can match SF of **210 L** water-only tank (Fig. 11) — **~57%** volume reduction cited in conclusions.
- Collector area range studied: **1–8 m²**; tank volume **50–240 L**; PCM fraction **0–0.7**.
- Pump control: \(\Delta T_{off} = \mathbf{2\,K}\); collector \(\tau\alpha = 0.8\), \(U_L = 5.0\,\mathrm{W/m^2K}\), \(F' = 0.84\).
- Daily load: **189 L** at **8 L/min** peak draw rate.

---

## 9. Baseline Comparison
- **Baseline method(s):** (1) Full **Abdelsalam et al.** transient numerical model; (2) **Polynomial regression** Eq. **(14)**; (3) **Single ANN** vs **multi-ANN**; (4) **MC** and **LHS** sampling vs **Sobol**.
- **Proposed method:** **Sobol-sampled**, hyperparameter-tuned **multi-ANN ensemble** (3×50 sigmoid, Adam LR **0.005**).
- **Improvement margin:**
  - vs numerical model: **~10⁵×** faster (**120 h → 5 s** for 84,480 points).
  - vs regression: **MAE 0.00114 vs 0.02010** (**R² 0.99990 vs 0.98481**).
  - vs single ANN: **~10%** lower MAE, **~21%** lower max error.
  - vs LHS/MC sampling: **44%** / **16%** MAE reduction (Sobol, low-N regime).
- **Conditions of comparison:** Same **84,480** test design points; same underlying physics and **Toronto spring day + 189 L** demand for all SF labels.

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — this paper develops an **ANN surrogate**; no new sensors, actuators, or embedded platform is built. Physical validation is **inherited** from Abdelsalam et al. [4] (prior experimental/numerical hybrid PCM-SDHW work). System control modeled as **ON/OFF pump** and **auxiliary heater + tempering valve**, not Raspberry Pi / ESP32 implementation.

---

## 11. Limitations Acknowledged by Authors
- Numerical modeling of solar TES over **long-term (annual)** operation is computationally demanding; this motivates ANN but the study itself uses **a single-day** weather profile (Introduction, Section 3).
- Framework can be extended to include **collector area**, **weather profile**, and **demand profile** as additional inputs for more comprehensive maps — **not done in current work** (Conclusions).
- PCM modules have **low conductivity and specific heat** vs water; excessive \(V_f\) reduces SF due to **incomplete melting/solidification** and PCM acting as poor sensible storage (Section 5c).
- Larger tanks suffer **higher surface losses**; SF peaks then drops with increasing \(V_{st}\) (Fig. 11, citing [31]).
- Improper **\(T_p\)** selection causes partial phase transformation and under-utilization of latent heat (Section 5c).
- ANN training risks **overfitting** if epochs are excessive (Section 4b).

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant (as implemented).** The ANN predicts **offline solar fraction** for design sweeps; pump logic is fixed **ON/OFF** (Eq. **1**), not PPO/DDPG or climate-adaptive MPC. Supports using ML as a **fast plant model** inside a future controller, not a deployed controller.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Strong precedent for **Python + TensorFlow/Keras** ANN on PCM-SDHW (transferable to **TFLite** edge inference), but **no RPi/ESP32/DS18B20** hardware; PCM is **capric-acid-like**, not Rubitherm RT / PLUSS OM.
- **RG3 (Poor alignment with household demand patterns):** **Partially relevant.** Uses a **realistic high-resolution draw profile** [19] scaled to **189 L/day** and **8 L/min** — closer to demand-aware design than pure step loads, but **not** Coimbatore/Jaisalmer/Kochi profiles or your three-zone comparison; authors note demand/weather as future ANN inputs.
- **RG4 (Limited real-world experimental validation):** **Partially relevant.** Training labels come from a model **validated experimentally in [4]**, but **this paper adds no new field tests**; single-day Toronto simulation limits RG4 claims for India climates.
- **RG5 (No predictive optimization under climatic uncertainty):** **Partially relevant.** Cites ANN literature for **solar irradiance** forecasting [7–9] but uses **deterministic one-day** weather; no ERA5/NASA POWER or forecast-driven optimization — useful as **surrogate for climate sweeps** if retrained on ERA5 for Coimbatore/Jaisalmer/Kochi.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(\mathrm{SF} = Q_{solar\to load}/Q_{demand}\) **(7)** | System performance KPI | RL reward / benchmark metric vs rule-based baseline |
| \(\Delta T_{off} \leq A_c F_R U_L / (\dot{m} C_w \Delta T_{on})\) **(1)** | Collector pump hysteresis | Rule-based baseline controller in Phase 1 |
| \(F_R = \dfrac{\dot{m} C_w (1 - e^{-A_c U_L F'/(\dot{m} C_w)})}{A_c U_L}\) **(2)** | Collector heat removal factor | Collector sub-model in grey-box simulator |
| \(X' = (X-\bar{X})/\sigma\) **(3)** | Input scaling for ML | XGBoost/ANN feature pipeline for climate + tank states |
| MAE / RMSE **(4)–(6)** | Surrogate accuracy metrics | Compare TFLite ANN vs XGBoost vs physics model |
| Eq. **(14)** polynomial SF | Fast design surrogate | Initial sizing before RL training; inferior to ANN here |
| Sigmoid **(9)** | Hidden activation | Reference if shallow ANN used on ESP32 |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **M.Y. Abdelsalam et al., "Hybrid thermal energy storage with phase change materials for solar domestic hot water applications: Direct versus indirect heat exchange systems," Renew. Energy, 2020 [4]** — Relevant because: **Experimentally validated** water–PCM SDHW tank architecture (coil HX, PCM modules) that this ANN replaces computationally.
2. **M.Y. Abdelsalam et al., "A novel approach for modelling thermal energy storage with phase change materials and immersed coil heat exchangers," Int. J. Heat Mass Transf., 2019 [17]** — Relevant because: **PCM + immersed coil** transient model equations underlying training data.
3. **W. Yaïci and E. Entchev, "Performance prediction of a solar thermal energy system using artificial neural networks," Appl. Therm. Eng., 2014 [14]** — Relevant because: Prior **ANN for SDHW** stratification and solar fraction (**±10%** SF accuracy cited in intro).
4. **S. Edwards et al., "Representative hot water draw profiles at high temporal resolution…," Sol. Energy, 2015 [19]** — Relevant because: **Household demand profiles** for aligning PCM discharge with realistic loads (RG3).
5. **A. Najafian et al., "Integration of PCM in domestic hot water tanks: Optimization for shifting peak demand," Energy Build., 2015 [6]** — Relevant because: **PCM placement and volume** optimization in DHW tanks; ANN used for discharge time in related work.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Computational barrier + PCM-SWH gap | "Transient PCM-SDHW models can require **>120 h** for **84,480** design cases vs **~5 s** with a tuned ANN (**R² ≈ 0.9999**)." |
| II. Literature Review | ANN surrogate for hybrid PCM tank | Method: **5-input** ANN (Sobol + Adam); Key insight: **+13%** SF at \(V_f: 0\to0.5\) in **90 L** tank vs **+5%** in **150 L** |
| III. Methodology | SF definition + pump Eqs. (1)–(2) | Use **SF** as KPI; implement **ON/OFF** collector control as rule baseline before PPO |
| IV. Dataset & Setup | Demand + PCM properties | **189 L/day**, **8 L/min** draw; PCM **182 kJ/kg**, \(T_p\) **25–35 °C**, \(V_f\) **0–0.7** |
| V. Results | ANN vs physics speed/accuracy | Surrogate **MAE = 0.00114**; **57%** tank volume reduction potential with proper PCM sizing |
