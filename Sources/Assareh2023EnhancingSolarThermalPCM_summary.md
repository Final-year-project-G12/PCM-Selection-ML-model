# Enhancing Solar Thermal Collector Systems for Hot Water Production Through Machine Learning-Driven Multi-Objective Optimization with Phase Change Material (PCM)

**Authors:** Ehsanolah Assareh, Amjad Riaz, Mehrdad Ahmadinejad, Siamak Hoseinzadeh, Mohammad Zaheri Abdehvand, Sajjad Keykhah, Tohid Jafarinejad, Rahim Moltames, Moonyong Lee  
**Year:** 2023  
**Journal/Conference:** Journal of Energy Storage, Vol. 73, Article 108990  
**DOI:** https://doi.org/10.1016/j.est.2023.108990  
**IEEE Citation:** E. Assareh et al., "Enhancing solar thermal collector systems for hot water production through machine learning-driven multi-objective optimization with phase change material (PCM)," J. Energy Storage, vol. 73, p. 108990, 2023, doi: 10.1016/j.est.2023.108990.

---

## 1. One-Line Summary
This paper uses MATLAB-based MOEA/D multi-objective optimization (plus RSM, TOPSIS, LINMAP, and AHP) on a flat-plate solar collector with PCM hot-water storage to trade off PCM discharge duration \(t_{PCM}\) versus net stored energy \(Q_{net}\), showing inverse Pareto coupling and sensitivity to tube diameter and collector area.

---

## 2. Problem Being Solved
- Solar thermal collectors with PCM storage require appropriate **energy discharge time** \(t_{PCM}\) and **net stored energy** \(Q_{net}\) in the PCM, but these objectives conflict under design and operating parameter choices.
- Prior work optimized collectors or PCM separately rather than jointly optimizing collector geometry, tank/contact area, and PCM storage behavior for hot-water production.
- Night-time hot-water availability depends on how long PCM can discharge latent/sensible heat after sunset, while daytime charging must maximize stored energy—requiring multi-objective, not single-objective, design.
- Lack of integrated decision support linking tube diameter, collector area \(A_c\), and PCM class selection to both discharge duration and stored energy in one framework.

---

## 3. Key Contributions
1. Integrated **flat-plate collector + PCM tank** model (disodium hydrogen phosphate dodecahydrate baseline PCM) with energy-balance equations for useful collector gain and PCM discharge time **(1)–(8)**.
2. **MOEA/D** decomposition of the bi-objective problem (minimize \(t_{PCM}\), maximize \(Q_{net}\)) with decision variables: tube inner diameter, contact area, collector area \(A_c\), and PCM-minus-water stored energy band.
3. Pareto analysis (**500** population points) proving **inverse relationship**: maximum \(t_{PCM}\) aligns with minimum \(Q_{net}\), and vice versa.
4. Parametric studies on **tube diameter** (nonlinear increase in \(t_{PCM}\) and \(Q_{net}\)) and **storage contact area** (linear sensitivity; \(Q_{net}\) more sensitive than \(t_{PCM}\)).
5. Comparison of three PCM classes (**hybrid salt, paraffin, fatty acid**) plus post-optimization screening via **RSM**, **TOPSIS**, **LINMAP**, and **AHP** for Pareto-point selection.

---

## 4. Methodology
### 4a. System / Experiment Setup
- **Configuration (Fig. 1):** Flat-plate solar collector; water storage/PCM tank with **disodium hydrogen phosphate** PCM; piping, valves, and **bypass line** for night-time network-water return to extract stored PCM heat after sunset.
- **Day operation:** Purified water flows through collector tubes, enters PCM tank, transfers heat to solid PCM (**29 °C melting temperature** stated in §2.2 narrative; **35 °C** in Table 5 for dodecahydrate salt).
- **Night operation:** Network water returns via bypass, extracts energy from liquefied PCM, then flows to consumption loops.
- **Software:** **MATLAB** simulator; **MOEA/D** for multi-objective search; **RSM** for response-surface refinement; **TOPSIS**, **LINMAP**, **AHP** for multi-criteria decision-making on Pareto solutions.
- **Assumptions (§2.1):** One-dimensional flow; sky treated as black body for long-wave radiation; certain collector loss properties taken **temperature-independent**.
- **No physical test rig** in this paper—computational optimization study with validation against published thermal data only.

### 4b. Mathematical Models & Equations
**Useful collector energy (Hottel–Whillier–Bliss form):**

- \(Q_u = A_c F_R \left[ S - U(T_c - T_a) \right]\) — **(1)**

**Heat removal factor:**

- \(F_R = \dfrac{\dot{m} C_p}{A_c U}\left[1 - e^{-\left[A_c U F' / (\dot{m} C_p)\right]}\right]\) — **(2)**

**Collector efficiency factor:**

- \(F' = \dfrac{1}{\frac{1}{U}\left[\frac{1}{t_W}\left(U_L (D + (W-D)F)\right) + \frac{1}{\pi D_i h_f}\right]}\) — **(3)**

**Top loss coefficient (radiative/convective glazing model):**

- \(U_t = \dfrac{\sigma(T_a + T_p)(T_a^2 + T_p^2)}{\left[\dfrac{1}{\varepsilon_p + 0.0425N(1-\varepsilon_p)^{-1}} + \dfrac{2N+f-1}{\varepsilon_g} - N\right]^{-1}}\) — **(4)**  
  (\(N\) = number of glass covers; \(\varepsilon_p\) = plate emissivity; \(\varepsilon_g = 0.88\); \(T_a\) ambient; \(\beta\) slope; \(h_w\) wind coefficient)

**Collector loss / geometric factor:**

- \(Q_L = U_l (T_{in} - T^\circ)\) — **(5)**
- \(f = \left(1 + 0.089 h_w - 0.1166 h_w / \varepsilon_p\right) + (1 + 0.07866 N)\) — **(6)**

**Optimization objectives:**

- \(Q_{net} = Q_u + Q_L\) — **(7)**  
- \(t_{PCM} = Q_{net} / Q_u\) — **(8)**  
  (\(t_{PCM}\) = duration water can stay warm without solar input)

**Multi-objective problem:**

- \(\min \mathbf{F}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x}))^T\) subject to \(\mathbf{x} \in \mathbb{R}^m\) — **(9)**  
  (implemented as MOEA/D scalar sub-problems with weight vectors)

### 4c. Algorithm / Control Method Steps
1. Define decision vector bounds (Table 1): diameter **0.005–0.02 m**, area **0.2–1 m²**, \(A_c\) **0.26–0.38 m²**, PCM-minus-water energy **3100–3300 kJ**.
2. For each candidate design, compute \(Q_u\), \(Q_L\), \(Q_{net}\), and \(t_{PCM}\) via **(1)–(8)** under stated operating constants (e.g., \(\dot{m}_w\), \(m_{PCM}\), \(T_{in,water}\)).
3. Run **MOEA/D**: decompose bi-objective problem into weighted scalar sub-problems; evolve population (**500** solutions reported for Pareto plot).
4. Extract Pareto front relating \(t_{PCM}\) vs \(Q_{net}\); identify design points for target night discharge (**6 h** or **7 h**).
5. Apply **RSM** to build empirical response surfaces between inputs (diameter, area) and objectives.
6. Rank/select Pareto points using **TOPSIS**, **LINMAP**, and **AHP** (ideal vs negative-ideal distance logic).
7. Repeat parametric sweeps for tube diameter and \(A_c\) with other parameters fixed (Tables 3–4).
8. Compare alternate solid PCMs (hybrid salt, paraffin, fatty acid) at matched conditions (Figs. 11–12).

*No real-time control loop, reinforcement learning, or online learning steps are implemented.*

### 4d. Data Sources & Dataset Details
| Source | Variables | Resolution | Scope | Period / size |
|--------|-----------|------------|-------|----------------|
| **MATLAB thermal model** (this study) | \(Q_u\), \(Q_L\), \(Q_{net}\), \(t_{PCM}\), \(T_{out}\) | Hourly time tags in validation table | Generic flat-plate + PCM tank | Parametric runs; **500** MOEA/D population |
| **Luo et al. [55] (2021)** | Collector outlet temperature vs time | **6:00–18:00** hourly | Air-type double-pass collector with PCM rod | Used for model validation (Table 2) |
| **Fixed operating set (Table 3)** | \(\dot{m}_w = 0.009\) kg/s, \(A_c = 0.287\) m², \(m_{PCM} = 10\) kg, \(T_{in,water} = 293.05\) K | Steady/parametric | Single baseline case | Diameter sensitivity runs |

*No ERA5, NASA POWER, ISRO, or site-specific Indian weather series used.*

### 4e. Validation Method
- **Literature temperature benchmark** against **Luo et al. (2021)** outlet temperatures (Table 2): e.g., **6:00** — current **305** vs **306**; **12:00** — **375** vs **374**; **18:00** — **330.5** vs **330** (units printed as °C in table; values correspond to ~**32–102 °C** if interpreted as K minus offset—authors state “good accuracy” of thermal modeling).
- **MOEA/D internal consistency:** Pareto set of **500** designs; decision-making methods (RSM/TOPSIS/LINMAP/AHP) converge to **coincident optimal point** on response surface (Fig. 10).
- **No RMSE/R²** reported for optimization; **no experimental validation** of the optimized Assareh system in field or lab.

---

## 5. PCM Details (if applicable)
- **Materials tested:** **Disodium hydrogen phosphate dodecahydrate** (hybrid salt, baseline in §2.2); additionally **paraffin (C20–C33)** and **uric acid (fatty acid)** in comparative study (Table 5, Figs. 11–12).
- **Melting temperature range:** Baseline narrative **29 °C** (§2.2); Table 5 hybrid salt **35 °C**; paraffin **50 °C**; fatty acid **44 °C**.
- **Latent heat:** Hybrid salt **278.84 kJ/kg**; paraffin **189 kJ/kg**; fatty acid **178 kJ/kg**.
- **Thermal conductivity:** Not reported in Table 5.
- **Specific heat (solid/liquid):** Hybrid salt **1.55 / 2.51 kJ/kg·K**; paraffin **2.4 / 2.4 kJ/kg·K**; fatty acid **1.7 / 2.3 kJ/kg·K**.
- **Density:** Hybrid salt **1522 kg/m³**; paraffin **912 kg/m³**; fatty acid **862 kg/m³**.
- **Performance metrics reported:** \(t_{PCM}\) target **6 h** or **7 h** night discharge; \(Q_{net}\) **3094 kJ** (7 h case) vs **3200 kJ** (6 h case) at RSM/DM optimum; hybrid salts yield **largest** increases in \(t_{PCM}\) and \(Q_{net}\) vs paraffin/fatty acids at high melting temperature (Figs. 11–12); \(m_{PCM} = 10\) kg in parametric tables.

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** **MOEA/D** (multi-objective evolutionary algorithm based on decomposition); supporting methods: **RSM**, **TOPSIS**, **LINMAP**, **AHP** — *not* neural networks, PPO, DDPG, or XGBoost despite “machine learning-driven” wording in the title.
- **Input features / state space:** Design/decision variables: tube inner **diameter** \(D\), storage **contact area** \(A\), collector area **\(A_c\)**, band on **PCM-minus-water stored energy** (Table 1); fixed runs use \(\dot{m}_w\), \(m_{PCM}\), \(T_{in,water}\), solar/input parameters embedded in \(Q_u\).
- **Output / action space:** Pareto-optimal **\(t_{PCM}\)** (minimize) and **\(Q_{net}\)** (maximize); selected operating points for **6–7 h** discharge scenarios.
- **Model architecture:** N/A — no ANN/CNN; empirical **RSM** surrogate models for response surfaces after MOEA/D.
- **Hyperparameters:** MOEA/D population **500** (Fig. 5); weight vectors per sub-problem (standard MOEA/D); RSM/AHP/TOPSIS/LINMAP procedural parameters not numerically tabulated.
- **Training data size:** **500** Pareto population members; no supervised ML train/test split.
- **Hardware used for training:** N/A — MATLAB simulation on unstated compute.
- **Performance metrics:** Pareto trade-off curves; RSM optimum **\(Q_{net} = 3094\) kJ** at **\(t_{PCM} = 7\) h** and **3200 kJ** at **6 h**; linear \(t_{PCM}\) and \(Q_{net}\) vs \(A_c\); nonlinear increasing \(Q_{net}\) vs diameter.

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **Not stated** — solar input enters through term **\(S\)** (irradiance, W/m²) inside **\(Q_u\) (1)** without naming NSRDB, measured weather files, or satellite products.
- **Variables used:** Implicit solar gain **\(S\)**, ambient **\(T_a\)**, collector **\(T_c\)**, wind-related **\(h_w\)** in loss coefficients.
- **Geographic scope:** **Not stated** — no city, climate zone, or country-specific weather series.
- **Temporal resolution:** Hourly comparison points in validation table (**6:00–18:00**); \(t_{PCM}\) in **hours**.
- **Time period covered:** Validation references one-day profile from **Luo et al. (2021)**; optimization runs are parametric, not multi-year.
- **Clear-sky index / derived metrics:** **Not computed.**

---

## 8. Key Results & Numbers
- MOEA/D Pareto population: **500** designs; **inverse trade-off** — maximum \(t_{PCM}\) (left side of Fig. 5) pairs with **minimum** \(Q_{net}\); maximum \(Q_{net}\) yields **shortest** night-time energy availability.
- Target **\(t_{PCM} = 7\) h** (RSM/decision-making optimum): **\(Q_{net} = 3094\) kJ** (Fig. 10).
- Target **\(t_{PCM} = 6\) h**: **\(Q_{net} = 3200\) kJ** — **106 kJ** higher stored energy for **1 h** shorter discharge target.
- Increasing **tube inner diameter** increases **\(t_{PCM}\)** nonlinearly (Fig. 7) and increases **\(Q_{net}\)** nonlinearly (Fig. 9); \(Q_{net}\) **more sensitive** at larger diameters.
- **Contact area \(A\):** both \(t_{PCM}\) and \(Q_{net}\) vary **linearly** with area; **\(Q_{net}\)** line steeper (more sensitive) than \(t_{PCM}\) (Fig. 8).
- Decision-variable bounds: diameter **0.005–0.02 m**; area **0.2–1 m²**; \(A_c\) **0.26–0.38 m²**; stored-energy band **3100–3300 kJ** (Table 1).
- Parametric baseline (Table 3): \(\dot{m}_w = 0.009\) kg/s, \(A_c = 0.287\) m², \(m_{PCM} = 10\) kg, \(T_{in,water} = 293.05\) K (**~20 °C**).
- Validation vs Luo et al.: outlet temperature agreement within **~1–1.5** units at most hours (e.g., **358.9** vs **360** at 10:00; **375** vs **374** at 12:00) — Table 2.
- PCM class comparison: **hybrid salts** produce **much greater** increases in \(t_{PCM}\) and \(Q_{net}\) than **paraffin** or **fatty acids** at high melting temperature; hybrid salt latent heat **278.84 kJ/kg** vs paraffin **189 kJ/kg** and fatty acid **178 kJ/kg**.
- Literature benchmarks cited (not this paper’s direct results): Lin et al. heat-transfer effectiveness **44.25% → 59.29%**; Shamsi et al. **+5%** discharged energy over **8 h** cycle and **+5.12%** stored in **4 h** charge vs single PCM.

---

## 9. Baseline Comparison
- **Baseline method(s):** **Non-optimized** / Pareto extremes on MOEA/D front (max \(t_{PCM}\) vs max \(Q_{net}\)); implicit comparison among **PCM material classes** (hybrid salt vs paraffin vs fatty acid); validation reference **Luo et al. (2021)** thermal profile only.
- **Proposed method:** **MOEA/D** multi-objective design optimization + **RSM/TOPSIS/LINMAP/AHP** decision-making on Pareto solutions.
- **Improvement margin:** At fixed discharge requirement, **6 h** vs **7 h** target changes optimum **\(Q_{net}\)** by **3200 − 3094 = 106 kJ** (~**3.4%** relative to 7 h case); material switch to hybrid salt gives **largest** \(t_{PCM}\) and \(Q_{net}\) vs other PCM classes (qualitative ranking, Figs. 11–12 — no single % tabulated).
- **Conditions of comparison:** Same MATLAB energy-balance model; parametric constants in Tables 3–4; PCM properties from Table 5.

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — this paper is purely simulation/optimization-based in **MATLAB**. No sensors (DS18B20, pyranometer), actuators (solenoid valves), embedded platforms (RPi/Arduino/ESP32), or field/lab test duration is reported. Physical system description (Fig. 1) is conceptual for modeling only.

---

## 11. Limitations Acknowledged by Authors
- Authors **do not include a dedicated “Limitations” section**; the following are explicit scope/assumption statements only.
- Modeling assumes **one-dimensional flow**, a **black-body sky**, and collector loss properties **independent of temperature** (§2.1), which may reduce fidelity under real variable weather and temperature-dependent losses.
- **“Data will be made available on request”** — no public dataset or reproducibility package is provided in the article.
- Comparative PCM study selects **one random material per class** (hybrid salt, paraffin, fatty acid), not an exhaustive PCM database (§3.2 narrative).
- Conclusion focuses on **design-parameter sensitivity** (diameter, area) and does not claim field demonstration of optimized hardware.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant.** MOEA/D is **offline design optimization**; there is no hourly/online controller, DRL policy, or pump/valve actuation loop comparable to your PPO charge/discharge/bypass agent.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Not Relevant.** Work stops at MATLAB simulation; no Raspberry Pi, ESP32, or sensor-actuator integration despite being a PCM–solar hot-water architecture similar in schematic to your FYP.
- **RG3 (Poor alignment with household demand patterns):** **Partially relevant.** Optimizing **\(t_{PCM}\)** for **6–7 h** night-time thermal availability loosely aligns with evening/morning hot-water needs, but there is **no measured residential draw profile**, flow schedule, or occupant-driven demand model (Coimbatore/Jaisalmer/Kochi).
- **RG4 (Limited real-world experimental validation):** **Relevant (as gap exemplar).** Only **literature temperature comparison** (Luo et al.) is shown; the optimized Assareh system is **not built or field-tested**, mirroring the simulation-heavy literature your prototype addresses.
- **RG5 (No predictive optimization under climatic uncertainty):** **Not Relevant.** Solar input **\(S\)** is not tied to ERA5/NASA POWER/forecasting; no uncertainty-aware or predictive dispatch—static parametric and evolutionary design only.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(Q_u = A_c F_R [S - U(T_c - T_a)]\) **(1)** | Flat-plate useful solar gain | Couple pyranometer GHI to collector thermal input in grey-box SWH |
| \(F_R = \frac{\dot{m} C_p}{A_c U}[1 - e^{-A_c U F'/(\dot{m} C_p)}]\) **(2)** | Heat removal factor | Tank–collector coupling in enthalpy balance |
| \(t_{PCM} = Q_{net}/Q_u\) **(8)** | Night discharge duration metric | RL reward: maximize hot-water availability hours after sunset |
| \(Q_{net} = Q_u + Q_L\) **(7)** | Net stored energy objective | PCM stored-energy state for PPO observation / reward |
| MOEA/D \(\min \mathbf{F}(\mathbf{x})\) **(9)** | Multi-objective design trade-off | Offline NSGA-II/PSO baseline vs online PPO for same \((t_{PCM}, Q_{net})\) objectives |
| \(U_t\) radiative loss **(4)** | Collector top losses | Optional detailed collector loss if moving beyond lumped model |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **Q. Luo et al., "Thermal modeling of air-type double-pass solar collector with PCM-rod embedded in vacuum tube," *Energy Convers. Manag.*, 2021 [55]** — Relevant because: Direct **PCM–collector validation** benchmark used in this paper’s temperature accuracy check.
2. **W. Lin et al., "Multi-objective optimisation of thermal energy storage using phase change materials for solar air systems," *Renew. Energy*, 2019 [23]** — Relevant because: Prior **MO + PCM** study reporting **44.25% → 59.29%** heat-transfer effectiveness and **4.53 → 6.11 h** charging time improvements.
3. **M. Mahfuz et al., "Performance investigation of TES with PCM for solar water heating application," *Int. Commun. Heat Mass Transf.*, 2014 [26]** — Relevant because: **Shell-and-tube PCM–SWH** experimental lineage closest to domestic hot-water storage.
4. **A. Mourad et al., "Recent advances on the applications of phase change materials for solar collectors," *J. Energy Storage*, 2022 [19]** — Relevant because: Review of **PCM–solar collector** limits and practical constraints for literature review framing.
5. **M.H. Zahir et al., "Challenges of PCMs to achieve zero energy buildings under hot weather," *J. Energy Storage*, 2023 [8]** — Relevant because: **Hot-climate PCM–building** context analogous to Coimbatore/Jaisalmer deployment challenges.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | PCM–SWH design trade-off | "Assareh et al. show \(t_{PCM}\) and \(Q_{net}\) are **inversely coupled** on the MOEA/D Pareto front—longer night discharge (**7 h**) yields **3094 kJ** stored vs **3200 kJ** at **6 h**." |
| II. Literature Review | Offline MO vs your online DRL | Method: **MOEA/D + RSM/TOPSIS**; Key insight: **no real-time control**, MATLAB-only |
| III. Methodology | Collector + PCM energy balances | Adopt **(1)–(2)** for \(Q_u\) and **(8)** as discharge-duration metric in grey-box model |
| IV. Dataset & Setup | PCM property table | Hybrid salt **278.84 kJ/kg** latent heat, **35 °C** \(T_m\); compare to Rubitherm RT/PLUSS OM ranges |
| V. Results | Design optimization baseline | Contrast your embedded PPO against their **500-point** Pareto and **106 kJ** swing between **6 h** and **7 h** discharge targets |
