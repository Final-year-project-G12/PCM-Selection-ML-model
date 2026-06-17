# Performance, Techno-Economic Viability, and Environmental Impact of Domestic Solar Tri-Generation System (DSTS): A Comparative Study of Copper and Galvanized Iron-Based Systems for Sustainable Building Applications

**Authors:** Balamurali Duraivel, Natarajan Muthuswamy  
**Year:** 2025  
**Journal/Conference:** Journal of Building Engineering, Vol. 113, Article 113964  
**DOI:** https://doi.org/10.1016/j.jobe.2025.113964  
**IEEE Citation:** B. Duraivel and N. Muthuswamy, "Performance, techno-economic viability, and environmental impact of domestic solar tri-generation system (DSTS): A comparative study of copper and galvanized iron-based systems for sustainable building applications," J. Build. Eng., vol. 113, p. 113964, 2025, doi: 10.1016/j.jobe.2025.113964.

---

## 1. One-Line Summary
This study builds and field-tests **copper (C-DSTS)** and **galvanized-iron (GI-DSTS)** concrete-roof tri-generation prototypes in **Vellore, India**, integrating solar water heating, **160 Wp** PV, and **50 TEGs**, achieving **46.4%** overall efficiency, **~6 °C** passive cooling, and **25-year** payback as low as **5.3 years** with **180,704–206,921 kg** lifetime net **CO₂** mitigation.

---

## 2. Problem Being Solved
- Building-integrated solar systems are often **single-function** (water heating only), while conventional **PVT/BIPVT** and **TEG-PVT** designs suffer from **complex rooftop integration**, **seasonal efficiency swings**, **extra cost**, and **weak passive cooling** (Abstract, Section 1).
- **TEG-integrated BIPVT** systems typically mount TEGs externally, giving **unstable ΔT**, **low TEG output (~5–8%** of waste heat), and **minimal large-scale indoor cooling** (Table 1).
- Indian residential energy use is rising (**~6–8%** for water heating, **~10%** for cooling; AC demand projected **5×** by 2030), increasing grid stress and emissions (Introduction).
- Lack of **combined experimental + techno-economic + environmental** evidence for **multifunctional** roof-embedded systems using **existing slab structure** rather than full-roof add-on layers.

---

## 3. Key Contributions
1. **Domestic Solar Tri-Generation System (DSTS):** Concrete slab roof embeds serpentine **copper or GI** absorber, **frameless monocrystalline PV** on top, **50 parallel TEGs** under slab — water heating + electricity + passive cooling without separate rooftop assemblies.
2. **Outdoor experiments (Vellore, March 2024):** Three flow rates **0.12 / 0.24 / 0.36 L/min**; performance vs conventional concrete slab; uncertainty **5.08%** overall.
3. **Quantified tri-generation metrics:** Thermal **η_T ≈ 24.75%** (C) / **24.20%** (GI); exergy **η_E up to 37.05%** / **32.30%**; overall **46.4%**; indoor cooling **6.2 °C** / **5.9 °C**; **108 L** heated to **50 °C** in **5 h**.
4. **Techno-economic model:** ALCC, LCC, payback, 25-year cumulative savings **55,113.57 USD** (present value **43,265.77 USD** per Section 4.2 narrative).
5. **Environmental accounting:** Embodied energy, lifetime **CO₂** mitigation **180,704.32 kg** (C-DSTS) and **206,921.09 kg** (GI-DSTS); carbon credits **> 4000 USD** at **23 USD/t**.

---

## 4. Methodology
### 4a. System / Experiment Setup
**Location:** Vellore Institute of Technology, **Vellore, Tamil Nadu, India** (12.9236°N, 79.1331°E).

**Slab / absorber (Table 3):**
- M20 concrete slab: **0.8 m × 1.65 m × 0.13 m**
- Absorber plate: **0.7 m × 1.55 m × 0.002 m** (Cu or GI); **5 serpentine pipes** (ID **0.0127 m**, OD **0.0147 m**)
- **PV:** Frameless mono **160 Wp**, **0.7 m × 1.5 m** (Pmax 160 W, Vmp 18.8 V, Imp 8.52 A)
- **TEG:** **50 units**, **10×5 grid**, parallel; Qmax 50 W, Vmax 14.4 V, Imax 6.4 A (at 25 °C spec)
- **Tank:** **220 L**; pump **1.5 hp**; insulation: polyurethane + thermocol; pipes: C-PVC

**Test arrangement:** Slab on **0.15 m** blocks over polyurethane; thermocol on four sides; mock indoor room under slab for cooling measurement.

**Procedure:** March **2024**; data **11:00–16:00**, **15-min** intervals (IS 3370-1, IS 12976 cited); prototype concrete **k = 2.2 W/m·K** from conductivity rig.

### 4b. Mathematical Models & Equations
**Uncertainty propagation:**

- External uncertainty \(=\sqrt{\sum \left(\frac{\partial C}{\partial i_j}\right)^2 u^2(i_j)}\) — **(1)**
- Internal uncertainty % \(=\dfrac{\sqrt{sd_1'^2+\cdots+sd_n'^2}}{\text{mean observations}}\times 100\) — **(2)**

**Fluid / heat transfer:**

- \(Re = \dfrac{\rho v L_p}{\mu}\) — **(3)**
- \(Pr = \dfrac{\mu C_p}{k_l}\) — **(4)**
- \(Nu = 1.86\left(\dfrac{Re\cdot Pr}{L/L_p}\right)^{1/3}\) — **(5)**
- \(h_c = \dfrac{k_l}{L_p} Nu\) — **(6)**
- \(\bar{U} = \dfrac{1}{\frac{1}{h_c}+\frac{L_p}{k_p}+\frac{L_{concrete}}{k_{concrete}}}\) — **(7)**

**Energy / efficiency:**

- \(Q_o = \dot{m} C_p (t_{out}-t_{in})\) — **(8)**
- \(\mathrm{HUF} = \dfrac{\dot{m} C_p (t_{out}-t_{in})}{\dot{A} G \hat{T} F}\) — **(9)**
- \(Q_l = \dot{A} \bar{U} (t_s - t_a)\) — **(10)**
- \(\eta_T = \dfrac{(Q_o - Q_l)}{\dot{A} G}\times 100\%\) — **(11)**
- \(F = \dfrac{Q_o}{\dot{A} G}\) — **(12)**
- \(\mathrm{COP} = \dfrac{Q_o}{Q_o + P_{pump}}\) — **(13)**
- \(E_o = \dot{m} C_p (t_{out}-t_a)\left(1-\dfrac{t_a}{t_s}\right)\) — **(14)**
- \(E_i = \dot{A} G \left(1-\dfrac{t_a}{t_s}\right)\) — **(15)**
- \(\eta_E = E_o/E_i\) — **(16)**
- \(\eta_P = \dfrac{\dot{V}\dot{I}}{\dot{A} G}\times 100\%\) — **(17)**

**Economics:**

- \(\mathrm{ALCC}_{DSTS} = \mathrm{ALCC} + [C_{EWH}+C_{cooling}] - C_{PV-TEG}\) — **(18)**
- \(\mathrm{ALCC} = \mathrm{LCC}\times C_{rf}\) — **(19)**
- \(\mathrm{LCC} = C_i + \sum_{t=1}^{n}[-(C_{o,t}+C_{m,t}+C_{r,t}) Df_t] - C_s Df_n\) — **(20)**
- \(C_{rf} = \dfrac{r(1+r)^n}{(1+r)^n-1}\) — **(21)**
- \(A_i = A_{initial}(1+i)^n\) — **(22)**; \(A_d = A_{initial}(1-d)^n\) — **(23)**
- \(A_c = \sum (A_i + A_d)\) — **(24)**; \(PV = A_c/(1+r)^n\) — **(25)**
- Payback \(P = C_i / A_{initial}\) — **(26)**

**Environment:**

- \(\mathrm{CO_2\ emission} = \hat{E}\times 2.04/n\) — **(27)**
- Mitigation \(= \hat{E}_T \times 2.04\) — **(28)**; \(\hat{E}_T = D_T \times n_s\) — **(29)**
- \(D_T = Q_h + Q_c\) — **(30)**
- Net lifetime mitigation \(= [(\hat{E}_T \times n) - \hat{E}\times 2.04\times 10^{-3}]\) — **(31)**
- CO₂ credit \(=\) Net mitigation \(\times\) cost per ton — **(32)**

### 4c. Algorithm / Control Method Steps
N/A — **no AI/ML or adaptive control**. Operation is manual/semi-manual:

1. Circulate water at set flow via **ball valves** and **flow sensors** (**0.12 / 0.24 / 0.36 L/min**).
2. Log temperatures (K-type TCs), irradiance (**pyranometer**), PV V/I (**multimeter**), DAQ every **15 min**.
3. TEGs passively convert slab temperature gradient; **no closed-loop controller** for charging/discharging or cooling setpoint.

### 4d. Data Sources & Dataset Details
| Source | Content | Scope |
|--------|---------|--------|
| On-site measurements | \(G\), \(T_a\), slab/PV/room/water temps, flow, PV V/I | **Vellore**, **March 2024**, **5 h/day** test windows |
| Economic assumptions (Table 6) | Costs, interest **8%**, inflation **5%**, real interest **3%**, electricity **0.048 USD/kWh** | **25-year** DSTS life; conventional **10-year** |
| Operating days | **245 days/year** (from **290** sunny days assumption scaled) | **5 h/day** operation for annual energy accounting |
| Embodied energy coefficients | Table 8 (kWh/kg materials) | Fabrication LCA inputs |
| Standards cited | IS 3370-1 (2009), IS 12976 (1990) | Concrete + solar water heater testing guidance |

No ERA5, NASA POWER, or ISRO Solar Calculator used.

### 4e. Validation Method
- **Experimental** comparison: conventional slab vs **C-DSTS** vs **GI-DSTS** under identical radiation (**778–997 W/m²** peaks).
- **Literature benchmarking** (Fig. 7): outlet temperature, roof temperature, room temperature, HUF, COP, \(\eta_T\), \(\eta_E\).
- **Maximum validation errors (Table 5):** thermal efficiency **12.02%** (C) / **11.22%** (GI); exergy **8.08%** / **7.18%**; outlet water temp error **14.4 °C** / **14.8 °C** (vs reference studies — authors attribute to configuration differences).
- **Uncertainty:** Overall **5.08%** (external **0.25%** + internal thermocouple **4.83%**).

---

## 5. PCM Details (if applicable)
N/A — the **DSTS prototype does not use phase-change materials**. Thermal buffering is provided by the **concrete slab** mass and water circulation.

**Literature context only (Table 2):** PCM-integrated BIPVT systems are surveyed at **50–70%** thermal efficiency, **14–18%** electrical, **12–22%** exergy, **$350–500/m²**, **5–7 year** payback, **30–50% CO₂** reduction — with note of **PCM degradation over time**. Cited hybrid PVT–PCM building study: Li et al., *Renew. Energy* **199**, 662–671 (2022) [ref. 8 in paper].

---

## 6. AI / ML / Control Details (if applicable)
N/A — no artificial intelligence, forecasting, or reinforcement learning. Control is **fixed flow-rate tests** with **ball valves** and a **1.5 hp** pump; space cooling is **passive** via slab + TEG temperature gradient.

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **On-site pyranometer** (not satellite reanalysis or NASA POWER).
- **Variables used:** Global solar irradiance **G** (**778.44–997.51 W/m²** peak during tests); ambient **\(T_a\)** **35.33–43.6 °C**; inlet water **30.66–32.90 °C**.
- **Geographic scope:** **Vellore, Tamil Nadu, India** (hot, high-insolation climate).
- **Temporal resolution:** **15-min** logging; test window **11 a.m.–4 p.m.**; economic extrapolation **245 days/year × 5 h/day**.
- **Time period covered:** **March 2024** experiments; **25-year** lifecycle analysis.
- **Clear-sky index / derived metrics:** Not computed; peak irradiance reported directly.

---

## 8. Key Results & Numbers
- **Overall system efficiency:** **46.4%**; TEG contribution **~11%** average.
- **Thermal efficiency (max at 0.36 L/min):** **24.75%** (C-DSTS), **24.20%** (GI-DSTS).
- **Exergy efficiency (max):** **37.05%** (C), **32.30%** (GI) at **0.36 L/min**.
- **Passive indoor cooling:** **6.2 °C** (C) vs conventional room (**31.7 °C → 25.69 °C** at 0.12 L/min); **5.9 °C** (GI).
- **Water heating:** **108 L** to **50 °C** in **~5 h**; outlet peak **48.2 °C** (C-DSTS); ΔT **9.97–11.42 °C** across tests.
- **Flow-rate sweep:** HUF **0.093 / 0.177 / 0.269** (C) and **0.090 / 0.170 / 0.260** (GI) at **0.12 / 0.24 / 0.36 L/min**; COP up to **3.513** (C) and **3.393** (GI) at **0.36 L/min**.
- **PV electrical efficiency:** **11.38–12.39%** (DSTS panels) vs **11.42–12.31%** conventional panel — cooling from water loop + TEGs limits PV overheating (top PV temps **70–74 °C** on DSTS).
- **TEG output (parallel bank):** avg **0.685 V**, **50 A**; cold-side TEG **~23.3–24.1 °C**.
- **Annual outputs (economic model):** **198 kWh** electricity; **26,460 L** hot water; cooling benefit up to **6.2 °C** for **5 h/day**.
- **Capital cost:** **1321.06 USD** (C-DSTS), **855.06 USD** (GI-DSTS) at **1 USD = ₹85.03**.
- **Annualized cost:** **167.26 USD** (C), **108.25 USD** (GI) vs **286.30 USD** conventional EWH + fan/AC.
- **Payback:** **8.18 years** (C), **5.29 years** (GI); lifespan **25 years**.
- **25-year cumulative savings:** **55,113.57 USD** (text Section 4.2); present value **43,265.77 USD**.
- **Daily savings:** **0.17 USD** (water heating) + **0.45 USD** (cooling).
- **Embodied energy:** **1658.83 kWh** (C fabrication), **1171.12 kWh** (GI).
- **Lifetime net CO₂ mitigation:** **180,704.32 kg** (C), **206,921.09 kg** (GI); carbon credits **4165.37 USD** (C), **4194.12 USD** (GI) at **23 USD/t**.
- **Slab top temperature (C-DSTS):** **55–59 °C** avg **49.09–53.28 °C**; bottom–top ΔT up to **17.38 °C**.

---

## 9. Baseline Comparison
- **Baseline method(s):** (1) **Conventional concrete slab** (no absorber/PV/TEG); (2) **Conventional electric water heater + fans/AC** (80% efficiency assumed, **10-year** life); (3) **Literature BIPVT/PVT** configurations (Tables 1–2).
- **Proposed method:** **C-DSTS** and **GI-DSTS** (embedded tri-generation roof slab).
- **Improvement margin:**
  - vs conventional slab: bottom/roof **3.57–4.71 °C** cooler (C); room up to **~6 °C** lower.
  - vs conventional electric systems: annualized cost **286.30 → 167.26 USD** (C) or **108.25 USD** (GI).
  - vs literature PCM-BIPVT band: proposed DSTS Table 2 entry claims **65–82%** thermal (design target row) vs **50–70%** PCM-BIPVT — **experimental** \(\eta_T\) achieved **~25%** (lower than table aspirational range).
  - Literature cited MODIS+NWP multimodal not in this paper; intro cites **13.2% RMSE** only in other survey context — **not applicable here**.
- **Conditions:** Same **Vellore** climate; same **5 h** solar window; flow **0.36 L/min** optimal for \(\eta_T\), \(\eta_E\).

---

## 10. Hardware / Experimental Setup (if applicable)
- **Physical components:** M20 **concrete roof slab**; **Cu or GI** serpentine absorber; **160 Wp** frameless PV; **50× TEG** (parallel); **220 L** tank; **1.5 hp** pump; **ball valves**; C-PVC pipes; polyurethane + thermocol insulation.
- **Sensor specs (Table 4):**
  - **Pyranometer:** **0–2000 W/m²**, accuracy **±0.1 W/m²**
  - **K-type thermocouples:** **−200 to 650 °C**, **±0.01 °C** (internal uncertainty **±4.83%**)
  - **Flow sensor:** **0–40 L/s**, **±0.01**
  - **Multimeter:** **200 mV–1000 V**, **200 μA–200 mA**
  - **DAQ** for logging
- **Embedded/compute platform:** **DAQ + manual multimeter** — no Raspberry Pi / Arduino / ESP32.
- **Test environment:** **Outdoor rooftop mock-up** at **VIT Vellore** with insulated underside “room” for passive cooling tests.
- **Test duration:** **March 2024**; **5 h/day** (11:00–16:00); **15-min** sampling; three flow-rate days per configuration.

---

## 11. Limitations Acknowledged by Authors
- Performance evaluated only under **Vellore** conditions (high irradiance, **35–43.6 °C** ambient); **“performance metrics may vary”** in low insolation, humidity, or cold climates (Conclusion).
- **Multi-location validation** and seasonal extremes **beyond scope**; thermal mass behavior may differ (Conclusion).
- **Retrofitting challenges** not fully studied: structural modifications, plumbing/electrical integration, roof-space limits (Conclusion).
- PV panels tested **without tilt**, though cooling maintained PV efficiency near conventional tilted panels.
- Validation vs literature shows **up to 12.02%** thermal efficiency deviation and **14.4 °C** outlet temperature error vs some reference systems (Table 5).
- Economic model assumes **245 operating days/year**, **5 h/day** — actual household use may extend beyond this (Section 4.1).

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant.** Flow rates are **manually set** (0.12–0.36 L/min); no PPO/DDPG, no climate-forecast-driven charging — fixed test protocol only.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Real **multi-sensor experimental rig** (pyranometer, thermocouples, flow, DAQ) in **India (Vellore)** parallels your sensing stack, but **no PCM**, **no AI**, and architecture is **BIPVT-TEG concrete slab**, not Rubitherm/PLUSS tank + ESP32/RPi.
- **RG3 (Poor alignment with household demand patterns):** **Partially relevant.** Delivers **108 L at 50 °C in 5 h** and models **26,460 L/year** — residential scale, but **no dynamic draw profile** (unlike Edwards-type demand); constant-flow experiments only.
- **RG4 (Limited real-world experimental validation):** **Highly relevant.** **Field experiments** in **Tamil Nadu** with quantified \(\eta_T\), \(\eta_E\), and indoor ΔT — supports feasibility of **Indian outdoor SWH-related** research; your FYP can contrast their **concrete sensible storage** with **PCM latent storage**.
- **RG5 (No predictive optimization under climatic uncertainty):** **Not Relevant.** No ERA5/forecasting or optimization under weather uncertainty; single-site **March 2024** campaign only.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(Q_o = \dot{m} C_p (t_{out}-t_{in})\) **(8)** | Useful heat to water | PCM-SWH energy balance / charging rate |
| \(\eta_T = (Q_o-Q_l)/(\dot{A}G)\) **(11)** | Collector thermal efficiency | KPI vs rule-based and RL policies |
| \(\eta_E = E_o/E_i\) **(14)–(16)** | Exergy efficiency | Second-law metric for PCM charging quality |
| \(\mathrm{COP} = Q_o/(Q_o+P_{pump})\) **(13)** | Pumping penalty | Include **solenoid/pump power** in reward function |
| \(\eta_P = \dot{V}\dot{I}/(\dot{A}G)\) **(17)** | PV efficiency vs irradiance | Couple **pyranometer** to electrical auxiliary offset |
| Eqs. **(18)–(26)** ALCC/LCC/payback | Lifecycle economics | Optional FYP techno-economic appendix for PCM-SWH |
| Eqs. **(27)–(32)** CO₂ mitigation | Environmental benefit | Cite Indian building-scale **CO₂** reduction context |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **J. Li et al., "A hybrid photovoltaic and water/air based thermal (PVT) solar energy collector with integrated PCM for building application," Renew. Energy, 2022 [8]** — Relevant because: Direct **PCM + PVT** building collector — closest architectural analog to PCM-SWH latent storage.
2. **M. Emam et al., "Year-round experimental analysis of a water-based PVT-PCM hybrid system: comprehensive 4E assessments," Renew. Energy, 2024 [10]** — Relevant because: **Experimental PVT-PCM** with energy/exergy/economic/environment metrics.
3. **L. Xu et al., "Hybrid PV thermal wall with double air channel and phase change material: seasonal experimental research," Renew. Energy, 2021 [25]** — Relevant because: **PCM + hybrid PVT** seasonal outdoor data for building integration.
4. **V.S. Chandrika et al., solar concrete water heater **57%** thermal efficiency in Tamil Nadu [57] (cited in Section 1 review)** — Relevant because: **South India** experimental solar concrete WH benchmark near your geography.
5. **B. Duraivel et al., "Extensive analysis of a reinvigorated solar water heating system using low-density polyethylene glazing," Energies, 2023 [4]** — Relevant because: Same author group’s **Indian SWH** experimental work preceding DSTS.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Indian residential energy + roof-integrated solar | "Water heating ~6–8% and cooling ~10% of Indian household energy; Duraivel & Muthuswamy (2025) show roof-embedded tri-generation in Vellore achieving 46.4% overall efficiency." |
| II. Literature Review | BIPVT vs PCM-SWH gap | Method: concrete slab + PV + TEG (no PCM); Key: passive **6.2 °C** cooling, **24.75%** thermal η — sensible mass only |
| III. Methodology | Eq. (8), (11) for collector KPIs | Use \(Q_o\) and \(\eta_T\) definitions for grey-box validation against DS18B20 + pyranometer |
| IV. Dataset & Setup | India outdoor benchmark | Vellore: **778–997 W/m²** peak G; pyranometer **±0.1 W/m²** — comparable instrumentation tier |
| V. Results / Discussion | Field validation precedent | Cite **37.05%** exergy and **108 L → 50 °C in 5 h** as Indian experimental SWH performance anchor (different technology than PCM tank) |
