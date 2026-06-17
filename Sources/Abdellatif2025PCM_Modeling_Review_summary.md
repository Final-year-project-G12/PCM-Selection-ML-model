# Modeling and Performance Analysis of Phase Change Materials in Advanced Thermal Energy Storage Systems: A Comprehensive Review

**Authors:** Houssam Eddine Abdellatif, Ahmed Belaadi, Adeel Arshad, Mostefa Bourchak  
**Year:** 2025  
**Journal/Conference:** Journal of Energy Storage, Vol. 121, Article 116517  
**DOI:** https://doi.org/10.1016/j.est.2025.116517  
**IEEE Citation:** H. E. Abdellatif, A. Belaadi, A. Arshad, and M. Bourchak, "Modeling and performance analysis of phase change materials in advanced thermal energy storage systems: A comprehensive review," J. Energy Storage, vol. 121, p. 116517, 2025, doi: 10.1016/j.est.2025.116517.

---

## 1. One-Line Summary
This review synthesizes latent and hybrid PCM thermal energy storage literature—enhancement methods (fins, nanoparticles, metal foam, encapsulation), numerical models (enthalpy, enthalpy-porosity, LBM, FEM), and shell-and-tube/hot-water-tank applications—while identifying AI/ML and field validation as open needs for practical PCM-SWH design.

---

## 2. Problem Being Solved
- PCMs offer high latent heat storage at near-constant temperature but suffer from **low thermal conductivity** (e.g., paraffin **0.15–0.24 W/m·K**), **phase-change leakage**, **subcooling**, and **limited latent heat** when heavily modified.
- Prior reviews often treat **numerical modeling** and **experimental enhancement** separately, without a unified comparison of hybrid TES (sensible + latent) vs pure latent systems for hot-water and solar applications.
- Engineers lack consolidated guidance on **which simulation method** (enthalpy, enthalpy-porosity, heat capacity, LBM) and **which enhancement** (fins vs NePCM vs metal foam vs encapsulation) to use for a given PCM-SWH design problem.
- **Real-world deployment gaps** remain: scaling, encapsulation durability, metal-foam model realism, nanoparticle-induced latent-heat loss, and limited integration of data-driven PCM selection/control.

---

## 3. Key Contributions
1. Integrated review of **latent heat TES** and **hybrid PCM–water tanks** with tables on shell-and-tube modifications, hybrid tank numerics/experiments, fin geometries, and dimensionless groups (Table 11: Nu, Ste, Fo, Bi, Ra, Gr, Pr, Re, Str, Ri, Pe, Mix, \(\eta_{ch}\), \(\eta_{storage}\)).
2. Structured survey of **PCM enhancement**: fins, multi-PCM cascades, nanoparticles (NePCM), porous metal matrices, macro/micro/nano-encapsulation, and shape-stabilized composites (SS-PCM).
3. Detailed exposition of **numerical methods**: enthalpy method Eqs. **(4)–(8)**, enthalpy-porosity Eqs. **(9)–(16)** / **(27)–(29)**, heat capacity, FDM, FVM, **LBM**, FEM, and molecular dynamics—with mesh-quality guidance (skewness, orthogonality).
4. Thermophysical synthesis for **composite PCMs**: single NePCM, hybrid NePCM, EPCM, porous PCM; effective-property models (e.g., Maxwell, Bruggeman, Hamilton–Crosser for nanofluids).
5. Application map for **solar and building thermal systems**, including **solar water heating (40–80 °C)** and cascaded PCM in solar collector storage tanks; future research roadmap explicitly includes **machine learning** for mushy-zone modeling and **pilot-scale validation**.

---

## 4. Methodology
### 4a. System / Experiment Setup
N/A — this is a **literature review** (48 pages, **>300 references**). It does not implement a new physical test rig. It organizes prior work on:
- **Shell-and-tube** and **triplex-tube** LHTES units.
- **Hybrid latent–sensible tanks** (PCM in water storage, macro-encapsulated PCM balls, cascaded packed beds).
- **Enhancement configurations** from cited primary studies (longitudinal/angled/tree/stair fins, CuO/graphene NePCM, Al6061 foam, etc.).

### 4b. Mathematical Models & Equations
**Sensible heat storage:**

- \(Q = m\, c_p\, \Delta T\) — **(1)**

**Thermochemical (illustrative):**

- \(A + Q \rightleftharpoons B + C\) — **(2)**
- \(\mathrm{Ca(OH)_2 \rightleftharpoons CaO + H_2O}\) — **(3)**

**Enthalpy method (Voller-type):**

- \(\dfrac{dH}{dt} = \nabla \cdot (k \nabla T)\) — **(4)**
- \(H(T) = h(T) + \rho\, f(T)\, L\) — **(5)**
- \(h(T) = \int_{T_m}^{T} \rho\, c\, dT\) — **(6)**
- \(f(T) = L\) if \(T > T_m\), else \(0\) if \(T < T_m\) (Heaviside) — **(7)**
- \(\dfrac{\partial h}{\partial t} = \nabla \cdot (\alpha \nabla h) - \rho L \dfrac{\partial f}{\partial t}\) — **(8)**

**Enthalpy-porosity (general vector form):**

- \(\nabla \cdot (\rho \vec{V}) = 0\) — **(9)**
- \(\dfrac{\partial (\rho \vec{V})}{\partial t} + \nabla \cdot (\rho \vec{V}) = -\nabla p + \mu \nabla^2 \vec{V} - \rho_0 \beta (T - T_{ref}) + S\) — **(10)**
- \(\dfrac{\partial (\rho H)}{\partial t} + \nabla \cdot (\rho \vec{V} H) = k \nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) — **(11)**

**Enthalpy–porosity energy (Cartesian example):**

- \(\dfrac{\partial T}{\partial t} + u \dfrac{\partial T}{\partial x} + v \dfrac{\partial T}{\partial y} + w \dfrac{\partial T}{\partial z} = \alpha \left(\dfrac{\partial^2 T}{\partial x^2} + \dfrac{\partial^2 T}{\partial y^2} + \dfrac{\partial^2 T}{\partial z^2}\right) - \rho L_f \dfrac{\partial f}{\partial t}\) — **(16)**

**Combined enthalpy formulation:**

- \(\dfrac{\partial (\rho H)}{\partial t} + \nabla \cdot (\rho \vec{u} H) = k \nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) — **(27)**
- \(H = h + \Delta H\); \(\Delta H = f L_f\); piecewise \(f(T)\) over \(T_{solid}\), mush, liquid — **(28)–(29)**

**Dimensionless groups (Table 11 excerpts):**

- \(\mathrm{Nu} = h d / k\); \(\mathrm{Ste} = c_p \Delta T / L\); \(\mathrm{Fo} = \alpha t / l^2\); \(\mathrm{Bi} = h l / k\); \(\mathrm{Ra} = g \beta \Delta T l^3 / (\nu \alpha)\)

**Metal-foam porosity (Calmidi–Mahajan):**

- \(a_{sf} = \dfrac{3\pi d_f}{\left[1 - e^{-(1-\varepsilon)/0.04}\right] (0.59 d_p)^2}\) — **(109)** (\(e = 0.339\) in related foam correlations)

### 4c. Algorithm / Control Method Steps
N/A — no new control algorithm is implemented. The review **surveys** optimization and simulation workflows from cited papers (e.g., Gao et al. **multi-objective optimization** of cascaded packed-bed TES: exergy **+5%**, TES capacity **−4%**). Future work (Section 12) recommends:
1. Explore **ML/AI** for mushy-zone and phase-change modeling.
2. Integrate PCM with batteries and renewables for hybrid energy management.
3. **Techno-economic optimization** of composition, geometry, and operation.
4. **Field pilot experiments** with industry partners.
5. **Life-cycle assessment** of PCM systems.
6. **Scale-up** manufacturing and modular deployment.

### 4d. Data Sources & Dataset Details
| Source type | Content | Scope |
|-------------|---------|--------|
| Prior journal papers (2016–2025 focus) | Experimental and CFD studies on PCM-LHTES | Global; heavy Elsevier/JEST, Appl. Therm. Eng., Renew. Energy |
| Tabulated PCM properties | Organic/inorganic/eutectic lists (Tables 3–4, 6) | Melting ranges **~6–256 °C** depending on material |
| Review comparisons | Shell-and-tube (Table 8), hybrid tanks (Tables 9–10), fin surveys (Table 5) | Design-parameter meta-analysis |
| Author’s related work [32] | ANN for inclined-enclosure PCM melting (J. Energy Storage **114**, 2025) | Cited as emerging AI-for-PCM example |

No ERA5, NASA POWER, or Indian climate datasets are used in this review itself.

### 4e. Validation Method
N/A as primary research — validation is **by synthesis of published studies**. The review reports benchmark outcomes from cited validation papers, for example:
- Santos et al.: enthalpy-method code validated against **solidification experiments** on finned tubes.
- Neri et al.: **three numerical models validated with experiments** on macro-encapsulated PCM in hot-water tank (but only **40%** latent heat utilization reported).
- Gao et al.: cascaded packed-bed model **validated through experiments**.
- Lee et al. [242]: numerical vs experimental melting in **finned CTES** tank.

---

## 5. PCM Details (if applicable)
- **Materials tested (surveyed, not single study):** Paraffin waxes (**C\(_n\)H\(_{2n+2}\)**), fatty acids (lauric, myristic, palmitic, stearic acids), sugar alcohols, salt hydrates, eutectics (CA-MA-PA + exfoliated graphite), commercial grades **RT15, RT18, RT22 HC, RT27, RT35HC, RT100**, n-eicosane, erythritol, maleic acid, NaNO\(_3\), etc.
- **Melting temperature range:** Application bands cited: refrigeration **−20 to 5 °C**; buildings/electronics **5–40 °C**; **solar water heating 40–80 °C**; broader PCM list spans **~6–256 °C** (Table 3).
- **Latent heat:** Examples — paraffin **~200 J/g**; palmitic acid **206 J/g**; erythritol **340 J/g**; maleic acid latent storage density **103 kWh/m³** (Table 2); Jebasingh eutectic composite **142.2 / 139.5 J/g** melting/solidification.
- **Thermal conductivity:** Paraffin **0.15–0.24 W/m·K**; 10 wt% exfoliated graphite in eutectic: **0.149 → 0.180 W/(m·K)** (+20.8%); RT100/EG composite shows conductivity increasing with packing density; graphene/EG enhancements up to **5000 W/m·K** for nanoparticles (Table 6, material property).
- **Specific heat (solid/liquid):** Water **4.18 kJ/kg·K** (sensible reference); organic PCMs typically **~2–2.5 kJ/kg·K** in tables; effective \(C_{p,eff}\) used in heat-capacity method.
- **Density:** Water **1000 kg/m³**; Al\(_2\)O\(_3\) nanoparticle **3980 kg/m³**; paraffin ~**850–912 kg/m³** range in cited encapsulation studies.
- **Performance metrics reported (from cited works aggregated):** Melting-time reductions up to **80.2%** (tree fins), **71%** (triplex-layer fins), **68%** (5% GNP + fins); finned shell-and-tube melting/solidification **−52% / −43%**; hybrid tank latent utilization **40%** only (Neri); charging efficiency and \(\eta_{storage}\) definitions in Table 11.

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** No new ML model trained in this review. **Surveyed/future:** AI and ML for **PCM selection** from large datasets [32]; **machine learning and AI** recommended for mushy-zone modeling (Section 12, item 1). Related author work: **ANN** for PCM melting in inclined enclosures (J. Energy Storage **115750**, 2025, ref. [32]).
- **Input features / state space:** Not specified for a unified model — review mentions general use of thermophysical databases, climate, and material properties for selection tools.
- **Output / action space:** N/A for this paper.
- **Model architecture:** N/A — cites ANN application externally; discusses **LBM**, **CFD/FLUENT**, **TRNSYS** enthalpy models, **MATLAB** implementations in literature.
- **Hyperparameters:** N/A.
- **Training data size:** N/A.
- **Hardware used for training:** N/A.
- **Performance metrics:** N/A — no original ML experiment.

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** N/A as a primary dataset — review cites studies using **solar irradiation** boundary conditions in CFD (e.g., Xie et al. PCM wall study; Liu et al. PVT with microencapsulated PCM slurry). Barzin et al. [97] mention **weather forecast** with PCM passive buildings (not detailed here).
- **Variables used:** Solar radiation / irradiation as boundary input in cited PCM-wall and solar-collector simulations; stratification metrics (Mix number, Richardson number) for storage tanks.
- **Geographic scope:** Not focused on a single country; includes global literature. **Application temperature** for solar DHW: **40–80 °C** PCM band [108].
- **Temporal resolution:** N/A at review level; cited TRNSYS/annual climate studies use hourly or building-scale timesteps in source papers.
- **Time period covered:** Literature through **2025** (received Oct 2024, accepted Mar 2025).
- **Clear-sky index / derived metrics:** Not computed in this review.

---

## 8. Key Results & Numbers
*Aggregated from studies surveyed; all bullets include numeric values reported in this review.*

- Paraffin wax thermal conductivity: **0.15–0.24 W/m·K** — core limitation for SWH charging rates.
- Sensible heat storage (water): energy density **~70 kWh/m³**; maleic acid latent **~103 kWh/m³**; NaNO\(_3\) latent **~108 kWh/m³** (Table 2).
- Meng et al. (shell-and-tube sensitivity): **+50%** \(c_p\) → **+4%** average heat-storage rate; **+50%** latent heat → **+6%** storage; **1.5×** conductivity → nearly **doubles** average heat-storage rate.
- Kirincic et al. (longitudinal fins, paraffin/water): melting time **−52%**, solidification **−43%** vs plain tube.
- Mhood et al.: optimized fin geometry → melting time reduced up to **50%**.
- Song et al. (tree-shaped fins, MTLHS): complete melting time **−80.2%**.
- Kim et al. (angled fins): **θ\(_f\) = −20°** → average power **+19.3%** vs horizontal fins.
- Gao et al. (cascaded packed-bed solar heating): exergy efficiency **+5%**, TES capacity **−4%** after multi-objective optimization.
- Das et al. (graphene nanosheets, 2 vol%): melting time **−41%** at **60 °C**, **−37%** at **70 °C** HTF.
- Nakhchi et al. (CuO + stair fins): energy storage **+9.1%**, capacity **474.1 kJ**.
- Singh et al. (5% GNP + optimized fins): total melting time **−68%**.
- Lee et al. (finned CTES, enthalpy-porosity): stratified fin design → mean power **+156.3%**.
- Bouzennada et al. (RT-27, inclined fins): melting time **−1.28% to −20.52%**; stored energy **+14.75% to +36.88%** (0° fin best).
- Xu et al. (triplex-layer PCMs + fins): melting time reduced up to **71%**.
- Yang et al. (paraffin in metal foam, angle): full melting time **−12.28%** (0°), **−22.81%** (30°), **−34.21%** (60°) vs reference.
- Jebasingh et al.: **10 wt%** exfoliated graphite → \(k\): **0.149 → 0.180 W/(m·K)** (+20.8%); latent heats **142.2 J/g** (melt), **139.5 J/g** (solidify).
- Xiao et al. (shape-stabilized PCM): light-to-thermal conversion **66.9% → 94.1%** with CuS; enthalpy **194.8 J/g** after **150** cycles (**−5.9%** enthalpy, **−2.6%** efficiency).
- Neri et al. (macro-encapsulated PCM in hot-water tank): only **40%** of PCM latent-heat potential utilized due to thermal transport limits.
- Global energy demand projection cited: **+28% by 2050** [361] motivating PCM deployment.
- Latent heat energy density: **~5–14×** sensible heat (literature comparison); thermochemical **~5×** phase-change systems (Section 2.4).

---

## 9. Baseline Comparison
- **Baseline method(s):** Within reviewed literature: **pure PCM** vs **fin-enhanced**, **NePCM**, **metal-foam composite**, **multi-PCM cascade**, **encapsulated** vs **non-encapsulated**, and **sensible-only** vs **latent** vs **hybrid** tanks (Tables 2, 8–10).
- **Proposed method:** Not a single proposed device — the review’s synthesis favors **hybrid latent + sensible tanks**, **enthalpy-porosity CFD** for convection-dominated melting, and combined **fins + nanoparticles** where latent-heat penalty is acceptable.
- **Improvement margin:** Illustrative spans from surveyed papers: melting time **−52% to −80%** with fins; energy storage **+9.1% to +36.88%** with structured fins/NePCM; hybrid cascaded TES exergy **+5%** with **−4%** capacity trade-off (Gao et al.).
- **Conditions of comparison:** Varies by cited study (geometry, PCM type, HTF temperature, natural vs forced convection); review emphasizes matching **Ste, Ra, Fo** and mesh quality when cross-comparing CFD results.

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — this paper is a **literature review** without a new experimental apparatus. It summarizes hardware from cited work (e.g., shell-and-tube LHTES, hybrid water tanks with macro-PCM capsules, DSC characterization rigs for **RT15/RT22 HC**, finned annular test sections). No Arduino/RPi/DS18B20 deployment in this article.

---

## 11. Limitations Acknowledged by Authors
- PCM phase transitions remain **unsteady and nonlinear**; innovative solutions still needed for conductivity, leakage, and long-term stability (Abstract, Conclusion).
- **Sensible heat storage** requires large temperature swings for capacity — future work should raise capacity without relying only on \(\Delta T\) (Section 11).
- **Fin design** still needs optimization across operating conditions; **multi-PCM** stages help but optimal stage count is unresolved.
- **Nanoparticles** often **reduce latent heat** in experiments; metal-foam **numerical models** need higher geometric realism.
- **Encapsulation** must become **robust, low-cost, long-life** for practical scale-up.
- Numerical methods must better capture **melted PCM flow** and **supercooling**; mushy-zone parameters remain uncertain.
- Gap between **laboratory studies and real-world engineering** — authors call for **pilot projects**, LCA, and industry collaboration (Sections 10–12).
- Hybrid tanks may use only a **fraction of PCM latent capacity** (e.g., **40%** in Neri et al. finding cited).

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant (as implemented).** The review covers passive/active HTF control only through cited studies, not online RL/MPC; it lists **AI/ML for mushy-zone and system optimization** as future work, not deployed embedded control.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Surveys **RT-series** and **NePCM** literature (e.g., TiO\(_2\)/RT-35HC) aligned with Rubitherm/PLUSS selection space, and co-author **ANN–PCM melting** work [32], but no Raspberry Pi / ESP32 closed-loop SWH prototype in this paper.
- **RG3 (Poor alignment with household demand patterns):** **Partially relevant.** Discusses **hot-water tanks**, stratification (Mix number, charging efficiency \(\eta_{ch}\)), and **40–80 °C** solar DHW PCM range, but **no morning/evening draw profiles** or demand-aware control — supports tank/PCM sizing context only.
- **RG4 (Limited real-world experimental validation):** **Highly relevant.** Explicitly states need for **practical experiments, pilot projects, and field validation**; itself is simulation/literature synthesis. Your FYP field/bench validation addresses a gap the authors highlight.
- **RG5 (No predictive optimization under climatic uncertainty):** **Partially relevant.** Mentions **weather forecast + PCM** passive building study [97] and AI for PCM selection; does not implement **ERA5/NASA POWER** or irradiance forecasting for control — supports your Phase 1b climate + forecast narrative as an extension beyond this review.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(Q = m c_p \Delta T\) **(1)** | Sensible energy in tank water | Baseline tank energy without PCM phase change |
| \(H(T) = h(T) + \rho f(T) L\) **(5)** with **(28)–(29)** | Total enthalpy with mushy-zone \(f(T)\) | Grey-box PCM state; melting fraction tracking |
| \(\dfrac{\partial (\rho H)}{\partial t} + \nabla\cdot(\rho \vec{u} H) = k\nabla^2 T - \rho L_f \dfrac{\partial f}{\partial t}\) **(27)** | Enthalpy-porosity energy | `scipy.solve_ivp` + event at \(T_m\); Barqawi-style model extension |
| \(\mathrm{Ste} = c_p \Delta T / L\) | Sensible/latent coupling | Dimensionless groups for RL reward scaling |
| \(\eta_{storage}(t) = (T_{avg}(t)-T_{ini})/(T_i-T_{ini})\) | Charging efficiency of TES | Metric for comparing PPO vs rule-based charging |
| \(a_{sf}\) foam surface area density **(109)** | Metal-foam enhanced PCM (if used) | Optional enhancement path vs pure PCM duct |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **M. E. Zayed et al., "Applications of cascaded phase change materials in solar water collector storage tanks: A review," Sol. Energy Mater. Sol. Cells, 2019 [44]** — Relevant because: Direct **PCM in solar water collector storage tanks** and cascaded CTSPCM configurations for Indian-relevant SWH literature review.
2. **H. Asgharian and E. Baniasadi, "A review on modeling and simulation of solar energy storage systems based on phase change materials," J. Energy Storage, 2019 [35]** — Relevant because: Maps **PCM-SWH simulation methods** (enthalpy, effective heat capacity) aligned with your grey-box + CFD validation.
3. **L. Kalapala and J. K. Devanuri, "Influence of operational and design parameters on PCM based heat exchanger for thermal energy storage – a review," J. Energy Storage, 2018 [36]** — Relevant because: **Shell-and-tube PCM-HX** design parameters for LHTES in solar thermal systems.
4. **A. Arshad et al., "Preparation and characteristics evaluation of mono and hybrid nano-enhanced phase change materials (NePCMs) for thermal management of microelectronics," Energy Convers. Manage., 2020 [29]** — Relevant because: **RT-35HC / hybrid NePCM** property characterization methodology applicable to Rubitherm/PLUSS selection.
5. **T. Bouhal et al., "PCM addition inside solar water heaters: numerical comparative approach," J. Energy Storage, 2018 [337]** — Relevant because: **Numerical PCM integration in solar water heater tanks** — close architectural analog to your PCM-SWH simulator.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | PCM conductivity gap + review gap on integrated modeling | "Commercial paraffin PCMs exhibit \(k \approx 0.15\)–0.24 W/m·K, limiting charge rates; Abdellatif et al. (2025) note few reviews unify enhancement, hybrid TES, and CFD validation for practical SWH." |
| II. Literature Review | Comprehensive PCM-TES modeling review entry | Method: enthalpy-porosity + enhancement survey; Key insight: fin/NePCM can cut melting time **52–80%** in literature but field pilots still needed |
| III. Methodology | Enthalpy-porosity Eqs. (27)–(29) + Ste, Fo, Bi | Adopt **(27)–(29)** with phase events; cite Table 11 dimensionless groups for nondimensionalizing RL states |
| IV. Dataset & Setup | SWH PCM temperature band + RT grades | "Solar DHW PCM applications commonly target **40–80 °C**; commercial RT/paraffin grades surveyed include **RT18–RT100** class materials." |
| V. Results | Literature benchmark spans for improvement claims | Cite fin-enhanced melting **−52% to −71%** and hybrid-tank latent utilization as low as **40%** to motivate demand-aware + climate-adaptive control |
