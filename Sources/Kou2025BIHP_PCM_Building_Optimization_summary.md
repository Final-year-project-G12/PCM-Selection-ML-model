# A Novel Solar Heating Building Integrated Heat Pipes and PCMs: Optimizing Thermophysical Properties and Reducing Energy Consumption

**Authors:** Fangcheng Kou, Nian Zhu, Xin Wang, Yu Zou, Jinhan Mo  
**Year:** 2025  
**Journal/Conference:** Building and Environment, Vol. 285, Article 113674  
**DOI/Link:** https://doi.org/10.1016/j.buildenv.2025.113674  
**IEEE Citation:** F. Kou et al., "A novel solar heating building integrated heat pipes and PCMs: Optimizing thermophysical properties and reducing energy consumption," Build. Environ., vol. 285, p. 113674, 2025, doi: 10.1016/j.buildenv.2025.113674.

---

## 1. One-Line Summary
This paper proposes **BIHP-PCM** (flat gravity **heat-pipe** + **PCM** interior wall), optimizes volumetric enthalpy \(\rho H\), phase-change temperature \(T_m\), and conductivity \(\lambda\) via **PSO** at **61** Chinese cold-region cities, and shows **ESR/IDTD of 30–100%** linearly tracking **RRTD\(_{HS}\)** (solar-radiation-to-temperature-difference ratio), with Tianjin HVAC case reaching **94.4%** energy savings (\(Q\): **1220 → 68 MJ**).

---

## 2. Problem Being Solved
- Solar heating faces **intermittent radiation**; passive PCM walls rely on **low natural-convection** coefficients and deliver limited comfort gains (literature cites only **1–3%** savings for passive PCM sunspaces).
- Active PCM systems (pumps/fans) improve performance (**12.7–40%** load cuts) but add **complexity, parasitic energy, and maintenance**.
- Prior **BIHP** (heat-pipe only) transfers solar heat efficiently by **thermal diode** conduction but lacks sufficient **latent storage**, causing daytime overheating and nighttime drops.
- PCM thermophysical properties (\(\rho H\), \(T_m\), \(\lambda\)) are climate-dependent; no prior work optimized PCM inside **BIHP** across many cities or linked performance to a **climate index** for zoning.

---

## 3. Key Contributions
1. **BIHP-PCM architecture:** L-shaped flat gravity HP (evaporator on south exterior wall, condenser embedded in east/west **PCM interior walls**) — daytime HP \(k_e \approx 2\times10^4\) W/(m·K) charges PCM by **conduction**; nighttime HP blocks reverse loss (**~1/170** forward vs reverse thermal resistance).
2. **Equivalent-specific-heat PCM model** (triangular \(c_p(T)\) over \(\Delta T=2\) °C) coupled with HP and indoor air balance (Eqs. **1–11**).
3. **PSO inverse optimization** of \(\rho H\), \(T_m\), \(\lambda\) for **with-HVAC** (minimize \(Q\), maximize **ESR**) and **without-HVAC** (minimize **IDDC**, maximize **IDTD**) — **20** particles, **30** iterations.
4. **61-city** severe-cold/cold China study using **DeST** climate data; optimal PCM universally favors **max \(\rho H=420\) MJ/m³** and **high \(\lambda\)** (5–9 W/(m·K) without HVAC; 3.5–9 with HVAC).
5. **Climate correlation:** **ESR\(_{OPT}=0.148·RRTD\(_{HS}\)** and **IDTD\(_{OPT}=0.150·RRTD\(_{HS}\)** with **\(R^2>0.98\)**; three application zones (**zero-carbon / good / suitable**).
6. **Experimental validation** (Beijing twin test houses): simulated vs measured room temperature **\(R^2>0.98\)**, mean error **0.3 °C** (BIHP-PCM) and **0.2 °C** (reference).

---

## 4. Methodology
### 4a. System
- South-facing room **4×3×3 m**, window–wall ratio **0.3**, \(T_L=18\) °C comfort lower bound.
- East/west interior walls: **12 cm** brick + **6 cm** PCM layer; HP condenser between brick and PCM.
- Non-optimized PCM: **KF·4H₂O** (potassium fluoride tetrahydrate); reference building = same geometry **without HP/PCM**.

### 4b. Heat-pipe & wall models
- Forward HP heat: \(Q_{HP,fw} = A_{sec} k_e (T_{eva}-T_{con})/l_{eff}\) **(1)** with \(k_e=2\times10^4\) W/(m·K).
- Reverse nighttime conduction **(3)** — ignored in energy balance (two orders smaller).
- PCM latent heat: \(H = \frac{1}{2}\Delta T \Delta c_p\) **(5)**; melting fraction **(7)**.
- Wall conduction **(8)**; outer/inner boundaries **(9–10)** with \(h_{out}=23.0\), \(h_{in}=8.7\) W/(m²·K).
- Indoor air energy balance **(11)** including ACH (0.5 below 26 °C operative temp, 5.0 above), window gap, and \(q_{HVAC}\).

### 4c. Optimization indices
- **Without HVAC:** IDDC **(12)** — integrated cold discomfort; IDTD **(15)** = % reduction vs reference.
- **With HVAC:** seasonal heating energy \(Q\) **(16)**; ESR **(17)** = % savings vs reference.
- **Climate index:** RRTD\(_{HS} = Q_{sol,ave}/(T_{set}-T_{out,ave})\) **(18)** over local heating season.

### 4d. PSO procedure
Six steps: initialize random \((\rho H, T_m, \lambda)\) in box **0–420 MJ/m³**, **10–30 °C**, **0–10 W/(m·K)** → simulate objective → select global best → velocity/position update → **30** iterations → output optimum.

### 4e. Validation experiment
- Twin **2.4×2.4×2.4 m** houses, Beijing, May 11–21; **CaCl₂·6H₂O** PCM **153 kg** (0.09 m³) in **0.20×0.15×0.02 m** boxes; acetone-filled aluminum HP (evap **1.68 m²**, cond **1.36 m²**); ACH **0.1 h⁻¹**.
- **MATLAB** FDM: spatial step **5 mm**, time step **60 s**.

---

## 5. PCM Details (if applicable)
| Parameter | Study range / optimum (examples) |
|-----------|----------------------------------|
| Volumetric enthalpy \(\rho H\) | Search **0–420 MJ/m³**; optimum always **420** (upper bound) |
| Phase-change temperature \(T_m\) | Search **10–30 °C**; Tianjin **19.0 °C** (no HVAC), **19.5 °C** (with HVAC); with HVAC cluster **19.0–20.5 °C** (~**\(T_L+2\) °C**) |
| Thermal conductivity \(\lambda\) | Search **0–10 W/(m·K)**; Tianjin optimum **6.3** (no HVAC), **6.8** (with HVAC) |
| Phase-change range \(\Delta T\) | Fixed **2 °C** |
| PCM layer thickness | **6 cm** on interior walls |
| Validation PCM | **CaCl₂·6H₂O**, \(T_m\approx27\) °C, \(\rho H=289\) MJ/m³ (Table 2 literature) |
| Literature PCMs cited | RT27 (\(\rho H=145\), \(\lambda=0.2\)), SP25A8, L30 (\(\rho H=420\)), TM/EG (\(\lambda=9.72\)) |

**Design rule:** maximize storage capacity and use **high \(\lambda\)** but not always maximum — overly high \(\lambda\) can **prematurely discharge** latent heat to the room.

---

## 6. AI / ML / Control Details (if applicable)
N/A — no machine learning.  
**Control/optimization:** **Particle Swarm Optimization (PSO)** — population **20**, iterations **30**; objectives **IDDC** (passive) or **seasonal heating energy \(Q\)** (HVAC).  
**HVAC rule:** maintain \(T_{op}\geq18\) °C when equipped; optimized PCM reduces HVAC runtime (**131 h** vs **1475 h** reference in Tianjin).

---

## 7. Solar / Climate Data Details (if applicable)
- **Source:** **DeST** building-simulation weather database (not ERA5/NASA POWER).
- **Geography:** **61 cities** in China **severe-cold** and **cold** zones (Fig. 5); includes Tianjin, Harbin, Urumqi, Xi’an, Kashgar, etc.
- **Variables:** heating-season **southern solar radiation** \(Q_{sol,ave}\), **outdoor air temperature** \(T_{out,ave}\), heating degree metrics; derived **RRTD\(_{HS}\)** **(18)**.
- **Temporal resolution:** heating-season integrals (hourly simulation, **60 s** experimental logging).
- **Project mapping:** analogous index for India — ratio of **GHI** (NASA POWER / ERA5) to \((T_{set}-T_{amb})\) for Coimbatore, Jaisalmer, Kochi could pre-score PCM-SWH potential without full TRNSYS.

---

## 8. Key Results & Numbers
- China urban/rural heating share: **27.0%** / **41.5%** of building operation energy.
- Passive PCM sunspace savings cited: **1–3%**; ventilated Trombe wall **12.7%**; heat-pump underfloor PCM **40%**.
- HP daytime thermal resistance **1/170** of nighttime; forward \(k_e\) up to **10⁴ W/(m·K)** cited in intro, **2×10⁴** in model.
- Model validation **\(R^2>0.98\)**; daily mean temp error **0.3 °C** (BIHP-PCM), **0.2 °C** (reference).
- **Tianjin without HVAC:** optimal \((\rho H,T_m,\lambda)=(420, 19.0, 6.3)\); **IDDC=192 K·h**; **IDTD=98.9%** vs **IDDC=2206 K·h** non-optimized (**87.4%** IDTD).
- **Tianjin with HVAC:** \(Q_{RB}=1220\) MJ; non-optimized \(Q=341\) MJ (**ESR=72.1%**); optimized \(Q=68\) MJ (**ESR=94.4%**).
- HVAC on-time \(\tau_{on}\): **1475 h** (ref) → **735 h** (initial BIHP-PCM) → **131 h** (optimized).
- Average operating load \(q'_a\): **19.2 / 10.7 / 12.1 W/m²** (ref / initial / optimized).
- **22/61** cities achieve **zero-carbon** heating without HVAC.
- Linear fits **\(R^2>0.98\):** ESR\(_{OPT}=0.148·RRTD\(_{HS}\); example RRTD=5 → **ESR≈74%**.
- **ESR/IDTD range across cities: 30–100%** (abstract/conclusion).
- **Zoning (Table 3):** zero-carbon RRTD≥**8.5** (ESR/IDTD **100%**); good **6.0–8.5** (ESR **60–100%**, IDTD **70–100%**); suitable **<6.0** (ESR **<60%**, IDTD **<70%**).
- Example city data: Tianjin RRTD\(_{HS}=6.10\), ESR\(_{OPT}=94.4%\), IDTD\(_{OPT}=98.9%\); Dalian ESR\(_{OPT}=100%\); Urumqi ESR\(_{OPT}=34.0%\).

---

## 9. Baseline Comparison
| Configuration | Tianjin seasonal heating \(Q\) | ESR vs reference | IDTD (no HVAC) |
|---------------|-------------------------------|------------------|----------------|
| Reference (no HP/PCM) | **1220 MJ** | 0% | — |
| BIHP-PCM initial (KF·4H₂O) | **341 MJ** | **72.1%** | **87.4%** |
| **BIHP-PCM PSO-optimized** | **68 MJ** | **94.4%** | **98.9%** |
| Passive PCM literature | — | ~**1–3%** | minimal |
| BIHP without PCM (prior work) | — | — | large but comfort gaps remain |

Optimized PCM adds **+22.3 percentage points** ESR over non-optimized BIHP-PCM at Tianjin.

---

## 10. Hardware / Experimental Setup (if applicable)
- **Validation houses:** 2.4 m cube, **150 mm** rock wool envelope, south glazing **2.10×1.50 m**, no windows on other walls.
- **Heat pipes:** aluminum, acetone working fluid, **10%** fill; black-painted evaporator.
- **PCM:** **CaCl₂·6H₂O**, encapsulated boxes.
- **Sensors:** WZY-1 automatic thermometer **±0.2 °C**; TES-132 solar energy meter **±10 W/m²**; **60 s** logging.
- **Temperature points:** indoor at **0.5, 1.0, 1.5 m** height (averaged); HP evap/cond; wall inner/outer surfaces.
- **Platform:** MATLAB simulation; **no RPi/Arduino** in loop — aligns with your bench-scale validation gap (RG4) but shows sensor specs usable for field tests.

---

## 11. Limitations Acknowledged by Authors
- Optimal \(\rho H\) hits **upper bound 420 MJ/m³** — higher enthalpy PCMs could improve further.
- Validation experiment **not during heating season** and envelope differs from typical BIHP-PCM; authors argue heat-transfer physics still valid.
- PCM position, thickness, and \(\Delta T\) fixed — only three properties optimized.
- PSO uses modest swarm (**20×30**) — global optimum not guaranteed.
- Results specific to **Chinese severe-cold/cold** climates and south-facing geometry.
- Without HVAC, comfort not guaranteed all season; HVAC case still needs backup heat in weak-solar weeks.

---

## 12. Direct Relevance to My Project
- **RG1 (real-time adaptive control):** **Not relevant** — seasonal PSO design, no online control or DRL; but **HVAC on-time reduction (131 vs 1475 h)** motivates demand-aware charging strategies.
- **RG2 (integrated PCM–AI–hardware):** **Partially relevant** — full **HP+PCM hardware** validated experimentally, but **no AI** and not a compact SWH tank; heat-pipe diode concept transferable to collector-to-storage coupling.
- **RG3 (household demand):** **Relevant** — optimizes \(T_m \approx T_{demand}-2\) °C and quantifies **latent discharge when room nears 18 °C**; maps to evening hot-water draw aligning PCM melt plateau with comfort/demand.
- **RG4 (field validation):** **Relevant benchmark** — real-house experiment with **±0.2 °C** sensors and pyranometer; **\(R^2>0.98\)** model agreement supports your grey-box + bench validation target.
- **RG5 (climate uncertainty):** **Highly relevant** — **RRTD\(_{HS}\)** linear predictor enables climate-adaptive PCM selection across cities; direct parallel to classifying **Coimbatore / Jaisalmer / Kochi** before deployment using NASA POWER or ERA5.

---

## 13. Equations to Reuse or Adapt
**HP forward heat flux:**
\[
Q_{HP,fw} = \frac{A_{sec}\, k_e\, (T_{eva}-T_{con})}{l_{eff}}
\]

**Energy saving ratio (HVAC case):**
\[
\mathrm{ESR} = \frac{Q_{RB}-Q_{HP}}{Q_{RB}}\times 100\%
\]

**Climate resource index:**
\[
\mathrm{RRTD}_{HS} = \frac{Q_{sol,ave}}{T_{set}-T_{out,ave}}
\]

**Empirical potential (optimized, with HVAC):**
\[
\mathrm{ESR}_{PCM,OPT} = 0.148 \cdot \mathrm{RRTD}_{HS}
\]

**PCM latent heat (equivalent specific heat triangle):**
\[
H = \tfrac{1}{2}\Delta T\,\Delta c_p
\]

**Reward/penalty ideas for DRL:** minimize \(Q_{HVAC}\) or IDDC; bonus when \(T_{op}\) stays above \(T_L\) without auxiliary heat; penalize premature melt (\(T_m\) too low) or early discharge (\(T_m\) too high).

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Gong et al., L-shaped flat gravity heat-pipe solar building, prior BIHP work** — thermal diode foundation [14–16].
2. **Kou et al., PSO optimization of BIHP conventional walls, *Build. Environ.* prior** — optimization framework [36, 41].
3. **Zeng et al., δ-function optimal envelope specific heat** — theoretical PCM equivalence [30, 31].
4. **Soares et al., PSO for PCM drywalls** — metaheuristic PCM sizing [32].
5. **Guo/Zhu passive PCM Trombe studies** — low passive savings baseline [8–10].

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite **27%/41.5%** heating energy share and **intermittent solar** challenge; contrast passive PCM **1–3%** vs BIHP-PCM **up to 94.4%** ESR.
- **Section II:** Position BIHP-PCM as **heat-pipe-enhanced PCM storage** alternative to convection-limited SWH tanks; include in lit-review table with PSO-optimized \(T_m\), \(\rho H\), \(\lambda\).
- **Section III:** Adapt **RRTD\(_{HS}\)** for Indian cities to pre-select **RT35–RT64HC / OM35–OM50** melt points before RL training.
- **Section IV:** Reference validation sensors (**±0.2 °C**, **±10 W/m²**, 60 s) for your **DS18B20 + pyranometer** logging protocol.
- **Section V:** Benchmark **ESR 72–94%** (Tianjin) or **IDTD 98.9%** as aspirational seasonal metrics if extending project from daily control to seasonal simulation.

---
