# State-of-the-Art in Solar Water Heating (SWH) Systems for Sustainable Solar Energy Utilization: A Comprehensive Review

**Authors:** Md. Rashid Al-Mamun, Hridoy Roy, Md. Shahinoor Islam, Md. Romzan Ali, Md. Ikram Hossain, Mohamed Aly Saad Aly, Md. Zaved Hossain Khan, Hadi M. Marwani, Aminul Islam, Enamul Haque, Mohammed M. Rahman, Md. Rabiul Awual  
**Year:** 2023  
**Journal/Conference:** Solar Energy, Vol. 264, Article 111998  
**DOI/Link:** https://doi.org/10.1016/j.solener.2023.111998  
**IEEE Citation:** M. R. Al-Mamun et al., "State-of-the-art in solar water heating (SWH) systems for sustainable solar energy utilization: A comprehensive review," Sol. Energy, vol. 264, p. 111998, 2023, doi: 10.1016/j.solener.2023.111998.

---

## 1. One-Line Summary
This comprehensive SWH review catalogs collector types (**FPC 45–60%**, **CPC 30–50%**, **ETC up to ~84% above FPC**), storage stratification, **PCM integration**, and **nanofluid** gains (**MWCNT +35%**, **Al₂O₃ +28.3%**, **CuO ETC +12.4%**) while identifying cost, stability, and adoption barriers for residential solar water heating.

---

## 2. Problem Being Solved
- Global fossil fuel reserves may deplete by **2050**; forecast energy demand **46 TWh (2100)** / **30 TWh (2150)** drives renewable transition.
- SWH is mature but under-penetrated vs PV due to scattered cost data, low awareness, and performance limits of conventional working fluids.
- Heat storage tank stratification losses and low-conductivity PCMs/nanofluids limit delivered hot-water temperature and system efficiency.
- Need consolidated design guidance on collectors, nanofluids, PCM tanks, and future hybrid SWH research directions.

---

## 3. Key Contributions
1. **Component-level review:** solar thermal collectors, storage tanks, heat exchangers, absorber plates, HTF selection.
2. **Collector benchmarking:** stationary **FPC 45–60%** (25–100 °C); tracking **CPC 30–50%** (60–300 °C); **ETC** higher efficiency band (50–200 °C).
3. **Nanofluid synthesis:** MWCNT, Al₂O₃, CuO, TiO₂, GO, Ag, etc. — quantified FPC/ETC/DASC improvements.
4. **PCM in SWH:** latent storage in tanks/collectors; stratification devices (diffusers, baffles, membranes).
5. **Future roadmap:** hybrid ETC+CPC, nano+PCM fluids, large-scale nanofluid stability criteria (cost, surfactant, sedimentation).
6. **Market perspective:** Arizona case — education/economics drive SWH adoption less than PV.

---

## 4. Methodology
- **Narrative comprehensive review** of peer-reviewed SWH literature (collectors, storage, nanofluids, PCM, CFD/TRNSYS modeling).
- Comparative tables of experimental FPC/ETC installations worldwide (outlet temperature, irradiance, measured efficiency).
- Critical synthesis of nanofluid preparation, volume fraction, stability, and DASC volumetric absorption studies.
- No original experiment — aggregates published performance data and design criteria.

---

## 5. PCM Details (if applicable)
- PCMs integrated in **storage tanks** or **collectors** to enhance thermal performance and reduce tank losses (cites Seddegh et al. latent-HW systems).
- Medium-temperature organic PCMs (**paraffin**, fatty acids) suited to **60–100 °C** SWH range.
- Review recommends investigating **nanofluids combined with PCMs** in SWH loops.
- Salt hydrates and paraffins classified; PCM stratification with tank internals reduces mixing losses.
- **Not RT35/OM35 specific** — project should map Rubitherm/PLUSS products to cited medium-T paraffin band.

---

## 6. AI / ML / Control Details (if applicable)
N/A as primary focus — review mentions **CFD simulation software** and numerical modeling of SWH but not closed-loop AI control.
- Indirect relevance: calls for optimized heat-transfer modeling; your **XGBoost + PPO** fills the intelligent-control gap this review does not cover.

---

## 7. Solar / Climate Data Details (if applicable)
- SWH applications cited across **China, India, Lebanon, Italy, London**, etc.
- Typical test irradiance **300–1100 W/m²**; tilt angles **30–60°**.
- Operating temperature band **60–280 °C** for solar thermal applications broadly; domestic SWH **<100 °C** (FPC) / **>150 °C** (ETC).
- **Project link:** aligns with **NASA POWER / ERA5 / ISRO Solar Calculator** for Coimbatore, Jaisalmer, Kochi resource assessment; India receives **4–7 kWh/m²/day** (cited in related Indian literature).

---

## 8. Key Results & Numbers
- **FPC thermal efficiency:** **45–60%** (25–100 °C operating range).
- **CPC (single-axis tracking):** **30–50%** (60–300 °C).
- **ETC vs FPC:** thermal efficiency up to **84% higher** than FPC.
- **MWCNT nanofluid (FPC):** effectiveness **+35%**; **Al₂O₃:** **+28.3%**.
- **CuO/water in ETC:** collector efficiency **+12.4%**.
- **Al₂O₃/oil nanofluid:** collector efficiency **23.83%** reported experiment.
- **Al₂O₃/synthetic oil:** relative thermal efficiency **+11%** vs conventional fluid.
- **Graphite nanofluid (0.01 vol%):** **122.7%** efficiency metric vs baseline in cited study.
- **MWCNT DASC:** **+10–29%** vs water base fluid.
- **RGO/water-EG DASC:** **70%** efficient at **1000 W/m²**.
- Example FPC lab results: **61.59–73.45%** (Table 2 studies); minichannel FPC **+16.1%** thermal efficiency gain.
- Petroleum consumption **105×** faster than renewable production (cited forecast).

---

## 9. Baseline Comparison
| Enhancement | Baseline | Improvement |
|-------------|----------|-------------|
| MWCNT nanofluid FPC | Water HTF | **+35%** efficiency |
| Al₂O₃ nanofluid FPC | Water HTF | **+28.3%** |
| CuO nanofluid ETC | Water HTF | **+12.4%** |
| ETC collector | FPC | Up to **+84%** efficiency |
| MWCNT DASC | Water | **+10–29%** |
| Stratified tank + internals | Mixed tank | Reduced heat loss (qualitative) |
| SWH vs PV adoption | PV growth | SWH "limited growth since 1970" |

---

## 10. Hardware / Experimental Setup (if applicable)
Review compiles diverse rigs:
- **FPC:** corrugated tubes, heat-pipe absorbers, minichannel absorbers, roll-band absorbers.
- **ETC:** flood-design, all-glass vacuum tubes.
- **DASC:** volumetric nanofluid absorption cells.
- **Working fluids:** water, air, glycol, hydrocarbon, nanofluids.
- **Sensors/methods:** outlet temperature, flow rate, solar irradiance — typical experimental SWH loops.
- **No embedded RPi/Arduino** — supports your custom bench instrumentation as novel contribution.

---

## 11. Limitations Acknowledged by Authors
- Nanofluid **agglomeration, sedimentation, cost, viscosity**, surfactant stability limit scale-up.
- SWH market growth lags PV despite technical maturity.
- Need holistic studies: **cost, lifetime, usability** vs electric/LPG heaters.
- PCM+nano combinations need more experimental validation in SWH loops.
- Performance-focused reviews alone won't drive residential adoption without policy/economics.

---

## 12. Direct Relevance to My Project
- **RG1:** **Gap confirmed** — review lacks real-time adaptive control; motivates **PPO** valve policy.
- **RG2:** **Relevant** — PCM in tanks/collectors and sensor-based modeling cited; your **PCM + AI + ESP32/RPi** prototype is explicitly missing.
- **RG3:** **Relevant** — domestic hot-water use cases (bathing, washing) and stratified storage align with **evening demand** PCM discharge.
- **RG4:** **Relevant** — extensive experimental FPC/ETC benchmarks (**45–73%**) for field/lab comparison; India-friendly deployment context.
- **RG5:** **Partial** — intermittent solar emphasis supports **climate-adaptive** forecasting; no ERA5/NASA workflow — your ERA5 + irradiance ML fills this.

---

## 13. Equations to Reuse or Adapt
**Collector efficiency (standard test form, from cited SWH literature):**
\[
\eta_{th} = F_R\left[\tau\alpha - U_L \frac{(T_{in}-T_{amb})}{G}\right]
\]

**Energy stored in PCM (latent):**
\[
Q_{stored} = m \cdot L \quad \text{(plus sensible terms over } T_m \text{)}
\]

**Nanofluid effective conductivity (mixture models in review):**
\[
k_{nf} = \phi k_p + (1-\phi) k_f
\]

Use \(\eta_{th}\) vs \((T_{in}-T_{amb})/G\) for grey-box validation against FPC/ETC curves.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Gautam et al., SWH technical improvements review, *Renew. Sust. Energy Rev.*, 2017** — prior SWH survey [7].
2. **Seddegh et al., latent-heat SDHW systems, *Renew. Sust. Energy Rev.*, 2015** — PCM tank SWH [15].
3. **Yousefi et al., Al₂O₃–H₂O nanofluid FPSC, *Renew. Energy*, 2012** — **+28%** nanofluid benchmark [22].
4. **Mehmood et al., heat-pipe ETC SWH with gas backup, *Energy Rep.*, 2019** — HP-ETC performance [13].
5. **Sharma, residential SWH adoption Arizona, *J. Clean. Prod.*, 2021** — market barriers [202].

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite global energy demand and SWH as dominant low-temperature solar application (**60–280 °C** band).
- **Section II:** Lit-review table row — collector efficiencies **FPC 45–60%**, **ETC +84% vs FPC**, nanofluid uplifts.
- **Section III:** Justify **PCM storage tank** and stratification in grey-box; optional nanofluid as future work.
- **Section IV:** Benchmark collector **η_th** against cited **69–73%** heat-pipe FPC experiments.
- **Section V:** Compare system COP/energy savings to **+12.4%** (CuO ETC) and **+35%** (MWCNT) as aspirational HTF enhancement bounds.

---
