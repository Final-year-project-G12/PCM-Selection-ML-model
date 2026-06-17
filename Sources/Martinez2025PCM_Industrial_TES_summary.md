# Phase Change Materials for Thermal Energy Storage in Industrial Applications

**Authors:** Franklin R. Martínez, Emiliano Borri, Saranprabhu Mani Kala, Svetlana Ushak, Luisa F. Cabeza  
**Year:** 2025  
**Journal/Conference:** Heliyon, Vol. 11, Article e41025  
**DOI:** https://doi.org/10.1016/j.heliyon.2024.e41025  
**IEEE Citation:** F. R. Martínez, E. Borri, S. M. Kala, S. Ushak, and L. F. Cabeza, "Phase change materials for thermal energy storage in industrial applications," Heliyon, vol. 11, no. e41025, 2025, doi: 10.1016/j.heliyon.2024.e41025.

---

## 1. One-Line Summary
This study compiles **65** mid-temperature (**60–80 °C**) and **36** high-temperature (**150–250 °C**) PCMs from literature and commercial datasheets, then experimentally characterizes **14** shortlisted materials (DSC, TGA/DSC, Hot Disk), showing large gaps versus published \(T_m\), \(\Delta H\), \(k\), and especially **thermal stability** data.

---

## 2. Problem Being Solved
- Industry emitted **9.0 Gt CO₂** in 2022 (~**25%** of global energy-system emissions), with slow efficiency and renewable uptake (Introduction, IEA [1]).
- PCM-TES can bridge supply–demand mismatch for industrial heat (**60–80 °C** and **150–250 °C** bands targeted for heat-pump-coupled storage), but **no consolidated property database** exists for these ranges (Section 2).
- Literature and vendor datasheets report inconsistent **melting enthalpy**, **degradation temperature**, and **thermal conductivity** — selection remains difficult (Abstract, Section 4).
- Many catalogued PCMs lack complete property sets (density, \(C_p\), \(k\), \(T_{deg}\), NFPA 704) in published tables (Tables 1–2 footnotes).

---

## 3. Key Contributions
1. **Screening database:** **65** PCMs for **60–80 °C** (Table 1) and **36** for **150–250 °C** (Table 2) from Scopus literature + commercial sheets (**Rubitherm**, **PCMproducts/PLUSS**, **CRODA**).
2. **Shortlists:** **8** mid-temperature + **6** high-temperature candidates (Tables 3–4) including **RT 54 HC, RT 55, RT 64 HC**, **E 58**, salt hydrates, **palmitic/stearic acid**, nitrate salts.
3. **Experimental characterization of 14 PCMs** with **METTLER TOLEDO DSC 3+** and **TGA/DSC 3+** (±**0.1 °C**, ±**3 J/g**); **Hot Disk TPS 2500 S** (Kapton 5506, mean deviation **1×10⁻⁴**); **3** thermal cycles per DSC sample at **1 K/min**.
4. **Cross-validation:** Literature vs measured variances up to **−98%** \(\Delta H\) (hydrate mixture), **+80%** \(\Delta H\) (RT 55 TGA), **+266%** \(k\) (NaNO₃–KNO₃ 60–40), **−91%** \(\Delta H\) (E 58).
5. **Open dataset:** Characterization data deposited at https://doi.org/10.34810/data1822 (Data availability).

---

## 4. Methodology
### 4a. System / Experiment Setup
**Type:** Materials screening + **laboratory thermophysical characterization** (no full-scale TES tank or SWH loop).

**Temperature targets:**
- **Mid:** **60–80 °C** (aligned with dairy pasteurization, drying, etc., and overlapping domestic SWH PCM band).
- **High:** **150–250 °C** (industrial process heat, solar salt applications).

**Purchased samples (Section 2.2):**
- Salts/acids: Mg(NO₃)₂·6H₂O, MgCl₂·6H₂O, palmitic acid, stearic acid, LiNO₃, NaNO₃, KNO₃ (Merck/VWR/Panreac).
- Commercial PCMs: **RT 54 HC**, **RT 55**, **RT 64 HC** (Rubitherm); **E 58** (PCM Products UK).

**DSC/TGA sample prep:** ~**15 mg** in Al crucibles (40 µL, sealed) or sapphire crucibles (70 µL, open) under **N₂**; scan from ~**50 °C below** to ~**50 °C above** literature \(T_m\) (Fig. 1).

**Hot Disk samples:** Compact flat discs (Fig. 2); **3** repeat measurements per PCM after parameter convergence.

### 4b. Mathematical Models & Equations
No transient TES or CFD model — **correlations used only for reporting deviations:**

- **Percent variance (property X):** \(\mathrm{Var}(\%) = \dfrac{X_{exp} - X_{lit}}{X_{lit}} \times 100\)

**Dimensionless groups cited in selection context (literature review, not derived here):**
- **Stefan number** \(\mathrm{Ste} = c_p \Delta T / L\) — discussed in related Cabeza/Zalba reviews [17, 21] for PCM heat transfer.

**Heat transfer correlations referenced for future HX design (from cited work, Section 1):**
- Dittus–Boelter-type relations appear in linked PCM-HX literature [81, 85] but are **not fitted** in this paper.

**Energy storage density (implicit selection criterion):**
- \(E_{latent} \approx \rho \cdot \Delta H_{melting}\) (J/m³) — used conceptually when comparing \(\Delta H\) and \(\rho\) in Tables 1–2.

### 4c. Algorithm / Control Method Steps
N/A — no control system or ML. **Selection workflow:**
1. Literature + vendor database search (Scopus; Rubitherm, PCMproducts, PLUSS, CRODA).
2. Filter by \(T_{melting}\) in target bands; compile \(T_m\), \(\Delta H\), \(C_p\), \(\rho\), \(k\), \(T_{deg}\), **NFPA 704** (Section 2.1).
3. Expert shortlist: include **commercial organics** + **inorganic/organic acids/salts** (Tables 3–4).
4. **DSC** (3 cycles, discard cycle-1 average for powder vs recrystallized sample) → **TGA/DSC** (25–250 °C mid, 25–400 °C high) → **Hot Disk** \(k_{solid}\).
5. Compare to literature/datasheet; flag decomposition before melt (e.g., salicylic acid).

### 4d. Data Sources & Dataset Details
| Source | Content | Scope |
|--------|---------|--------|
| Scopus scientific literature | PCM property tables | Global publications |
| **Rubitherm** datasheets [24] | RT series PCMs | Commercial organics |
| **PCMproducts / PLUSS** [34, 46, 77] | PlusICE, E 58, etc. | Commercial |
| **CRODA CRODATHERM** [16] | Paraffin products | Commercial |
| NASA CR-51363 handbook [25, 35] | Legacy PCM data | Reference |
| Open repository | Measured DSC/TGA/k | https://doi.org/10.34810/data1822 |

**Counts (Section 3.1):** Mid — **45** literature + **20** commercial entries → **65** total; High — **30** + **6** → **36** total.

### 4e. Validation Method
- **Internal:** DSC/TGA instrument accuracy ±**0.1 °C**, ±**3 J/g**; Hot Disk repeatability target mean deviation **1×10⁻⁴**; DSC **3-cycle** repeatability check.
- **External:** Compare measured vs literature/datasheet; Table 19 summary with **Var (%)** columns.
- **Example variances (Table 19, DSC vs literature):**
  - **RT 54 HC:** \(T_m\) **55.4 °C** (lit. **54 °C**), \(\Delta H\) **172.1 J/g** (**−5%**), \(k\) **0.23 W/m·K** (**+15%**).
  - **RT 55:** \(\Delta H\) **114.5 J/g** (**−28%**); TGA \(\Delta H\) **+80%** vs datasheet.
  - **RT 64 HC:** \(\Delta H\) **166.9 J/g** (**−31%**), \(k\) **0.33 W/m·K** (**+64%**).
  - **E 58:** \(\Delta H\) **13.8 J/g** (**−91%** vs **145 J/g** rated) — **unsuitable** as tested.
  - **Palmitic acid:** \(\Delta H\) **182.7 J/g** at **64.0 °C**; \(k\) **0.26 W/m·K** (**+73%** vs **0.15** lit.).
  - **Stearic acid:** **70.4 °C**, **194.2 J/g**, \(k\) **0.26 W/m·K**.
  - **NaNO₃–KNO₃ (60–40):** \(T_m\) **223.2 °C**, \(\Delta H\) **85.8 J/g** (**−21%**), \(k\) **0.88 W/m·K** (**+266%** vs **0.24** lit.).
  - **LiNO₃:** \(T_m\) **~250 °C**, \(\Delta H\) **276.9 J/g** (**−8 to −25%** vs lit. **370 J/g**), \(k\) **0.84 W/m·K**.

---

## 5. PCM Details (if applicable)
*Primary focus: characterized + catalogued materials. Rubitherm grades directly match FYP PCM family.*

### Mid-temperature band (60–80 °C) — selected & tested
| Material | \(T_m\) (°C) lit. / exp. | \(\Delta H\) (J/g) lit. / exp. | \(k_{solid}\) (W/m·K) lit. / exp. | \(T_{deg}\) notes |
|----------|------------------------|-------------------------------|----------------------------------|-------------------|
| **RT 54 HC** | 54 / **55.4** | 182 / **172.1** | 0.20 / **0.23** | Onset **130.3 °C**; use below **130 °C** |
| **RT 55** | 55 / **55.2** | 158 / **114.5** | 0.20 / **0.27** | TGA \(T_m\) **+10 °C** vs DSC |
| **RT 64 HC** | 64 / **55.5**† | 242 / **166.9** | 0.20 / **0.33** | †DSC peak lower than grade name |
| **E 58** | 58 / **57.7** | 145 / **13.8** | 0.69 / failed Hot Disk | Rated \(T_{deg}\) **120 °C**; **not validated** |
| **Palmitic acid** | 55–69 / **64.0** | 163–222 / **182.7** | 0.15–0.17 / **0.26** | Multiple lit. \(T_m\) values |
| **Stearic acid** | 67.8 / **70.4** | 198.9 / **194.2** | 0.17 / **0.26** | Stable cycling |
| **Mg(NO₃)₂·6H₂O + MgCl₂·6H₂O (80–20)** | 60 / **no clear peak** | 150 / **2.8** | n.a. | **−98%** \(\Delta H\); dehydration dominates |
| **Mg(NO₃)₂·6H₂O + MgCl₂·6H₂O (60–40)** | 60 / **59.6** | 132.3 / **28.9** | n.a. | **−78%** \(\Delta H\) |

†Nominal vs measured melt discrepancy flagged in Table 19.

### High-temperature band (150–250 °C) — tested subset
| Material | \(T_m\) (°C) | \(\Delta H\) (J/g) exp. | \(k_{solid}\) (W/m·K) exp. |
|----------|--------------|-------------------------|--------------------------|
| **LiNO₃–NaNO₃–KNO₃ (20-28-52)** | **175.9** | **103.7** | **0.69** |
| **Salicylic acid** | decomposes before melt | — | — |
| **LiNO₃–NaNO₃ (49–51)** | **175.9** | **66.7** (**−75%** vs lit.) | **0.56** |
| **NaNO₃–KNO₃ (50–50)** | **212.7** | **65.4** (**−35%**) | **0.91** |
| **NaNO₃–KNO₃ (60–40)** | **223.2** | **85.8** | **0.88** |
| **LiNO₃** | **249.6** | **276.9** | **0.84** |

### Catalog examples (Table 1, not all tested)
- **RT 60:** \(T_m\) **60 °C**, \(\Delta H\) **160 J/g**, \(\rho_s\) **880 kg/m³**, \(k\) **0.20 W/m·K**, \(T_{deg}\) **80 °C**
- **RT 80:** \(T_m\) **80 °C**, \(\Delta H\) **220 J/g**, \(\rho_s\) **900 kg/m³**
- **PureTemp 60:** \(\Delta H\) **220 J/g**; **CRODATHERM 60:** **217 J/g** at **60 °C**
- **Ba(OH)₂·8H₂O:** \(\Delta H\) up to **301 J/g** at **78 °C**; \(\rho_s\) **2180 kg/m³**

- **Performance metrics reported:** Melting enthalpy **J/g**, degradation/onset temperature **°C**, solid thermal conductivity **W/m·K**, NFPA 704 hazard class **1–3**; no system COP or tank efficiency (materials study only).

---

## 6. AI / ML / Control Details (if applicable)
N/A — materials characterization and literature compilation only; no machine learning, forecasting, or TES control.

---

## 7. Solar / Climate Data Details (if applicable)
N/A — industrial process-heat framing; no solar irradiance, ERA5, NASA POWER, or Indian climate datasets. **Indirect link:** mid-temperature band **60–80 °C** overlaps solar DHW / SWH PCM operating range cited in other Cabeza-group work.

---

## 8. Key Results & Numbers
- **65** mid-temperature + **36** high-temperature PCMs catalogued; **14** experimentally characterized (**8** + **6**).
- **RT 54 HC:** DSC \(\Delta H\) **172.1 J/g** (**−5%** vs **182 J/g** datasheet); \(k\) **0.23 W/m·K** (**+15%**); degradation onset **130.3 °C**.
- **RT 55:** DSC \(\Delta H\) **114.5 J/g** (**−28%**); TGA/DSC \(\Delta H\) **284.9 J/g** (**+80%**); \(k\) **0.27 W/m·K** (**+34%**).
- **RT 64 HC:** DSC \(\Delta H\) **166.9 J/g** (**−31%** vs **242 J/g**); \(k\) **0.33 W/m·K** (**+64%**).
- **E 58:** \(\Delta H\) **13.8 J/g** (**−91%** vs **145 J/g**) — material **not reliable** per authors’ tests.
- **Palmitic acid:** \(T_m\) **64.0 °C**, \(\Delta H\) **182.7 J/g**, \(k\) **0.26 W/m·K** (**+73%** vs **0.15** literature).
- **Stearic acid:** \(T_m\) **70.4 °C**, \(\Delta H\) **194.2 J/g**, \(k\) **0.26 W/m·K**.
- **Mg(NO₃)₂/MgCl₂ (80–20):** \(\Delta H\) **2.8 J/g** (**−98%** vs **150 J/g** literature) — **no usable phase-change peak**.
- **LiNO₃:** DSC \(\Delta H\) up to **456.5 J/g** on TGA branch (**+52%** vs **370 J/g** lit.); \(k\) **0.84 W/m·K** vs **1.70** lit. (**−51%**).
- **NaNO₃–KNO₃ (60–40):** \(k\) **0.88 W/m·K** (**+266%** vs **0.24 W/m·K** literature).
- **Salicylic acid:** **decomposes before melting** — excluded as PCM.
- DSC equipment accuracy: ±**0.1 °C**, ±**3 J/g**; heating rate **1 K/min**; **3** cycles.
- Industry context: **9.0 Gt** industrial CO₂ (2022); target TES bands **60–80 °C** and **150–250 °C**.

---

## 9. Baseline Comparison
- **Baseline method(s):** Published literature values and **manufacturer datasheets** (Rubitherm, PCM Products, etc.) vs **this study’s DSC/TGA/Hot Disk** measurements.
- **Proposed method:** Unified experimental protocol (3-cycle DSC, TGA stability, Hot Disk \(k\)) on **14** shortlisted PCMs.
- **Improvement margin:** Not “better performance” — exposes **gaps**: e.g., **RT 55** enthalpy **−28%** (DSC); **E 58** **−91%**; hydrate mix **−98%**; conductivities **+15% to +266%** vs literature for several salts.
- **Conditions:** Same nominal chemistry; powder vs recrystallized cycle-1 excluded from averages (Section 3.4).

---

## 10. Hardware / Experimental Setup (if applicable)
- **Physical components:** METTLER **STARe DSC 3+**; **STARe TGA/DSC 3+**; **Hot Disk TPS 2500 S** (sensor Kapton **5506 F2**); Al crucibles **40 µL**; sapphire **70 µL**; N₂ purge.
- **Sensor specs:** Temperature ±**0.1 °C**; enthalpy ±**3 J/g**; TGA balance ±**0.00001 g**.
- **Embedded/compute platform:** N/A — lab calorimetry only.
- **Test environment:** GREiA lab, **Universitat de Lleida**, Spain (authors also affiliated with **University of Antofagasta**, Chile).
- **Test duration:** **3** thermal cycles per PCM (DSC); TGA scans **25–250 °C** or **25–400 °C**; **3** Hot Disk repeats per sample.

---

## 11. Limitations Acknowledged by Authors
- **Degradation temperature** was the **most difficult parameter** to find in literature and is critical for safety and operating limits (Section 4).
- **DSC vs TGA/DSC enthalpy** can disagree strongly (e.g., RT 55 **−28%** vs **+80%**); authors state **DSC is more precise** for enthalpy (Section 3.3.1–3.3.2).
- **First DSC cycle** excluded from averages because powder packing differs from recrystallized material (Section 3.4).
- Many catalogued PCMs lack complete property rows in Tables 1–2 (missing \(C_p\), \(\rho\), \(k\), or \(T_{deg}\)).
- Study defines materials for **industrial** TES — **next step** still requires mapping to **final application operating temperature** before tank design (Section 4).
- **E 58**, **salicylic acid**, and some hydrates **failed** stability or phase-change criteria — not all shortlisted materials are viable.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant.** Pure materials screening; no controllers, pumps, or charging logic.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Provides **verified Rubitherm RT 54 HC / RT 55 / RT 64 HC** properties (close to **RT35–RT64HC** family) for simulator parameterization — but **no** RPi/ESP32/DS18B20 tank prototype.
- **RG3 (Poor alignment with household demand patterns):** **Not Relevant.** Industrial heat processes (pasteurization, drying, etc.), not DHW draw profiles.
- **RG4 (Limited real-world experimental validation):** **Partially relevant.** Rigorous **lab DSC/TGA/k** validation of commercial PCMs (including RT grades) supports using **measured** not datasheet-only properties in your model — but **no** full SWH field test.
- **RG5 (No predictive optimization under climatic uncertainty):** **Not Relevant.** No weather data or forecast-driven optimization.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(Q_{stored} \approx m \cdot \Delta H_{melting}\) | Latent storage capacity | Size PCM mass in tank for target kWh |
| \(\mathrm{Var}(\%) = (X_{exp}-X_{lit})/X_{lit} \times 100\) | Property uncertainty | Sensitivity bounds for **RT35/OM35** in grey-box model |
| \(Nu = 1.86\left(\frac{Re\cdot Pr}{L/L_p}\right)^{1/3}\) (from related HX refs [43]) | Tube PCM convection | If modeling coil in tank (optional) |
| \(Q_o = \dot{m} c_p (t_{out}-t_{in})\) | Sensible heat (test fluids) | Calibrate charging experiments |
| Enthalpy averaging: mean(cycles 2–3) | Stable PCM cycling | Lab protocol for validating PLUSS/Rubitherm batches |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **L. F. Cabeza, A. Castell, et al., "Materials used as PCM in thermal energy storage in buildings: a review," Renew. Sustain. Energy Rev., 2011 [17]** — Relevant because: Foundational **building PCM** database overlapping SWH temperatures.
2. **J. Pereira da Cunha, P. Eames, "Thermal energy storage for low and medium temperature applications using phase change materials – a review," Appl. Energy, 2016 [15]** — Relevant because: **Low/medium-temperature PCM-SWH/TES** applications review.
3. **B. Zalba, J. M. Marín, L. F. Cabeza, H. Mehling, "Review on thermal energy storage with phase change," Appl. Therm. Eng., 2003 [21]** — Relevant because: Classic **PCM enthalpy + heat transfer** reference for FYP theory section.
4. **L. Miró, C. Barreneche, et al., "Health hazard, cycling and thermal stability as key parameters when selecting a suitable PCM," Thermochim. Acta, 2016 [103]** — Relevant because: **Thermal cycling and stability** selection criteria for long-life SWH PCM.
5. **J. Li, et al., "A hybrid photovoltaic and water/air based thermal (PVT) solar energy collector with integrated PCM for building application," Renew. Energy, 2022 [8]** — Relevant because: **PCM + solar thermal** system at building scale from same research network.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Industrial decarbonization + PCM data gap | "Martínez et al. (2025) note missing compiled PCM data for 60–80 °C and 150–250 °C, with measured \(\Delta H\) differing up to 91% from datasheets for commercial grades." |
| II. Literature Review | Rubitherm validation entry | Method: DSC/TGA/Hot Disk on **RT 54 HC, RT 55, RT 64 HC**; Key: **RT 55** \(\Delta H\) **114.5 J/g** (−28% vs catalog **158 J/g**) |
| III. Methodology | Use measured properties, not datasheet-only | Adopt **RT 64 HC** \(k_{solid}\)=**0.33 W/m·K**, \(\Delta H\)=**166.9 J/g** for Coimbatore/Jaisalmer/Kochi simulations |
| IV. Dataset & Setup | Mid-temp PCM band overlap | **60–80 °C** band includes **RT 60/65/70/80** catalog entries (e.g., **RT 80:** **220 J/g**, **900 kg/m³**) |
| V. Results / Discussion | Uncertainty justification | Cite **−5% to −31%** \(\Delta H\) variance on Rubitherm grades as motivation for **batch calibration** before deployment |
