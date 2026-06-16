# Using the Taguchi Method and Grey Relational Analysis to Optimize the Parameter Design of Flat-Plate Collectors with Nanofluids and Phase Change Materials in an Integrated Solar Water Heating System

**Authors:** Guan-Rong Chen, Ting-Wei Liao, Chien-Chun Hsieh, Jagadish Barman, Chao-Yang Huang, Chung-Feng Jeffrey Kuo  
**Year:** 2025  
**Journal/Conference:** Energy Conversion and Management: X, Vol. 26, Article 100910  
**DOI/Link:** https://doi.org/10.1016/j.ecmx.2025.100910  
**IEEE Citation:** G.-R. Chen et al., "Using the Taguchi method and grey relational analysis to optimize the parameter design of flat-plate collectors with nanofluids, and phase change materials in an integrated solar water heating system," Energy Convers. Manag.: X, vol. 26, p. 100910, 2025, doi: 10.1016/j.ecmx.2025.100910.

---

## 1. One-Line Summary
This study combines **RT35HC** PCM, **CuO** nanofluid, and flat-plate collectors in a **TRNSYS**-simulated integrated SWH, optimizes **9** factors via **L36 Taguchi DOE** and **grey relational analysis (GRA)**, and achieves **94.2%** thermal storage efficiency and **31.7 h** heat retention at **30 °C** target—**+28%** efficiency and **+14.6 h** retention vs the non-optimized baseline.

---

## 2. Problem Being Solved
- SWH systems suffer from **low thermal storage efficiency** and **insufficient heat retention** after sunset under intermittent solar input.
- Prior work optimized nanofluids or PCMs **separately**; few studies jointly integrate both with systematic multi-objective parameter design for flat-plate SWH.
- Single-response Taguchi optimization cannot simultaneously maximize **thermal storage efficiency** and **heat retention time**—requiring GRA multi-quality fusion.
- Physical multi-factor experiments are costly; need validated simulation-based DOE (TRNSYS) with confirmation runs.

---

## 3. Key Contributions
1. **Novel integrated architecture:** nanofluid + **Rubitherm RT35HC** PCM + flat-plate collector in one closed-loop SWH (FPC, **0.04 m³** tank, PCM tubes, pump, pyranometer, PT100 sensors).
2. **L36 orthogonal array** with **9 control factors** × **36** TRNSYS runs; S/N (larger-the-better), MEA, and ANOVA for each quality characteristic.
3. **GRA multi-objective optimization** merging thermal storage efficiency and heat retention into a single **grey relational grade (GRG)**.
4. **Lumped thermal + electrical analog model** for FPC layers (glass, air gap, absorber, fluid, insulation) with TRNSYS validation within **5%** of physical measurements.
5. **Confirmed optimum:** PCM on, **20%** PCM volume, **14** PCM tubes, **CuO** nanofluid, **0.02 kg/s** flow, **9** collector tubes, **copper** plate, tilt **22.4°**, azimuth **0°** (south).
6. **Performance claim:** first reported **nanofluid + PCM** combined SWH optimization reaching **94.2%** storage efficiency—exceeding literature PCM-only (~64–79%) and nanofluid-only (~50–86%) benchmarks cited in Table 26.

---

## 4. Methodology

### 4a. System Setup
- **Collector:** FPC **505 × 320 mm**, tube OD **25 mm**, ID **24 mm**; baseline **9** tubes, **0.15 m²** area.
- **Storage:** Tank **0.04 m³**, height **500 mm**; PCM pipe height **450 mm**, baseline **12** tubes, volume **0.0037 m³** per configuration.
- **PCM:** **Rubitherm RT35HC** organic paraffin (heating/cooling curves from manufacturer data; minimal hysteresis).
- **Working fluids:** Water, **Al₂O₃** nanofluid, **CuO** nanofluid (properties Tables 8–9).
- **Instrumentation (physical rig):** PYR2-420 pyranometer (300–2900 nm, RS485), Galltec-Mela ambient T/RH, **PT100** sensors (±0.25 K), RP flow meter (±2.5% FSD), FORMOSA RS-15/6GWS pump, BCT TF 200S data logger.
- **Simulation:** TRNSYS modular architecture (Fig. 4); parameters tuned to match physical system.

### 4b. Taguchi DOE (Table 7)

| Factor | Symbol | Levels |
|--------|--------|--------|
| PCM material | A | No paraffin / **RT35HC** |
| PCM volume | B | **10%**, 15%, **20%** of tank volume |
| PCM tube count | C | 12, **14**, 16 |
| Working fluid | D | Water, Al₂O₃, **CuO** |
| Mass flow rate | E | **0.02**, 0.025, 0.03 kg/s |
| Collector tubes | F | **9**, 10, 11 |
| Plate material | G | **Cu**, Al, stainless steel (M) |
| Tilt angle | H | 20.4°, **22.4°**, 24.4° |
| Azimuth | I | −45°, **0°** (south), +45° |

- Daily water demand assumption: **≥30 L**; PCM volume levels anchored at **10%** baseline + **5%** increments.

### 4c. Quality Metrics
1. **Thermal storage efficiency** — larger-the-better S/N (Eq. 1).
2. **Heat retention time** — hours after sunset until tank T drops below **30 °C** target.

### 4d. GRA Procedure
- Normalize both S/N sequences (Eq. 15, larger-the-better).
- Grey relational coefficient \(\zeta_i(k)\) with distinguishing coefficient **ζ = 0.5** (Eq. 16).
- Grey relational grade \(\gamma_i\) averaged over responses (Eq. 17).
- Select factor levels maximizing mean GRG (Table 23).

### 4e. Validation
- TRNSYS vs physical system: accept if error **< 5%**.
- Single-quality confirmation: 5 runs; S/N must fall in **95% CI**.
- Multi-quality confirmation: efficiency **94.2%**, retention **31.7 h**; S/N **39.481** (efficiency), **30.021** (retention).

---

## 5. PCM Details (if applicable)

| Property | RT35HC (Rubitherm) |
|----------|-------------------|
| Type | Organic paraffin (solid–liquid) |
| Role | Latent TES in dedicated PCM tubes in storage tank |
| Factor A | Level 1 = no PCM; Level 2 = **with PCM** (dominant performance gain) |
| Volume levels | **10%**, 15%, **20%** of total tank volume (30 L demand basis) |
| Tube count | 12 / **14** / 16 tubes (heat transfer area) |
| Optimal (GRA) | **20%** volume, **14** tubes |
| Behavior | Heating/cooling enthalpy curves nearly overlap (Fig. 17) — low hysteresis |
| PCM-less vs PCM | Figs. 18–20 show both efficiency and retention improve substantially when PCM enabled |

**Project alignment:** RT35HC is in your **Rubitherm RT35–RT64HC** screening set; Chen’s **20% PCM volume** and **14-tube** layout provide a documented DOE baseline for tank geometry in grey-box modeling.

---

## 6. AI / ML / Control Details (if applicable)
N/A — **classical DOE + GRA + TRNSYS simulation**; no ML/DRL/MPC.  
**Relevance:** Taguchi+GRA is your **Objective 1 PCM selection** methodology (per Presentation §8.1); DRL (Objective 2) would replace static optimal flow/tilt with adaptive control under climate uncertainty.

---

## 7. Solar / Climate Data Details (if applicable)
- **Location:** **Taiwan** (Taipei area tilt study); optimal tilt **22.4°** from Liu et al. Taiwan regional analysis (**20.2°–22.4°** range).
- **Climate inputs in model:** Solar radiation on collector, ambient temperature, wind (in lumped FPC heat-loss equations); pyranometer **PYR2-420** on physical rig.
- **Azimuth:** **0°** (true south) best; ±45° degrades capture.
- **Target retention temperature:** **30 °C** minimum after sunset.
- **Not used:** ERA5, NASA POWER, ISRO — local Taiwan weather implicit in TRNSYS/physical validation.
- **India mapping:** Re-run tilt/azimuth levels for **Coimbatore (~11°N)**, **Kochi (~10°N)**, **Jaisalmer (~26.9°N)**; retain GRA structure with climate-specific TRNSYS or grey-box weather files.

---

## 8. Key Results & Numbers
- **36** Taguchi experiments (L36 orthogonal array).
- **Global cumulative solar thermal capacity:** **522 GW_th** (+3% growth) [IRENA cite in intro].
- **Thermal energy share:** **48.7%** of global supply; solar **10.5%** of modern renewable heat [8].
- **TRNSYS validation:** simulation within **5%** of physical data.
- **Single-quality optimum — thermal storage efficiency:** predicted S/N **39.179**; confirmation mean efficiency **92.2%**, S/N **39.294** (CI **38.84–39.52**).
- **Single-quality optimum — heat retention:** predicted S/N **33.809**; confirmation **29.6 h**, S/N **29.425** (CI **28.83–38.79**).
- **GRA multi-quality confirmation:** thermal storage efficiency **94.2%**; heat retention **31.7 h**; S/N **39.481** / **30.021**.
- **vs non-optimized system:** efficiency **+28%**; retention **+14.6 h** (abstract).
- **ANOVA — thermal storage efficiency:** collector plate material **44.68%** contribution (F=385.7); PCM material **25.51%** (F=440.5); collector tube count **13.98%**; working fluid **11.62%**.
- **ANOVA — heat retention:** collector plate material **59.98%**; working fluid **13.86%**; collector tubes **10.9%**; PCM material **2.56%**.
- **GRG factor ranking:** collector plate material rank **1** (Δ=0.3041); working fluid rank **2**; collector tubes rank **3**.
- **Optimal nanofluid:** CuO \(\rho\) **1210 kg/m³**, \(c_p\) **3.41 kJ/kg·°C** outperforms Al₂O₃ and water.
- **Optimal flow:** **0.02 kg/s** (lower pump energy, better heat absorption vs 0.03 kg/s).
- **Literature comparison (Table 26):** nanofluid SWH **50.27–85%**; PCM SWH **64.53–79%**; this work **94.2%** combined.

---

## 9. Baseline Comparison

| Configuration | Thermal storage efficiency | Heat retention (T ≥ 30 °C) | Notes |
|---------------|---------------------------|----------------------------|-------|
| Non-optimized integrated SWH | ~**73.6%** (derived: 94.2/1.28) | ~**17.1 h** (31.7 − 14.6) | Abstract baseline |
| GRA-optimized (PCM+CuO) | **94.2%** | **31.7 h** | **+28%** eff., **+14.6 h** |
| Without PCM (factor A1) | Lower S/N across runs | Shorter retention | Figs. 18–20 |
| Water vs CuO nanofluid | Lower η_storage | Shorter retention | CuO highest \(k\) |
| Al plate vs Cu plate | η drop (S/N 34.57 min vs 36.78 max) | Major retention impact | ANOVA 44.7–60% |
| 11 vs 9 collector tubes | Reduced efficiency (thermal interference) | — | Optimal **9** tubes |
| Tilt 24.4° vs 22.4° | Less radiation capture | — | Taipei-optimal **22.4°** |
| Azimuth ±45° vs south | Reduced performance | — | **0°** best |
| Prior PCM-only literature | 64.53–79% | — | Table 26 |
| Prior nanofluid-only literature | 50.27–85% | — | Table 26 |

---

## 10. Hardware / Experimental Setup (if applicable)

| Component | Specification |
|-----------|---------------|
| Flat-plate collector | 505×320 mm; 9–11 copper/aluminum/SS tubes |
| PCM | **RT35HC** in **12–16** tubes; **10–20%** tank volume |
| Tank | **40 L** (0.04 m³); HX tubes OD 22 mm |
| Pump | FORMOSA RS-15/6GWS |
| Pyranometer | PYR2-420, Class C, 10 µV/(W/m²) sensitivity |
| Temperature | PT100 (±0.25 K, 30–500 K range stated) |
| Ambient | Galltec-Mela PC-ME (±0.2 K, RH ±2%) |
| Flow meter | RP variable area, ±2.5% FSD |
| Data logger | BCT TF 200S |
| Platform | Physical prototype + **TRNSYS** simulation (primary optimization) |
| Test conditions | Taiwan; south-facing; validated to **5%** |

---

## 11. Limitations Acknowledged by Authors
- Optimization primarily via **simulation** (36 TRNSYS runs); physical confirmation limited to **5** runs per quality metric.
- **Taiwan-specific** tilt/azimuth optima (**22.4°**, south) — not directly transferable without re-optimization.
- Model assumptions: uniform layer temperatures, perfect edge insulation, no dust, equal front/back ambient (Section 3.5.1).
- **Nanofluid** stability, agglomeration, and long-term pumping wear not deeply studied.
- Combined **94.2%** metric is **thermal storage efficiency** in their TRNSYS definition—not identical to ISO 9459 annual solar fraction.
- No **adaptive/real-time** control; static optimal parameters only.
- Authors note gap filled vs PV/T nanofluid+PCM work (Liu 2023 **64.76%**) but SWH field lacked combined optimization before this study.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Relevant as baseline** — Chen optimizes **fixed** flow (**0.02 kg/s**), tilt, and PCM volume offline; your DRL agent can treat these as action bounds or initial policy, then adapt online to irradiance/load.

- **RG2 (No integrated PCM–AI–hardware prototype):** **Highly relevant** — Same material stack (**RT35HC**, nanofluid option, FPC, instrumented rig) maps to your PCM-SWH hardware story; Chen lacks AI layer—you add DRL + embedded closure.

- **RG3 (Poor alignment with household demand patterns):** **Relevant** — **30 L** daily demand and **30 °C** retention threshold mirror domestic hot-water targets; extend to time-of-use demand profiles in reward shaping.

- **RG4 (Limited real-world experimental validation):** **Partially relevant** — Physical TRNSYS calibration within **5%** exists, but optimization runs are simulation-heavy; supports using your grey-box as training env with field validation as differentiator.

- **RG5 (No predictive optimization under climatic uncertainty):** **Relevant** — Taguchi+GRA is **climate-static** (single Taiwan profile); your ERA5/NASA POWER forecast + PCM classifier generalizes across **Coimbatore/Kochi/Jaisalmer** where tilt **11°–27°** differs from **22.4°**.

---

## 13. Equations to Reuse or Adapt

**S/N ratio (larger-the-better):**
\[
\frac{S}{N}_{LTB} = -10 \log_{10}\left(\frac{1}{n}\sum_{i=1}^{n}\frac{1}{y_i^2}\right) \tag{1}
\]

**Main effect:**
\[
F_i = \frac{1}{m}\sum_{k=1}^{m}\eta_{ik}, \qquad \Delta F = F_{i,\max} - F_{i,\min} \tag{2–3}
\]

**Grey normalization (larger-the-better):**
\[
x_i^*(k) = \frac{x_i(k) - \min x_i(k)}{\max x_i(k) - \min x_i(k)} \tag{15}
\]

**Grey relational coefficient:**
\[
\xi_i(k) = \frac{\Delta_{\min} + \zeta\Delta_{\max}}{\Delta_i(k) + \zeta\Delta_{\max}}, \quad \zeta = 0.5 \tag{16}
\]

**Grey relational grade (your Presentation notation):**
\[
\gamma_i = \frac{1}{n}\sum_{k=1}^{n} w(k)\,\xi_i(k) \tag{17}
\]

**Working fluid energy balance (lumped FPC):**
\[
C_f \frac{dT_f}{dt} = \frac{1}{R_{cov}}(T_{ab} - T_f) + \dot{m}_f C_{pf}(T_{fo} - T_{fi}) \tag{43}
\]

**Absorber solar gain:**
\[
I_T = \alpha_{ab}\,\tau_g\, G\, A_{ab} \tag{37}
\]

**Heat retention metric (project adoption):**
\[
t_{ret} = t\left(T_{tank}(t) < T_{target}\right) - t_{sunset}, \quad T_{target} = 30\,°\text{C}
\]

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **L.F. Cabeza et al.** — PCM volume/module count effects in SWH [37].
2. **Liu et al. (2023)** — Taguchi+GRA on PV/T with paraffin + nanofluid (**64.76%** heat storage efficiency) [44].
3. **C. Kuo et al.** — prior Taguchi+GRA on flat-plate collectors [23].
4. **Moghadam et al.** — tilt angle effects on FPC efficiency [18].
5. **A. El-Fakharany et al.** — SAH with PCM up to **64.53%** efficiency [22].

---

## 15. Suggested Use in My IEEE Paper

- **Section I (Introduction):** Cite the dual challenge of low storage efficiency and poor overnight retention; note global **522 GW_th** solar thermal capacity.

- **Section II (Literature Review):** Position as the primary **Taguchi + GRA** reference for PCM–nanofluid–FPC SWH; contrast static DOE (**94.2%**, **31.7 h**) with lack of adaptive AI control.

- **Section III (Methodology):** Reproduce GRA equations (15–17) for **Objective 1** PCM selection alongside XGBoost; use Chen’s **9-factor** table as template for Indian climate re-optimization (tilt, PCM volume, flow).

- **Section IV (Dataset & Setup):** Reference **RT35HC** Rubitherm curves, **PYR2-420**-class pyranometer, **PT100**/DS18B20 equivalence, **30 °C** retention threshold, **30 L** demand—mirror sensor stack in prototype section.

- **Section V (Results):** Benchmark grey-box/DRL against Chen’s **94.2%** storage efficiency and **31.7 h** retention; report improvement over **~73.6%** / **~17.1 h** non-optimized baseline; cite **+28%** / **+14.6 h** as published DOE gains.

---
