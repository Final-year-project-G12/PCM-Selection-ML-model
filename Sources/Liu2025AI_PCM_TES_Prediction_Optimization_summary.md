# The Contribution of Artificial Intelligence to Phase Change Materials in Thermal Energy Storage: From Prediction to Optimization

**Authors:** Shuli Liu, Junrui Han, Yongliang Shen, Sheher Yar Khan, Wenjie Ji, Haibo Jin, Mahesh Kumar  
**Year:** 2025  
**Journal/Conference:** Renewable Energy, Vol. 238, Article 121973  
**DOI/Link:** https://doi.org/10.1016/j.renene.2024.121973  
**IEEE Citation:** S. Liu et al., "The contribution of artificial intelligence to phase change materials in thermal energy storage: From prediction to optimization," Renew. Energy, vol. 238, p. 121973, 2025, doi: 10.1016/j.renene.2024.121973.

---

## 1. One-Line Summary
This comprehensive review maps AI applications across PCM-based latent heat storage—from **ANN/XGBoost/SVM** property prediction and **CALPHAD** integration to **GA/PSO/DRL** structural and operational optimization—reporting melting-point error reductions up to **42–71%**, NEPCM conductivity prediction **R² ≈ 0.99**, cascaded LHS energy gains of **5–18%**, and ANN–MPC operating-cost cuts of **9.1–14.6%**, while identifying gaps in real-time embedded control and standardized datasets.

---

## 2. Problem Being Solved
- LHS with PCMs faces low conductivity, supercooling, leakage, and complex melting/solidification dynamics that make trial-and-error design slow and expensive.
- Traditional CFD and experiments alone cannot efficiently explore high-dimensional PCM composites, encapsulation layouts, and system operating strategies.
- AI methods are proliferating but lack a unified synthesis comparing prediction vs optimization algorithms across solar thermal, building, and industrial LHS domains.
- Operational control strategies (flow, inlet temperature, charge/discharge scheduling) can dominate performance yet are under-optimized relative to material selection.

---

## 3. Key Contributions
1. **Two-pillar framework:** (A) **AI for prediction** — PCM/CPCM/NEPCM thermophysical properties, melting behavior, temperature fields; (B) **AI for optimization** — structure/layout (fins, foam, cascaded PCM) and operation/control.
2. **Algorithm taxonomy:** ANN, BP-ANN, ELM, SVM, XGBoost, RF, GBR, LSTM, CNN, GA, PSO, DE, CFD-coupled GA, DRL, GEP, MARS, CART.
3. **Composite PCM prediction survey:** molten-salt eutectics, nano-enhanced organics, microencapsulated cement/concrete PCMs, metal-foam composites.
4. **Optimization tables (Tables 3–4):** intelligent algorithms for fin geometry, foam porosity, cascaded CLHS, shell-and-tube layouts—with quantified energy/exergy gains.
5. **Limitations & future directions:** need physics-informed ML, embedded real-time control, cross-climate validation, standardized open datasets, and DRL for dynamic LHS operation.

---

## 4. Methodology
- **Type:** Narrative comprehensive review (Renewable Energy, **29** pages, **120+** references).
- **Scope:** AI in PCM-based LHS/TES across buildings, solar thermal, cold storage, batteries, waste heat—not exclusively solar water heating.
- **Organization:** §2 TES/LHS background → §3 AI prediction (properties, NEPCM, temperature/behavior) → §4 AI optimization (structure §4.1, operation/control §4.2) → challenges/future.
- **Validation:** Synthesizes cited primary studies' reported \(R^2\), RMSE, MAPE, energy % improvements; no new experiments in this paper.

---

## 5. PCM Details (if applicable)
### Materials & properties predicted/optimized (selected from review)
| Category | Examples | Key properties AI-targeted |
|----------|----------|--------------------------|
| Molten salt eutectics | KCl-NaF, NaNO₃-KNO₃-KCl | \(T_m\), latent heat, composition |
| Organic / paraffin | Octadecane, RT50/RT65/RT80 cascades | \(k_{eff}\), \(T_m\), melt fraction |
| NEPCM | CuO, Al₂O₃, TiO₂, SiO₂, Fe₂O₃ in paraffin | Effective conductivity **0.5–12 wt%** |
| Carbon NEPCM | MWCNT, graphene, CNF, GNP | \(R^2\) up to **0.99** vs RSM **0.79** |
| Building CPCM | Microencapsulated paraffin in cement | Compressive strength, activation energy |
| Cascaded CLHS | RT50, RT65, RT80 commercial PCMs | Stage height, NTU, PCM mass |

**Reported accuracy examples:**
- KCl-NaF ANN: \(T_m = 648 \pm 2\) °C, \(L = 365 \pm 5\) kJ/kg [50]
- BP-PBO vs BP-GA: melting-point error **−42%** / **−38%**; latent heat error **−71%** / **−68%** [51]
- NEPCM octadecane + metal oxides: max errors **2.31%** (liquid), **0.812%** (solid) [69]
- 10–20 wt% MPCM in cement: activation energy **−10%** / **−28%** [cited microencapsulation study]

---

## 6. AI / ML / Control Details (if applicable)

| Application | Algorithms | Inputs (examples) | Outputs | Reported metrics |
|-------------|------------|-------------------|---------|------------------|
| Eutectic salt design | ANN, BP-GA, BP-PBO | Electronegativity, ion radius, charge | \(T_m\), \(L\), composition | Error ↓ **42–71%** [51] |
| NEPCM conductivity | MARS, CART, ANN, KNN, SVM, XGBoost | NP concentration, size, PCM phase, \(k_{pcm}\) | \(k_{eff}\) | ANN **R² 0.99**; liquid/solid errors **<2.31%** [69] |
| Temperature field / melt fraction | ANN, SVM, RF, LSTM-BP, CNN | Geometry, \(T_{in}\), flow, time | \(T(x,t)\), liquid fraction | SVM **R² 0.99**, RMSE **2.19–3.17** [cited] |
| Building PCM demand | MLP, LSTM, CNN | Weather, occupancy | Cooling load | Energy ↓ **4.7–25.2%** [cited] |
| Operating control | ANN + εDE metaheuristic + **MPC** | TES tank states, tariffs | Charge/discharge rate | Cost ↓ **9.1–14.6%** [32] |
| Structural optimization | GA, PSO, DE, CFD-GA | Fin length, foam %, PCM stage layout | Energy, exergy, entransy | Stored energy ↑ **5–18%** (Table 4) |
| Dynamic control | **DRL** (cited) | LHS state | Valve/flow policy | Scores within **1.6–4.3%** of GA Pareto [cited] |

**Training notes:** Datasets range from literature compilations (hundreds of salt systems) to CFD-generated samples; many studies use **90/10** train-test splits; hyperparameter tuning via GA on ANN topology common.

---

## 7. Solar / Climate Data Details (if applicable)
- **Direct solar datasets:** Not primary focus; solar LHS studies cited include flat solar CLHS [173], medium-temperature spherical encapsulated PCM solar units [175], packed-bed solar plant CLHS [172].
- **Climate variables in cited control studies:** Ambient temperature, operational schedules, electricity tariffs—for building-coupled TES MPC [32].
- **Geographic scope:** Global literature (China BIT-led review); no India-specific cities.
- **Your project mapping:** Use Liu's **8-feature weather vector** pattern (GHI, DNI, DHI, \(T_a\), wind, RH, hour, month) from related SWH studies together with this review's PCM prediction/optimization sections for grey-box + XGBoost + DRL design.

---

## 8. Key Results & Numbers
- Global energy storage installed capacity **209.4 GW** by end-2021 (**+9.6%** YoY); pumped hydro **86.2%** [7].
- Thermal energy = **50%** of terminal energy utilization [10].
- ANN–MPC building TES: operating cost reduction **9.1–14.6%** [32].
- BP-PBO salt prediction: melting-point error **−42%** vs BP; latent heat error **−71%** vs BP-GA [51].
- NEPCM ANN: liquid-phase error **2.31%**, solid **0.812%** [69].
- ANN vs RSM for NEPCM \(k\): **R² 0.99** vs **0.79** [67].
- SVM temperature prediction: **R² 0.99**, RMSE **2.19–3.17** (review synthesis).
- XGBoost/RF regression: highest \(R^2\), lowest RMSE among tree models for PCM performance [cited].
- Cascaded CLHS GA: **+5.12%** stored energy vs single PCM [172]; flat solar CLHS **+6%**, **+18%**, **+11%** vs RT50/RT65/RT80 single PCM [173].
- Shell-and-tube CFD-GA: charged energy **+13.79%**, exergy **+14.85%**, entransy **+14.45%** [174].
- Spherical encapsulated CLHS: charging energy **+≥14%** [175].
- Cascaded heat sink GA: thermal management time **+12.4%**; cooling time **−31.9 min** [176].
- PSO optimal PCM parameters example: \(C_{ps}=2.5\), \(C_{pl}=3.1\) kJ/kg·K, \(L=238\) kJ/kg, \(T_m=20.85\) °C [180].
- Cold storage GA: optimal water flow **0.095 kg/s**, inlet ΔT **−1.25 °C** [181].
- ORC-LHS multi-objective GA: exergy efficiency **0.3351**, cost rate **1.529 $/h** [179].
- Branch/Y-shaped fins + ML: melting time **−52.8%**, heat dissipation **+110%** [cited fin studies].
- Metal foam optimization: solidification time **−7.62%**; charging time **−34%** [cited].

---

## 9. Baseline Comparison
| Study area | Baseline | AI method | Improvement |
|------------|----------|-----------|-------------|
| Salt \(T_m\), \(L\) prediction | BP ANN | BP-PBO | Error **−42%** / **−71%** [51] |
| NEPCM conductivity | RSM | ANN | **R² 0.79 → 0.99** [67] |
| Building TES operation | Rule-based / no MPC | ANN + MPC | Cost **−9.1 to −14.6%** [32] |
| Single PCM CLHS | RT50 alone | GA cascaded design | **+6%** stored energy [173] |
| Shell-and-tube LHS | Unoptimized layout | CFD-coupled GA | Energy **+13.79%**, exergy **+14.85%** [174] |
| Temperature field | CFD alone | SVM / RF hybrid | **R² 0.99**, RMSE **2.19–3.17** |
| Pumping/control (Barqawi-class) | Fixed speed | ANN flow multiplier | **+2.5–4.1%** energy (external SWH cite) |

---

## 10. Hardware / Experimental Setup (if applicable)
N/A at review level. Cited experimental systems include:
- DSC-validated molten salts [52]
- Shell-and-tube and packed-bed CLHS loops [172, 178]
- Flat-plate solar CLHS [173]
- Building-integrated sensible/latent tanks with BMS sensors for MPC [32]
- No standard **RPi/Arduino/ESP32** embedded deployment survey — identified as future need.

---

## 11. Limitations Acknowledged by Authors
- Many AI models are **data-hungry** and trained on narrow material/system classes.
- **Black-box** models lack interpretability and extrapolation beyond training bounds.
- CFD-coupled optimization is computationally expensive.
- **Real-time embedded control** and field validation rare vs offline simulation.
- Need **physics-informed** and **hybrid CALPHAD+AI** frameworks.
- Standardized benchmark datasets for PCM-AI missing.
- DRL cited but not yet mainstream in PCM-SWH household applications.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Highly relevant** — §4.2 documents ANN–MPC, GA/PSO operational optimization, and emerging DRL; your PPO valve/pump policy extends this for PCM charge/discharge/bypass.

- **RG2 (No integrated PCM–AI–hardware prototype):** **Highly relevant** — Review covers full prediction+optimization stack but notes absence of low-cost embedded prototypes; directly motivates RPi/ESP32 + DS18B20 + solenoid deployment.

- **RG3 (Poor alignment with household demand patterns):** **Relevant** — Building PCM studies with demand forecasting (MLP/LSTM, **4.7–25.2%** energy reduction) inform demand-shaped reward functions for DRL.

- **RG4 (Limited real-world experimental validation):** **Relevant** — Most cited AI–PCM work is simulation or lab-scale; your field evaluation across Indian climates addresses stated gap.

- **RG5 (No predictive optimization under climatic uncertainty):** **Highly relevant** — Couples with irradiance forecasting reviews; Liu's MPC and multi-objective exergy optimization support forecast-driven PCM selection (XGBoost) + predictive DRL.

---

## 13. Equations to Reuse or Adapt

**Pre-melt sensible balance (representative lumped model cited across studies):**
\[
\frac{dT_p}{dt} = \frac{h A}{m C_p}(T_{wf} - T_p)
\]

**Latent charging at constant \(T_m\):**
\[
\frac{dQ}{dt} = h A \max(0, T_{wf} - T_m), \quad Q_{max} = m L
\]

**Effective NEPCM conductivity (ML target):**
\[
k_{eff} = f(k_{pcm}, k_{np}, \phi, T, \mathrm{phase})
\]
where \(\phi\) = nanoparticle volume fraction (**0.5–12 wt%** in cited octadecane study).

**Exergy-based objective (CLHS optimization):**
\[
\max f_E = \frac{E_{stored,exergy}}{E_{input}} \quad \text{preferred over pure energy or entransy in [178]}
\]

**MPC cost reduction metric (building TES):**
\[
\Delta C = \frac{C_{baseline} - C_{ANN-MPC}}{C_{baseline}} \in [9.1\%, 14.6\%]
\]

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Lee et al., ANN + εDE + MPC for building TES, cost −9.1–14.6%** — operational control benchmark.
2. **Tamizharasan & Kini, deep learning for PCM-enhanced SWH, *Int. J. Energy Res.*, 2023** — closest SWH+DL parallel.
3. **Vempally & Dhanarathinam, ML PCM selection, *J. Therm. Anal. Calorim.*, 2023** — aligns with your XGBoost classifier.
4. **Barqawi, ANN pump control for PCM-SWH simulation, 2025** — retrofit SWH ML control baseline.
5. **Yan et al., ML melting time in triplex-tube LHS, *Appl. Energy*** — geometry-aware PCM ML predictor.

---

## 15. Suggested Use in My IEEE Paper
- **Section I (Introduction):** AI transforms LHS from static design to predictive+optimized systems; thermal storage **209.4 GW** global context.
- **Section II (Literature Review):** Two-column table: Liu prediction (XGBoost/ANN for PCM props) vs Liu optimization (GA/PSO/MPC/DRL for layout/operation).
- **Section III (Methodology):** Cite exergy-objective preference for multi-objective DRL reward design; adopt NEPCM \(k_{eff}\) ML error benchmarks (**R² 0.99**) for material feature validation.
- **Section IV (Dataset & Setup):** Structure PCM property database like Liu §3.1 (Rubitherm/PLUSS + eutectic blends); 8-feature climate vector for labels.
- **Section V (Results):** Target exceeding CLHS GA gains (**+13.79%** energy) and MPC cost savings (**9.1–14.6%**) via integrated forecast+DRL on SWH hardware.

---
