# The Role of Nanotechnology and Artificial Intelligence in Optimizing Thermal Energy Systems

**Authors:** Hayder I. Mohammed, Farhan Lafta Rashid, Hussein Togun, Ephraim Bonah Agyekum, Arman Ameen, Karrar A. Hammoodi, Rujda Parveen, Saif Ali Kadhim, Walaa N. Abbas  
**Year:** 2025  
**Journal/Conference:** Applied Energy, Vol. 400, Article 126576  
**DOI/Link:** https://doi.org/10.1016/j.apenergy.2025.126576  
**IEEE Citation:** H. I. Mohammed et al., "The role of nanotechnology and artificial intelligence in optimizing thermal energy systems," Appl. Energy, vol. 400, p. 126576, 2025, doi: 10.1016/j.apenergy.2025.126576.

---

## 1. One-Line Summary
This narrative review synthesizes **~180** studies (2013–2024) on **nano-enhanced PCMs/nanofluids** (e.g., **+28.8%** conductivity) combined with **AI** (ANN, PSO, XGBoost, DRL) for solar collectors, SWH, and latent storage—reporting **>97%** prediction accuracy in cited works, **28%** HVAC energy savings (ROI **<3 years**), and identifying gaps in **scalability, cost, and field validation**.

---

## 2. Problem Being Solved
- Conventional **PCMs** have **low thermal conductivity**, slow charge/discharge, and poor real-time controllability in solar thermal and SWH systems.
- **FPSCs/SWHS** suffer heat losses and limited working-fluid conductivity; passive PCM envelopes often yield only **1–3%** savings in cited passive-building studies.
- **Nanotechnology** improves materials but faces **agglomeration, cost, toxicity, and cycling durability** issues.
- **AI models** are often trained on steady-state or siloed datasets, lacking integration with **hardware prototypes** and **climate-adaptive control** under extreme transients.
- Need a unified roadmap for **NePCM + AI** hybrid TES spanning prediction, optimization, and deployment economics.

---

## 3. Key Contributions
1. **Dual-pillar framework:** nanotechnology (NePCM, nanofluids, nano-coatings) + AI (ML/DL/DRL, PSO, MPC) for TES optimization.
2. **Structured literature synthesis:** Scopus, WoS, ScienceDirect, IEEE, Springer; Boolean search on NePCM, nanofluid, solar collector, AI/ML/DL, HVAC; **~180 core papers** (2013–2024).
3. **KPI taxonomy (Table 1):** thermal/energetic efficiency, heat-transfer enhancement, ROI, payback, LCA, reliability.
4. **PCM classification (Table 2):** organic/inorganic; low/medium/high \(T_m\); paraffin, salt hydrates, fatty acids.
5. **AI algorithm map (Section 4.3):** predictive (ANN, SVM, RF, **XGBoost**), control (RL, ANFIS, PSO, GA), fault detection (**LSTM**, autoencoders).
6. **Synergy case studies:** nanofluid HX (+**20%** efficiency), AI solar thermal (+**25%**), Fraunhofer NePCM-HVAC (**28%** energy cut), ML+CFD surrogate (**>99%** compute savings).
7. **Challenge roadmap:** nanoparticle stability, AI compute cost, data scarcity, sensor drift, regulatory/LCA gaps.

---

## 4. Methodology
### 4a. Review approach
- **Narrative** (not PRISMA-systematic) but structured inclusion/exclusion.
- **Inclusion:** empirical or comparative reviews on TES, solar thermal, NePCM, AI control/modelling (2013–2024).
- **Exclusion:** non-peer-reviewed, insufficient technical data.

### 4b. Technical domains covered
1. **NePCM physics:** equivalent enthalpy, nanoparticle dispersion (0D–2D: CuO, Al₂O₃, CNT, graphene, MXene).
2. **Nanofluids:** Buongiorno model; Brownian motion/thermophoresis; base fluids water/EG/oil.
3. **AI pipeline:** data from CFD/experiments → train ANN/XGBoost/LSTM → PSO/GA hyperparameter or geometry optimization → optional RL closed-loop control.
4. **Hybrid ML+CFD:** CFD labels train surrogate; surrogate replaces expensive simulations in design loops.

### 4c. Validation cited in review
- Multiple third-party studies (Kalani **130** experimental PV/T points; Fraunhofer **12-month** campus pilot; field-tested nano-coated solar thermal per [263]).

---

## 5. PCM Details (if applicable)
| Enhancement | Nanoparticle / system | Reported effect (from cited studies) |
|-------------|----------------------|--------------------------------------|
| Paraffin NePCM | **3% TiO₂** | Thermal conductivity **+25%** |
| Paraffin NePCM | **CuO** | Conductivity up to **+28.8%** (abstract headline) |
| Neopentyl glycol | CuO | Significant conductivity gain [106] |
| Basic PCM | **SWCNT / MWCNT** | **+134% / +339%** conductivity vs base PCM |
| PCM + fin metal foam | — | Melting time **−83.35%** |
| Paraffin + CuO (cycling) | — | Latent heat **−12%** after **200** thermal cycles (agglomeration) |
| Solar still | CuO + PCM mix | Freshwater productivity **+108%** |
| Medium-T PCM class | Paraffin, RT-class organics | \(T_m\) between room temp and **100 °C** — SWH-relevant band |

**Stability mitigation:** surfactants, encapsulation, metal foam matrices, ultrasonic dispersion; CNT/graphene reported stable **>500** cycles in cited work.

---

## 6. AI / ML / Control Details (if applicable)
| Algorithm | Application in review | Reported metrics |
|-----------|----------------------|------------------|
| **ANN + PSO** | PV/T nanofluid collector (Kalani); molten-salt/wind hybrid | PSO AAPD **0.47–0.51%**; GOA **0.05–0.27%** |
| **ANFIS / RBF** | PV/T outlet temperature (130 experiments) | Best among compared models [24] |
| **XGBoost / LGB / GBR** | PV/T energy prediction (15,540 samples) | LGB \(R^2=0.983\) thermal; MLP electrical \(R^2=0.0906\) |
| **ANN** | ITES-HVAC, solar irradiance forecast | ITES **R=0.94–0.99**, MSE **<20%**; solar ANN **MAE=0.9558** |
| **MPC** | SMR + two-tank TES; microgrid electro-thermal | 24-h vs 8-h horizon: **−6.71%** cost, **−15.68%** temp RMSE; **−31.57%** PV curtailment |
| **DRL / GAN+RL** | Fuel-cell thermal management; smart-grid bidding | Reward-based adaptive control |
| **ML + CFD** | Compact TES discharge | **>99%** computational time reduction [203] |
| **ANN surrogate** | Al₂O₃-Cu/water FPSC CFD | **MAPE <2.5%**; efficiency **+17.6%** at **1.2 vol%**; optimization **85%** faster |
| **ODNN + Sand Cat Swarm** | PV/T cooling design | Lower MAE/MSE, higher \(R^2\) [202] |

**Abstract claim:** some cited models achieve prediction accuracies **above 97%** under complex conditions.

---

## 7. Solar / Climate Data Details (if applicable)
- Review cites **solar irradiance, ambient temperature, wind, humidity** as AI inputs for collectors and green-roof models (e.g., Shanghai meteorological data [204]).
- **No single project dataset** — aggregates literature using measured weather, TMY, and operator databases (e.g., California ISO for SMR study).
- **Project link:** supports using **NASA POWER / ERA5 GHI, \(T_{amb}\), wind** as DRL/XGBoost state features for Coimbatore, Jaisalmer, Kochi; Fraunhofer case uses **weather + occupancy** for HVAC AI.

---

## 8. Key Results & Numbers
- **Al₂O₃ nanofluid (1.5 vol%)** in FPSC: efficiency **+31.64%** [11].
- **Al₂O₃/water FPSC (Yousefi):** **+28%** efficiency; **MWCNT:** **+35%**.
- **Hybrid Ag/graphite/CNT nanofluid:** FPSC **+5%** efficiency.
- **NePCM CuO in paraffin:** conductivity up to **+28.8%** (headline); **+25%** with **3% TiO₂**.
- **CNT-enhanced PCM:** conductivity **+134%** (SWCNT) and **+339%** (MWCNT) vs base.
- **PCM + metal foam fins:** melting time **−83.35%**.
- **Latent heat degradation:** **−12%** after **200** cycles (CuO-paraffin agglomeration).
- **Kalani PV/T ANN/PSO:** **130** experimental datasets; reliable outlet-temperature prediction.
- **Fraunhofer NePCM + AI HVAC:** **−28%** energy, **+21%** comfort, **ROI <3 years**.
- **Google AI data-centre cooling:** up to **−40%** cooling energy.
- **Nanofluid HX case study:** **+20%** energy efficiency [262].
- **AI nano-coated solar thermal field test:** **+25%** vs conventional [263].
- **MgO-CuO/water in heat-pipe ETC:** average efficiency **+20%**; payback **−27%** [295].
- **Fe₃O₄-water HX:** Nusselt number **+13%** [294].
- **ML+CFD TES:** **>99%** compute-time savings [203].
- **ANN-FPSC hybrid CFD:** **MAPE <2.5%**, efficiency **+17.6%**, optimal nanoparticle **1.2 vol%**.
- **MPC microgrid:** **−5.86%** operating cost, **−31.57%** PV curtailment.
- **Power plant NOx ANN:** \(R^2=0.97\).
- **Review scope:** **~180** publications; open-access **CC BY**.

---

## 9. Baseline Comparison
| Approach | Baseline | Improvement cited |
|----------|----------|-------------------|
| Al₂O₃ nanofluid FPSC | Pure water working fluid | **+28–31.64%** thermal efficiency |
| NePCM (CuO/TiO₂) | Plain paraffin PCM | **+25–28.8%** conductivity |
| CNT NePCM | Base PCM | **+134–339%** conductivity |
| AI-optimized BIHP-PCM HVAC (Fraunhofer) | Conventional HVAC | **−28%** energy |
| ML surrogate vs full CFD | Full CFD per design point | **>99%** time reduction |
| AI solar tracking + nanofluid | Traditional solar thermal | **+25%** efficiency |
| Passive PCM building (literature cite) | Reference building | **1–3%** only — motivates active AI+nano |

---

## 10. Hardware / Experimental Setup (if applicable)
Review aggregates setups rather than one unified experiment:
- **Double-pipe / shell-and-tube HX** with Fe₃O₄, Al₂O₃ nanofluids.
- **Heat-pipe evacuated-tube collectors** with hybrid MgO-CuO nanofluid [295].
- **FPSC** with Al₂O₃-Cu hybrid nanofluid; **nano-coated TiO₂** absorbers.
- **Smart-campus Fraunhofer pilot:** NePCM storage + predictive AI HVAC (12-month deployment).
- **Sensors implied:** temperature, flow, irradiance, nano-sensors for fouling detection.
- **No RPi/Arduino SWH prototype** in this review — gap your FYP addresses (RG2).

---

## 11. Limitations Acknowledged by Authors
- **Nanoparticle cost** and industrial-scale synthesis remain barriers.
- **Nanofluid stability:** agglomeration, viscosity, pressure drop, long-term sedimentation.
- **NePCM cycling:** phase separation, **12%** latent-heat loss after 200 cycles (cited).
- **Environmental/toxicity** of CuO vs lower-risk Al₂O₃; disposal pathways unclear.
- **AI:** steady-state training data miss extremes; **computational overhead**; reproducibility gaps; **data protection** and integration with legacy plant.
- **Lack of open, standardized TES ML datasets** for benchmarking.
- Review is **narrative**, not systematic — selection bias possible.

---

## 12. Direct Relevance to My Project
- **RG1 (real-time adaptive control):** **Highly relevant** — reviews **DRL, MPC, RL** for flow/control; cites real-time nanofluid flow regulation; your PPO valve control fits this gap.
- **RG2 (integrated PCM–AI–hardware):** **Highly relevant** — identifies missing **embedded end-to-end** SWH prototypes; cites PV/T and HVAC pilots but not low-cost **RPi + DS18B20 + solenoid** PCM tank.
- **RG3 (household demand):** **Partially relevant** — HVAC occupancy/weather AI discussed; limited **hot-water draw profile** optimization; extend to evening demand peaks.
- **RG4 (field validation):** **Relevant** — Fraunhofer **12-month** and field solar tests cited; flags insufficient extreme-scenario validation; benchmark your **Coimbatore/Jaisalmer/Kochi** trials against **28%** savings claims.
- **RG5 (climate uncertainty):** **Highly relevant** — LSTM forecasting, MPC with irradiance horizon, federated/edge AI; supports **ERA5/NASA POWER**-driven XGBoost + DRL under monsoon/dust/climate zones.

---

## 13. Equations to Reuse or Adapt
**PCM latent storage (review notation):**
\[
Q = m \cdot L \quad \text{(sensible + latent during phase change)}
\]

**Equivalent specific-heat PCM model (triangular \(c_p(T)\) over \(\Delta T\)):**
\[
H = \tfrac{1}{2}\Delta T \cdot \Delta c_p
\]

**Nanofluid effective property (conceptual — cite Buongiorno / Maxwell-Garnett in grey-box):**
\[
k_{nf} = \phi k_p + (1-\phi) k_f \quad \text{(baseline mixture form; review discusses enhanced models)}
\]

**AI performance metrics used across cited studies:**
\[
\mathrm{MAPE} = \frac{100}{n}\sum\left|\frac{y_i-\hat{y}_i}{y_i}\right|, \quad
R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

**DRL reward template:** energy saved vs baseline minus comfort violation penalty — aligns with your PCM charge/discharge reward.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Kalani et al., ANN+PSO for PV/T nanofluid collector, *Appl. Therm. Eng.*, 2017** — 130-point experimental ML baseline [24].
2. **Al-Waeli et al., ANN for PV/T nano-PCM/nanofluid, *Sol. Energy*, 2018** — hybrid material + AI [25].
3. **He et al., AI methods for TES prediction/design/control, *Renew. Sust. Energy Rev.*, 2022** — TES-AI survey [33].
4. **Olabi et al., AI prediction/optimization/control of TES, *Therm. Sci. Eng. Prog.*, 2023** — direct TES-AI review [23].
5. **Bharathiraja et al., hybrid NePCM flat-plate SWH, *J. Energy Storage*, 2024** — SWH + nano-PCM experimental [277].

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite **low PCM conductivity** and **1–3%** passive savings vs **28–35%** nano/AI gains to motivate integrated PCM-SWH control.
- **Section II:** Use as **umbrella review** for nano-AI TES; position your work as closing the **hardware integration + climate-adaptive DRL** gaps flagged in Sections 6–7.
- **Section III:** Justify **XGBoost** (review ranks it top for speed/accuracy) for PCM class selection; **PPO** under DRL subsection; optional NePCM as future work (RT/OM baseline first).
- **Section IV:** Mirror KPIs from **Table 1** (thermal efficiency, temperature uniformity, ROI); sensor quality argument from Google/Fraunhofer cases.
- **Section V:** Benchmark against **+17.6%** ANN-FPSC efficiency, **28%** Fraunhofer HVAC savings, or **>97%** predictor accuracy — state your RMSE/MAPE/% energy improvement relative to rule-based valve control.

---
