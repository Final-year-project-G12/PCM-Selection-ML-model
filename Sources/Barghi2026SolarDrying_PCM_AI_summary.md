# Thermal Energy Storage-Centric Solar Drying with Phase Change Materials: Intelligent Optimization via Neural and Evolutionary Regression Models

**Authors:** Mohammad Saleh Barghi Jahromi, Ayla Sayedolasgari, S. Madhankumar, Hadi Samimi Akhijahani, Payman Salami  
**Year:** 2026 (online Nov 2025)  
**Journal/Conference:** Journal of Energy Storage, Vol. 141, Article 119192  
**DOI/Link:** https://doi.org/10.1016/j.est.2025.119192  
**IEEE Citation:** M. S. Barghi Jahromi et al., "Thermal energy storage-centric solar drying with phase change materials: Intelligent optimization via neural and evolutionary regression models," J. Energy Storage, vol. 141, p. 119192, 2026, doi: 10.1016/j.est.2025.119192.

---

## 1. One-Line Summary
This review synthesizes **PCM-buffered solar dryers** (palmitic acid, paraffin, micro-PCM) with **ANN, SVM, LSTM, RF, CatBoost, and EPR** surrogates—reporting collector gains up to **66.52%**, drying-time cuts **63%**, **EPR R² > 0.98**, and ANN metrics up to **R² = 0.9999**—while noting dataset scarcity for embedded real-time PCM control.

---

## 2. Problem Being Solved
- Conventional food drying consumes large fossil energy; post-harvest losses **30–40%** without adequate preservation.
- Solar dryers suffer **intermittent irradiance** and unstable chamber temperatures.
- **PCM thermal storage** can extend operation after sunset but adds design complexity (placement, thickness, melting point).
- Physics-based CFD models are accurate but slow; pure empiricism lacks generalization — need **ML + grey-box (EPR)** tools for PCM dryer/collector optimization.

---

## 3. Key Contributions
1. **PCM integration taxonomy** for solar dryers: absorber-mounted, plenum, cabinet walls, copper-tube encapsulation.
2. **Quantified PCM benefits** across studies: stable temperature, **−63%** drying time, **+7.81%** drying efficiency, night-time heat release.
3. **ML algorithm survey:** DT, RF, SVR, KNN, **ANN/BPNN**, FNN, RNN, **LSTM**, **CatBoost**, hybrid ANN-KNN.
4. **EPR (Evolutionary Polynomial Regression)** as interpretable grey-box: **R² > 0.98** for outlet temperature and thermal efficiency vs CFD (**R² > 0.94**) and ANN (**R² ≤ 0.99**).
5. **Case study hub:** authors' Jerusalem artichoke ETC+PCM dryer — **payback 22 months** (**15%** improvement), **EPR R² > 0.98**.
6. **Co-authorship link:** Sri Krishna College of Engineering and Technology, **Coimbatore** — regional relevance to project city.

---

## 4. Methodology
- Narrative review of solar dryer classifications (direct, indirect, mixed-mode, greenhouse, CPV/T).
- PCM selection criteria: \(T_m\) near operating temperature, latent heat, conductivity, encapsulation geometry.
- ML pipeline patterns: experimental/CFD data → train ANN/SVM/LSTM → predict MC, MR, \(T_{out}\), \(\eta_{th}\) → compare RMSE, MAPE, \(R^2\).
- **EPR:** GA searches equation structure; least-squares fits coefficients (Eq. **18**).
- Benchmark tables (Table 9 PCM studies; Table 10 ANN in dryers).

---

## 5. PCM Details (if applicable)
| Study / PCM | Configuration | Key numbers |
|-------------|---------------|-------------|
| Mixed-mode coffee dryer [104] | PCM unit + air recycling | Charge **0.033–0.161 kWh**; discharge **0.051–0.237 kWh**; collector η **55.27–66.52%** (0% recycle best) |
| Palmitic acid + graphene–Al₂O₃ nanofluid [105] | Flat-plate air collector | **1.5 vol%** optimal; \(k=0.75\) W/m·K; \(\eta_{th}=62.5%\), exergy **20.7%**, dryer η **46.8%** |
| PVT air collector [59] | PCM thickness sweep | **0.005 m** layer → **151 Wh** thermal output vs **75 Wh** (0.02 m) |
| Mango pulp [109] | **200 g micro-PCM** | **>5 h** above **60 °C**; drying time **−63%** vs no PCM |
| Paraffin in copper tubes [102] | **176 g / 494 g** PCM | Stored **3.52 kJ**, released **4.73 kJ**; exergy η **21% → 28%** |
| Barghi ETC cabinet dryer [9] | PCM for Jerusalem artichoke | SEC **14.51 → 13.38 MJ/kg**; exergy η **35.3–59.7%**; activation energy **33.4 kJ/mol** |
| NePCM Al₂O₃-paraffin CPV/T [44] | Nano-enhanced PCM | Thermal η **20%**, exergy **8%**; ANN **R²=0.999** vs SVM **0.974** for MC |

**Placement insight:** absorber-mounted PCM melts faster; plenum PCM releases longer after **20:00** but lower peak \(T\).

---

## 6. AI / ML / Control Details (if applicable)
| Method | Application | Reported performance |
|--------|-------------|----------------------|
| **Decision Tree** | Peanut solar dryer | **R² = 0.9972** [112] |
| **ANN (1-7-5, LM)** | PCM solar collector | **R² = 0.832–0.899** [195] |
| **ANN** | TES collector heating capacity | RMSE **7840.56**, **R² = 0.9995**; efficiency **R² = 0.9999** [197] |
| **ANN vs CFD** | Dehydration system | ANN more accurate, faster, cheaper than CFD [196]; η **21.11–25.20%** |
| **ANN vs SVM** | CPV/T + NePCM mushroom drying | ANN **R² = 0.999** (MC), beats SVM |
| **LSTM / RNN** | Solar radiation, drying kinetics | Strong on time-series; humidity RMSE **< 0.645** cited |
| **CatBoost** | Red beetroot drying | Beats XGBoost/LightGBM on **R², MSE, MAE** [191] |
| **EPR** | Collector \(T_{out}\), \(\eta_{th}\) with PCM | **R² > 0.98**; beats CFD **R² > 0.94**; competitive with ANN |
| **RF** | Solar water heating performance | Cited [119086] in references |

**No DRL/PPO** — feedforward ANN used for solar transients [177–179]; gap for your valve control.

---

## 7. Solar / Climate Data Details (if applicable)
- Inputs across studies: **solar irradiance \(I_g\)**, ambient \(T_a\), humidity, air velocity, recycling ratio, incidence/slope/azimuth.
- Mixed-mode dryer [104]: charging **09:00–14:30**, discharging until **18:00**.
- **Coimbatore** co-author institution — aligns with humid tropical drying/solar conditions in your Kochi/Coimbatore tests.
- No ERA5/NASA POWER — uses on-site pyranometer/logging per cited experiments.

---

## 8. Key Results & Numbers
- Post-harvest losses without drying: **30–40%**.
- Air recycling **0%** vs **100%**: collector efficiency **66.52%** vs **55.27%** with PCM [104].
- Hybrid nanofluid **1.5 vol%**: conductivity **0.75 W/m·K**, heat transfer **345.5 W**, heat loss **58.7 W**.
- PCM thickness **0.005 m** vs **0.03 m**: thermal output **151 Wh** vs **55 Wh**.
- Mango drying: time reduction up to **63%** with PCM.
- Banana/paraffin PCM2: exergy efficiency **28%** vs **21%** without PCM.
- Barghi PCM dryer: payback **22 months** (**15%** better); SEC **13.38 MJ/kg**; drying efficiency gain **1.51–7.81%**.
- EPR vs alternatives: **R² > 0.98** (outlet \(T\), \(\eta\)).
- ANN drying kinetics: up to **R² = 0.99998**, MSE **1×10⁻⁶** [172].
- Freeze-drying ANN: **R² = 0.999** for MR, MC, DR [166].
- CPV/T NePCM: greenhouse temperature held **100 min** after irradiance drop [44].

---

## 9. Baseline Comparison
| System | Baseline | With PCM / AI |
|--------|----------|---------------|
| Mixed-mode dryer collector | No PCM, 100% recycle η **55.27%** | PCM + 0% recycle **66.52%** |
| Solar air dryer | Ethylene glycol + palmitic acid | Hybrid nanofluid 1.5% + PCM: η **46.8%** vs lower baseline |
| Mango drying duration | No PCM | Micro-PCM: **−63%** time |
| Exergy efficiency | **21%** (no PCM) | **28%** (PCM2) |
| Outlet temperature model | CFD **R² ~0.94** | **EPR R² > 0.98** |
| Moisture content prediction | SVM **R² = 0.974** | **ANN R² = 0.999** |

---

## 10. Hardware / Experimental Setup (if applicable)
Review aggregates:
- **Indirect/cabinet dryers** with ETC, flat-plate collectors.
- **PCM encapsulation:** copper tubes in cabinet corners, grooved aluminum trays, plenum chambers.
- **Sensors:** temperature loggers, MC measurement, sometimes image-based LSTM monitoring.
- **Nanofluid loops:** **7 L/min** flow, graphene–alumina hybrid.
- **No RPi/solenoid SWH rig** — closest parallel is **ETC + PCM + airflow control**; transferable to your **DS18B20 + valve** bench.

---

## 11. Limitations Acknowledged by Authors
- **Limited experimental datasets** for ML training in PCM dryers.
- ANN needs **large data volume**; risk of overfitting on small PCM tests.
- Advanced computation for **dynamic PCM phase-change** still challenging.
- EPR less robust than LSTM on **high-dimensional time-series**.
- Review focuses on **drying**, not domestic SWH — direct hardware transfer requires adaptation.

---

## 12. Direct Relevance to My Project
- **RG1:** **Relevant** — ANN feedforward for solar transients [177–179] but no closed-loop actuator; your **PPO solenoid** advances beyond review.
- **RG2:** **Highly relevant** — PCM + sensors + optimization workflow; dryer ETC+PCM cabinet is analog to **collector + PCM tank**; **Coimbatore** co-author ties to project geography.
- **RG3:** **Partial** — drying load profiles differ from evening bath draw; PCM **night discharge** pattern directly transferable.
- **RG4:** **Relevant** — rich experimental PCM numbers (SEC, exergy, payback **22 months**) for economic validation framing.
- **RG5:** **Highly relevant** — **EPR grey-box** and **ANN surrogates** under variable \(I_g\) mirror your XGBoost grey-box + climate forecasts.

---

## 13. Equations to Reuse or Adapt
**EPR general form (Eq. 18):**
\[
y = \sum_{j=1}^{m} F\big(X, F(X), a_j\big) + a_0
\]

**ANN neuron output (Eq. 11):**
\[
q_i = f\!\left(\sum_j W_{ij} P_i\right)
\]

**Network error (Eq. 12):**
\[
E_r = \frac{1}{N}\sum_{i=1}^{N}(E_i - q_i)^2
\]

**BP weight update (Eq. 13):**
\[
\omega = \omega - \eta \frac{\partial e}{\partial \omega}
\]

**PCM energy balance (review nomenclature):**
\[
Q = m \lambda \frac{df}{dt} + mc_p \frac{dT}{dt}
\]

Use EPR for interpretable \(T_{out}(\text{GHI}, T_{amb}, \dot{m})\); ANN for PCM state observer.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Barghi Jahromi et al., ETC+PCM cabinet dryer + ANN/EPR/CFD, prior experimental** — core case **R² > 0.98**, payback **22 months** [9].
2. **Karaağaç et al., CPV/T + NePCM + ANN/SVM drying, *Sol. Energy*** — **R² = 0.999** ANN [44].
3. **Suherman et al., mixed-mode PCM coffee dryer, *Renew. Energy*, 2025** — recycling ratio study [104].
4. **Soudagar et al., palmitic acid + hybrid nanofluid dryer, *Appl. Therm. Eng.*, 2025** — **62.5%** thermal η [105].
5. **Lillo-Bravo et al., RF for solar water heating performance, *Renew. Energy*, 2023** — SWH ML crossover [119086].

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite **30–40%** post-harvest loss and PCM night-discharge for intermittent solar (parallel to SWH evening demand).
- **Section II:** Position as **PCM + ML review** complementary to Liu/Odoi SWH reviews; highlight **EPR interpretability** vs black-box DRL.
- **Section III:** Adopt **EPR Eq. (18)** or **ANN 1-7-5 LM** as surrogate for grey-box tank; inputs \(T_{amb}, I_g, \dot{m}\).
- **Section IV:** PCM placement trade-off (absorber vs plenum) informs your **RT35/OM35** tank coil location; target **R² > 0.98** like EPR benchmark.
- **Section V:** Compare energy metrics to **SEC 13.38 MJ/kg** analog (kWh per L hot water) and **22-month** payback as economic aspirational bound.

---
