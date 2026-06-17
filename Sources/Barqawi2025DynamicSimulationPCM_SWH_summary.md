# Dynamic Simulation of Phase Change Material-Integrated Solar Water Heating Systems: A Machine Learning Approach to Energy Conversion Optimization

**Authors:** Falah A. Barqawi  
**Year:** 2025  
**Journal/Conference:** Muthanna Journal of Engineering and Technology, Vol. 13, No. 3, pp. 1–14  
**DOI/Link:** https://doi.org/10.52113/3/eng/mjet/2025-13-03/-1-14  
**IEEE Citation:** F. A. Barqawi, "Dynamic simulation of phase change material-integrated solar water heating systems: A machine learning approach to energy conversion optimization," Muthanna J. Eng. Technol., vol. 13, no. 3, pp. 1–14, 2025, doi: 10.52113/3/eng/mjet/2025-13-03/-1-14.

---

## 1. One-Line Summary
This paper develops and validates a simulation-only feedforward neural-network controller that modulates pump flow multipliers in a three-phase PCM–solar water heating model, achieving **2.5–4.1%** (**3.3%** average) higher energy storage than fixed-speed conventional control across five synthetic climate scenarios.

---

## 2. Problem Being Solved
- PCM-integrated solar water heaters suffer from low PCM thermal conductivity (typically **0.1–0.5 W/m·K**), supercooling, and thermal degradation, limiting charge/discharge rates.
- Conventional fixed-speed pump control cannot adapt to variable solar irradiance and ambient conditions, causing supply–demand mismatch and overdimensioned collectors or backup heating.
- Machine learning deployment for PCM thermal energy storage optimization is underutilized; prior work emphasizes material/geometric enhancements (nanoparticles, fins, metal wool) rather than intelligent, retrofit-compatible software control.
- Solar intermittency leaves stored thermal energy misaligned with end-use timing without adaptive operational optimization.

---

## 3. Key Contributions
1. A complete three-phase lumped-parameter mathematical model (pre-melting, isothermal melting, post-melting) with dynamic sinusoidal solar input, automatic phase-transition event detection, and per-step energy balance accounting.
2. A feedforward neural-network controller using eight environmental/temporal inputs to predict optimal pump **flow_multiplier** values that retune the water thermal time constant in real time.
3. Comparative simulation across **five** environmental scenarios and **five** PCM geometry configurations (P01–P05) with identical thermophysical properties but varying volume and surface area.
4. Quantified ML-vs-baseline gains: **+2.5% to +4.1%** energy storage (MJ/kg), **1.03–1.04×** enhancement factors, and **12–18%** pumping energy reduction at flow multipliers **0.3–0.6**.
5. Positioning of ML software control as **retrofit-compatible** (very high retrofit potential, very low implementation cost) versus physical PCM enhancements requiring hardware modification.

---

## 4. Methodology
### 4a. System / Experiment Setup
- **System type:** Horizontal cylindrical storage tank (length **1.0 m**, diameter **0.5 m**) with internal solar collector coil and distributed PCM containers (volumes **0.025–0.05 m³**); schematic adapted from Chen et al. [21].
- **Heat transfer areas/coefficients:** Coil area \(A_c = 2.5\ \mathrm{m^2}\), water–coil HTC \(h_c = 1500\ \mathrm{W/m^2·K}\), PCM–water HTC \(h_p = 800\ \mathrm{W/m^2·K}\); PCM surface areas **2.5–5.0 m²** depending on configuration.
- **Flow/HT correlations:** Dittus–Boelter correlation for turbulent pipe flow; water velocity **0.02–0.05 m/s**, Reynolds number **2000–5000**.
- **Assumptions:** No external hot-water draw load; ambient heat losses neglected to isolate PCM effects; reference climate **33°N, 44°E** (Middle Eastern conditions).
- **Simulation duration:** **50,400 s (14 h)** per scenario; fixed reporting time step **100 s**; solver uses adaptive internal stepping.
- **Control comparison:** Conventional **fixed-speed pump** vs **ML-optimized variable flow** via predicted flow multiplier.
- **Software:** Python **SciPy `solve_ivp`** with **Runge–Kutta (RK45)** and event detection for phase changes.

### 4b. Mathematical Models & Equations
**Phase 1 — Pre-melting (\(T_p < T_{melt}\)):**

- Water: \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w) + \eta(T_p - T_w)\right]\) — **(1)**
- PCM: \(\displaystyle \frac{dT_p}{dt} = \frac{1}{\tau_{ps}}(T_w - T_p)\) — **(2)**

**Solar / coil input:**

- \(T_c(t) = T_{amb} + \dfrac{\mathrm{efficiency} \times I_{solar}(t)}{20}\) — **(3)**
- \(I_{solar}(t) = I_{max}\sin\!\left(\pi \dfrac{t_{hours} - \mathrm{sunrise}}{\mathrm{sunset} - \mathrm{sunrise}}\right)\) — **(4)**

**Time constants and coupling:**

- \(\tau_w = \dfrac{M_w C_w}{h_c A_c}\)
- \(\tau_{ps} = \dfrac{M_p C_{ps}}{h_p A_p}\)
- \(\eta = \dfrac{h_p A_p}{h_c A_c}\)

**Phase 2 — Melting (\(T_p = T_{melt}\)):**

- \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w) + \eta(T_{melt} - T_w)\right]\) — **(5)**
- \(\displaystyle \frac{dT_p}{dt} = 0\) — **(6)**
- \(\displaystyle \frac{dQ_p}{dt} = h_p A_p \max(0,\, T_w - T_{melt})\) — **(7)**
- \(Q_{p,\max} = H_f M_p\) — **(8)**

**Phase 3 — Post-melting (\(T_p > T_{melt}\)):**

- \(\displaystyle \frac{dT_w}{dt} = \frac{1}{\tau_w}\left[(T_c(t) - T_w) + \eta(T_p - T_w)\right]\) — **(9)**
- \(\displaystyle \frac{dT_p}{dt} = \frac{1}{\tau_{pl}}(T_w - T_p)\) — **(10)**
- \(\tau_{pl} = \dfrac{M_p C_{pl}}{h_p A_p}\)

**Energy balances:**

- Phase 1: \(E_{Water} = C_w M_w (T_w - T_{init})\) — **(11)**; \(E_{PCM} = C_{ps} M_p (T_p - T_{init})\) — **(12)**
- Phase 2: \(E_{Water} = C_w M_w (T_w - T_{init})\) — **(13)**; \(E_{PCM} = E_{p,melt,init} + \max(0, Q_p)\) — **(14)**
- Phase 3: \(E_{Water} = C_w M_w (T_w - T_{init})\) — **(15)**; \(E_{PCM} = E_{p,melt,init} + E_{p,melt3} + C_{pl} M_p (T_p - T_{melt})\) — **(16)**

where \(E_{p,melt,init} = C_{ps} M_p (T_{melt} - T_{init})\) and \(E_{p,melt3} = H_f M_p\).

**ML control linkage:**

- \(\mathbf{X} = [\mathrm{GHI}, \mathrm{DNI}, \mathrm{DHI}, T_{amb}, W_{spd}, RH_{um}, \mathrm{Hour}, \mathrm{Month}]\) — **(17)**
- \(\tau_{w,\mathrm{optimized}} = \left(\dfrac{\mathrm{flow\_multiplier}}{\tau_w}\right) \times \mathrm{base\_time\_constant}\) — **(18)**
- \(\mathrm{flow\_multiplier} = \mathrm{ML\_model}(\mathbf{x}_{normalized})\) — **(19)**

### 4c. Algorithm / Control Method Steps
1. Initialize PCM properties (Table 1) and tank/coil parameters; set \(T_{init} = 40\,^\circ\mathrm{C}\).
2. Load **8,760** hourly environmental records; normalize features for the NN.
3. For each simulation timestep: compute \(I_{solar}(t)\) and \(T_c(t)\); detect PCM phase (solid / melting / liquid) via event functions.
4. Integrate ODEs **(1)–(2)**, **(5)–(7)**, or **(9)–(10)** with `solve_ivp` (RK45, adaptive stepping).
5. **Conventional path:** fixed pump speed / baseline time constant.
6. **ML path:** predict `flow_multiplier` from **(17)–(19)**; update \(\tau_w\) and continue integration.
7. At each step, compute **(11)–(16)**; check energy conservation and phase consistency.
8. After 14 h, compute energy improvement %, temperature improvement %, and enhancement factor vs baseline.

**Neural network hyperparameters (stated):** 3 hidden layers **64 → 32 → 16**, activation **ReLU**, output = scalar flow multiplier; optimizer **Adam**, loss **MSE**, **100** epochs; **90%** validation prediction accuracy on held-out data.

### 4d. Data Sources & Dataset Details
| Source | Variables | Resolution | Scope | Period / size |
|--------|-----------|------------|-------|----------------|
| Synthetic / Meteonorm-style annual set (per text) | GHI, DNI, DHI, \(T_{amb}\), wind, RH, Hour, Month | Hourly | Climate representative of **33°N, 44°E** | **8,760** samples (1 year) |
| Table 2 scenario parameters | Peak irradiance, seasonal factor, collector efficiency, \(T_{amb}\) | Per scenario | Five named cases (Summer Sunny, Winter Cloudy, etc.) | **14 h** each, \(I_{max}\) **400–800 W/m²** |
| Rule-based synthetic targets | Optimal pump speed / flow multiplier, target system efficiency **30–80%** | Hourly | Same annual set | Used as supervised NN labels |

### 4e. Validation Method
- **Primary:** Simulation comparison of ML-optimized vs conventional fixed-speed control under five environmental scenarios for PCM variants P01–P05 (ML metrics tabulated for **P01–P03**).
- **NN validation:** **90%** prediction accuracy on validation split (MSE-trained Adam model).
- **Numerical checks:** Energy conservation at each timestep; relative tolerance **\(1\times10^{-6}\)**, absolute tolerance **\(1\times10^{-9}\)**.
- **No field experiment:** Authors state simulation-only validation; experimental confirmation listed as future work.

---

## 5. PCM Details (if applicable)
- **Materials tested:** Five labeled configurations **P01–P05** (generic organic-type PCM properties; geometry varies, chemistry not named as commercial grade).
- **Melting temperature range:** **44.0 °C** (all P01–P05)
- **Latent heat values:** **165,000 J/kg** (165 kJ/kg)
- **Thermal conductivity values:** Not reported for modeled PCM (literature context cites **0.1–0.5 W/m·K** for typical PCMs)
- **Specific heat (solid/liquid):** **2100 / 2300 J/kg·K**
- **Density:** **850 kg/m³**
- **Performance metrics:** Total energy storage up to **\(1.55\times10^7\) J** (P01, Summer Sunny); **12.3 MJ/kg** baseline without PCM vs PCM integration **+26%**; scenario efficiency heatmap **80–100%** (Summer Sunny) vs **20–40%** (Winter Cloudy); ML energy improvements **+2.5% to +4.1%** (Table 3).

| Config | \(V_p\) (m³) | \(A_p\) (m²) |
|--------|-------------|-------------|
| P01 | 0.05 | 5.0 |
| P02 | 0.03 | 3.0 |
| P03 | 0.025 | 2.5 |
| P04 | 0.025 | 2.5 |
| P05 | 0.035 | 3.5 |

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm name:** Feedforward **ANN** (supervised regression) for pump **flow_multiplier** vs **fixed-speed conventional pump** baseline.
- **Input features / state space:** GHI, DNI, DHI, \(T_{amb}\), wind speed \(W_{spd}\), relative humidity \(RH_{um}\), Hour (0–23), Month (1–12) — Eq. **(17)**.
- **Output / action space:** Continuous **flow_multiplier** (training distribution skewed **0.3–0.6**); scales water thermal time constant via **(18)–(19)**.
- **Training details:** Input (8) → hidden **64 → 32 → 16** (ReLU) → scalar output; **Adam**, **MSE**, **100** epochs; **8,760** hourly samples; synthetic rule-based labels; **90%** validation accuracy.
- **Performance metrics:** System-level **+3.3%** average energy storage improvement; enhancement factors **1.03–1.04×**; pumping energy **−12% to −18%** vs fixed speed.

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** Annual hourly meteorological-style dataset (GHI, DNI, DHI, temperature, wind, humidity); scenario parameters from Table 2; Meteonorm-style reference [27].
- **Climate variables:** GHI, DNI, DHI, \(T_{amb}\), wind speed, RH, Hour, Month; scenario peak irradiance, seasonal factor, collector efficiency.
- **Geographic scope:** **33°N, 44°E** (Middle East reference — not Indian cities).
- **Temporal resolution:** **Hourly** (training); **14 h** diurnal simulation per scenario with sinusoidal \(I_{solar}(t)\); \(I_{max}\) **400–800 W/m²** across five scenarios.

---

## 8. Key Results & Numbers
- ML energy storage improvement: **+3.3%** (P01), **+4.1%** (P02), **+2.5%** (P03); **average +3.3%** across P01–P03 (Table 3).
- Specific energy (Table 3): Normal **15.50** → ML **16.00 MJ/kg** (P01); **7.77 → 8.08** (P02); **6.93 → 7.11** (P03).
- Peak water temperature: **49.7 → 49.9 °C** (P01, **+0.4%**); **51.0 → 51.8 °C** (P02, **+1.6%**); **49.7 → 49.9 °C** (P03, **+0.5%**).
- Enhancement factors: **1.03** (P01, P03), **1.04** (P02); average **1.03×**.
- Pumping energy reduction with ML flow control: **12–18%** at flow multipliers **0.3–0.6**.
- Response time: target temperatures reached **15–20 minutes earlier** (~**2.3%** improvement relative to 14 h operation).
- Maximum energy storage: **\(1.55\times10^7\) J** (P01, Summer Sunny); Winter Cloudy range **\(2.5\times10^6\)–\(7.0\times10^6\) J**.
- Scenario efficiency (heatmap): **80–100%** (Summer Sunny); **20–40%** (Winter Cloudy).
- PCM vs no-PCM baseline: conventional SWH **12.3 MJ/kg** → P01 PCM integration **+26%** under Summer Sunny.
- Nanoparticle literature benchmark (comparison only): **32%** thermal conductivity gain, **72%** thermal efficiency (Dayer et al.) — not achieved by this ML method.

---

## 9. Baseline Comparison
- **Baseline method(s):** **Conventional fixed-speed pump control** (“Normal Method”); additional reference **SWH without PCM** (**12.3 MJ/kg** Summer Sunny).
- **Proposed method:** **ML-optimized variable flow** via ANN-predicted `flow_multiplier` retuning \(\tau_w\).
- **Improvement margin:** **+2.5% to +4.1%** energy (MJ/kg); **+0.4% to +1.6%** peak water temperature; **1.03–1.04×** enhancement factor; **12–18%** lower pumping energy.
- **Conditions:** Same PCM properties, tank geometry, five Table 2 scenarios, 14 h simulation, identical three-phase model; only control law differs.

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **simulation-only study.** Authors describe a retrofit-compatible concept requiring \(T_w\), \(T_p\), and environmental inputs but deploy **no physical prototype**, **no RPi/Arduino/ESP32**, and **no lab or field test**.

---

## 11. Limitations Acknowledged by Authors
- **Simulation-based only** — requires experimental validation in real SWH installations under diverse climates.
- ML controller trained on **synthetic optimization targets**, not measured operational data.
- PCM variants share identical **\(T_{melt}=44°C\)** — limits generalizability to other chemistries (e.g., RT35/OM35).
- Future work: **RNNs/Transformers**, **weather forecasting** for predictive control, extension to HVAC/industrial heat.

---

## 12. Direct Relevance to My Project
- **RG1 (No real-time adaptive control):** **Relevant** — ANN maps live weather to flow multipliers (**+2.5–4.1%**); your **DRL PPO** on embedded hardware extends this with charge/discharge/bypass modes.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Relevant (gap)** — Full software pipeline without hardware; your closed-loop prototype fills this.
- **RG3 (Poor alignment with household demand patterns):** **Not relevant** — No hot-water draw profiles or demand scheduling.
- **RG4 (Limited real-world experimental validation):** **Highly relevant** — Authors explicitly call for field trials; your multi-city evaluation addresses this.
- **RG5 (No predictive optimization under climatic uncertainty):** **Partially relevant** — Uses current/historical hourly weather, not forecasts; aligns with your ERA5/NASA POWER + forecast-driven DRL extension.

---

## 13. Equations to Reuse or Adapt
- **Pre-melt water/PCM dynamics:** Eqs. **(1)–(2)** for grey-box Gym environment.
- **Collector coil drive:** \(T_c(t)=T_{amb}+\eta_{col}I_{solar}(t)/20\) — **(3)**; couple to pyranometer/forecast GHI.
- **Diurnal solar:** \(I_{solar}(t)=I_{max}\sin(\pi(t_{hr}-t_{sr})/(t_{ss}-t_{sr}))\) — **(4)**.
- **Latent melting:** **(6)–(8)** with `solve_ivp` event detection for phase changes.
- **Stored PCM energy:** **(14)** in reward function.
- **ML/DRL feature vector:** \(\mathbf{X}=[\mathrm{GHI},\mathrm{DNI},\mathrm{DHI},T_{amb},W_{spd},RH,Hour,Month]\) — **(17)**.
- **Control action analogy:** \(\mathrm{flow\_multiplier}=\mathrm{ML\_model}(\mathbf{x}_{norm})\) — **(19)** → map to solenoid valve / pump speed / bypass mode in PPO action space.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Tamizharasan & Kini, "Deep learning approach for PCM-enhanced SWH," *Int. J. Energy Res.*, 2023** — DL + PCM-SWH parallel to your DRL line.
2. **Vempally & Dhanarathinam, ML PCM selection, *J. Therm. Anal. Calorim.*, 2023** — data-driven PCM selection like your XGBoost classifier.
3. **Goel et al., PCM in solar thermal review, *Appl. Therm. Eng.*, 2023** — Introduction/lit-review framing.
4. **Chen L. et al., solar thermal collector system design, *Renewable Energy*, 2023** — tank–coil–PCM schematic source.
5. **Meteonorm global meteorological database, 2023** — hourly climate data analogue to **ERA5/NASA POWER/ISRO**.

---

## 15. Suggested Use in My IEEE Paper
- **Section I (Introduction):** ML-for-PCM-TES underutilized; Barqawi reports **+3.3% average** stored energy vs fixed-speed pump with retrofit-compatible software control.
- **Section II (Literature Review):** ANN **flow_multiplier** on 8-feature weather vector; **+4.1%** max (P02) without hardware change.
- **Section III (Methodology):** Adopt three-phase ODEs **(1)–(16)** with RK45 tolerances **\(10^{-6}\)/\(10^{-9}\)** for grey-box training environment.
- **Section IV (Dataset & Setup):** Benchmark PCM Table 1 (\(T_{melt}=44°C\), \(H_f=165\) kJ/kg); map Summer Sunny **800 W/m²** → Jaisalmer, Winter Cloudy **400 W/m²** → monsoon cases.
- **Section V (Results):** Exceed **+3.3%** mean energy and **1.03×** enhancement; secondary metric **12–18%** pumping savings for valve/pump actuation.

---
