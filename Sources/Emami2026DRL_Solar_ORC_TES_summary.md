# Deep Reinforcement Learning-Based Smart Control of Solar-Driven Power Cycle with Thermal Energy Storage: A Los Angeles Case Study

**Authors:** Araz Emami, Ata Chitsaz, Amirali Nouri  
**Year:** 2026 (published online 18 December 2025)  
**Journal/Conference:** Energy Conversion and Management: X, Vol. 29, Article 101478  
**DOI:** https://doi.org/10.1016/j.ecmx.2025.101478  
**IEEE Citation:** A. Emami, A. Chitsaz, and A. Nouri, "Deep reinforcement learning-based smart control of solar-driven power cycle with thermal energy storage: A Los Angeles case study," Energy Convers. Manag.: X, vol. 29, p. 101478, 2026, doi: 10.1016/j.ecmx.2025.101478.

---

## 1. One-Line Summary
This paper trains a MATLAB–CoolProp DDPG supervisor on 8760 h of NSRDB solar data to jointly regulate ORC superheat, turbine inlet pressure, and net efficiency via pump mass-flow commands, achieving ~6 percentage-point higher annual mean efficiency and stable paraffin-TES cycling versus a fixed-flow passive baseline in a Los Angeles solar-ORC case study.

---

## 2. Problem Being Solved
- Solar-driven organic Rankine cycles (ORCs) face tightly coupled control of working-fluid superheat, turbine inlet pressure, and efficiency under steep irradiance ramps (up to ±100 W·m⁻²·min⁻¹), which conventional decoupled PID loops handle poorly.
- Fixed-mass-flow baseline operation with passive TES dispatch causes turbine inlet pressure swings (~1.9–4.0 MPa vs 2.5 MPa design), superheat deviations exceeding ±10 K from a +10 K target, and long near-zero efficiency periods.
- Prior DRL work (e.g., Wang et al.) addressed superheat only under short synthetic disturbances, not joint multi-objective control under realistic year-long solar variability.
- No prior single DRL agent was demonstrated to simultaneously coordinate superheat safety, pressure integrity, and thermodynamic efficiency on real solar-thermal input while preserving full nonlinear cycle physics.

---

## 3. Key Contributions
1. Multi-objective DDPG supervisory controller with five-dimensional state and continuous normalized mass-flow action, integrated with CoolProp 6.4 for non-linear R245fa ORC thermodynamics without model reduction.
2. Composite 8760 h GHI training profile from NSRDB (Los Angeles, 34.05°N, 118.25°W) capturing clear-sky, transient, and low-flux regimes (15% of hours <100 W·m⁻²; 18% >800 W·m⁻²).
3. Parabolic-trough collector + rule-based paraffin PCM-TES buffering upstream of the evaporator, with lumped TES charge/discharge/idle logic and ambient loss model.
4. Full-year closed-loop results vs fixed-flow baseline: ~6 percentage-point mean annual efficiency gain, pressure held within ~4% of 2.5 MPa, superheat within ±0.2 K of +10 K, and disciplined 0–250 MJ TES SOC cycles under DRL.
5. Post-hoc multi-objective genetic algorithm (GA) on DRL-controlled operating data mapping pressure–efficiency–temperature Pareto front (non-dominated cluster near 2.55 MPa, 10.1–10.7 K superheat, ~28.5% peak efficiency).

---

## 4. Methodology
### 4a. System / Experiment Setup
- **Plant:** Solar-ORC with parabolic-trough collector field (LS-2 style, η₀=0.765), shell-and-tube evaporator, axial turbine (ηₜ=0.75), water-cooled condenser (25 °C loop, 5 K approach), gear pump (ηₚ=0.70, ηₑ=0.90).
- **Working fluid:** R245fa; nominal evaporation **170 °C / 2.5 MPa**, condensation **40 °C / 450 kPa**; nominal thermal rating **~10 kW** at **20 m²** aperture (~550–600 W·m⁻² peak spring insolation).
- **TES:** Paraffin wax PCM upstream of evaporator; rule-based charging (excess heat, SOC≤1), discharging (deficit heat, SOC above minimum), idle otherwise; SOC band **0–250 MJ** in seasonal results.
- **Software:** MATLAB environment + **CoolProp 6.4** property calls; explicit Euler integration **Δt = 3600 s** (hourly GHI alignment); refined **Δt = 600 s** changes annual yield <0.4%.
- **Simulation scope:** 8760 h annual runs; baseline = constant mass flow + uncoordinated TES; proposed = DDPG pump-frequency command with safety filter.
- **Site:** Los Angeles, California (NSRDB v3, 34.05°N, 118.25°W); composite trace **8970 h** before hourly averaging (Weeks A/B/C from DOY 70–76, 96–100, 110–114).

### 4b. Mathematical Models & Equations
**Collector efficiency (EN 12975 quadratic):**

- \(\eta_{col} = \eta_0 - a_1 \dfrac{\Delta T}{I_b} - a_2 \dfrac{\Delta T^2}{I_b}\) — **(1)**  
  (\(\eta_0=0.765\), \(a_1=0.71\) W·m⁻²·K⁻¹, \(a_2=0.0015\) W·m⁻²·K⁻², \(\Delta T = T_m - T_a\))

**Useful collector heat:**

- \(Q_u = \eta_{col}\, I_b\, A_{ap}\) — **(2)**

**HTF energy balance:**

- \(m_{HTF} c_{p,HTF} \dfrac{dT_{out}}{dt} = \dot{Q}_u - \dot{m}\, c_{p,HTF}(T_{out}-T_{in})\) — **(3)**  
  (\(\dot{m}\) commanded by DRL agent)

**Evaporator outlet / wall dynamics:**

- \(T_{evap,out} = T_{sat}(P_{evap}) + \Delta T_{sh}\) — **(4)**
- \(C_{evap} \dfrac{dT_{evap}}{dt} = Q_{in} - \dot{m}\, h_{fg}\) — **(5)**

**Turbine:**

- \(\dot{W}_{turb} = \dot{m}(h_{in}-h_{out})\) — **(6)**
- \(h_{out} = h(P_{cond}, s_{in}) / \eta_t\) (isentropic reference) — **(7)**

**Condenser:**

- \(\dot{Q}_{cond} = \dot{m}(h_{out,turb}-h_{cond,out})\) — **(8)**

**Pump:**

- \(\dot{W}_{pump} = \dfrac{\dot{m}(h_{evap,in}-h_{cond,out})}{\eta_p \eta_e}\) — **(9)**

**Net efficiency (observed by controller):**

- \(\eta_{net}(t) = \dfrac{\dot{W}_{turb}(t)-\dot{W}_{pump}(t)}{\dot{Q}_{in}(t)}\) — **(10)**

**State vector (MDP observation):**

- \(\mathbf{s}_t = [I_b(t),\, T_{evap}(t),\, P_{in}(t),\, \Delta T_{sh}(t),\, \eta_{net}(t)]^T\) — **(11)/(12)**

**Action mapping:**

- \(a_t \in [-1,1] \Rightarrow \dot{m}_t = 0.075 + 0.025\, a_t\ \mathrm{kg{\cdot}s^{-1}}\) (range **0.05–0.10 kg·s⁻¹**) — **(13)**

**Reward:**

- \(r_t = -0.50|\Delta T_{sh}-10| - 0.30\dfrac{|P_{in}-2.5|}{0.1} + 0.20\,\eta_{net}\) — **(14)**  
  (instant penalty **−25** if \(P_{in}>3.0\) MPa or mass-flow ramp violates ~15% min⁻¹ limit)

**TES heat rate:**

- \(\dot{Q}_{TES} = \begin{cases} \dot{Q}_{ch}-\dot{Q}_{loss}, & \text{Charging} \\ -(\dot{Q}_{dis}+\dot{Q}_{loss}), & \text{Discharging} \\ 0, & \text{Idle} \end{cases}\) — **(19)**
- \(\dot{Q}_{loss} = UA(T_{PCM}-T_{amb})\) — **(20)**

**Exploration noise (Ornstein–Uhlenbeck):**

- \(dN_t = \theta(\mu_N - N_t)\,dt + \sigma_t \sqrt{dt}\,\varepsilon_t,\ \varepsilon_t \sim \mathcal{N}(0,1)\) — **(15)**

**Safety filter (pressure predictor & ramp limit):**

- \(\hat{P}_{in} = P_{in} + \kappa(\tilde{\dot{m}}_t - \dot{m}_{t-1}),\ \kappa \approx 12\) MPa·s·kg⁻¹ — **(17)**
- \(\dot{m}^{safe}_t\) limited by ±15% \(\dot{m}_{max}\,\Delta t\) ramp — **(18)**

### 4c. Algorithm / Control Method Steps
1. Build NSRDB composite GHI → hourly irradiance sequence; convert to collector thermal input via PTC model **(1)–(3)**.
2. Initialize ORC + paraffin TES states; set targets: **+10 K superheat**, **2.5 MPa** turbine inlet pressure.
3. At each hourly step, observe \(\mathbf{s}_t\) **(12)**; actor MLP outputs \(a_t \in [-1,1]\).
4. Map to \(\dot{m}_t\) **(13)**; apply OU noise **(15)** with \(\sigma_t\) annealed **0.20 → 0.05** (episodes 3000–9000 per paper schedule).
5. Apply two-layer safety filter **(17)–(18)**; discard unsafe transitions from replay buffer.
6. Simulate plant with **(4)–(10)** + rule-based TES **(19)–(20)**; compute reward **(14)** (normalized zero-mean, unit-variance over first 3000 episodes).
7. Store transitions in replay buffer (**10⁶**); update actor–critic (DDPG) with \(\gamma=0.99\), soft update \(\tau=0.005\).
8. Train until convergence (~**9000** episodes): moving-average return **> +8** for 5 consecutive episodes, TD loss plateau, constraint violation rate **< 0.5%** of timesteps.
9. Evaluate on blind **400 h** irradiance record from different meteorological year.
10. Apply GA multi-objective optimization on archived DRL-controlled \((P_{in}, \eta, T_{fluid})\) data for Pareto mapping.

**DDPG hyperparameters (Table 3):** Actor/Critic MLP **64 → 32**, ReLU, layer normalization; critic merges state+action paths; replay **1×10⁶**; OU \(\sigma\): 0.2 initial, linear decay; safety filter enabled; penalty weights also listed as \(w_\eta=0.5\), \(w_c=5.0\), \(w_u=0.1\) in Table 3 (ablation of full reward **(14)** noted as future work).

### 4d. Data Sources & Dataset Details
| Source | Variables | Resolution | Scope | Period / size |
|--------|-----------|------------|-------|----------------|
| **NSRDB v3** | GHI (composite); clearness index \(k_t\) for segment selection | 1 min raw → **1 h** after concat | Los Angeles (**34.05°N, 118.25°W**) | 1998–2022 archive; **8760 h** training trace (mean **512 W·m⁻²**, σ **282 W·m⁻²**, kurtosis **2.9**) |
| Composite weeks | Clear (DOY 70–76), mixed cumulus (96–100), stratocumulus (110–114) | Hourly | Same site | **8970 h** pre-average |
| Blind test set | Irradiance | Hourly | Different meteorological year | **400 h** |
| GA optimization set | \(P_{in}\), \(\eta_{net}\), working-fluid temperature | From DRL simulation logs | DRL-controlled ORC only | Full-year operational archive |

### 4e. Validation Method
- **Training convergence:** 5 random seeds converge at episodes **8997–9001** (mean **9000**, σ <8); moving-average episode return **> +8** for ≥5 episodes with episode-to-episode change **< 0.2** reward units.
- **Constraint compliance:** Worst-case violation rate (pressure >3 MPa or ramp limit breach) **< 0.5%** of timesteps over 20-episode window.
- **Generalization:** Frozen policy on **400 h** blind irradiance — average reward **−3%**, **no** pressure/ramp violations, cycle-average efficiency **> 22%**.
- **Baseline comparison:** Full **8760 h** fixed-flow + passive TES vs DDPG on same GHI profile (Figs. 8–18).
- **TES dispatch fit:** Predicted vs actual hourly TES usage **R² ≈ 0.99**, regression slope **0.995**, intercept **0.50 kWh**.
- **Sensitivity (intro):** Unseen GHI with overcast periods lengthened **30%** degrades efficiency **< 2** percentage points; pressure stays below safety valve limit.
- **No physical experiment:** Simulation-only validation in MATLAB–CoolProp.

---

## 5. PCM Details (if applicable)
- **Materials tested:** **Paraffin wax** (commercial PCM for TES upstream of ORC evaporator; not a SWH tank PCM).
- **Melting temperature range:** **45–60 °C**
- **Latent heat:** **180–210 kJ/kg**
- **Thermal conductivity:** **0.2–0.4 W/m·K**
- **Specific heat (solid/liquid):** **1.7–2.5 / 2.1–2.9 kJ/kg·K**
- **Density:** **820–900 kg/m³ (solid); 760–800 kg/m³ (liquid)**
- **Performance metrics reported:** TES SOC cycled **0–250 MJ** under DRL; rule-based charge/discharge; round-trip losses via **UA** model; DRL achieves regular daily SOC cycles vs irregular baseline overfill/underfill (Figs. 11, 18).

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm:** **Deep Deterministic Policy Gradient (DDPG)** continuous-control RL; post-hoc **multi-objective genetic algorithm (GA)** on DRL trajectories.
- **Input features / state space:** \(I_b(t)\), \(T_{evap}(t)\), \(P_{in}(t)\), \(\Delta T_{sh}(t)\), \(\eta_{net}(t)\) — **5D state (Eq. 12)**. (Solar training driver is **GHI** from NSRDB; state uses beam irradiance \(I_b\) for PTC model.)
- **Output / action space:** Continuous \(a_t \in [-1,1]\) → mass flow **0.05–0.10 kg·s⁻¹** **(13)**.
- **Model architecture:** Actor & Critic: **2 hidden layers 64 → 32**, ReLU, layer normalization; actor output **tanh**; critic linear Q-output; state and action pathways merged after second hidden layer.
- **Hyperparameters:** \(\gamma = 0.99\); soft update \(\tau = 0.005\); replay buffer **1×10⁶**; OU noise \(\sigma\): **0.20 → 0.05**; **~9000** training episodes; reward weights in **(14)**: 0.50 (superheat), 0.30 (pressure), 0.20 (efficiency).
- **Training data size:** **8760** hourly steps per episode × **~9000** episodes.
- **Hardware used for training:** N/A — MATLAB simulation; **~40 s wall-time per 8760 h episode** stated.
- **Performance metrics:** Annual mean \(\eta_{net}\) **+6 percentage points** vs baseline; superheat **±0.2 K** vs **±10 K** baseline; pressure within **~4%** of **2.5 MPa**; efficiency band **20–30%** under DRL vs baseline **0–30%** wide scatter; intro claim **16% → >22%** mean efficiency and **38%** improvement vs tuned **PID** benchmark (introduction validation statement).

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **National Solar Radiation Database (NSRDB) Version 3**; Perez clearness index \(k_t\) for segment classification.
- **Variables used:** **GHI** (primary composite input); \(k_t\) thresholds: clear \(k_t>0.65\), partly cloudy **0.15–0.65**; state/orientation uses **\(I_b\)** (beam) in collector **(1)–(2)**.
- **Geographic scope:** **Los Angeles, California, USA** (mid-latitude, high annual insolation **~1900 kWh·m⁻²·a⁻¹**).
- **Temporal resolution:** **1 min** NSRDB filtered → **1 h** simulation timestep.
- **Time period covered:** NSRDB archive **1998–2022**; training composite from selected DOY windows; **8760 h** annual simulation.
- **Clear-sky index / derived metrics:** \(k_t\) for week selection; composite mean GHI **512 W·m⁻²**, σ **282 W·m⁻²**; **15%** of hours **<100 W·m⁻²**, **18%** **>800 W·m⁻²**.

---

## 8. Key Results & Numbers
- Annual mean net ORC efficiency increased by **~6 percentage points** with DDPG vs fixed-flow baseline (Conclusion / §4.2).
- DRL holds turbine inlet pressure within **~4%** of **2.5 MPa** setpoint; baseline swings **1.9–4.0 MPa** seasonally (May–Aug peaks **>3.5 MPa** baseline; DRL summer peaks **~3.6 MPa** max vs tighter clustering).
- Superheat regulated to **±0.2 K** of **+10 K** target under DRL vs baseline deviations **>±10 K** (Fig. 16 seasonal blocks).
- Net efficiency operated in **20–30%** band under DRL vs baseline clusters near **0%** for extended winter/night periods (Abstract, Conclusion).
- **Jan–Apr** efficiency: baseline **5–22%** → DRL **13–28%**; **May–Aug**: DRL **20–30%** vs baseline drops **<10%** at times; **Sep–Dec**: baseline **~6%** min → DRL **15–27%**.
- Training converges at episode **9000** (five seeds: **8997–9001**); blind **400 h** test: reward **−3%**, efficiency **>22%**.
- TES hourly usage prediction: **R² ≈ 0.99**, slope **0.995**, intercept **0.50 kWh** (Fig. 14).
- TES SOC under DRL: disciplined **0–250 MJ** daily cycles; baseline overcharges above **250 MJ** in summer (Fig. 18).
- GA Pareto peak: **~28.5%** efficiency at **~2.55 MPa**, superheat **10.1–10.7 K**; efficiency ridge **η ≈ 31%** near **2.55 MPa / 10 K** (Fig. 19).
- Design sensitivity (Fig. 20): optimal plateau **~30%** η near **200 MJ** TES and **1200 m²** collector field (ranges 100–300 MJ, 900–1500 m²).
- Pumping energy: DRL commands smooth mass flow; intro sensitivity — overcast +**30%** duration reduces efficiency **<2** percentage points.
- Relative to tuned **PID**: introduction reports **~38%** average efficiency increase and superheat within **0.01 K** during training evaluation (distinct from baseline fixed-flow comparison).

---

## 9. Baseline Comparison
- **Baseline method(s):** **Fixed mass-flow rate** ORC operation; **uncoordinated** TES charge/discharge (no active optimization); conventional single-loop PID cited in literature but baseline in results is passive/fixed-flow.
- **Proposed method:** **DDPG supervisory controller** with OU exploration, safety filter, and coordinated pump mass-flow modulation; **rule-based TES** (not RL-learned).
- **Improvement margin:** **+6** percentage points annual mean \(\eta_{net}\); pressure stability **~4%** vs **±40%+** relative swings; superheat **±0.2 K** vs **±10 K**; seasonal efficiency uplift **up to ~15** percentage points in low-GHI quarters.
- **Conditions of comparison:** Same **8760 h** Los Angeles GHI composite, same R245fa ORC + paraffin TES model, same MATLAB–CoolProp physics; only control layer differs.

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — this paper is purely simulation-based (MATLAB + CoolProp). No physical sensors, actuators, embedded platforms (RPi/Arduino/ESP32), or field tests are reported. Authors position the approach as a **retrofit-compatible SCADA/software** upgrade pathway requiring measured or simulated plant data only.

---

## 11. Limitations Acknowledged by Authors
- Detailed **ablation analysis** of reward weights, noise model, normalization, and safety filters is **beyond the scope** of this study and left to future work.
- Future work must add **direct online learning** for evolving solar/load profiles, **weather forecasting** for predictive dispatch, and comparison with **MPC** and **PPO**.
- Extension to **variable-geometry expanders** and **real-time multi-objective optimization** along the Pareto front under changing operator priorities is not yet demonstrated.
- Framework validated only in **simulation** (Los Angeles case); field-scale transferability claimed but not experimentally proven in this paper.
- Fig. 13 discussion notes **minor systematic bias** in learned irradiance-related scatter due to underrepresented low-frequency atmospheric regimes in training data.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Relevant.** DDPG provides continuous real-time pump mass-flow adaptation from plant states, cutting superheat/pressure excursions versus fixed control—direct methodological precedent for your **PPO/DDPG charge–discharge–bypass** policy, though applied to ORC power not domestic SWH.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Partially relevant.** Integrates **PCM-TES + DRL in software** (MATLAB) but **no embedded prototype** (RPi/DS18B20/solenoid); supports your gap that AI–PCM coupling remains simulation-bound in published ORC work.
- **RG3 (Poor alignment with household demand patterns):** **Not Relevant.** Objectives are turbine **superheat, pressure, and η_net** for electricity generation; no domestic hot-water draw or morning/evening load profiles.
- **RG4 (Limited real-world experimental validation):** **Relevant as contrast.** Full **8760 h** simulation with blind **400 h** test, but **zero hardware validation**—strengthens your FYP claim that PCM–AI–SWH needs Indian field/bench data beyond Emami-style desktop studies.
- **RG5 (No predictive optimization under climatic uncertainty):** **Partially relevant.** Training uses historical NSRDB variability; authors explicitly propose **weather-forecast integration** as future work and note the DRL agent does **not** explicitly predict GHI (forecasting is a separate module in their discussion). Your **XGBoost + ERA5/NASA POWER** pipeline addresses this gap for Indian sites.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(\eta_{net} = (\dot{W}_{turb}-\dot{W}_{pump})/\dot{Q}_{in}\) **(10)** | Instantaneous useful output ratio | RL reward term for COP/efficiency maximization in grey-box SWH |
| \(r_t = -w_{sh}|\Delta T_{sh}-T^*| - w_P|P_{in}-P^*| + w_\eta \eta_{net}\) **(14)** | Multi-objective safety + performance reward | Template for PPO reward: penalize \(T_w\) error, PCM constraint violations, reward delivered energy |
| \(a_t \mapsto \dot{m}_t = 0.075 + 0.025 a_t\) **(13)** | Bounded continuous actuation | Analogous mapping for normalized valve/pump command on ESP32 |
| \(\dot{Q}_{TES}\) charge/discharge/idle **(19)**–**(20)** | PCM storage with ambient loss | Rule-based PCM bypass/charge modes before DRL overrides in hybrid controller |
| OU noise **(15)** + ramp safety **(18)** | Exploration without actuator damage | Stable-Baselines3 exploration + rate limits on solenoid/pump commands |
| \(\eta_{col} = \eta_0 - a_1\Delta T/I_b - a_2(\Delta T)^2/I_b\) **(1)** | Solar collector thermal input | Couple pyranometer/forecast GHI to collector thermal input in Indian cities |
| GA Pareto over \((P_{in}, \eta, T)\) | Post-hoc multi-objective trade space | Offline PCM geometry/PCM-type selection (NSGA-II/PSO) complementing online PPO |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **Wang X. et al., "Control of superheat of organic Rankine cycle under transient heat source based on deep reinforcement learning," *Appl. Energy*, 2020** — Relevant because: Foundational **DRL superheat control** for ORC that this paper extends to joint pressure–efficiency objectives under real solar profiles.
2. **Zalba B. et al., "Review on thermal energy storage with phase change," *Appl. Therm. Eng.*, 2003 [32]** — Relevant because: Canonical **PCM property ranges** (latent heat, conductivity) cited for paraffin TES sizing in their ORC model.
3. **Hernandez A. et al., "Experimental validation of MPC for waste heat recovery ORC," *Appl. Therm. Eng.*, 2021 [21]** — Relevant because: Benchmark **advanced model-based control** the authors propose comparing against PPO/MPC in future work.
4. **Imran M. et al., "Dynamic modeling and control strategies of ORC systems," *Appl. Energy*, 2020 [13]** — Relevant because: Reviews **PID limitations** (e.g., **18 K** superheat overshoot on 50% heat step) motivating RL for thermal plants.
5. **Dorokhova M. et al., "DRL control of EV charging in the presence of PV," *Appl. Energy*, 2021 [33]** — Relevant because: Demonstrates **DDPG + OU noise** for solar-volatile systems—methodological parallel to your SB3 DDPG/PPO training setup.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Gap: single-loop ORC control fails under solar transients | "Fixed-flow solar-ORC baselines exhibit turbine pressure swings of **1.9–4.0 MPa** and superheat errors **>±10 K**, cutting seasonal yield up to **25%** vs design (field data cited by Emami et al.)." |
| II. Literature Review | DRL for solar-thermal plants (not SWH) | Method: **DDPG** on 5-state ORC model; Key insight: **+6 pp** annual \(\eta_{net}\), pressure within **~4%** of setpoint |
| III. Methodology | Reward shaping + bounded actions | Adopt weighted penalty form **(14)** and action scaling **(13)** for PPO valve/pump commands |
| IV. Dataset & Setup | Long-horizon solar training | **8760 h** NSRDB composite; cite need to replicate with **ERA5/NASA POWER** for Coimbatore/Jaisalmer/Kochi |
| V. Results | Simulation-only DRL benchmark | Contrast your embedded PCM-SWH prototype against Emami et al.'s **MATLAB-only** validation and **±0.2 K** superheat-level control precision |
