# The Potential of Machine Learning to Predict Melting Response Time of Phase Change Materials in Triplex-Tube Latent Thermal Energy Storage Systems

**Authors:** Peiliang Yan, Chuang Wen, Hongbing Ding, Xuehui Wang, Yan Yang  
**Year:** 2025  
**Journal/Conference:** Applied Energy, Vol. 390, Article 125863  
**DOI/Link:** https://doi.org/10.1016/j.apenergy.2025.125863  
**IEEE Citation:** P. Yan et al., "The potential of machine learning to predict melting response time of phase change materials in triplex-tube latent thermal energy storage systems," Appl. Energy, vol. 390, p. 125863, 2025, doi: 10.1016/j.apenergy.2025.125863.

---

## 1. One-Line Summary
This study builds a **60-case** enthalpy-porosity CFD dataset for a **Y-fin triplex-tube** PCM unit (**RT82**, melt time **15–45 min**) and compares **PR, SVR, RFR, and XGBoost** (Bayesian-tuned) to predict melting response time—**XGBoost** achieves **92%** accuracy with **~5 min** max error, while SHAP-style importance ranks **fin width 51%** and **HTF temperature 47%** vs **fin angle 2%**.

---

## 2. Problem Being Solved
- PCM poor conductivity slows melting in triplex-tube LHS, causing supply–demand temporal mismatch in buildings and solar thermal systems.
- CFD with enthalpy-porosity is accurate but expensive for design sweeps over fin geometry and HTF conditions.
- Empirical correlations are scenario-specific and inaccurate when extended to new fin configurations.
- Need fast, quantitative meta-models to guide fin design and operational setpoints for melting response time.

---

## 3. Key Contributions
1. **Y-shaped fin triplex-tube** PCM-TES cross-section model (copper tubes **200/150/50.8 mm** OD) with **RT82** PCM and fixed **2%** Y-fin area fraction.
2. **CFD dataset:** **60** numerical cases; melting response time **15–45 min** under varied fin width, fin angle, and HTF temperature.
3. **Four ML regressors:** Polynomial Regression (PR), SVR, Random Forest (RFR), **XGBoost** — hyperparameters tuned via **Bayesian optimization** (**400** steps: **200** random + **200** fine search).
4. **XGBoost** identified as best meta-model (**92%** accuracy; lowest test-set error).
5. **Feature importance:** fin width **51%**, HTF temperature **47%**, fin angle **2%** — design guidance for surface area over angle styling.

---

## 4. Methodology
### 4a. System
- Triplex-tube LTES: outer/middle/inner copper tubes; PCM in middle annulus; HTF in inner/outer channels.
- **PCM:** Rubitherm-class **RT82** (\(T_s=350.15\) K, \(T_l=358.15\) K, \(L=176\) kJ/kg, \(k=0.2\) W/m·K, \(\rho=770\) kg/m³).
- **Fins:** Y-shaped on inner/middle tubes, staggered; branch length **2×** root length.

### 4b. CFD (enthalpy-porosity)
- Continuity **(1)**; momentum **(2)–(3)** with Boussinesq source \(\rho g \beta (T-T_m)\); porosity sink **(4)**; liquid fraction \(\lambda(T)\) piecewise linear between \(T_s\) and \(T_l\).
- Same method as prior work [38]; generates labeled melt times for ML.

### 4c. ML Pipeline
1. Variable independence check before modeling.
2. Inputs: **fin width**, **fin angle**, **HTF temperature**; output: **melting response time** (min).
3. Train/test split; Bayesian hyperparameter search maximizing **\(R^2\)**.
4. Evaluate with **MSE (10)** and **\(R^2\) (11)**; residual and parity plots.
5. Permutation-based feature importance on best **XGBoost** model.

### 4d. Software/Hardware
- **Python 3.9**; PC: Intel i5-8300H @ 2.30 GHz, **24 GB** RAM, Windows 11.

---

## 5. PCM Details (if applicable)
| Property | RT82 (Table 2) |
|----------|----------------|
| Density | **770 kg/m³** |
| Specific heat | **2000 J/kg·K** |
| Thermal conductivity | **0.2 W/m·K** |
| Latent heat | **176,000 J/kg** |
| Solidus / liquidus | **350.15 K / 358.15 K** (~**77–85 °C**) |
| \(\beta\) | **0.001 1/K** |

**Design variables (Table 1):**
- Fin width: **0.5, 1, 1.5, 2** (mm per table; abstract also cites **5–15 mm** range in narrative)
- Fin angle: **30°, 60°, 90°** (abstract: **10°–30°** branch-angle study context)
- HTF temperature: **363, 365.5, 368, 370.5, 373 K** (**90–100 °C**)

**Performance target:** melting response time **15–45 min** across **60** CFD cases.

---

## 6. AI / ML / Control Details (if applicable)
| Algorithm | Key tuned hyperparameters | Performance notes |
|-----------|---------------------------|-------------------|
| PR | degree **2** | Most unbiased train/test MSE ratio **1.15×** |
| SVR | \(\gamma=0.154\), \(C=186\), \(\epsilon=0.369\) | Severe overfitting; train/test MSE ratio **11.8×** |
| RFR | 65 trees, max_depth **7**, max_features **0.999** | Max residual **>15 min**; worst high-value bias |
| **XGBoost** | 497 trees, lr **0.389**, max_depth **46**, subsample **0.933** | **92%** accuracy; max error **~5 min**; best test **\(R^2\)** |

**Metrics:** MSE Eq. **(10)**; \(R^2\) Eq. **(11)**.  
**No real-time control** — offline surrogate for design optimization.

---

## 7. Solar / Climate Data Details (if applicable)
N/A — building/solar-TES motivated in introduction but dataset uses **prescribed HTF inlet temperatures (90–100 °C)**, not outdoor weather time series.  
**Project link:** melt-time surrogate can inform grey-box PCM charge duration estimates when HTF is driven by collector/pyranometer models.

---

## 8. Key Results & Numbers
- Global temperature rise **+1.09 °C** since pre-industrial (IPCC AR6 cite).
- Dataset: **60** simulations; melt time **15–45 min**.
- **XGBoost accuracy: 92%** (abstract/conclusion).
- **XGBoost max prediction error ~5 min** vs **>15 min** for RFR/SVR/PR outliers.
- **SVR** train/test MSE gap **11.8×** (overfitting).
- **PR** train/test MSE gap **1.15×** (best unbiasedness).
- **XGBoost** train/test MSE gap **~4.84×**.
- **Feature importance (XGBoost):** fin width **51%**, HTF temperature **47%**, fin angle **2%**.
- Y-fin cross-section fixed at **2%** of system area.
- Tube ODs: **200 / 150 / 50.8 mm** (wall **2 / 2 / 1.2 mm**).

---

## 9. Baseline Comparison
| Method | vs XGBoost | Result |
|--------|------------|--------|
| Full CFD (enthalpy-porosity) | Ground truth for 60 cases | Minutes per case; ML replaces for sweeps |
| Polynomial Regression | Higher max error (~15 min cases) | Less accurate; best MSE fairness (1.15×) |
| SVR | Overfits (11.8× MSE gap) | Unsuitable without regularization redesign |
| Random Forest | Max error **>15 min** | Poor at high melt times |
| **XGBoost** | Best test \(R^2\) and **92%** accuracy | **~5 min** max residual |

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **CFD-generated dataset only**; no physical triplex-tube experiment or embedded platform in this paper. Copper tube/fin material properties from tables; validation is train/test ML split against simulation labels.

---

## 11. Limitations Acknowledged by Authors
- Optimal algorithm is **context-specific** to this triplex-tube Y-fin geometry.
- Models trained on one PCM (**RT82**) may not transfer directly to RT35/OM35 without retraining.
- SVR shows **significant overfitting** on test set.
- Dataset size (**60** cases) is modest; scalability relies on adding more CFD points.
- Authors note meta-model **selection framework** generalizes better than direct weight transfer across configurations.

---

## 12. Direct Relevance to My Project
- **RG1:** **Indirect** — predicts melt time, not closed-loop control; informs charge-phase timing in DRL state/reward.
- **RG2:** **Relevant** — demonstrates **XGBoost** for PCM thermal surrogate (same toolkit as your PCM classifier); no hardware integration.
- **RG3:** **Indirect** — links melt speed to HTF temperature (collector outlet); map HTF **47%** importance to demand-aligned charging.
- **RG4:** **Not relevant** — simulation-only, no field validation.
- **RG5:** **Relevant** — fast melt-time predictor complements climate-driven HTF forecasting; fin width importance supports geometry sensitivity in grey-box model.

---

## 13. Equations to Reuse or Adapt
**Liquid fraction (enthalpy-porosity):**
\[
\lambda = \begin{cases} 0 & T < T_s \\ \dfrac{T-T_s}{T_l-T_s} & T_s \le T \le T_l \\ 1 & T > T_l \end{cases}
\]

**Momentum sink (porosity):**
\[
A = -C\frac{(1-\lambda)^2}{\lambda^3 + \varepsilon}
\]

**ML metrics:**
\[
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2, \quad
R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

**Melting response time target:** \(t_{melt} = f(\text{fin width}, \text{fin angle}, T_{HTF})\) — use as grey-box calibration output or reward penalty for slow charge.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **Ermis et al., ANN for finned-tube PCM storage, *Int. J. Heat Mass Transf.*, 2007** — early PCM+ANN thermal prediction.
2. **Yan et al., leaf-vein bionic fin PCM-TES, *Appl. Energy*, 2023** — prior Y-fin qualitative study [38].
3. **Mahdi & Nsofor, nano+foam triplex-tube melting, *Appl. Energy*, 2017** — triplex-tube enhancement benchmark.
4. **Liu et al., AI–PCM TES review, *Renew. Energy*, 2025** — broader ML+PCM context.
5. **Chen et al., Taguchi+GRA PCM-SWH, *Energy Convers. Manag.: X*, 2025** — SWH optimization with RT35-class PCM.

---

## 15. Suggested Use in My IEEE Paper
- **Section I:** Cite temporal mismatch between solar availability and thermal demand as motivation for fast PCM charge modeling.
- **Section II:** Position Yan as **XGBoost melt-time surrogate** reference alongside your PCM **selection** XGBoost (Presentation cites Yan 2025).
- **Section III:** Use feature-importance pattern (**width 51%, HTF 47%**) to justify classifier features (\(k\), \(L\), \(T_m\), predicted collector outlet).
- **Section IV:** Benchmark surrogate training on **60+** CFD/experimental points for your tank geometry; report **\(R^2\)** and max error in minutes.
- **Section V:** Target **>92%** accuracy or **<5 min** melt-time RMSE vs their XGBoost baseline when calibrating grey-box against TRNSYS or bench data.

---
