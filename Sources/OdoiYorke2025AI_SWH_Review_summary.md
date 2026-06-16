# Artificial Intelligence for Solar Water Heating Systems: A Review of Global Research Trends, Advances, and Future Perspectives

**Authors:** Flavio Odoi-Yorke  
**Year:** 2025  
**Journal/Conference:** Energy Conversion and Management: X, Vol. 28, Article 101378  
**DOI/Link:** https://doi.org/10.1016/j.ecmx.2025.101378  
**IEEE Citation:** F. Odoi-Yorke, "Artificial intelligence for solar water heating systems: A review of global research trends, advances, and future perspectives," Energy Convers. Manag.: X, vol. 28, p. 101378, 2025, doi: 10.1016/j.ecmx.2025.101378.

---

## 1. One-Line Summary
This dual-method review combines PRISMA-guided bibliometric analysis of **245** Scopus-indexed AI–SWHS papers (2000–2024) with a qualitative systematic review, showing exponential post-**2019** growth led by China (**151** pubs), India (**105**), and the USA (**65**), and mapping five research clusters—neural prediction, multi-objective optimisation, intelligent control/fault detection, TRNSYS–ANN hybrids, and deep learning/PV/T—while identifying gaps in real-world validation, PCM–AI integration, and developing-region deployment.

---

## 2. Problem Being Solved
- SWHS performance is limited by design inefficiencies, dynamic environmental conditions, and weak predictive/control capabilities under variable irradiance and demand.
- Prior SWHS reviews were largely qualitative and technical; they lacked quantitative mapping of global AI research trends, collaboration networks, and thematic evolution.
- Regional disparities in AI–SWHS research (especially Africa at **~5.3%** of output) risk technology designs that ignore local climate, economics, and household usage patterns.
- No unified evidence base existed for how ML, ANN, optimisation, and control methods collectively improve thermal efficiency, exergy, and lifecycle cost across collector types and storage configurations.

---

## 3. Key Contributions
1. **Dual methodology:** PRISMA data collection + **Bibliometrix (R 4.3.2)** + **VOSviewer 1.6.20** bibliometrics complemented by qualitative systematic review of AI applications in predictive modelling, optimisation, control, and system design.
2. **Curated corpus:** **255** initial Scopus hits (Oct 21, 2024) → **245** English peer-reviewed documents after filtering duplicates, non-research types, and non-English items.
3. **Temporal and geographic mapping:** Linear trend \(y = 1.0554x - 2113.6\), \(R^2 = 0.776\); peak **34** publications in **2024**; China/India/USA = **~48%** of activity; Africa **36** pubs total (**5.3%**).
4. **Five keyword clusters** identified via VOSviewer (26 keywords, ≥4 occurrences from 650): (1) computational intelligence & optimisation, (2) thermal performance & exergy, (3) intelligent control & fault detection, (4) TRNSYS–simulation hybrids, (5) deep learning & PV/T integration.
5. **Synthesised performance benchmarks** from cited primary studies (Table 1): ANN \(R^2\) up to **0.9993**, DNN+PCM MAPE **<15%**, MPC/Q-learning control, Random Forest MAPE **2.94–5.86%**, and documented **12–22%** electricity savings in lab smart-control demos.
6. **Future research agenda** explicitly calls for physics-informed AI, long-term field validation, demand-aligned control, and equitable deployment in solar-rich developing regions—including India and Africa.

---

## 4. Methodology
### 4a. Data Collection (PRISMA)
- **Database:** Scopus (chosen over WoS/Google Scholar for engineering coverage and metadata).
- **Search date:** October 21, 2024; window **2000–2024**.
- **Boolean query:** Two concept clusters—(A) AI terms (fuzzy logic, ANN, genetic algorithms, machine learning, deep learning, reinforcement learning, XGBoost, LSTM, PSO, etc.) AND (B) SWHS terms (solar water heater, solar thermal collector, solar hot water, etc.).
- **Inclusion:** research articles, conference papers, reviews, book chapters in English with explicit AI integration in thermal water heating.
- **Exclusion:** notes, errata, editorials; studies without AI or without SWHS thermal focus.

### 4b. Bibliometric Analysis
- **Bibliometrix:** annual trends, country production, keyword analysis, thematic mapping, factorial analysis.
- **VOSviewer:** co-occurrence networks; association strength normalisation; attraction **2**, repulsion **0**; resolution **1**; minimum cluster size **1**.

### 4c. Qualitative Systematic Review
- Secondary synthesis of peer-reviewed AI–SWHS studies grouped by cluster: prediction, optimisation, control, hybrid simulation, deep learning.
- Cross-references **150+** primary studies; Table 1 tabulates AI method, inputs, metrics, and outcomes for representative works.

### 4d. Validation
- Bibliometric reproducibility via PRISMA flow (Fig. 3a).
- No new experiments; validation is literature-based consistency checking and cross-study metric comparison.

---

## 5. PCM Details (if applicable)
- PCM is **not the primary focus** of this review paper, but the systematic review cites PCM-integrated SWHS studies:
  - **Uniyal et al.:** nine ML models for **U-tube PCM** solar collectors; ANN and SVR \(R^2\) up to **0.9540**.
  - **Tamizharasan et al. [116]:** **DNN for SWHS with PCM** — training accuracy **0.83888–0.98692**, RMSE below **15%** threshold, MAPE mainly **<5%** (vs industry **20–30%** MAPE benchmark).
  - **Kanimozhi et al. [121]:** ANN for TES with **paraffin and honey wax** — **265** trials; Chi-square **1.5–4.8**; MAPE **<12%**; RMSE **0.65–1.9**; time contributes **>40%** to heat improvement in charge/discharge.
  - **Shirinbakhsh et al. [139]:** effect of hot-water demand and **PCM integration** on SDHW performance (cited in references).
  - **Ramesh et al. [97]:** PCM in **heat pipe evacuated tube collectors** for superior storage (cited).
- Review conclusion: PCM + AI remains under-validated in long-term field tests relative to simulation-heavy PCM-ML papers.

---

## 6. AI / ML / Control Details (if applicable)
### Algorithms Surveyed (dominant in corpus)
| Category | Methods |
|----------|---------|
| Prediction | ANN, DNN, LSTM, ELM, SVR, Random Forest, XGBoost, LightGBM, Extra Trees, ANFIS, graybox + ALNN |
| Optimisation | GA, PSO, NSGA-II, multi-objective PSO, micro-time variant PSO |
| Control | Fuzzy logic, **MPC**, **Q-learning / reinforcement learning**, NNC, AFLC, PID |
| Hybrid | TRNSYS + ANN, physics-informed (called for, rarely implemented) |

### Representative Inputs / States (from synthesised studies)
- Collector area (**1.81–4.38 m²**), flow rate (**0.01–0.015 kg/s** optimal in one PSO study), inlet/outlet temperatures, ambient temperature, solar radiation, tank stratification layers, PCM state, electricity demand, nanofluid concentration (e.g. **CeO₂/water 0.01%**).

### Representative Outputs / Actions
- Heat collection rate, heat loss coefficient, outlet temperature, annual energy, solar fraction, pump flow rate, fault class, lifecycle cost.

### Training / Data Scale (examples cited)
- **915** samples (ELM heat collection/loss) [118]; **30** thermosiphon systems ISO 9459-2 [29]; **36** systems Random Forest training [123]; **608** tank design combinations [126]; **265** PCM TES trials [121].

### Performance Metrics Reported
- \(R^2\): **0.776** (publication trend) to **0.9993** (ANN thermosiphon prediction)
- RMSE: e.g. **0.30** (ELM heat collection), **0.67** (ELM heat loss), DNN+PCM **<15%**
- MAPE: RF **2.94–5.86%**; DNN+PCM **<5%**; fault fusion **89.7–93.7%** accuracy
- Efficiency gains: PSO minichannel collector **+10–12%**; groove absorber XAI **+20%**; nanofluid ANN max **78.2%** at **2 L/min**

---

## 7. Solar / Climate Data Details (if applicable)
- **Bibliometric scope:** global; leading countries China, India, USA, Iran, Italy, Spain, Germany, Brazil, Mexico, Indonesia, Saudi Arabia, Egypt.
- **Climate variables in cited studies:** solar radiation, ambient temperature, seasonal variation (4-season ANN for heat pipe collectors), clear-sky vs cloudy performance (TRNSYS vs ANN crossover at very cloudy conditions).
- **Forecast integration mentioned:** GraphCast and Pangu-Weather (up to **10-day** weather prediction) as enablers for adaptive SWHS—not primary data sources in this review.
- **India relevance:** **105** publications (**16%** of mapped output); cited Indian-linked work includes Singh PCM-SWH review [40], Uniyal PCM-ETC ML study, Pathak/Chopra HP-ETSC ML studies.
- **Temporal resolution in cited works:** hourly (TRNSYS–ANN), daily/annual (RF replacing ISO 9459-5 tests), long-term thermosiphon campaigns.
- **Project-aligned sources not used directly:** ERA5, NASA POWER, ISRO Solar Calculator, Global Solar Atlas—not referenced in this paper; review is bibliometric, not geospatial modelling.

---

## 8. Key Results & Numbers
- **245** final Scopus documents (from **255** initial, **252** after type filter).
- Publication trend: **3** papers in **2000** → **34** in **2024**; surge from **2019** (**25** papers) onward.
- Linear regression slope **+1.0554** papers/year; \(R^2 = 0.776\) (**77.6%** variance explained).
- **China 151** (**~22%**), **India 105** (**16%**), **USA 65** (**10%**) — **~48%** combined.
- **Africa 36** total (**5.3%**); Egypt **18**, South Africa **4**, Algeria **5**, Morocco **3**.
- Global SWH capacity **560 GW_th** in **2023** (+**18 GW_th** YoY); water heater market **$23.7B** (2023) → **$32.1B** by **2029** at **5.2%** CAGR.
- Global energy demand projected **+~50%** (2020–2050) per EIA citation.
- **Levenberg–Marquardt ANN** nanofluid collector: correlation **>0.98** [68]; Cu-MWCNT ANN \(R^2\) up to **0.9989** [69].
- **XGBoost/BRT** hybrid nanofluid: \(R^2\) **0.9914–0.9997** [70].
- **Multi-objective PSO** combisystem: 1→10 collectors → lifecycle energy **−63%**, cost **+84%** [73].
- **Enhanced PSO** PV/T: pump energy **−17.93%**, thermal efficiency **+7.86%** [75].
- **DNN + PCM SWHS:** training accuracy **0.839–0.987**, testing **0.720–0.987**, MAPE **<15%** [116].
- **Random Forest** vs ISO 9459-5 testing: MAPE **2.94–5.86%**, \(R^2\) **0.995–0.998** for annual energy [123].
- **TRNSYS vs ANN** (tropical 14-day): TRNSYS MAE **1.5 °C**, ANN MAE **1.7 °C**; both \(R^2 > 0.95\) [101].
- **30** thermosiphon systems ANN: training \(R^2 = 0.9993\), validation **0.9913** [29].
- **Fault detection** SVM-DS fusion: accuracy **89.7–93.7%** vs traditional **77.6–84.7%** [153].
- **Deep RL** smart water heater control: **12–22%** electricity savings (lab, cited as needing field validation) [88].
- **VOSviewer keywords:** **26** of **650** met ≥**4** occurrences threshold.

---

## 9. Baseline Comparison
| Comparison | Baseline | AI / Optimised | Improvement |
|------------|----------|----------------|-------------|
| Heat transfer correlations vs ANN [28] | Conventional \(R^2\) **0.808–0.522** | ANN \(R^2\) **0.993** train, **0.978** validation | **+18.5–45.6** percentage points |
| Plain water vs CeO₂ nanofluid ANN [84] | Plain water baseline | **78.2%** max efficiency at **2 L/min** | **+21.5%** thermal efficiency |
| TRNSYS vs ANN outlet temp [101] | TRNSYS MAE **1.5 °C** | ANN MAE **1.7 °C** | TRNSYS better in variable cloudiness; ANN better very cloudy |
| ANFIS/TRN vs ANN seasonal [163] | ANFIS, thermal resistance network | ANN autumn \(R^2 = 0.989\), VAF **0.99489** | Max divergence **3.56%** thermal, **1.52%** exergy |
| ISO 9459-5 experimental testing [123] | Standard test campaign | Random Forest predictor | MAPE **2.94–5.86%**; reduces test burden |
| PSO collector count [73] | 1 collector | 10 collectors | Energy **−63%**, cost **+84%** (trade-off) |
| Industry MAPE benchmark [116] | **20–30%** MAPE typical | DNN+PCM **<5%** MAPE | Substantially below industry norm |
| Traditional fault detection [153] | **77.6–84.7%** accuracy | SVM-DS fusion **89.7–93.7%** | **+5–16** percentage points |

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **review paper without original hardware experiment.**  
Cited embedded/field setups in literature include:
- **IoT wireless temperature monitoring** for domestic SWHS [146]
- **Arduino/Android** platforms reducing test time from **15 days** to near real-time [28]
- **Low-cost three-way valve PID** passive SWH controller [150]
- Laboratory smart water heaters with DRL for electricity/carbon reduction [88]
- Thermosiphon and flat-plate / ETC / heat-pipe collector test rigs across global labs (no single unified prototype in this review).

---

## 11. Limitations Acknowledged by Authors
- **Limited experimental validation** of AI models; most work is simulation or short-duration lab tests [147, 148, 152].
- **No standardised public datasets** for AI–SWHS benchmarking.
- **Physics-informed neural networks** underexplored despite TRNSYS–ANN success (\(R^2 > 0.93\)).
- **Training data requirements** not systematically characterised (hundreds to hundreds of thousands of samples).
- **Africa and Latin America underrepresented** in research output vs solar potential.
- Multi-objective studies often omit embodied energy, water use, and social acceptance.
- **Smart grid / demand response integration** inadequately addressed.
- Socio-economic, policy, and cost-benefit dimensions scarcely explored.
- Demonstrated **12–22%** electricity savings need validation under realistic user behaviour and climates.

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Highly relevant** — Cluster 3 and Table 1 document fuzzy-MPC [147], Q-learning RL for solar fields [148], DRL smart water heaters [88], and demand-aware scheduling; authors state most systems remain simulation-only, directly motivating your **DRL charge/discharge/bypass** controller on live sensors.

- **RG2 (No integrated PCM–AI–hardware prototype):** **Highly relevant** — Review maps PCM+ML (DNN [116], paraffin/honey wax ANN [121], U-tube PCM ML [72]) but notes fragmentation between materials, algorithms, and deployment; supports your **closed-loop RPi/ESP32 + PCM + AI** integrated prototype as a gap-filling contribution.

- **RG3 (Poor alignment with household demand patterns):** **Relevant** — Paper emphasises AI for demand prediction, user behaviour, and hot-water scheduling [26, 27]; cites Shirinbakhsh PCM+ demand interaction [139] and RF models using daily load volume [123]; aligns with your **demand-conditioned DRL reward** and Indian household profiles (Coimbatore, Kochi, Jaisalmer).

- **RG4 (Limited real-world experimental validation):** **Highly relevant** — Authors explicitly flag short lab studies vs long-term field trials; Terfai-style embedded validation is rare in AI–SWHS corpus; strengthens justification for your **multi-city field/bench evaluation** objective.

- **RG5 (No predictive optimization under climatic uncertainty):** **Highly relevant** — Calls for weather-forecast-driven MPC [104, 147], GraphCast/Pangu-Weather integration, and multimodal climate inputs; supports your **ERA5/NASA POWER/ISRO forecast → PCM selection + DRL** pipeline under variable irradiance.

---

## 13. Equations to Reuse or Adapt

**Publication growth trend (bibliometric):**
\[
y = 1.0554x - 2113.6,\quad R^2 = 0.776
\]
where \(x\) = year, \(y\) = annual publication count.

**Grey relational grade (your project already uses GRA — cross-cite Chen/Singh via this review’s optimisation cluster):**
\[
\xi_i = \frac{\Delta_{\min} + \zeta \Delta_{\max}}{\Delta_i + \zeta \Delta_{\max}}, \qquad
\gamma_i = \frac{1}{n}\sum_{k=1}^{n}\xi_i(k)
\]

**Standard ML error metrics cited across reviewed studies (for your benchmark table):**
\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, \qquad
MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

**Thermosiphon ANN target (Kalogirou & Panteliou, synthesised in review):**
- Predict annual useful solar energy \(Q_u\) from collector area \(A_c\), system configuration, and climate class; reported \(R^2 = 0.9993\) training, **0.9913** validation across **1.81–4.38 m²** systems — useful baseline for grey-box vs ANN comparison in your simulation environment.

**PCM TES ANN feature importance (Kanimozhi et al., via review):**
- Time dominates charge/discharge improvement (**>40%** contribution) — supports time-series state features \((T_w, T_p, f, \dot{m}, GHI)\) in your DRL state vector.

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **B. Singh et al., "Application of phase change materials in solar water heating systems — A comprehensive review,"** 2025 — direct PCM-SWH literature anchor for India-focused review table.
2. **A. Al-Mamun et al., "State-of-the-art in solar water heating (SWH) systems…,"** Sol. Energy, 2023 — baseline SWH technology review paired with this AI review.
3. **M. Liu et al., "The contribution of artificial intelligence to phase change materials in thermal energy storage…,"** 2025 — AI+PCM TES prediction-to-optimization pipeline.
4. **A. Terfai et al., ANN–MPC shallow pond experimental work,** 2025 — embedded real-time control validation benchmark (cited indirectly via your corpus; Odoi cites MPC/ANN control cluster).
5. **S. Uniyal et al., ML for U-tube PCM solar collectors** — nine-model comparison with ANN/SVR \(R^2\) up to **0.9540**; closest ML+PCM collector study in review.

---

## 15. Suggested Use in My IEEE Paper

- **Section I (Introduction):** Cite global SWH capacity **560 GW_th** (2023), market growth **$23.7B → $32.1B** (CAGR **5.2%**), and post-**2019** explosion of AI–SWHS publications (**34** in 2024) to motivate intelligent PCM-SWH research.

- **Section II (Literature Review):** One-line entry: *"Odoi-Yorke (2025) bibliometrically maps 245 AI–SWHS studies into five clusters (prediction, optimisation, control, TRNSYS-hybrid, deep learning), reporting India as the second-largest contributor (105 papers) while noting <6% African output and limited field validation of adaptive controllers."*

- **Section III (Methodology):** Borrow PRISMA-style literature screening logic for your survey section; adopt reported ML metric suite (\(R^2\), RMSE, MAPE, MAE) for controller and forecaster evaluation consistency with SWHS AI literature.

- **Section IV (Dataset & Setup):** Contrast your **ERA5 / NASA POWER / ISRO** climate pipeline against review’s finding that standardised irradiance datasets are missing; position India (**16%** global AI–SWHS share) as context for Coimbatore/Kochi/Jaisalmer case studies.

- **Section V (Results):** Benchmark against synthesised targets: DNN+PCM MAPE **<15%** [116], RF annual energy MAPE **2.94–5.86%** [123], MPC/RL electricity savings **12–22%** [88] (lab), and ANN thermosiphon \(R^2\) **0.9913** validation [29] for grey-box and DRL superiority claims.

---
