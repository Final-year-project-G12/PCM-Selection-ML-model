# Multimodal Learning Techniques for Time Series Forecasting in Renewable Energy Systems: A Comprehensive Survey

**Authors:** Majdi Mansouri, Khadija Attouri, Shady S. Refaat  
**Year:** 2025  
**Journal/Conference:** IEEE Access, Vol. 13, pp. 151970–151991 (article sequence)  
**DOI:** https://doi.org/10.1109/ACCESS.2025.3602914  
**IEEE Citation:** M. Mansouri, K. Attouri, and S. S. Refaat, "Multimodal learning techniques for time series forecasting in renewable energy systems: A comprehensive survey," IEEE Access, vol. 13, pp. 151970–151991, 2025, doi: 10.1109/ACCESS.2025.3602914.

---

## 1. One-Line Summary
This survey categorizes and compares **multimodal fusion strategies** (early, late, hybrid/attention, cross-modal, self-supervised) and **deep architectures** (CNN, LSTM/GRU, Transformer, VAE, GNN) for renewable energy time-series forecasting, while cataloguing benchmark datasets, metrics, deployment cases, and open challenges including alignment, missing modalities, and lack of standardized multimodal benchmarks.

---

## 2. Problem Being Solved
- Renewable generation (solar, wind, hybrid) is **stochastic and weather-driven**; single-modality models fail on non-stationarity, missing sensors, and site transfer (Section II-B).
- Heterogeneous data—**NWP**, **satellite imagery**, **SCADA/sensors**, **text/logs**, **grid data**—exist at mismatched spatial/temporal resolutions, making naive fusion unreliable (Sections III, VIII-A–B).
- Prior surveys cover either **single-modality forecasting** or high-level AI overviews, not a **technically grounded taxonomy** of multimodal fusion + deep models + benchmarks for renewables (Abstract, Table 1).
- Operational deployment needs **interpretability, uncertainty quantification, and scalable inference**, which black-box multimodal models often lack (Sections VIII-E, IX-D).

---

## 3. Key Contributions
1. **Comparative survey positioning (Table 1):** Contrasts recent renewable-forecasting surveys on domains, modalities, fusion techniques, deep models, and benchmark/metric coverage—claiming unique focus on **multimodal fusion + deep architectures** for solar/wind/hybrid.
2. **Modality taxonomy (Section III):** Numerical sensors (irradiance, wind, power), **NWP** (GHI, DNI, wind at hub height), **satellite/sky imagery** (GOES, Meteosat, Himawari, MODIS, Landsat), and **textual** SCADA/maintenance/weather bulletins with NLP pipelines.
3. **Fusion-strategy synthesis (Section IV):** Early (concatenation), late (modality-specific models + aggregation), hybrid/intermediate (attention, co-learning), cross-modal/co-attention, and self-supervised/contrastive/**multimodal VAE** approaches; **Table 6** compares reported RMSE/MAE/accuracy across fusion types (values in source table).
4. **Architecture review (Section V):** Engineered multimodal features → CNN (spatial), LSTM/GRU (temporal), **Transformer/cross-attention**, multimodal AE/VAE, and **GNN/GAT** for spatial sensor topology.
5. **Applications, metrics, datasets, challenges, future roadmap (Sections VI–IX):** Solar PV with irradiance + clouds + weather; wind with SCADA + NWP; horizons; hybrid solar–wind–battery; grid-aware forecasting; real deployments (Japan, Australia, China, Korea); gaps in **standardized multimodal benchmarks**, **federated learning**, **foundation models**, and **physics-informed multimodal fusion**.

---

## 4. Methodology
### 4a. System / Experiment Setup
N/A — **literature survey** (23 pages, **IEEE Access**, CC BY 4.0). No new experiment. Scope: **solar PV**, **wind farms**, **hybrid renewable + storage**, and **grid-aware** forecasting using multimodal inputs.

Representative **physical relationships** used to frame applications:
- **PV:** \(P = \eta A G\) with effective irradiance \(G\) modulated by clouds, aerosols, shading (Section VI-A).
- **Wind turbine power curve** \(P(v)\): zero below \(v_{cut-in}\), cubic region to \(P_{rated}\), rated plateau, zero above \(v_{cut-out}\) (Section VI-B).
- **Hybrid net power:** \(P_{net}(t) = P_{solar}(t) + P_{wind}(t) + P_{storage}(t)\); battery SoC update with \(\eta_{charge}\), \(\eta_{discharge}\) (Section VI-D).

### 4b. Mathematical Models & Equations
**NWP forecast error and ML bias correction:**

- \(\text{Forecast Error} = y_{true} - \hat{y}_{NWP}\)
- \(y_{corrected} = f_{ML}(\hat{y}_{NWP}, \text{auxiliary features})\) — RF, GBM, or DNN

**Cloud motion (satellite nowcasting):**

- \(I_{predicted}(t+\Delta t) = I(t) + \vec{v}_{cloud} \cdot \Delta t\)

**Text vectorization (TF-IDF):**

- \(\mathrm{TF\text{-}IDF}(t,d) = \mathrm{tf}(t,d) \cdot \log\dfrac{N}{\mathrm{df}(t)}\) — **(1)**

**Multimodal feature vector (traditional ML):**

- \(\mathbf{x} = [x^{(1)}, x^{(2)}, \ldots, x^{(M)}]^\top\); \(y = f(\mathbf{x}) + \varepsilon\)

**CNN convolution:**

- \(F_{i,j,k} = \sigma\left(\sum_{m,n,c} I_{i+m,j+n,c} \cdot K_{m,n,c,k} + b_k\right)\)

**LSTM gates:**

- \(f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)\), \(i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)\)
- \(c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c x_t + U_c h_{t-1} + b_c)\)
- \(h_t = o_t \odot \tanh(c_t)\)

**Transformer attention:**

- \(\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\dfrac{QK^\top}{\sqrt{d_k}}\right) V\)

**Multimodal VAE loss:**

- \(\mathcal{L}_{VAE} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \mathrm{KL}(q_\phi(z|x) \,\|\, p(z))\)

**GNN layer:**

- \(H^{(l+1)} = \sigma\left(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)\)

**Solar PV power (simplified):**

- \(P = \eta \cdot A \cdot G\) — Section VI-A

**Short-term forecast objective:**

- \(\min_{\hat{P}_{t+\tau}} \mathbb{E}\left[(P_{t+\tau} - \hat{P}_{t+\tau})^2\right]\), \(\tau \leq\) few hours

**Long-term decomposition:**

- \(P_t = T_t + S_t + R_t\) (trend, seasonal, residual)

**AC power flow (grid-aware excerpt):**

- \(P_i = \sum_{j=1}^{N} V_i V_j (G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})\)
- \(Q_i = \sum_{j=1}^{N} V_i V_j (G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})\)

**Forecasting metrics (Section VII-A):**

- \(\mathrm{RMSE} = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}\)
- \(\mathrm{MAE} = \dfrac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i|\)
- \(\mathrm{MAPE} = \dfrac{100\%}{N}\sum_{i=1}^{N}\left|\dfrac{y_i - \hat{y}_i}{y_i}\right|\)
- \(\mathrm{NRMSE} = \mathrm{RMSE}/(y_{max}-y_{min})\) or \(\mathrm{RMSE}/\bar{y}\)

### 4c. Algorithm / Control Method Steps
N/A as a single implemented pipeline — the survey **describes** workflows:

**Typical multimodal forecasting pipeline:**
1. Collect modalities (sensors, NWP grids, satellite/sky images, optional text).
2. Preprocess: calibration, resampling, **spatial/temporal alignment**, normalization \(X'=(X-\bar{X})/\sigma\).
3. Choose fusion: **early** (concatenate after encoders), **late** (separate models + average/stacking), **hybrid** (attention/gating in latent space), or **cross-modal attention**.
4. Train deep encoders (CNN/LSTM/Transformer/GNN) with loss **MSE/MAE/RMSE**; optional **self-supervised/contrastive** pretraining when labels scarce.
5. Evaluate with **RMSE, MAE, MAPE, NRMSE**, skill scores; deploy with compression (pruning, quantization, distillation) for edge/real-time use (Section VIII-C).

**Hybrid plant operation (cited direction, not implemented here):** **reinforcement learning** and **MPC** for storage dispatch using forecasts (Section VI-D).

### 4d. Data Sources & Dataset Details
| Source / dataset (surveyed) | Modalities | Notes |
|----------------------------|------------|--------|
| **NREL NSRDB** [119] | Solar radiation, satellite-derived | Cited in references; multimodal solar research |
| **SolarAnywhere** | Satellite + ground | Table 8 discussion |
| **GEFCom** (Global Energy Forecasting Competition) [123] | Energy + weather | Table 9 |
| **SolarDB** | Solar forecasting benchmark | Table 9 |
| **Pecan Street**, **RENES** | Hybrid home/grid + storage interactions | Table 8 |
| **MODIS**, **GOES**, **Meteosat MSG**, **Himawari-8/9**, **Landsat 8/9** | Satellite imagery | Section III-C |
| **NWP models** (e.g., **ECMWF**, **AROME** — cited in refs) | GHI, DNI, wind, temperature, humidity | Sections III-B, refs [48], [51] |
| **SCADA / smart meters** | Power, wind, irradiance, temperatures | Tables 2–3, deployment cases |

**Not used in this survey as primary sources:** ERA5, NASA POWER, ISRO Solar Calculator, Global Solar Atlas (India).

**Geographic deployment examples:** **Kyushu (Japan)**, **Western Australia**, **China**, **South Korea** (Section VI-F, Table 7).

### 4e. Validation Method
N/A as primary research — validation is **by synthesis** of published studies using **RMSE, MAE, MAPE, NRMSE, R², skill scores**. Example **deployed/cited** outcomes:

- **MODIS + NWP** hybrid vs single-modality: **13.2% RMSE improvement** [9] (Introduction).
- **Kyushu, Japan** CNN–LSTM ensemble (weather + calendar + power variables): **R² = 0.787**, **MAE = 1.936**, **RMSE = 2.630** [113] (Section VI-F).
- **Table 6 / Table 9:** Aggregated fusion-type and dataset-level metrics from literature (numeric cells in PDF tables not text-extracted).

---

## 5. PCM Details (if applicable)
N/A — this survey addresses **renewable power forecasting** (solar PV, wind, hybrid plants, grid) and does **not** study phase-change materials or solar hot water thermal storage.

---

## 6. AI / ML / Control Details (if applicable)
- **Algorithm (surveyed families):** Early/late/hybrid/**cross-modal attention** fusion; **CNN**, **LSTM/GRU**, **Transformer** (Informer, Temporal Fusion Transformer cited), **multimodal AE/VAE**, **GNN/GAT**; traditional **RF/SVM/MLP** on engineered features; **NLP** (TF-IDF, LDA, BERT-style embeddings); **RL/MPC** mentioned for hybrid storage dispatch (literature pointer only).
- **Input features / state space:** Irradiance (**GHI**, **DNI**), temperature, humidity, wind speed/direction, cloud cover, **satellite/sky images**, **NWP grids**, turbine SCADA (rotor speed, pitch, power), calendar features, load/grid states, optional **text** embeddings.
- **Output / action space:** **Power generation**, **irradiance**, **price** (Kyushu case), grid response variables — **forecasts**, not PCM tank control actions.
- **Model architecture:** Modality-specific encoders + fusion module; e.g., **CNN–LSTM** ensembles, **ConvLSTM**, transformer **cross-attention** \(Q,K,V\) across image and time-series branches.
- **Hyperparameters:** Not fixed (survey); discusses **learning rate**, hidden layers, **Adam**-class optimizers, attention sparsity, **pruning/quantization/distillation** for deployment.
- **Training data size:** Varies by cited study; emphasizes need for **large multimodal corpora** for foundation models (Section IX-A).
- **Hardware used for training:** **GPUs/clusters** noted as typical requirement; **edge computing** suggested to reduce centralized load (Section VIII-C).
- **Performance metrics (examples from cited work):**
  - **13.2%** lower RMSE (MODIS + NWP vs single-modality) [9]
  - Deployment: **R² = 0.787**, **MAE = 1.936**, **RMSE = 2.630** [113]
  - Metrics framework: **RMSE, MAE, MAPE, NRMSE**, skill scores (Section VII)

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** **NWP** outputs; **satellite** platforms (GOES **5–15 min** revisit; Meteosat **15 min**, **1–3 km**; Himawari **10 min**; MODIS; Landsat **30 m**); ground **pyranometer/irradiance sensors**; benchmarks **NREL**, **SolarAnywhere**, **GEFCom**, **SolarDB**, **Pecan Street**, **RENES** — **not** ERA5/NASA POWER/ISRO/Global Solar Atlas in body text.
- **Variables used:** **GHI**, **DNI**, temperature, humidity, wind speed/direction, cloud cover, precipitation, pressure, power output, satellite-derived cloud/albedo features.
- **Geographic scope:** Global literature; explicit deployment regions include **Japan, Australia, China, South Korea**; satellite sections reference **Americas, Europe/Africa/Middle East, East Asia/Oceania**.
- **Temporal resolution:** **Sub-second** turbine data to **hourly** weather stations; satellite **10–15 min** typical; **very-short-term solar nowcasting 0–30 min** [55]; short-term **seconds–hours**, long-term **days–months** (Section VI-C).
- **Time period covered:** Survey literature through **2025** (received Jul 2025, accepted Aug 2025).
- **Clear-sky index / derived metrics:** Not a dedicated survey topic; **NWP bias correction** and cloud-motion nowcasting discussed; **skill scores** mentioned for benchmarking.

---

## 8. Key Results & Numbers
*Survey paper — bullets cite quantitative claims and aggregated literature results stated in the text.*

- Hybrid **MODIS satellite + NWP** model: **13.2% RMSE improvement** vs single-modality models [9] (Introduction).
- **Kyushu, Japan** operational-style deployment (CNN–LSTM multimodal ensemble): **R² = 0.787**, **MAE = 1.936**, **RMSE = 2.630** [113] (Section VI-F).
- **GOES** satellite revisit: **5–15 min**; **Meteosat MSG**: **15 min** temporal, **1–3 km** spatial; **Himawari-8/9**: **10 min** revisit (Section III-C).
- **Landsat 8/9**: **30 m** spatial resolution for terrain/vegetation (Section III-C).
- **Very-short-term solar nowcasting** horizon: **0–30 min** [55] (Section III-C).
- Satellite imagery availability example: broad regions imaged every **15 min to 1 h** vs turbine sensors at **sub-second** intervals (Section VIII-A).
- **NWP** grids: resolutions from **few km to tens of km**, updates every **few hours** (Section VIII-A).
- Survey scope: **23** pages; compares fusion strategies across **solar, wind, and hybrid** systems (Abstract, Section VI).
- **Western Australia** case: LSTM on smart-meter **import/export, rooftop PV, consumption, temperature** — beats classical baselines on **seconds-to-minutes** horizons [114] (Section VI-F, qualitative superiority).
- Deployment summary captured in **Table 7** (4 regional case studies); benchmark inventory in **Tables 8–9** (NREL, GEFCom, SolarDB, etc.).
- Future work cites **foundation models**, **federated multimodal learning**, **few-shot transfer**, **physics-informed multimodal fusion** as open research axes (Section IX).

---

## 9. Baseline Comparison
- **Baseline method(s):** **Single-modality** forecasts (sensors-only or NWP-only); **early fusion** vs **late fusion** vs **hybrid/attention fusion** (Table 6); classical statistical/physics models vs **deep multimodal** pipelines; **regression/NWP raw** vs **ML bias-corrected NWP**.
- **Proposed method:** Not one method — survey concludes **hybrid and attention-based fusion** often outperform naive early/late fusion when cross-modal interactions matter; **late fusion** is more **modular/robust to missing modalities** but may underperform without adaptive weighting (Sections IV-F, X).
- **Improvement margin:** Literature example: **13.2%** RMSE reduction (multimodal vs unimodal) [9]; Kyushu multimodal **R² = 0.787** vs implicit baselines in source study [113].
- **Conditions of comparison:** Varies by cited paper (solar vs wind, horizon, geography); survey stresses lack of **standardized multimodal benchmarks** for fair cross-study comparison (Sections VIII-D, VIII-F).

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **survey only**; no project-built test rig. Discusses operational data acquisition:

- **Sensors:** Pyranometers/irradiance, wind speed/direction, turbine SCADA, smart meters, temperature/humidity.
- **Communication:** **Modbus**, **IEC 61850** (Section III-A).
- **Compute:** Training on **GPU clusters**; inference via **model compression** and **edge/distributed** processing for real-time grid use (Section VIII-C).
- **Test environment:** Cited **field deployments** (Japan, Australia, China, Korea) plus simulation/NWP pipelines — **not** PCM-SWH bench tests.
- **Test duration:** Not applicable at survey level; horizons discussed from **0–30 min** nowcasting to **2035** long-term policy projections (Korea case [115]).

---

## 11. Limitations Acknowledged by Authors
- **Spatiotemporal resolution mismatch** across satellite, NWP, and sensor streams causes alignment artifacts and reduced accuracy (Section VIII-A).
- **Asynchronous updates and missing modalities** degrade models trained on complete inputs (Section VIII-B).
- **High computational cost** of deep multimodal models limits real-time deployment without compression/distillation (Section VIII-C).
- **Lack of standardized multimodal benchmark datasets and evaluation protocols** hinders reproducibility and fair comparison (Sections VIII-D, X).
- **Black-box models** limit **interpretability and operator trust**; need SHAP/LIME, attention visualization, uncertainty quantification (Sections VIII-E, IX-D).
- Text modality challenges: **ambiguous terminology**, **low labels**, **multilingual logs**, **timestamp misalignment**, **privacy restrictions** (Section III-D).
- Real deployments still face **data quality, interpretability, and scalability** barriers (Section VI-F).

---

## 12. Direct Relevance to My Project

- **RG1 (No real-time adaptive control):** **Not Relevant (as implemented).** Survey targets **forecasting**, not closed-loop PCM-SWH control; **RL/MPC** appear only as cited tools for **battery/hybrid dispatch**, not domestic hot water valves or charging logic.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Not Relevant.** No PCM tank, **DS18B20**, or embedded SWH prototype — focus is grid-scale **PV/wind** multimodal prediction.
- **RG3 (Poor alignment with household demand patterns):** **Not Relevant.** Does not model **DHW draw profiles** or end-use scheduling; smart-meter cases address **grid import/export**, not morning/evening hot water peaks.
- **RG4 (Limited real-world experimental validation):** **Partially relevant.** Highlights **operational multimodal deployments** (e.g., Japan **R² = 0.787**) but **not** PCM-SWH field trials; reinforces that AI renewables work is moving to operations while **thermal PCM-SWH** validation remains a separate gap.
- **RG5 (No predictive optimization under climatic uncertainty):** **Highly relevant.** Core thesis: fuse **NWP + satellite + sensors** for robust forecasts under weather variability — directly supports your **ERA5/NASA POWER + pyranometer XGBoost** layer and using forecasts as **PPO state inputs** for climate-adaptive PCM charging; paper notes **adaptive fusion under uncertainty** as future direction (Abstract) but does not use Indian cities or ERA5 by name.

---

## 13. Equations to Reuse or Adapt

| Equation | What It Models | Maps To (My Project) |
|----------|---------------|----------------------|
| \(P = \eta A G\) | PV power from irradiance | Link **pyranometer G** to available solar gain for collector/PCM charging |
| \(y_{corrected} = f_{ML}(\hat{y}_{NWP}, \text{aux})\) | NWP bias correction | **XGBoost** correction of ERA5/NASA POWER vs local Coimbatore/Jaisalmer/Kochi measurements |
| \(X'=(X-\bar{X})/\sigma\) | Feature normalization | ANN/XGBoost/TFLite input pipeline |
| \(\mathrm{Attention}(Q,K,V)\) | Cross-modal fusion | Fuse **irradiance time series** + optional sky-camera/satellite embeddings |
| \(\mathrm{RMSE},\ \mathrm{MAE},\ \mathrm{MAPE}\) | Forecast skill metrics | Report Phase 1b forecast accuracy before RL |
| \(P_{net}=P_{solar}+P_{wind}+P_{storage}\) | Hybrid plant balance | Analogous structuring if adding **battery** later; PCM as thermal storage not covered |
| \(I_{pred}(t+\Delta t)=I(t)+\vec{v}_{cloud}\Delta t\) | Cloud motion nowcasting | Optional **0–30 min** horizon layer above hourly ERA5 |

---

## 14. Citations This Paper Uses (That I Should Also Cite)

1. **T. Jing et al., "SolarFusion-Net: Enhanced solar irradiance forecasting via automated multi-modal feature selection and cross-modal fusion," IEEE Trans. Sustain. Energy, 2025 [14]** — Relevant because: Direct **multimodal solar irradiance** forecasting architecture aligned with your GHI-driven PCM control.
2. **K. Wang et al., "A robust photovoltaic power forecasting method based on multimodal learning using satellite images and time series," IEEE Trans. Sustain. Energy, 2025 [13]** — Relevant because: Fuses **satellite + time series** for PV — analogous to pyranometer + satellite/ERA5 fusion.
3. **J. Qin et al., "Enhancing solar PV output forecast by integrating ground and satellite observations with deep learning," Renew. Sustain. Energy Rev., 2022 [6]** — Relevant because: **Ground + satellite** solar forecasting precedent for Indian site calibration.
4. **J. Heo et al., "Multi-channel convolutional neural network for integration of meteorological and geographical features in solar power forecasting," Appl. Energy, 2021 [9]** — Relevant because: Source of cited **13.2% RMSE** gain from multimodal meteorological/geographical fusion.
5. **Y. Dong, "Robust dynamic modeling and optimal scheduling of wind-solar-storage systems via multi-modal data fusion under uncertainty," Proc. NESP, 2025 [125]** — Relevant because: **Solar–storage + uncertainty + multimodal fusion** closest thematic match to climate-adaptive thermal/electrical storage control.

---

## 15. Suggested Use in My IEEE Paper

| Section | What to Use | Exact Claim or Stat |
|---------|-------------|---------------------|
| I. Introduction | Multimodal forecasting gap for renewables | "Mansouri et al. (2025) note heterogeneous NWP, satellite, and sensor streams remain difficult to align, with few standardized multimodal benchmarks for operational forecasting." |
| II. Literature Review | Survey row: fusion taxonomy | Method: early/late/hybrid/cross-attention multimodal fusion; Key insight: hybrid fusion often beats naive concatenation; MODIS+NWP example **13.2% RMSE** gain [9] |
| III. Methodology | Metrics + normalization | Use **RMSE/MAE** for XGBoost irradiance; normalize inputs with \(X'=(X-\bar{X})/\sigma\) |
| IV. Dataset & Setup | Modalities list | Fuse **GHI/DNI**, \(T_{amb}\), humidity/wind from **ERA5/NASA POWER** with local **pyranometer** as multimodal analog to survey sensors |
| V. Results | Literature benchmark | Cite multimodal deployment **R² = 0.787**, **MAE = 1.936**, **RMSE = 2.630** [113] as grid-scale forecast benchmark context (not PCM-SWH) |
