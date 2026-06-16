# A Systematic Review of Solar Irradiance Forecasting Across Time Horizons Using Physical, Satellite, and AI-Based Methods

**Authors:** Mohammad Hasan Ghodusinejad, Nasrin Rashvand, Fatemeh Salmanpour, Shaghayegh Danehkar, Hossein Yousefi  
**Year:** 2026 (Solar Compass 17, 2026; received 2025)  
**Journal/Conference:** Solar Compass, Vol. 17, Article 100154  
**DOI/Link:** https://doi.org/10.1016/j.solcom.2025.100154  
**IEEE Citation:** M. H. Ghodusinejad et al., "A systematic review of solar irradiance forecasting across time horizons using physical, satellite, and AI-based methods," Sol. Compass, vol. 17, p. 100154, 2026, doi: 10.1016/j.solcom.2025.100154.

---

## 1. One-Line Summary
This systematic review taxonomizes solar irradiance forecasting by temporal horizon (intra-hour to multi-day), input data type (NWP, satellite, ASI), and model architecture (physical, statistical, ML/DL, hybrid), reporting benchmark errors such as NOAA GHI RMSE **107–125 W/m²** (rRMSE **21–25%**), TAPM **45 km** lowest RMSE resolution, NWP+TSI **+21%** accuracy gain, and XGBoost regional forecasts with **R² = 0.9993** and MAPE **0.0119**, while advocating physics-informed deep learning for operational solar applications.

---

## 2. Problem Being Solved
- Solar power integration requires accurate GHI/DNI forecasts across multiple horizons to stabilize grids and optimize dispatch.
- Irradiance variability from clouds, aerosols, humidity, terrain, and ozone creates large prediction errors that increase costs and curtailment.
- Physical NWP models (WRF, GFS, ECMWF) provide physics consistency but under-resolve clouds and aerosols at short horizons.
- Pure statistical/ML models capture local nonlinearities but may fail to generalize across climates without physical constraints.
- No unified taxonomy linked forecast horizon, data modality, and architecture to expected accuracy outcomes for practitioners.

---

## 3. Key Contributions
1. **Horizon taxonomy:** very-short/intra-hour (≤60 min), short-term (hours), medium-term (days), long-term — mapped to use cases (nowcasting, dispatch, planning).
2. **Atmospheric driver review:** aerosols (AOD550), cloud cover (TSI, optical flow), humidity/sky transparency, wind, ozone, terrain effects on WRF bias.
3. **Model family comparison:** physical (NWP, satellite-to-irradiance), statistical, ML/DL (kNN, RF, GBM, MLP, LSTM), and **hybrid physics+AI** post-processing.
4. **Quantitative error synthesis:** Tables for satellite GHI models (MBE, MAE, RMSE, rRMSE, Xcor); SVR best for **1–15 min** TSI+NWP; multimodel NWP improvements.
5. **Future direction:** physics-informed deep learning, multimodal fusion (satellite + NWP + ground), adaptive real-time forecasting for variable atmospheres.

---

## 4. Methodology
- **Type:** Narrative systematic review (not PRISMA-quantified paper count); integrates physical, satellite, and AI literature.
- **Scope:** GHI/DNI forecasting for PV and CSP; ASI (All-Sky Imager), geostationary satellite, reanalysis/NWP inputs.
- **Structure:** Section 3 atmospheric factors → Section 4 prediction models (physical §4.1, statistical §4.2, AI §4.3, hybrid pros/cons §4.4) → conclusions.
- **Validation approach:** Compares published RMSE/MAE/rRMSE/R²/MAPE across cited primary studies; no new experiments.

---

## 5. PCM Details (if applicable)
N/A — review focuses on **solar irradiance forecasting** for grid/PV/CSP, not PCM materials.  
**Indirect link:** Accurate GHI/DNI forecasts feed your **Objective 1** PCM classifier and **Objective 2** DRL state (forecasted solar input, charge/discharge timing).

---

## 6. AI / ML / Control Details (if applicable)
| Method class | Examples cited | Reported performance |
|--------------|----------------|----------------------|
| Classical ML | kNN, RF, GBM, MLP/ANN, SVR | SVR best for **1–15 min** nowcasts with TSI+NWP [22] |
| Gradient boosting | XGBoost (Turkey Mediterranean) | **R² = 0.9993**, MAPE **0.0119** [106] |
| Deep learning | LSTM (ASI), CNN-LSTM hybrids | Promising for nonlinear cloud dynamics [10, 88] |
| Hybrid | NWP + ML post-processing, physics-informed DL | **+21%** vs base TSI model when NWP integrated [26]; reduces WRF cloud bias via EnKF CWP assimilation [24] |
| Persistence / LR | Baselines for ultra-short horizon | Outperformed by SVR in cloud-tracking pipeline |

**Input features (typical):** GHI, clear-sky index \(K_c\), solar zenith angle, cloud fraction/type, wind, RH, aerosol AOD, satellite radiance, NWP fields.

---

## 7. Solar / Climate Data Details (if applicable)
- **Data sources:** NWP (WRF, GFS, ECMWF, TAPM, MM5), geostationary satellites, **Total Sky Imager (TSI)**, ground pyranometers, AOD from chemistry transport models (EURAD), SolarAnywhere benchmarks.
- **Variables:** GHI, DNI, DHI, AOD550, cloud water path (CWP), fractional sky cover, effective transfer ratio, humidity profiles, wind, ozone.
- **Geographic examples:** San Diego CA (satellite streamlines), California coast Sc clouds, Sicily wind+solar PCA [29], Turkey Mediterranean 8 cities [106], Tibetan Plateau [28], central Mediterranean dust events.
- **Temporal resolution:** TSI **5 min** native; extended to **15 min** with NWP [22]; intra-hour to multi-day horizons classified explicitly.
- **India relevance:** Review does not focus on ISRO/ERA5/NASA POWER directly; your project should map cited hybrid NWP+ML pattern to **ERA5 reanalysis + NASA POWER + ISRO Solar Calculator** for Coimbatore, Kochi, Jaisalmer.

---

## 8. Key Results & Numbers
- Global PV capacity growth: **>900 GW** added over ten years; **+180 GW** in 2021 alone [4].
- **NOAA GHI Model 2** (San Diego): mean measured **493.49 W/m²**, modeled **516.78 W/m²**, MBE **−23.29 W/m²** (**−4.72%**), MAE **61.72 W/m²** (**12.51%**), RMSE **107.41 W/m²** (**21.77%** rRMSE), Xcor **0.95** (Table 1).
- **NOAA GHI Model 1:** RMSE **124.53 W/m²** (**25.24%** rRMSE), Xcor **0.934**.
- **SUNY GHI** (outliers removed): RMSE **130.52 W/m²** (**31.38%** rRMSE), Xcor **0.932**.
- **TSI + NWP integration:** average **+21%** improvement vs base short-term model [26].
- **SVR** outperforms PM and LR for **1–15 min** forecasts in Xu et al. cloud-tracking study [22] (Fig. 2 RMSE comparison).
- **TAPM NWP:** **45 km** resolution yielded **lowest RMSE** vs finer scales [39].
- **WRF terrain correction:** horizontal surface RMSE reduced ~**20%** (~**25 W/m²**) winter/autumn; tilted surface best at **9 arcsec** with RMSE **45%** (**57 W/m²**) [31].
- **Aerosol optical depth > 0.1:** model errors up to **100 W/m²** [30].
- **XGBoost** (8 Turkish cities): **R² = 0.9993**, MAPE **0.0119** — best among statistical vs ML comparison [106].
- **Erbs + Liu-Jordan** hybrid: most accurate among 12 transfer models across sunny/cloudy/overcast/rainy days [105].

---

## 9. Baseline Comparison
| Approach | Horizon | vs Alternative | Outcome |
|----------|---------|----------------|---------|
| SVR + TSI + NWP | 1–15 min | Persistence, linear regression | SVR lowest RMSE [22] |
| NWP-augmented TSI | Short-term | Base TSI only | **+21%** accuracy [26] |
| ML (XGBoost) | Regional daily/hourly | Statistical models | **R² 0.9993** vs weaker statistical fits [106] |
| NOAA GHI satellite model | Day-ahead | SUNY GHI | RMSE **107 vs 213 W/m²** (outlier case) |
| WRF + EnKF CWP assimilation | Short-term | Raw WRF | Improved mid-latitude GHI [24] |
| Hybrid physics + AI | Multi-horizon | Pure NWP or pure ML | Recommended compromise §4.4 |
| Erbs + Liu-Jordan | All-weather | 11 other transfer models | Best RMSE/MAE across 4 weather classes [105] |

---

## 10. Hardware / Experimental Setup (if applicable)
N/A — **review paper.** Cited field setups include ground pyranometers, **Total Sky Imagers**, geostationary satellite receivers, and NWP assimilation systems — no unified experimental rig.

---

## 11. Limitations Acknowledged by Authors
- Cloud cover and aerosols remain dominant error sources for NWP at short horizons.
- Physical models struggle with stratocumulus (Sc) thickness/presence in coastal zones [20].
- ML models need representative training data; poor cross-climate generalization risk.
- Integration of real-time adaptive data streams still immature.
- Need higher-resolution hybrid models and better multimodal fusion (satellite + NWP + ground).
- Complex terrain and dust events still under-modeled in many operational chains.

---

## 12. Direct Relevance to My Project
- **RG1 (No real-time adaptive control):** **Indirect** — forecasts enable proactive control; review supports forecast-driven MPC/DRL policies.
- **RG2 (No integrated PCM–AI–hardware prototype):** **Indirect** — defines irradiance input pipeline (pyranometer + forecast API) for embedded controller.
- **RG3 (Poor alignment with household demand patterns):** **Partial** — multi-horizon taxonomy helps align morning/evening demand with day-ahead and intra-hour GHI forecasts.
- **RG4 (Limited real-world experimental validation):** **Relevant** — cites ground validation benchmarks (RMSE in W/m²) you can replicate with pyranometer vs forecast at Indian sites.
- **RG5 (No predictive optimization under climatic uncertainty):** **Highly relevant** — core paper for **Objective 1 & 2**; justifies ERA5/NASA POWER/ISRO forecast features and hybrid NWP+ML approach for climate-adaptive PCM+DRL under cloud/aerosol uncertainty.

---

## 13. Equations to Reuse or Adapt
**Error metrics (review nomenclature):**
\[
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}, \quad
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]
\[
\mathrm{rRMSE} = \frac{\mathrm{RMSE}}{\bar{y}}\times 100\%, \quad R^2 = 1 - \frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

**Clear-sky index (used in cloud-ML studies):**
\[
K_c = \frac{GHI_{measured}}{GHI_{clear\,sky}}
\]

**Forecast horizon classes (adopt in evaluation protocol):**
- Very-short: \(t \leq 60\) min (nowcasting for valve/pump control)
- Short: hours (same-day PCM charge planning)
- Medium: 1–3 days (PCM selection / pre-charge strategy)

---

## 14. Citations This Paper Uses (That I Should Also Cite)
1. **D.S. Kumar et al., solar irradiance resource and forecasting review, *IET Renew. Power Gener.*, 2020** — complementary forecasting survey.
2. **J. Xu et al., TSI + NWP multi-layer cloud tracking, 2015** — ultra-short horizon benchmark.
3. **L. Nonnenmacher & C.F. Coimbra, streamline satellite forecasting, *Sol. Energy*, 2014** — satellite optical-flow GHI method.
4. **A. Mellit, ML/DL for PV output forecasting overview, 2021** — links irradiance forecast to power prediction.
5. **Mansouri et al., multimodal renewable forecasting survey, 2025** — aligns with your multimodal climate+sensor fusion agenda.

---

## 15. Suggested Use in My IEEE Paper
- **Section I (Introduction):** Cite PV capacity growth and irradiance uncertainty as barrier to optimal PCM-SWH dispatch.
- **Section II (Literature Review):** Horizon taxonomy table; hybrid NWP+ML as state-of-art for **RG5**.
- **Section III (Methodology):** Adopt RMSE/MAE/rRMSE metrics for XGBoost irradiance model; use \(K_c\) and GHI forecast horizons as DRL state inputs.
- **Section IV (Dataset & Setup):** Position **ERA5/NASA POWER/ISRO** as India-specific analogues to WRF/satellite pipelines reviewed; target beating **~22% rRMSE** day-ahead satellite benchmarks at your sites.
- **Section V (Results):** Report forecast accuracy vs cited baselines (e.g., XGBoost **R² 0.9993** regional study as aspirational upper bound with caveat on climate transfer).

---
