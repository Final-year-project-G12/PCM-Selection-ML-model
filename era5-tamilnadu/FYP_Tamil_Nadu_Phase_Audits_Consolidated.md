#### **FINAL YEAR PROJECT** 

#### **Tamil Nadu ERA5 → PCM Selection Pipeline** 

# Consolidated Phase 1–8 Audit Documentation (v3.2) 

Source-preserving compilation of the Tamil Nadu audit files, verified and updated against the pipeline's actual current code and on-disk outputs (this pass, v3.2). Two Phase 7 physics-solver bugs were found and fixed during this verification (see the Phase 7/8 section and the new addendum at the end of this document); everything else was checked against the live scripts and confirmed to match what is claimed below, with corrections applied inline wherever a claimed number had gone stale relative to the current run (notably the Phase 4 cluster point counts). Nothing is silently reconciled — every correction made in this pass is called out explicitly, either inline or in the addendum. 

## **Included Source Files** 

- 00_MASTER_OVERVIEW(1).md 

- 01_PROJECT_CONTEXT(1).md 

- 02_DATA_SOURCES_AND_VARIABLES(1).md 

- 03_PHASE_1_AUDIT(1).md 

- 04_PHASE_2_AUDIT(1).md 

- 05_PHASE_3_AUDIT(1).md 

- 06_PHASE_4_AUDIT(1).md 

- 07_PHASE_5_AUDIT(1).md 

- 08_PHASE_6_AUDIT(1).md 

- 19_PHASE_7_8_AUDIT.md 

## **1. 00_MASTER_OVERVIEW(1).md** 

Source: 00_MASTER_OVERVIEW(1).md 

## **00 — Master Overview: ERA5 Tamil Nadu Climate → PCM Selection Pipeline** 

### **Project Objective** 

Final-year B.Tech CSE project: 

"Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating" 

Objective 1 builds a climate-region-aware PCM recommendation framework: turning 10 years of reanalysis climate data into population-weighted climate regimes, deriving PCM performance targets per regime, and ranking candidate phase-change materials against those targets with an auditable, multi-method, uncertainty-aware pipeline. 

Governing document: Objective1_PCM_Climate_Framework_Plan_v3.docx ("the framework doc"), version 3.0. The Tamil Nadu pipeline has been fully implemented from Phase 1 through Phase 8. Critical v3.0 bugs were corrected in v3.1 (August 2026). 

### **What the ERA5 Pipeline Achieves** 

1. Population-Weighted Sampling: Samples Tamil Nadu at 133 population-weighted points (representing 87.5% of the state's population) to ensure findings are representative of where domestic demand actually resides. 

2. Double-Source Validation: Pulls ERA5 reanalysis and NASA POWER satellite/model data for the same coordinates and times, validating one against the other. 

3. Two-Tier Climate Signature: Redefines 10 years of hourly/daily data into instantaneous sun-event statistics (Tier 1) and true daily-integral indices (Tier 2). 

4. Climate Regimes (Level A & B): Clusters points into spatial climate regimes (Level A) using Gaussian Mixture Models (GMM) and performs seasonal sensitivity analysis (Level B). 

5. PCM Feasibility & Screening: Filters the current 62-candidate database (55 manufacturer-derived + 7 literature) against physical, corrosion, and safety constraints. The feasibility CSV retains a full percandidate audit; current runs retain 9-15 candidates per cluster. 

6. Multi-Criteria Decision Making (MCDM): Ranks feasibility survivors using four independent methods (TOPSIS, GRA, PROMETHEE II, VIKOR) with Monte Carlo uncertainty propagation. 

7. Grey-Box Physics Validation: Solves a lumped-enthalpy tank simulation using backward Euler, driven by the real 10-year daily weather of each regime's medoid point, evaluating Spearman rank concordance. 

8. Recommendation Cards: Generates markdown summary cards for each climate regime. 

### **Complete Pipeline Map** 

##### Phase 1 — DATA COLLECTION 

00a_build_population_grid.py  → population_grid_points.csv (133 pts, 87.5% pop coverage) 00b_build_suntimes.py         → suntimes.csv (1,457,547 rows: 133 pts × 3653 days × 3 events) 01_download_era5_tamilnadu.py → data/raw/era5/points/*.nc (instant + accum NetCDF) 01b_download_nasapower.py     → data/raw/nasapower/*.json (1330 files, full hourly cache) 00_unzip_accum.py             → (fixes CDS zip-disguised-as-.nc quirk) 

- ↓ 

Phase 2 — PREPROCESSING & CROSS-SOURCE VALIDATION 

02_combine_tamilnadu.py       → climate_tamilnadu_points.csv (accum_to_flux, v3.1) 02b_build_daily_aggregates.py → daily_aggregates_tamilnadu.csv (POWER-only daily integrals) 03_plots_raw.py               → raw diagnostic plots & C_era5_vs_power_stats.csv 03b_agreement_analysis.py     → era5_power_agreement_tamilnadu.csv, bias decision (NEW v3.1) 03b_interactive_raw_qa.py     → interactive Plotly/Folium HTML maps/plots 04_preprocess_tamilnadu.py    → tamilnadu_cleaned_physical.csv (13-step QC + Step 2b QM) 04c_postprocess_plots.py      → post-cleaning QA plots ↓ 

Phase 3 — CLIMATE SIGNATURE CONSTRUCTION 04b_climate_signature.py      → climate_signature_tamilnadu.csv (300 L/day draw, SHARE_PCM=0.5) 04d_signature_interactive.py  → interactive signature exploration maps ↓ 

Phase 4 — CLIMATE REGIME CLUSTERING 

05_cluster_tamilnadu.py       → cluster_assignments (K_FINAL=5, covariance_type=diag) 05b_cluster_interactive.py    → interactive GMM cluster map 11_level_b_seasonal_analysis.py → level_b_seasonal_topk.csv, level_b_seasonal_summary.md 

↓ 

Phase 5 — FEASIBILITY FILTERING 

→ 07_feasibility_filter.py feasibility_survivors_by_cluster.csv (8 Table-12 filters) 

↓ 

Phase 6 — MULTI-CRITERIA RANKING ENGINE 

08_mcdm_ranking.py            → mcdm_topk_by_cluster.csv, monte_carlo_stability.csv ↓ 

Phase 7 — PHYSICS-BASED VALIDATION 

10_physics_validation.py      → physics_validation_results.csv (UA_TANK=2.0 W/K, v3.1) 

↓ 

Phase 8 — RECOMMENDATION CARDS 

09_recommendation_cards.py    → recommendation_cards.md 

**Phase 1–8 Status and Headline Findings** 

|Phase|Script(s)|Status|Headline Finding|
|---|---|---|---|
|1 — Data Collection|`00a`, `00b`, `01`,<br>`01b`,<br>`00_unzip_accum`|**COMPLETE**<br>|133 points, 240 NetCDF<br>fles, 1330 NASA POWER<br>JSON fles.|
|2 — Preprocessing & QA|`02`, `02b`, `03`,<br>`03b`, `04`, `04c`|**COMPLETE (v3.1 fxes<br>applied)**|Deaccumulation<br>replaced with<br>`accum_to_fux()`. Per-<br>season quantile<br>mapping in Step 2b. Re-<br>run required for new<br>outputs.|
|3 — Climate Signature|`04b`, `04d`|**COMPLETE**<br>|300 L/day draw with<br>`SHARE_PCM=0.5`;<br>current generated<br>cluster targets are<br>approximately 301-326<br>kJ/kg.|
|4 — GMM Clustering|`05`, `05b`, `11`|**COMPLETE (v3.1 fxes<br>applied)**|K=5 regimes,<br>`covariance_type="diag"<br>`. Level B seasonal re-<br>rank uses corrected|



||||draw volume.|
|---|---|---|---|
|5 — Feasibility|`06`, `07`|**COMPLETE**|62 PCM records are<br>audited per cluster;<br>current pass counts are<br>9-15 and varybycluster.|
|6 — MCDM Ranking|`08`|**COMPLETE**<br>|4-method Borda + 5000-<br>draw Monte Carlo.|
|7 — Physics Validation|`10`|**COMPLETE (v3.1 fxes<br>applied)**|Tank ambient heat loss<br>active<br>(`UA_TANK_W_K=2.0`).<br>Re-run for updated<br>Spearmanρ.|
|8 — Rec Cards|`09`|**COMPLETE**|Aggregates Phases 4–7<br>into<br>`recommendation_card<br>s.md`.|



### **Corrected Issues (v3.1 — August 2026)** 

All five critical bugs from the v3.0 audit are fixed in source code. See 20_IMPLEMENTATION_ISSUES.md for details. 

9. Deaccumulation → `accum_to_flux()` in `02_combine_tamilnadu.py` 

10. Quantile mapping → Step 2b in `04_preprocess_tamilnadu.py` + `03b_agreement_analysis.py` 

11. 1000× flow rate → 300 L/day in `04b_climate_signature.py` and `11_level_b_seasonal_analysis.py` 

12. GMM overfitting → `covariance_type="diag"` in `05_cluster_tamilnadu.py` 

13. Tank heat loss → `UA_TANK_W_K=2.0` in `10_physics_validation.py` 

### **Still Open** 

See 22_FINAL_READINESS_REPORT.md: PCM database expansion, external cluster validation, elevation proxy, monsoon precipitation download, full Level-B GMM. 

### **Plot Documentation** 

See 23_PLOTS_GUIDE.md for the interpretation and exact location of plots produced by the raw QA, preprocessing, climate-signature, clustering, comprehensive, Objective 1, and comparison scripts. 

### **Literature Support** 

|Pipeline Component|KeyReference|Source File|
|---|---|---|
|Populationgrid|GADM + WorldPop|`03_PHASE_1_AUDIT.md`|
|Solargeometry|Reda & Andreas(2004)SPA|`12_SOLAR_GEOMETRY.md`|
|Cross-source validation|Ghodusinejad et al. (2026)|`sources/<br>Ghodusinejad2026SolarIrradianc<br>eForecasting_summary.md`|
|Climate signature / sizing|Avargani et al. (2021), Singh et al.<br>(2025)|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|MCDM stack|Chen et al. (2025) Taguchi+GRA|`sources/<br>Chen2025TaguchiGRA_PCM_Nan<br>ofuid_SWH_summary.md`|
|Physics validation|Barqawi (2025)|`sources/<br>Barqawi2025DynamicSimulation|



|||PCM_SWH_summary.md`|
|---|---|---|
|Full matrix|—|`17_LITERATURE_MAPPING.md`|



## **2. 01_PROJECT_CONTEXT(1).md** 

Source: 01_PROJECT_CONTEXT(1).md 

## **01 — Project Context** 

### **Identity** 

"OBJECTIVE 1 — IMPLEMENTATION PLAN," Climate-Region-Aware PCM Recommendation Framework, Version 3.0, Group 12, B.Tech CSE Final Year, Amrita School of Engineering. Governing document: Objective1_PCM_Climate_Framework_Plan_v3.docx. 

### **Focus: Tamil Nadu** 

Section 1.3 names Tamil Nadu as the coastal tropical archetype. Tamil Nadu exhibits: 

14. A coastal tropical belt with high temperatures and high relative humidity. 

15. An interior semi-arid dry zone (e.g., Coimbatore/Tiruppur plains). 

16. A high-relief montane climate (e.g., Nilgiris hills: Ooty/Coonoor), which represents a cold temperate microclimate. 

17. An out-of-phase monsoon cycle (heavy rain during the North-East monsoon in Oct-Dec, whereas most of India receives rain during the South-West monsoon in Jun-Sep). 

This climatic diversity makes Tamil Nadu an ideal candidate for testing climate-adaptive PCM selection. Unlike Rajasthan (which is mostly arid/semi-arid), Tamil Nadu has high spatial humidity gradients and a distinct seasonal cycle. 

### **Scope and Deliverables (D1–D8)** 

Objective 1 covers the entire recommendation pipeline: 

- D1 (Validated Climate Dataset): ERA5 vs NASA POWER combined. 

- D2 (Climate Signature): Tier-1 sun-event and Tier-2 daily-integral features. 

- D3 (Climate Regimes): GMM-discovered spatial and seasonal clusters. 

- D4 (Feasibility Pool): Physically screened PCM database. 

- D5 (MCDM Rankings): Multi-method consensus ranks. 

- D6 (Physics-Validated Rankings): Simulation-verified performance. 

- D7 (Recommendation Cards): Regime summary cards. 

- D8 (Methodology Report): Thesis-ready documentation. 

### **Novelty Positions (N1–N6)** 

18. N1: Discovered regimes (GMM) rather than arbitrary administrative/geographical zones. 

19. N2: Two-tier signature (sun-event + daily-integral) rather than single monthly temperatures. 

20. N3: Corrected 42–70°C SWH-specific melting band rather than the common 18–28°C building-comfort band. 

21. N4: Four-method MCDM consensus reporting (TOPSIS, GRA, PROMETHEE II, VIKOR) rather than a single method. 

22. N5: Falsifiable physics-based validation rather than self-referential MCDM ranks. 

23. N6: Population-weighted sampling rather than uniform spatial grids. 

### **Literature Support** 

|Topic|Reference|Source|
|---|---|---|
|Tamil Nadu climate diversity|Singh et al. (2025) — SWH<br>regional context|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|PCM-SWH system scope|Al-Mamun (2023) state of art|`sources/<br>AlMamun2023SWH_StateOfArt_s<br>ummary.md`|
|AI for SWH optimization|Odoi & Yorke (2025)|`sources/<br>OdoiYorke2025AI_SWH_Review_s<br>ummary.md`|
|Framework deliverables D1–D8|Objective1_PCM_Climate_Frame<br>work_Plan_v3.docx|—|



## **3. 02_DATA_SOURCES_AND_VARIABLES(1).md** 

Source: 02_DATA_SOURCES_AND_VARIABLES(1).md 

## **02 — Data Sources and Variables** 

### **Variable Mapping Table** 

The following variables are collected, processed, and validated across the pipeline: 

|Variable|ERA5 Variable<br>Name|Original Unit|Stored Unit|Transformation|Reason /<br>Physical<br>Meaning|
|---|---|---|---|---|---|
|**GHI**|`ssrd`|J/m²<br>(accumulated)|W/m²|`(accum_to_fu<br>x(ssrd) / 3600)`|Global<br>Horizontal<br>Irradiance<br>(v3.1: stateless<br>clip,NOT dif)|
|**DNI**|`msdwswrf`|W/m² (mean<br>rate)|W/m²|`.clip(0)` (no<br>division)|Direct Normal<br>Irradiance<br>(direct solar<br>beam)|
|**DHI**|Derived|W/m²|W/m²|`(GHI - DNI *<br>cos(SZA))`|Difuse<br>Horizontal<br>Irradiance<br>(scattered solar<br>input)|
|**T_amb**|`t2m`|K|°C|`kelvin -<br>273.15`|Ambient air<br>temperature at<br>2 m|
|**T_dew**|`d2m`|K|°C|`kelvin -<br>273.15`|Dewpoint<br>temperature at<br>2 m|
|**RHum**|Derived|%|%|Magnus-Tetens<br>formula on T/Td|Relative<br>humidity|
|**W_spd**|`u10`, `v10`|m/s|m/s|`sqrt(u10² +<br>v10²)`|Wind speed at<br>10 m|
|**W_dir**|`u10`, `v10`|m/s|Degrees|`(atan2(u,v) +<br>360)% 360`|Wind direction<br>in degrees|
|**P_atm**|`sp`|Pa|hPa|`sp / 100.0`|Surface<br>atmospheric<br>pressure|
|**cloud_cover*<br>*|`tcc`|Fraction (0–1)|Fraction (0–1)|None|Total cloud<br>fraction|
|**precipitation<br>**|`tp`|m<br>(accumulated)|mm|`accum_to_fu<br>x(tp)* 1000`|Hourly<br>precipitation|
|**SZA**|Derived (pvlib)|Degrees|Degrees|Solar position<br>algorithm(SPA)|Solar Zenith<br>Angle|
|**solar_azimut<br>h**|Derived (pvlib)|Degrees|Degrees|Solar position<br>algorithm(SPA)|Solar azimuth<br>angle|
|**GHI_clearsky<br>**|Derived (pvlib)|W/m²|W/m²|Ineichen clear-<br>skymodel|Ideal/maximum<br>horizontal solar|



||||||input|
|---|---|---|---|---|---|
|**CSI**|Derived|Fraction|Fraction|`GHI /|Clearness|
|||||GHI_clearsky`|Index|



### **NASA POWER Variables (Tier 2 cross-source validation)** 

- `ALLSKY_SFC_SW_DWN` (GHI, W/m²) 

- `CLRSKY_SFC_SW_DWN` (Clear-sky GHI, W/m²) 

- `T2M` (Ambient temperature at 2 m, °C) 

- `RH2M` (Relative humidity at 2 m, %) 

- `WS10M` (Wind speed at 10 m, m/s) 

### **Literature Support** 

|Variable Group<br>|Reference|Source|
|---|---|---|
|ERA5 radiation felds|Ghodusinejad et al. (2026) —<br>reanalysis validation|`sources/<br>Ghodusinejad2026SolarIrradianc<br>eForecasting_summary.md`|
|NASA POWER GHI reference|NASA POWER documentation|`14_ERA5_POWER_VALIDATION.<br>md`|
|Solar geometry (SZA, clearsky)|Reda & Andreas (2004); Ineichen<br>model|`12_SOLAR_GEOMETRY.md`|
|RH from T/Td|Magnus-Tetens formula|Standard meteorologicalpractice|



## **4. 03_PHASE_1_AUDIT(1).md** 

Source: 03_PHASE_1_AUDIT(1).md 

## **03 — Phase 1 Audit: Data Collection** 

Scripts: 00a_build_population_grid.py, 00b_build_suntimes.py, 01_download_era5_tamilnadu.py, 01b_download_nasapower.py, 00_unzip_accum.py. 

### **Purpose** 

Determine the coordinates (where) and timestamps (when) to sample climate data, then retrieve ERA5 and NASA POWER historical records for Tamil Nadu. 

### **Inputs** 

- GADM boundary file (v4.1 India admin-1). 

- WorldPop 2020 UN-adjusted 100 m population density raster for India. 

- CDS API access credentials. 

- NASA POWER API. 

### **Processing Details** 

24. Population-Weighted Sampling (`00a_build_population_grid.py`): 

- Aggregates population onto a 0.25° grid aligned to ERA5's grid origin (`lat=90.0, lon=-180.0`). This guarantees a 1:1 spatial grid mapping between population cells and ERA5 grid nodes. 

- Keeps the minimal set of highest-population cells covering `COVERAGE_TARGET = 0.875` (87.5%) of the state's population. 

- Tamil Nadu Results: Produces 133 points (`TNP_0001` to `TNP_0133`). 

25. Sun-Event Times (`00b_build_suntimes.py`): 

- For every point × every date in 2016–2025, computes the exact UTC sunrise, solar noon, and sunset using `pvlib`'s SPA algorithm. 

- Row Count: 1,457,547 rows (133 points × 3653 days × 3 events). Alt=0 is assumed for sunrise/sunset times, which is a standard simplification. 

26. ERA5 Download (`01_download_era5_tamilnadu.py`): 

- Downloads three narrow UTC hour windows around sunrise, solar noon, and sunset, using circular mod-24 logic to handle day wraparound (important for westernmost points). 

- Downloads both instant and accumulated fields (240 files). 

27. NASA POWER Download (`01b_download_nasapower.py`): 

- Pulls full hourly weather parameters (87,660 hours per point) for all 133 points across the 10-year span (1,330 JSON files). 

28. CDS Zip-Quirk Fix (`00_unzip_accum.py`): 

- Scans and extracts netCDF files that the CDS API returned as disguised ZIPs. 

### **Differences from Rajasthan** 

- Point count: 133 points for Tamil Nadu vs 320 points for Rajasthan. This reflects Tamil Nadu's smaller geographic footprint. 

- Elevation: Rajasthan has a dedicated `00c_attach_elevation.py` script that downloads and extracts real elevation (m) using ERA5 geopotential. Tamil Nadu does not have an elevation attachment script. Instead, `02_combine_tamilnadu.py` uses a flat elevation approximation of 150 m for solar calculations. 

### **Status** 

#### COMPLETE 

### **Literature Support** 

|Component|Reference|Source|
|---|---|---|
|Population-weighted sampling|WorldPop 2020 UN-adjusted<br>raster|GADM/WorldPop documentation|
|Solar position algorithm|Reda & Andreas (2004) SPA via<br>pvlib|`12_SOLAR_GEOMETRY.md`|
|ERA5 reanalysis download|ECMWF ERA5 hourly data|`02_DATA_SOURCES_AND_VARI<br>ABLES.md`|
|NASA POWER hourly cache|NASA POWER API documentation|`14_ERA5_POWER_VALIDATION.<br>md`|
|NoveltyN6population weighting|Framework doc v3.0|`01_PROJECT_CONTEXT.md`|



## **5. 04_PHASE_2_AUDIT(1).md** 

Source: 04_PHASE_2_AUDIT(1).md 

## **04 — Phase 2 Audit: Preprocessing and Cross-Source Validation** 

Scripts: 02_combine_tamilnadu.py, 02b_build_daily_aggregates.py, 03_plots_raw.py, 03b_agreement_analysis.py, 03b_interactive_raw_qa.py, 04_preprocess_tamilnadu.py, 04c_postprocess_plots.py, 04c_interactive_postprocess_qc.py. 

### **Purpose** 

Combine ERA5 and NASA POWER weather variables at the sun-event instants, compute true daily averages/integrals, perform quality control, and impute missing values. 

### **Processing Details** 

29. Combine Script (`02_combine_tamilnadu.py`) — v3.1 corrected: 

- Snaps coordinates to the nearest ERA5 grid node, concatenates NetCDFs, applies `accum_to_flux()` (stateless clip — NOT diff-based deaccumulation), computes solar geometry via `pvlib`, and merges with NASA POWER within a 3-hour match window. 

30. Daily Aggregates (`02b_build_daily_aggregates.py`): 

- Reads full hourly NASA POWER series. Integrates GHI trapezoidally to daily kWh/m²/day; calculates DTR, HDD18, CDD24, cloudy fraction, CCI. 

31. Cross-Source Agreement (`03b_agreement_analysis.py`) — NEW v3.1: 

- Stratified MBE/RMSE/Pearson-r table; decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW); GHI scatter by season. 

32. 13-Step Preprocessing (`04_preprocess_tamilnadu.py`) — v3.1 corrected: 

- Steps 1–13 unchanged (inspection, physical validation, Hampel, imputation, features, lags, scaling, QC gate). 

- Step 2b (NEW): Per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution; saves `ghi_quantile_mapping_report.csv`. 

### **Corrected Audit Findings (v3.1)** 

33. Deaccumulation Bug — FIXED: 

- `02_combine_tamilnadu.py` now uses `accum_to_flux(s) = s.clip(lower=0)`. 

- Pre-fix stats (for reference): noon GHI r = 0.3963, MBE = −231.89 W/m². Post-fix expected: r > 0.80 (Rajasthan reference: r = 0.8102). 

34. Quantile-Mapping — FIXED: 

- Step 2b in `04_preprocess_tamilnadu.py` applies per-season QM after physical validation. 

- `03b_agreement_analysis.py` documents the cross-source decision branch. 

### **Status** 

COMPLETE (v3.1 fixes applied — re-run 02_combine → 04_preprocess for updated outputs) 

### **Literature Support** 

|Method|Reference|Source|
|---|---|---|
|ERA5 vs satellite GHI validation|Ghodusinejad et al. (2026)|`sources/<br>Ghodusinejad2026SolarIrradianc<br>eForecasting_summary.md`|
|Quantile mapping / bias<br>correction|Mansouri et al. (2025)|`sources/<br>Mansouri2025MultimodalRenewa<br>bleForecasting_summary.md`|
|Hampel MAD outlier detection|StandardQCpractice|`15_QUALITY_CONTROL.md`|
|MICE imputation|Rubin (1987); sklearn<br>IterativeImputer|`15_QUALITY_CONTROL.md`|



## **6. 05_PHASE_3_AUDIT(1).md** 

Source: 05_PHASE_3_AUDIT(1).md 

## **05 — Phase 3 Audit: Climate Signature Construction** 

Script: 04b_climate_signature.py, 04d_signature_interactive.py. 

### **Purpose** 

Collapse each point's 10-year hourly/daily weather into a single climate signature vector, which defines the location's climatology and determines the PCM performance targets. 

### **Processing Details** 

35. Tier 1 (Sun-Event Statistics): Means and percentiles of sun-event temperatures, GHI, humidity, wind. HSI (Thom 1959 Discomfort Index). 

36. Tier 2 (Daily-Integral Merge): True daily integrals from `02b` — GHI, SAI, cloudy fraction, CCI, HDD18, CDD24, DTR. 

37. Derived targets (v3.1 corrected): 

- `Tm_target = 50.0 + 7.0 = 57.0°C` 

- `L_required = (DRAW_MASS_KG × CP_WATER × ΔT) / ASSUMED_PCM_MASS_KG` 

- `DRAW_VOLUME_L = 300` (Avargani et al. 2021 domestic baseline) 

38. Five Interaction Terms: GHI×kt_std, DTR×cloudy_frac, RH×(Ta−Tm), wind×(Ta−Tsoil), CCI×(1−SAI). 

39. PCA Reduction: 4 components on temperature/climate block (>95% variance). 

40. Standardization: z-scoring for GMM clustering matrix. 

### **Current Finding** 

- Was: `DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60` → 0.001 kg/s → `L_required` ≈ 52 kJ/kg (latent-heat filter bypassed). 

- Fixed: `DRAW_VOLUME_L = 300`, `DRAW_MASS_KG = 300 kg` (realistic domestic scale). 

- Current model: `SHARE_PCM = 0.5`; PCM supplies half of the delivery energy while sensible storage and concurrent charging supply the remainder. 

- Current generated result: cluster `L_required` values are approximately 301-326 kJ/kg. These are runspecific outputs, not a universal constant. 

- Also applied in: `11_level_b_seasonal_analysis.py` (seasonal `L_required` uses the same share model). 

### **Status** 

COMPLETE (v3.1 fixes applied — re-run 04b for updated signatures) 

### **Literature Support** 

|Component|Reference|Source|
|---|---|---|
|300 L/daydraw volume|Avargani et al.(2021)|`17_LITERATURE_MAPPING.md`|
|HSI / discomfort index|Thom(1959)|`17_LITERATURE_MAPPING.md`|
|PCM melting band 42–70°C|Singh et al. (2025) Table 2|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|Worst-month sizing|Durin et al.(2018)|`17_LITERATURE_MAPPING.md`|
|Climate-feature → PCM mapping|Liu et al.(2025)|`sources/|



Liu2025AI_PCM_TES_Prediction_ Optimization_summary.md` 

## **7. 06_PHASE_4_AUDIT(1).md** 

Source: 06_PHASE_4_AUDIT(1).md 

## **06 — Phase 4 Audit: Climate Regime Clustering** 

Scripts: 05_cluster_tamilnadu.py, 05b_cluster_interactive.py, 11_level_b_seasonal_analysis.py. 

### **Purpose** 

Group the 133 population points into distinct climatic regimes using GMM clustering (Level A) and evaluate whether these regimes experience seasonal shifts that change the recommended PCM (Level B). 

### **Level A: GMM Clustering (v3.1 corrected)** 

- Fits K components from 2 to 10; computes BIC and silhouette scores. 

- Tamil Nadu Choice: K_FINAL = 5 regimes. 

- v3.1 fix: `covariance_type="diag"` (was `"full"`, which overfit 133×27 features). 

- Current (post-fix, current on-disk `cluster_profiles_tamilnadu.csv`) profiles: 

- Cluster 0: 8 pts (pop. 12.66M); Cluster 1: 42 pts (pop. 19.81M); Cluster 2: 39 pts (pop. 18.06M); Cluster 3: 22 pts (pop. 10.25M); Cluster 4: 22 pts (pop. 10.45M). Total 133 points, matching Phase 1. 

- **Open item, verified during this audit (not fixed):** BIC decreases monotonically from k=2 to k=10 with no interior minimum (same pathology documented in the Rajasthan pipeline), so BIC alone cannot select k. `05_cluster_tamilnadu.py` does not compute a bootstrap-ARI stability pass to break ties the way Rajasthan's three-tier k-selection rule does. Within the accepted 0.15–0.40 silhouette band, k=6 (silhouette 0.305) and k=9 (0.312) both score higher than the hard-coded K_FINAL=5 (0.262). This is flagged, not changed, here: re-clustering at a different k would change every downstream phase's headline recommendations (D6) and is a decision for the project owner, not something to silently redo mid-audit. See `20_IMPLEMENTATION_ISSUES.md` issue #8. 

### **Level B: Seasonal Sensitivity (v3.1 corrected)** 

- Recomputes `L_required_season` per season using 300 L/day draw (matching `04b`). 

- Single-method TOPSIS re-rank per (cluster, season); reports #1 PCM flips. 

- NE monsoon out-of-phase cycle provides physical basis for seasonal variation. 

### **Corrected Finding (v3.1 — GMM Overfitting)** 

- Was: `covariance_type="full"` → 1890 covariance parameters on 133 samples → membership saturation. 

- Fixed: `covariance_type="diag"` in `05_cluster_tamilnadu.py`. 

### **Status** 

COMPLETE (v3.1 fixes applied — re-run 05 and 11 after Phase 3 re-run) 

|**Literature Support**|||
|---|---|---|
|Component|Reference|Source|
|GMM climate regime discovery|Liu et al. (2025) — AI PCM TES|`sources/<br>Liu2025AI_PCM_TES_Prediction_<br>Optimization_summary.md`|
|Population-weighted clustering|NoveltyN1(framework doc)|`01_PROJECT_CONTEXT.md`|
|Seasonal PCM sensitivity|Singh et al. (2025) — monsoon<br>SWH|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|Diagonal GMM regularization|Standard small-npractice|`METHODS.md` §05|



## **8. 07_PHASE_5_AUDIT(1).md** 

Source: 07_PHASE_5_AUDIT(1).md 

## **07 — Phase 5 Audit: Feasibility Filtering** 

Scripts: 06_build_pcm_database.py, 07_feasibility_filter.py. 

### **Purpose** 

Hard-screen candidate PCMs from a database against each cluster's climate-adaptive targets (melting point and latent heat) to ensure only physically viable PCMs proceed to ranking. 

### **The PCM Database** 

- Imputes missing manufacturer properties (Rubitherm RT, Pluss savE) via MICE+RF+PMM blend. 

- Appends 7 literature PCMs (fatty acids, paraffins). 

- Total candidates: 62 PCMs: 55 manufacturer-derived records completed from the MICE+RF+PMM detailed input plus 7 literature records from Singh et al. Table 2. Manufacturer imputation flags and provenance are retained; genuinely unreported literature properties remain missing. 

### **Screen Constraints (Table 12)** 

41. Melting window: `Tm  [Tm_target − 5, Tm_target + 8]°C` (relaxable ±2K, up to 4 steps).∈ 42. Absolute band: `Tm  [42, 70]°C`.∈ 

43. Latent heat floor: `L ≥ 0.7 × L_required` — now binding after v3.1 L_required fix. 

44. Cycling stability: `cycles ≥ 300` (flagged if NaN). 

45. Supercooling veto: `supercooling ≤ 8K` (flagged if NaN). 

46. Corrosion veto: excludes `check_manually` in high-HSI clusters. 

47. Safety exclusion: flammability keyword veto. 

### **Current Finding** 

- The feasibility output audits 62 candidates per cluster, with pass/fail detail for every filter. Despite its filename, `feasibility_survivors_by_cluster.csv` is not survivors-only. 

- Current actual survivors (`passes_all=True`) are 15, 9, 13, 13, and 9 for clusters 0-4 respectively. 

- Current cluster `L_required` values are approximately 301-326 kJ/kg. The latent-heat floor is `max(100, 0.7 × L_required)` and is achievable for a subset of candidates. 

### **CRITICAL UPDATE: L_required Methodology Correction (2026-08-31, OPTION A)** 

The v3.1 L_required fix documented above has been superseded by a more fundamental methodology correction (2026-08-31). Phase 3's all-latent assumption (PCM supplies 100% of night discharge alone) was replaced with a literature-anchored fractional-share model: SHARE_PCM = 0.5, meaning PCM supplies ~50% of delivery, tank sensible heat + concurrent charging supply the remainder (per Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021). 

Current interpretation: SHARE_PCM = 0.5 is active in the upstream sizing calculation. The older approximately 2500 kJ/kg all-latent value and the approximately 1250 kJ/kg planning estimate are superseded by the values written to the current signature and feasibility artifacts. See 04b_climate_signature.py and config.py for the active implementation. 

### **Status** 

COMPLETE for the current generated artifacts. Re-run 06_build_pcm_database.py and 07_feasibility_filter.py whenever the PCM source or upstream climate signatures change. 

### **Literature Support** 

|Component|Reference|Source|
|---|---|---|
|PCM property database|Martinez (2025) — Rubitherm<br>measured data|`sources/<br>Martinez2025PCM_Industrial_TES<br>_summary.md`|
|Literature PCMs Table 2|Singh et al. (2025)|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|Melting band 42–70°C SWH|Abdellatif (2025) PCM modeling<br>review|`sources/<br>Abdellatif2025PCM_Modeling_Re<br>view_summary.md`|
|Corrosion in humid climates|Hamzat (2025) PCM solar storage|`sources/<br>Hamzat2025PCM_SolarEnergySto<br>rage_summary.md`|
|Property imputation|Eldokaishi (2022) ANN SWH|`sources/<br>Eldokaishi2022WaterPCM_ANN_<br>SWH_summary.md`|



## **9. 08_PHASE_6_AUDIT(1).md** 

Source: 08_PHASE_6_AUDIT(1).md 

## **08 — Phase 6 Audit: Multi-Criteria Ranking Engine** 

Script: 08_mcdm_ranking.py. 

### **Purpose** 

Rank the surviving PCM candidates in each cluster using four independent multi-criteria decision-making (MCDM) methods under weight and property uncertainty. 

### **Processing Details** 

48. Target-Based Fitness: 

- Converts melting temperature to a Gaussian fitness score: 

f_Tm = exp( - (Tm - Tm_target)² / (2 * σ²) ), where σ = 4.0 K. 

#### 49. Criteria Evaluated: 

- `f_Tm` (melting point fitness) - benefit. 

- `latent_heat_margin_ratio = latent_heat / L_required` (climate-relative benefit). 

- `rho_H_MJ_m3` (volumetric latent heat) - benefit. 

- `TC_W_mK` (thermal conductivity) - benefit. 

- `cycles_confidence` (log-scaled cycling reliability) - benefit. 

50. Four MCDM Methods: 

- TOPSIS: Closeness to Euclidean ideal/anti-ideal. 

- GRA: Grey relational grade vs max reference. 

- PROMETHEE II: Net outranking flow (V-shape, q=0.10, p=0.30). 

- VIKOR: Compromise index Q (v=0.5) with acceptable-advantage check. 

51. Weights: 

- Entropy weights (data-driven) blended with AHP prior weights (Table 13 priors) at `λ = 0.5`. 

52. Consensus & Uncertainty: 

- Primary rank: Borda count across the 4 methods. 

- Cross-check: Copeland pairwise majority. 

- Monte Carlo: 5,000 Dirichlet weight draws + Gaussian property perturbations (Tm ±1K, latent heat ±5%, conductivity ±10%). Calculates Top-3 inclusion probability and Top-1 retention. 

### **Results** 

- Ranks the current feasibility survivors for each cluster. The generated `mcdm_topk_by_cluster.csv` contains the Top-3 for each of the five clusters (15 rows total). 

- The current climate-relative latent-heat criterion is `latent_heat / L_required`, so the score retains cluster-specific demand information rather than treating raw latent heat as equally useful everywhere. 

- Do not describe the ranking as seven survivors per cluster; the feasibility file contains all 62 audited candidates per cluster and the number passing all filters varies by cluster. 

- Monte Carlo stability reports run-specific Top-3 inclusion and Top-1 retention probabilities; quote values from `monte_carlo_stability.csv` for the particular run being reported. 

- In the current run, `n-Octacosane (C28)` is the consensus rank-1 PCM in all five clusters. This is a statewide consensus result; it does not imply that all alternatives have equal stability or physical performance. 

### **Status** 

#### COMPLETE 

### **Literature Support** 

|Component|Reference|Source|
|---|---|---|
|TOPSIS|Hwang & Yoon (1981); Chen et al.<br>(2025) SWH MCDM|`sources/<br>Chen2025TaguchiGRA_PCM_Nan<br>ofuid_SWH_summary.md`|
|GRA|Deng (1982); Chen et al. (2025)|`sources/<br>Chen2025TaguchiGRA_PCM_Nan<br>ofuid_SWH_summary.md`|
|PROMETHEE II|Brans & Mareschal(2005)|Standard MCDM literature|
|VIKOR|Opricovic & Tzeng (2004)|Standard MCDM literature|
|Monte Carlo uncertainty|Chopra et al. (2023) techno-<br>economic MC|`sources/<br>Chopra2023HPETC_MonteCarlo_<br>TechnoEconomic_summary.md`|
|Entropy+AHP weight blend|Framework doc Table 13|`17_LITERATURE_MAPPING.md`|



## **10. 19_PHASE_7_8_AUDIT.md** 

Source: 19_PHASE_7_8_AUDIT.md 

## **19 — Phase 7 & 8 Audit: Physics-Based Validation and Output** 

The Tamil Nadu pipeline has fully implemented both phases. 

### **Phase 7: Grey-Box Physics Validation (`10_physics_validation.py`) — current run** 

53. Model Structure: 3-phase lumped-enthalpy tank (Barqawi 2025): sensible solid → isothermal melting → sensible liquid. 

54. Numerical Method: Backward Euler (implicit), hourly `dt = 3600 s`. 

55. v3.1 fix: Ambient tank heat loss `UA_TANK_W_K = 2.0 W/K` added to prevent artificially high solar fractions and enable PCM cycling. 

56. **v3.2 finding (this audit): the v3.1 fix above did not actually work.** Two independent algebra/structure bugs in the backward-Euler solver — a spurious term in the closed-form `Tw_new` solve, and no isolation of the collector-tank coupling at night — were silently reproducing the exact pre-v3.1 failure signature even with `UA_TANK_W_K` active. Both are the same bug classes already documented and fixed in the Rajasthan pipeline's `physics_lib.py`. Both are now fixed here too; see `20_IMPLEMENTATION_ISSUES.md` issues #6–#7 for the full derivation, a numerical proof (buggy formula gives `Tw_new=69.2°C` from a 45°C collector with no other heat source — thermodynamically impossible; corrected formula gives 44.5°C), and the before/after evidence. 

57. Current validation outcome (post v3.2 fix, current on-disk artifacts): 

- Spearman ρ by cluster: 0 = **-0.016**, 1 = **+0.717** (partial agreement, p=0.030), 2 = **+0.355**, 3 = **-0.171**, 4 = **0.000**. Mean **+0.177** across clusters (was mean -0.151 pre-fix). Still weak/mixed overall — an honestly-reportable finding per Table 17 of the framework plan, not evidence the fix is wrong. 

- Solar fractions now span approximately 30.5%–80.1% (was pinned at 85.3-99.6%); 41% of simulations fall within the published 54-84% benchmark band (was 0%). 

- Complete cycles/year now range 3–260 (was 0-1), consistent with real annual PCM freeze-melt cycling. The remaining band gap is a tank/collector calibration question (stated, literature-anchored parameters not empirically fit to Tamil Nadu deployments) — not a further solver bug. 

### **Corrected Root Causes (v3.1 + v3.2)** 

|Cause|Fix|Version|
|---|---|---|
|Disabled latent-heat constraint|300 L/day draw → realistic L_required|v3.1|
|Missing tank heat loss|UA_TANK_W_K = 2.0 W/K (ineffective until v3.2)|v3.1|
|GHI feature contamination|accum_to_flux + quantile mapping|v3.1|
|Spurious term in backward-Euler `Tw_new` closed form|Numerator corrected to use old `Tp` only|v3.2|
|No night/idle collector-coupling isolation|`NIGHT_ISOLATION_FRACTION = 0.05` gates coupling when Tc < Tw|v3.2|



### **Phase 8: Recommendation Cards (`09_recommendation_cards.py`)** 

- Aggregates cluster profiles, MCDM rankings, physics validation, Monte Carlo stability into `recommendation_cards.md`. 

- Re-run `09` after `10` to include updated Spearman ρ and solar fractions. 

- The current cards were regenerated after the updated ranking and physics runs and contain five cluster recommendations. 

### **Status** 

COMPLETE for the current generated artifacts. Re-run 10 → 09 whenever the PCM database, climate signatures, or ranking outputs change. 

### **Literature Support** 

|Component|Reference|Source|
|---|---|---|
|Grey-box tank ODE|Barqawi (2025)|`sources/<br>Barqawi2025DynamicSimulation<br>PCM_SWH_summary.md`|
|Solar fraction benchmark 54–84%|Singh et al. (2025)|`sources/<br>Singh2025PCM_SWH_Comprehe<br>nsiveReview_summary.md`|
|Spearman rank validation|Framework doc §10|`17_LITERATURE_MAPPING.md`|



|Backward Euler stability|Ghodusinejad (2026) — physics-|`sources/|
|---|---|---|
||informed models|Ghodusinejad2026SolarIrradianc<br>eForecasting_summary.md`|
|Recommendation cards|Odoi & Yorke (2025) AI SWH|`sources/|
||review|OdoiYorke2025AI_SWH_Review_s<br>ummary.md`|

## **11. Addendum — v3.2 Verification Pass (this audit)**

This addendum documents a full re-verification of the Tamil Nadu pipeline against its own documentation and against the Rajasthan pipeline's already-published, already-fixed bug history, performed by reading the live scripts and current on-disk data rather than trusting the prior documentation's claims at face value. It follows the same standard the rest of this project holds itself to: self-corrections are reported as findings, not smoothed over.

### What was checked and confirmed correct (no change needed)
- **Deaccumulation** (`02_combine_tamilnadu.py`): `accum_to_flux()` is a stateless `.clip(lower=0)`, not a diff — matches the corrected Rajasthan convention.
- **Quantile mapping** (`04_preprocess_tamilnadu.py` Step 2b): per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER is implemented and produces `ghi_quantile_mapping_report.csv`.
- **L_required derivation** (`config.py`, `04b_climate_signature.py`): `SHARE_PCM = 0.5`, `Tm_target = 57.0°C`, draw sizing `Q = 300 kg × 4.186 × (50 − T_mains)`, `L_required = SHARE_PCM × Q / 50 kg` — matches the Rajasthan-corrected (2026-08-31) methodology exactly. Current cluster values: 301.4, 321.6, 302.2, 304.1, 326.0 kJ/kg for clusters 0–4.
- **GMM covariance**: `covariance_type="diag"` in `05_cluster_tamilnadu.py`, correctly justified for 133 samples × 27+ standardized features.
- **VIKOR sign** (`08_mcdm_ranking.py`): `Q = v·(S−S*)/(Sw−S*) + (1−v)·(R−R*)/(Rw−R*)` — the corrected form; does **not** have the sign-inversion bug the Rajasthan audit found and fixed in its own MCDM script.
- **PCM database**: 62 rows (55 manufacturer-derived + 7 literature), same composition and row count as the shared Rajasthan/Tamil Nadu `PCM_data` source; the nested `PCM_data/PCM_data/` path-extraction quirk documented in the Rajasthan audit is present here too but already worked around (a copy of the detailed CSV exists at the path `06_build_pcm_database.py` actually reads).
- **Feasibility survivor counts and cluster targets**: verified directly against `feasibility_survivors_by_cluster.csv` and `cluster_profiles_tamilnadu.csv` — 15/9/13/13/9 survivors and 301–326 kJ/kg L_required, matching the claims in Sections 7–8 above.
- **MCDM Top-1 finding**: `n-Octacosane (C28)` is confirmed as the current consensus rank-1 PCM in all five clusters (verified against the live `mcdm_topk_by_cluster.csv`; unaffected by the Phase 7 fix below since Phase 6 does not depend on Phase 7).

### What was found and fixed (v3.2)
Two bugs in `10_physics_validation.py`'s backward-Euler solver — both the *same bug classes* already documented and fixed in the Rajasthan pipeline's `physics_lib.py`, independently reintroduced here:
1. **Closed-form solve error**: the pre-melt/post-melt `Tw_new` numerator carried a spurious `dt·c·Tw` term inside the PCM-coupling factor. Verified numerically: with the script's own default parameters this pushed `Tw_new` to 69.2°C from a 45°C collector with no other heat source (impossible); the corrected formula gives 44.5°C.
2. **Missing night/idle isolation**: the collector-tank coupling was applied identically day and night, letting the tank drain heat back out through the idle collector loop at nearly the daytime charging rate, on top of the separate ambient-loss term.

Together, these bugs meant the v3.1 "add ambient tank heat loss" fix had never actually taken effect — every simulated PCM was still landing at 85–100% solar fraction with 0–1 cycles/year (0% in the published 54–84% benchmark band), identical to the pre-v3.1 failure signature. Both are now fixed (see `20_IMPLEMENTATION_ISSUES.md` #6–#7 for the full derivation). After the fix: solar fractions span 30.5–80.1%, 41% of runs land in the benchmark band (was 0%), complete cycles/year range 3–260 (was 0–1), and mean Spearman ρ across clusters moved from −0.151 to **+0.177**. Both `10_physics_validation.py` and `09_recommendation_cards.py` have already been re-run; the on-disk `physics_validation_results.csv`, `physics_validation_spearman.csv`, and `recommendation_cards.md` reflect the fix.

### What remains open (verified gaps, not fixed this round — each is a scope/judgment decision, not a silent defect)
1. **K_FINAL=5 is hand-set, not selected by a bootstrap-ARI tie-break.** k=6 (silhouette 0.305) and k=9 (0.312) both score higher than k=5 (0.262) within the accepted band; BIC alone cannot decide (monotonically decreasing, same pathology as Rajasthan). Re-clustering at a different k would change every downstream phase's headline recommendation and is left for the project owner to decide.
2. **No cross-phase provenance/fingerprint hard-fail** between Phases 4→5→6→7→8, unlike Rajasthan's `provenance_lib.py`. No evidence this has caused an actual mismatch in the current artifacts, but it is a real reproducibility gap for future re-runs.
3. **MCDM criteria reduced to 5 of the framework doc's 8** (`cost`, `corrosion`, `supercooling` dropped entirely rather than carried as near-zero-weight criteria). Deliberate and documented, but means Rajasthan's own "supercooling dominates the entropy weight and the physics model can't simulate it" finding cannot recur here in the same form.
4. **`LATENT_HEAT_ABSOLUTE_MIN_KJ_KG = 100`** in `config.py` is a Tamil-Nadu-only addition beyond Table 12's literal `L ≥ 0.7 × L_required` rule. Currently a no-op (0.7×L_required ≈ 211–228 kJ/kg already exceeds it) but should be named as a deviation if Table 12 is quoted verbatim in the write-up.
5. **Residual Phase 7 calibration**: 59% of simulations still fall outside the 54–84% benchmark band. The solver is now structurally correct; the gap is in the stated, literature-anchored (not empirically fit) tank/collector parameters (`M_W_KG`, `A_C_M2`, `COLLECTOR_EFF`, draw schedule).
6. Previously-documented open items unchanged by this pass: PCM database further expansion, external Köppen/NBC cluster validation, flat elevation proxy, `monsoon_index` proxy-only status, and the heuristic (not literal) 5th-percentile charging-feasibility filter — see `22_FINAL_READINESS_REPORT.md`.

### Verdict
**Structurally correct as of v3.2.** All five v3.0 bugs are genuinely fixed (not just documented as fixed), the two Phase 7 solver bugs found in this pass are fixed and re-run, and the physics-validation output now behaves like an honest, if partially uncalibrated, physical model rather than one masking a bug. The open items above are real but are scope/calibration questions for the project owner, not defects to be silently patched.



