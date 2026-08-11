# METHODS — Algorithm and Technique Justification per Script

**Project**: Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating  
**Pipeline**: ERA5 Tamil Nadu — Objective 1 (Climate-Region-Aware PCM Recommendation)  
**Purpose of this document**: For every script in the pipeline, states (a) which algorithmic method was chosen, (b) why it was chosen, and (c) compares it against two plausible alternatives that were explicitly rejected and why.

---

## 00a — `00a_build_population_grid.py` — Population Sampling

### Method Chosen: **Population-Weighted Grid Sampling (WorldPop 0.25°)**

The script clips the WorldPop 2020 UN-adjusted 100 m raster to the Tamil Nadu boundary (from GADM v4.1), aggregates pixel populations onto a 0.25° grid aligned exactly to ERA5's native grid origin, then selects the minimum set of highest-population cells covering ≥ 87.5% of the state population.

**Why chosen:**
- Sampling proportional to population ensures climate regimes represent where actual domestic hot-water demand lives — a PCM recommendation is meaningless if it optimises for uninhabited desert cells.
- Grid-aligning to ERA5's origin (lat = 90.0°, lon = -180.0°, multiples of 0.25°) guarantees a 1:1 pixel-to-ERA5-node mapping, eliminating interpolation error at the download stage.
- Using 87.5% coverage keeps the point count manageable (133 points) while covering the overwhelming majority of residential demand.

**Alternative 1 rejected: Uniform Grid Sampling**
A naive 0.25° uniform grid over the Tamil Nadu bounding box would produce ~200+ points including sea cells, boundary cells, and sparsely inhabited high-elevation Western Ghats cells. These consume download and compute budget for regions with negligible domestic demand. The old `TN_LOCATIONS` (260+ named cities) in the original `02_combine_tamilnadu.py` was effectively this. Rejected because it wastes resources on low-demand areas and biases climate signatures toward uninhabited regions.

**Alternative 2 rejected: Administrative-Unit Centroid Sampling**
Using district centroids (38 districts in Tamil Nadu) would give one point per administrative unit. This misses within-district climate variation (e.g., a coastal district stretches from sea level to inland plains) and is not comparable across states with different numbers of districts (Rajasthan has 33). Rejected because it conflates political boundaries with climate boundaries, which the framework plan (Section 6.2) explicitly prohibits.

---

## 00b — `00b_build_suntimes.py` — Solar Event Times

### Method Chosen: **pvlib Solar Position Algorithm (SPA, Reda & Andreas 2004)**

For every point × every date in 2016–2025, computes the exact UTC sunrise, solar noon, and sunset using `pvlib.location.Location.get_sun_rise_set_transit()`, which internally calls the full NREL SPA.

**Why chosen:**
- SPA is the NREL reference standard for solar position with sub-0.01° accuracy (Reda & Andreas 2004). It correctly accounts for atmospheric refraction at sunrise/sunset, equation of time, and the obliquity of the ecliptic.
- Computing exact event times per point per date eliminates fixed-clock-hour bias. A Tamil Nadu coastal point sees solar noon at a different UTC time than a Western Ghats point, and both shift 20–25 minutes between summer and winter solstice.
- pvlib is the standard Python solar energy library; SPA is its most accurate model.

**Alternative 1 rejected: Fixed clock hours (e.g., always download 06:00, 12:00, 18:00 UTC)**
This was the original pipeline's approach. At Tamil Nadu's longitude (~77–80°E), local noon is approximately 06:30–06:48 UTC, so a fixed 06:00 UTC download misses the solar peak by up to 48 minutes, systematically undersampling peak GHI. Rejected because it introduces a time-of-observation bias into the climate signature.

**Alternative 2 rejected: Astronomical sunrise/sunset formula (Spencer 1971 or Cooper 1969)**
The simplified declination/hour-angle formula ignores atmospheric refraction and uses an approximate equation of time. Errors are up to ±3 minutes for near-equinox dates and ±15 minutes for high-altitude points. While acceptable for many engineering purposes, the SPA is computationally negligible and is already available in pvlib. Rejected in favour of the higher-accuracy standard.

---

## 01 — `01_download_era5_tamilnadu.py` — ERA5 Data Retrieval

### Method Chosen: **CDS API Point-Level NetCDF Download, Sun-Event-Aligned Hour Windows**

Downloads two variable types per month per year (240 total calls): instant-analysis variables (temperature, humidity, wind, pressure) and accumulated forecast variables (solar radiation, precipitation). Hour windows span the seasonal range of sunrise, noon, and sunset events across all 133 points plus a ±1-hour margin.

**Why chosen:**
- The CDS API (Copernicus Climate Data Store) provides ERA5's authoritative archive with full 10-year hourly coverage at 0.25° resolution — no other source offers this combination of length, resolution, and free access.
- Downloading by sun-event-aligned windows (vs. all 24 hours per day) reduces each monthly download by ~75%, making the 10-year × 240-call job feasible without exceeding CDS request size limits.
- Point-level extraction (all 133 points in one geographic bounding box per request) is the most efficient CDS access pattern for moderate point counts; per-point requests would require 133 × 240 = 31,920 API calls.

**Alternative 1 rejected: Google Earth Engine (GEE) ERA5-Land extraction**
GEE provides ERA5-Land (9 km resolution) and ERA5 hourly (31 km, coarser) via the `ee.ImageCollection` API. GEE is efficient for spatial aggregations but requires a GEE account and has strict quota limits per project. ERA5-Land variables differ from ERA5 (e.g., no SSRD in ERA5-Land's native export), causing schema mismatches with the pipeline. Rejected due to quota limitations and schema incompatibility.

**Alternative 2 rejected: ERA5-Complete MARS tape access**
Direct MARS (Meteorological Archival and Retrieval System) tape access provides full-precision ERA5 data without CDS pre-processing, and uses forecast step semantics where accumulated fields truly reset at 00 and 12 UTC. However, MARS access requires ECMWF internal credentials not available to external users. Rejected as unavailable.

---

## 01b — `01b_download_nasapower.py` — NASA POWER Retrieval

### Method Chosen: **NASA POWER REST API, Full Hourly Cache per Point**

Downloads complete hourly weather time series (87,660 hours per point) for all 133 points across 2016–2025 from the NASA POWER (Prediction Of Worldwide Energy Resources) API in JSON format.

**Why chosen:**
- NASA POWER provides satellite-assimilated surface meteorology including ALLSKY_SFC_SW_DWN (global horizontal irradiance) at 0.5° resolution, independently derived from MERRA-2 + GEWEX SRB + CERES, making it a genuinely independent cross-check against ERA5's model-driven GHI.
- Hourly resolution (not just daily) allows precise event-time matching with the ERA5 sun-event instants.
- Full caching of the complete 10-year hourly series in JSON eliminates repeated API calls per rerun.

**Alternative 1 rejected: PVGIS (Photovoltaic Geographical Information System)**
PVGIS provides hourly TMY (Typical Meteorological Year) data derived from SARAH-3 satellite data. However, PVGIS TMY data is a single synthetic representative year, not an actual 10-year historical series. The pipeline requires real daily variability (droughts, monsoon years) for a defensible physics validation. Rejected because TMY data cannot reproduce inter-annual variability.

**Alternative 2 rejected: SoDa MERRA-2 download**
SoDa (Solar radiation Data) provides MERRA-2-based solar data via commercial or academic API. MERRA-2 at its native 0.625° × 0.5° resolution is coarser than ERA5 and has known biases over the Indian subcontinent during monsoon months. Rejected in favour of NASA POWER which post-processes MERRA-2 with corrections derived from CERES.

---

## 02 — `02_combine_tamilnadu.py` — ERA5 Combine & Solar Geometry

### Method Chosen: **Nearest-Hour Snap with pvlib SPA Solar Geometry**

Concatenates all monthly NetCDF files per point into a continuous time series, applies deaccumulation to flux variables, computes solar geometry (SZA, azimuth, clearsky GHI via Ineichen model) using `pvlib`, and snaps ERA5 and POWER readings to the exact sun-event timestamps.

> ⚠️ **DOCUMENTED BUG (see `20_IMPLEMENTATION_ISSUES.md`)**: The `deaccumulate()` function uses `pd.Series.diff()`, which corrupts GHI when the downloaded NetCDF files already contain hourly fluxes (not running forecast totals). **The correct replacement is `accum_to_flux(s) = s.clip(lower=0)`, matching the Rajasthan pipeline fix.** This is noted here for the methods report; the corrected version of the script is the one that should be cited.

**Why Ineichen clear-sky model chosen:**
- The Ineichen-Perez model (Ineichen & Perez 2002) is the standard in pvlib and requires only location (lat, lon, altitude) plus a Linke turbidity coefficient. It outperforms the Haurwitz and Kasten models at tropical latitudes where aerosol loads and water vapour are high.
- pvlib provides a preloaded Linke turbidity climatology (from Remund et al. 2003) requiring no external data files.

**Alternative 1 rejected for clear-sky: REST2 model**
REST2 (Reference Evaluation of Solar Transmittance, 2 bands) provides higher accuracy but requires five atmospheric input columns (precipitable water, aerosol optical depth at 700 nm, nitrogen dioxide, ozone, surface pressure) that are not in the ERA5 download schema. Rejected because the required inputs are unavailable without a separate atmospheric chemistry download.

**Alternative 2 rejected for clear-sky: Simplified Solis model**
Simplified Solis (Ineichen 2008) is accurate and fast but requires precipitable water as an explicit input. Rejected for the same data-availability reason as REST2.

---

## 02b — `02b_build_daily_aggregates.py` — True Daily Integrals

### Method Chosen: **Trapezoidal Integration of Hourly NASA POWER GHI**

Reads the full hourly NASA POWER JSON cache per point. For each calendar day with ≥ 20 valid hours, integrates GHI using `numpy.trapz` to obtain the true daily GHI (kWh/m²/day), and computes true DTR, HDD18, CDD24, cloudy fraction, CCI, and seasonal amplitude index (SAI).

**Why chosen:**
- Trapezoidal integration on an hourly grid is both exact (for piecewise-linear interpolation of the true diurnal curve) and robust to missing individual hours, unlike the half-sine approximation `(2/π) × GHI_noon × daylen` that the old Tier-1 proxy used.
- Computing HDD/CDD from the true hourly series (rather than from the three-event proxy) eliminates the annualisation error that the Rajasthan audit documented (sum over 10 years gives ~10× the correct annual HDD).

**Alternative 1 rejected: Half-sine proxy integration**
The half-sine formula `Daily GHI ≈ (2/π) × GHI_noon × daylen_hours` was used in the Tier-1 proxy. It assumes a symmetric, single-peaked diurnal curve, which is inaccurate on partially cloudy days (where the curve is multi-peaked) or days with asymmetric cloud cover. RMSE vs. true trapezoidal integral can exceed 1.5 kWh/m²/day on cloudy monsoon days. Rejected as too imprecise.

**Alternative 2 rejected: Daily satellite product (SARAH-3, CM SAF)**
The SARAH-3 (Surface Solar Radiation Data Set-Heliosat, Edition 3) provides a validated daily surface irradiance product at 5 km resolution from Meteosat. However, accessing SARAH-3 data requires a EUMETSAT account and has incomplete India coverage in early years (2016–2017). Rejected due to availability constraints.

---

## 03 — `03_plots_raw.py` — Raw QA Statistics

### Method Chosen: **Pearson r, MBE, RMSE Cross-Source Agreement**

Computes Pearson correlation, Mean Bias Error (ERA5 − POWER), and RMSE between matched ERA5 and NASA POWER values for GHI, temperature, relative humidity, and wind speed at every sun-event timestamp.

**Why chosen:**
- MBE reveals systematic bias direction (ERA5 overestimates or underestimates POWER).
- RMSE captures scatter including both bias and random error.
- Pearson r measures linear association between the two sources, independent of absolute calibration.
- Together, these three metrics are the international standard for reanalysis validation (WMO OSCAR guidelines; Holmgren et al. pvlib 2018).

**Alternative 1 rejected: Bland-Altman analysis**
Bland-Altman plots are designed for method-comparison studies where both measurements share equal uncertainty (e.g., two clinical measurement devices). ERA5 and NASA POWER are not symmetric — ERA5 is a reanalysis (physical model) and POWER is a satellite product. The standard in climate science is to define one source as "reference" (POWER, higher spatial resolution for validation) and compute bias relative to it. Rejected as inappropriate for asymmetric source comparison.

**Alternative 2 rejected: Kolmogorov-Smirnov (KS) distributional test**
KS tests whether two samples come from the same distribution. It is useful for detecting distributional differences but does not decompose the disagreement into bias vs. random error components, making it less actionable for the diagnosis of whether the disagreement is a systematic offset (fixable by quantile mapping) or random scatter. Rejected in favour of the more diagnostic (MBE + RMSE + r) triplet.

---

## 04 — `04_preprocess_tamilnadu.py` — 13-Step QC and Feature Engineering

### Method Chosen: **Hampel Filter (MAD-based) + MICE Imputation**

Physical range gating removes physically impossible values. Hampel filter (median ± k × MAD over a 7-day rolling window) detects non-physical temporal outliers. Missing values are imputed via a cascade: linear interpolation → forward/backward fill → spatial zone median → IterativeImputer (MICE).

**Why chosen:**
- Hampel filter is robust to non-Gaussian distributions (ERA5 variables over Tamil Nadu have right-skewed solar distributions and heavy-tailed precipitation). Unlike z-score outlier detection, it uses the Median Absolute Deviation (MAD), making it resistant to masking by the very outliers it is trying to remove.
- MICE (Multiple Imputation by Chained Equations), implemented as `sklearn.impute.IterativeImputer`, is the gold standard for multivariate imputation because it models the joint distribution of all variables, preserving inter-variable correlations (e.g., temperature and humidity are strongly anti-correlated in Tamil Nadu).

**Alternative 1 rejected: IQR-based outlier detection**
IQR (1.5 × IQR rule) is a univariate, non-rolling outlier detector. It cannot distinguish a legitimate temperature extreme (e.g., 43°C heat wave) from a sensor spike. The Hampel filter's rolling window provides temporal context. Rejected because IQR would flag legitimate climate extremes as outliers.

**Alternative 2 rejected: KNN imputation**
K-Nearest Neighbours imputation fills missing values from the k spatially nearest points. It ignores temporal continuity — if a sensor is missing for 3 consecutive hours due to a data gap, KNN fills all 3 from the same spatial neighbours, which is correct spatially but ignores that temporal autocorrelation is stronger than spatial correlation for short gaps. Rejected in favour of MICE's temporal-plus-spatial joint model.

---

## 04b — `04b_climate_signature.py` — Climate Signature Construction

### Method Chosen: **Two-Tier Signature (Sun-Event + Daily Integral) with PCA Compression**

Tier 1 (sun-event statistics): computes per-point means/percentiles of temperature, GHI, humidity, wind at sunrise/noon/sunset. Tier 2 (daily integrals from `02b`): true GHI integral, HDD, CDD, DTR, cloudy fraction, CCI, SAI, seasonality. PCA compresses the correlated temperature/pressure block into 4 components.

> ⚠️ **DOCUMENTED BUG (fixed in corrected version)**: `DRAW_RATE_KG_PER_S = 60.0 / 1000 / 60` equals `0.001 kg/s`, which is 1000× too small. The correct value for the stated 300 L daily domestic draw is to use a flat volume: `DRAW_VOLUME_L = 300.0`. See fix notes in `20_IMPLEMENTATION_ISSUES.md`.

**Why two-tier signature chosen:**
- Sun-event statistics capture the state of the atmosphere at the moments most relevant to solar collector operation (charging at noon, delivery at sunset, overnight retention at sunrise).
- Daily integrals capture integrated energy quantities that determine sizing (total daily GHI, total heating/cooling degree days) — these cannot be derived from 3-event snapshots without the half-sine approximation error.
- PCA removes collinearity: temperature, dewpoint, and pressure are highly correlated, so including all of them raw inflates their combined weight in the GMM distance metric.

**Alternative 1 rejected: Monthly climatological means only**
Using only 12-month mean temperature and GHI values (as in most traditional PCM selection papers, e.g., Sharma et al. 2017) discards all information about intra-month variability (DTR, CCI, cloudy fraction). This is particularly damaging for Tamil Nadu, where the NE monsoon brings concentrated rainfall in Oct–Dec, producing high month-to-month variability that monthly means smooth out. Rejected as insufficient for climate-adaptive PCM selection.

**Alternative 2 rejected: Full-resolution time series clustering (DTW)**
Dynamic Time Warping (DTW) applied to the full 10-year daily GHI series would cluster points by their temporal curve shape, potentially identifying the out-of-phase monsoon pattern. However, DTW on 3,653-day series requires an O(n² × m²) matrix (n = 133 points, m = 3,653 days), which is computationally prohibitive, and the resulting distance metric is not easily interpretable for the PCM target derivation step. Rejected as computationally intractable and not interpretable.

---

## 05 — `05_cluster_tamilnadu.py` — Climate Regime Clustering

### Method Chosen: **Gaussian Mixture Model (GMM) with BIC + Silhouette Selection**

Fits GMM for K = 2..10, selects K by BIC minimum and silhouette score within the acceptable band [0.15, 0.40]. Final fit uses K_FINAL = 5.

> ⚠️ **DOCUMENTED BUG (fixed in corrected version)**: Original script uses `covariance_type="full"`. With 133 samples and 27+ feature dimensions, a full covariance GMM overfits (1,890 parameters). **Correct setting: `covariance_type="diag"`.** See `20_IMPLEMENTATION_ISSUES.md`.

**Why GMM chosen over hard clustering:**
- Climate in Tamil Nadu is a continuous gradient — the boundary between the coastal humid belt and the interior semi-arid zone is not a sharp line. Points near that boundary have genuine partial membership in both regimes. GMM's soft-membership probabilities capture this uncertainty explicitly and are passed to Phase 5/6 as boundary-awareness weights.
- BIC penalises over-complex models, preventing the algorithm from discovering spurious micro-clusters in sparse data.
- GMM's probabilistic framework allows the model selection (BIC) to be separated from the clustering (hard labels), enabling a more rigorous model comparison.

**Alternative 1 rejected: K-Means**
K-Means assumes spherical, equally-sized clusters and assigns hard membership. These assumptions are violated in Tamil Nadu climate space: the Nilgiris montane climate is a small cluster in feature space surrounded by a large coastal humid cluster. K-Means would either merge the Nilgiris into the coast cluster or split the coast cluster artificially to compensate. The script actually computes K-Means as a reported comparison baseline (see `kmeans_comparison_tamilnadu.csv`), confirming lower silhouette scores.

**Alternative 2 rejected: Hierarchical Agglomerative Clustering (HAC)**
HAC (Ward linkage) builds a dendrogram and does not require specifying K in advance. However, once the tree is cut, membership is hard (no soft probabilities for boundary points), and HAC does not provide a natural probabilistic model that can be queried for new points. HAC also requires O(n²) memory for the distance matrix, which is manageable for 133 points but does not scale to the eventual 4-state joint clustering (n ≈ 800+). Rejected in favour of a method that scales and provides soft membership.

---

## 06 — `06_build_pcm_database.py` — PCM Database Construction

### Method Chosen: **Random Forest PMM-like Imputation for Missing Manufacturer Properties**

Builds a 25-PCM database (18 from Rubitherm/Pluss manufacturer datasheets + 7 from peer-reviewed literature). Missing properties (density, specific heat, thermal conductivity) for manufacturer PCMs are imputed using a 3-donor Predictive Mean Matching (PMM)-like blend trained with Random Forest regressors on the available complete rows.

**Why RF imputation chosen:**
- PCM properties are non-linearly correlated (e.g., thermal conductivity correlates with density, which correlates with phase — solid vs. liquid). Random Forest captures these non-linear relationships, unlike linear regression imputation.
- PMM-like approach (predicting the value, then substituting the nearest observed value rather than the raw prediction) avoids imputed values outside the physically plausible range.

**Alternative 1 rejected: Mean/median imputation**
Filling missing properties with column means would destroy the correlation between density and thermal conductivity that exists across PCM families (paraffins have lower density and conductivity than fatty acids, which have lower density and conductivity than salt hydrates). Mean imputation would produce physically implausible combinations. Rejected.

**Alternative 2 rejected: MICE (multivariate chain)**
MICE is the gold standard for tabular imputation and was used in preprocessing (Phase 2). However, the PCM database has only 18 manufacturer rows, too few for MICE's iterative convergence. Random Forest with the PMM correction provides a stable single-pass imputation for such small databases. Rejected as unstable for n < 30.

---

## 07 — `07_feasibility_filter.py` — PCM Feasibility Screening

### Method Chosen: **8-Filter Hard-Screen with Step-Down κ-Relaxation**

Applies 8 sequential hard filters (melting window, absolute band, latent heat floor, cycling, supercooling, corrosion, safety, charging). If fewer than 5 candidates survive, the melting window is relaxed by 2K per step (up to 4 steps) — see Section 8's κ-calibration rule in the framework plan.

> ⚠️ **DOCUMENTED BUG (fixed in corrected version)**: The `L_required` value fed from `04b` was calculated with a 1000× unit error (`DRAW_RATE_KG_PER_S = 0.001 kg/s`), producing `L_required ≈ 52 kJ/kg`. This makes the latent heat floor `0.7 × 52 = 36 kJ/kg`, which all 25 PCMs clear trivially. The corrected script uses `DRAW_VOLUME_L = 300` to compute `L_required`, giving a physically realistic floor that actually screens candidates.

**Why hard-screen before MCDM chosen:**
- MCDM methods (TOPSIS, GRA, PROMETHEE II, VIKOR) are compensatory — a PCM with a melting point 15°C below target but excellent latent heat can score well in TOPSIS, even though it cannot physically absorb heat at the right temperature. Hard-screening eliminates this false positive before ranking.
- The 8 filters directly implement the framework plan's Table 12. Deviating from them would require a plan amendment.

**Alternative 1 rejected: Soft-penalty scoring instead of hard-screens**
A penalty function could reduce a PCM's score for being outside the melting window rather than excluding it. This allows compensatory trade-offs (e.g., great latent heat compensates for off-target Tm) that are physically unrealisable. Rejected because a PCM that doesn't melt at the right temperature provides zero latent heat storage, regardless of its rated capacity.

**Alternative 2 rejected: Fuzzy logic membership for melting window**
Fuzzy membership could give partial credit to PCMs near the melting window boundary. This is a valid approach for uncertainty quantification but adds complexity that is not required by the framework plan. The κ-relaxation mechanism already handles the boundary case in a documented, step-wise fashion. Rejected in favour of the plan-compliant step-down approach.

---

## 08 — `08_mcdm_ranking.py` — Multi-Criteria Decision Making

### Method Chosen: **4-Method Stack (TOPSIS + GRA + PROMETHEE II + VIKOR) with Borda Consensus + 5000-Draw Monte Carlo**

Ranks the feasibility survivors using four independent MCDM methods, aggregates via Borda count (and cross-checks with Copeland pairwise), and propagates weight/property uncertainty through 5,000 Dirichlet + Gaussian Monte Carlo draws.

**Why 4-method stack chosen:**
- TOPSIS (Euclidean distance-based), GRA (grey relational grade), PROMETHEE II (pairwise outranking), and VIKOR (compromise-ranking) represent four methodologically distinct schools: distance-from-ideal, reference-series, outranking, and compromise. Agreement across all four is strong evidence of a robust ranking; disagreement reveals criterion sensitivity that a single method would silently miss.
- The framework plan (v3.0 Section 2.2) explicitly states that using fewer methods is "a regression."

**Why Gaussian fitness for Tm chosen:**
- Melting temperature is a target-based criterion: closer to Tm_target is better in both directions. Converting to `f_Tm = exp(−(Tm − Tm_target)² / (2σ²))` transforms it into a genuine benefit criterion (higher = better) compatible with all four MCDM methods. Using raw Tm as a benefit or cost criterion would be mathematically incorrect.

**Alternative 1 rejected: AHP (Analytic Hierarchy Process) alone**
AHP derives weights from pairwise expert comparisons. It is excellent for weight elicitation but is not a ranking method — it assigns weights to criteria and then typically feeds them to a simple weighted sum. A weighted sum is the most compensatory aggregation possible and would allow a PCM with a very high latent heat to outrank one with a better-matched melting point. Rejected in favour of the TOPSIS/PROMETHEE stack.

**Alternative 2 rejected: ELECTRE III**
ELECTRE III is a full outranking method that handles veto thresholds and produces a partial preorder (not a total ranking), which is the most theoretically rigorous output for MCDM under uncertainty. However, ELECTRE III requires per-criterion concordance and discordance thresholds that are difficult to elicit without expert consensus, and it does not naturally integrate with the Monte Carlo stability framework. Rejected in favour of PROMETHEE II and VIKOR, which are fully ranked and Monte-Carlo-compatible.

---

## 10 — `10_physics_validation.py` — Grey-Box Tank Simulation

### Method Chosen: **Lumped-Enthalpy 3-Phase Tank Model (Backward Euler, Hourly)**

Implements a two-node lumped model (tank water Tw + PCM node Tp/melt-fraction f) coupled by a coil heat exchanger, solved with backward (implicit) Euler at hourly time steps over a full representative year of real daily weather for each cluster's medoid point.

> ⚠️ **DOCUMENTED BUG (noted for correction)**: The current model lacks an ambient heat loss term. Without `−U·A·(Tw − Ta)·dt` in the tank energy balance, the tank never cools overnight, producing artificially high solar fractions (90–99%) and near-zero complete freeze-melt cycles per year. The corrected model should add `UA_TANK_W_K = 2.0` and `−UA_TANK_W_K × (Tw − tamb) × dt` in each hourly step.

**Why backward Euler chosen:**
- The tank water–coil–PCM system has a fast time constant (~minutes for the coil coupling), far shorter than the hourly time step. Forward (explicit) Euler is conditionally stable only when `dt < 2τ`; with τ ≈ 3–5 minutes and dt = 3,600 s, explicit Euler would be wildly unstable. Backward Euler is unconditionally stable for this linear system and requires only a single 2×2 matrix solve per step (closed-form, no iterative solver needed).
- Forward Euler or Runge-Kutta methods are appropriate only when the time step is smaller than the fastest time constant — they are not appropriate here.

**Alternative 1 rejected: EnergyPlus simulation**
EnergyPlus is the U.S. DOE's whole-building energy simulation engine. It does not natively support latent-heat PCM inside a domestic hot water tank node network without a custom EnergyPlusFMU or EMS scripting. The framework plan (Section 10.1) explicitly rejects EnergyPlus for this reason. It would also require a full building geometry input file unrelated to the PCM storage problem. Rejected as specified in plan v3.0.

**Alternative 2 rejected: TRNSYS Type 840 (PCM storage)**
TRNSYS is a transient energy simulation package with a specialised PCM storage component (Type 840). It handles the three-phase enthalpy formulation natively. However, TRNSYS requires a commercial licence (several thousand USD), is not available in the project's computing environment, and cannot be called programmatically from Python for the 5-cluster × 7-PCM × 365-day batch run required here. Rejected due to licensing and integration constraints.

---

## 09 — `09_recommendation_cards.py` — Output Aggregation

### Method Chosen: **Markdown Report Generation via Python string formatting**

Reads cluster profiles, MCDM rankings, Monte Carlo stability, and physics validation results, and writes a human-readable structured `recommendation_cards.md` file with one card per cluster.

**Why chosen:**
- Markdown is version-controllable, renders in GitHub and all major documentation platforms, and can be pasted directly into a thesis appendix.
- A programmatic report ensures complete reproducibility — the numbers in the document are always consistent with the current pipeline run.

**Alternative 1 rejected: Jupyter notebook report**
Notebooks support rich cell-by-cell execution and inline plots, but they are not reproducible as a standalone artifact (require a running kernel), are difficult to diff in version control, and cannot be called as part of a batch pipeline. Rejected in favour of a pure Python script output.

**Alternative 2 rejected: PDF generation via LaTeX**
LaTeX would produce a publication-quality typeset PDF but requires a full TeX installation and significant formatting code. The framework plan specifies "recommendation cards" as a quick-reference human-readable output, not a publication. Rejected as over-engineered for the stated deliverable.

---

## 11 — `11_level_b_seasonal_analysis.py` — Seasonal Sensitivity (Level B)

### Method Chosen: **Seasonal TOPSIS Re-ranking with Per-Season L_required**

For each of the 4 seasons (Winter: Nov–Feb, Summer: Mar–May, Monsoon: Jun–Sep, Retreat: Oct–Nov for Tamil Nadu's NE monsoon), recomputes L_required from season-specific temperature and demand parameters, then re-ranks the feasibility survivors using single-method TOPSIS.

**Why single-method TOPSIS for seasonal sensitivity:**
- The 4-method stack in Phase 6 is computationally intensive and justified for the final reported ranking. For a sensitivity check (does the #1 PCM change by season?), a fast single method suffices.
- TOPSIS is chosen because it is the most widely cited method in PCM-MCDM literature, making the seasonal sensitivity results directly comparable to published studies.

**Alternative 1 rejected: Repeat full 4-method stack per season**
Running the full TOPSIS + GRA + PROMETHEE II + VIKOR + 5,000-draw Monte Carlo for each of 4 seasons × 5 clusters = 20 runs would add ~20 minutes of compute time and 20 separate Monte Carlo result files. The marginal scientific value over a single-method sensitivity check is low, since the goal is only to identify whether #1 flips. Rejected as unnecessarily expensive.

**Alternative 2 rejected: Sensitivity analysis via parameter perturbation on Phase 6 weights**
An alternative Level-B check would perturb the MCDM weights (increasing the weight on `L_required` for monsoon season) rather than recomputing L_required per season. This is not physically motivated — L_required genuinely changes with season because demand temperature and ambient conditions change. Rejected in favour of the physically grounded seasonal recomputation.

---

*This document covers all 13 primary scripts in the Tamil Nadu pipeline (Phase 1 through Phase 8). All algorithmic choices are made in conformance with `Objective1_PCM_Climate_Framework_Plan_v3.docx` unless explicitly flagged as an audited bug.*
