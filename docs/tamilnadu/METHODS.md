# METHODS — Algorithm and Technique Justification per Script

**Project**: Climate-Adaptive Intelligent Control and Optimization of PCM Thermal Storage for Solar Water Heating  
**Pipeline**: ERA5 Tamil Nadu — Objective 1 (Climate-Region-Aware PCM Recommendation)  
**Purpose of this document**: For every script in the pipeline, states (a) which algorithmic method was chosen, (b) why it was chosen, (c) compares it against two plausible alternatives that were explicitly rejected — each with **formal reference papers and literature-based justifications** — and **(d) compares our method against what published PCM-SWH and climate-data literature actually uses** — so the methodology can be defended in a viva or report.

> **How to cite in your thesis**: Each script section ends with a **Reference Papers — Chosen vs Rejected** table. Use ✅ rows for methodology justification; use ❌ rows to show you considered and rejected alternatives with published precedent (not arbitrary choices).

---

## Part 0 — Master Comparison: Our Methods vs Published Literature

This table is the quick-reference for supervisors. Each row maps one pipeline decision to what the literature typically does, whether our `sources/` folder covers it, and why our choice is justified or an improvement.


| Pipeline Stage                | **Our Method (Tamil Nadu v3.1)**                      | **What Literature Typically Uses**                                    | **Key Reference(s)**                                                                        | **Match / Gap / Our Advantage**                                                                       |
| ----------------------------- | ----------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **Spatial sampling**          | Population-weighted 0.25° grid (133 pts, 87.5% pop)   | Uniform grid, district centroids, or single-city case studies         | Odoi-Yorke (2025) — India 105 pubs but mostly single-site; WorldPop/GADM standard           | **Gap filled**: literature rarely weights by domestic demand; we align climate with where people live |
| **Solar event timing**        | pvlib SPA per point × date (Reda & Andreas 2004)      | Fixed clock hours or monthly means                                    | Ghodusinejad (2026) — SZA/cloud in NWP validation; Chen (2025) — TRNSYS hourly              | **Improvement**: eliminates time-of-observation bias vs fixed-hour downloads                          |
| **Climate data source**       | ERA5 reanalysis + NASA POWER cross-check              | TRNSYS TMY, PVGIS synthetic year, or local pyranometer only           | Ghodusinejad (2026) — hybrid NWP+satellite; Mansouri (2025) — multimodal fusion             | **Improvement**: 10-year real history + independent satellite reference (not TMY)                     |
| **Radiation unit conversion** | `accum_to_flux()` stateless clip                      | Manual deaccumulation (MARS convention) or raw use without validation | Ghodusinejad (2026) — reanalysis GHI RMSE 107–125 W/m²; Themeßl et al. (2012)*              | **Bug fix**: CDS point downloads differ from MARS; literature assumes one convention                  |
| **Bias correction**           | Per-season empirical quantile mapping                 | Fixed-weight blend (0.6 ERA5 + 0.4 POWER) or no correction            | Mansouri (2025) — ML bias correction y_{corr}=f_{ML}(\hat{y}_{NWP}); Themeßl et al. (2012)* | **Aligned**: QM is standard post-processing when systematic bias exists                               |
| **Daily GHI integration**     | Trapezoidal integration of hourly POWER               | Half-sine proxy from noon GHI only                                    | Ghodusinejad (2026) — hourly aggregation standard; Singh (2025) — sizing needs daily energy | **Improvement**: exact integral vs 3-snapshot proxy                                                   |
| **Cross-source metrics**      | MBE + RMSE + Pearson r                                | MAPE, R² alone, or KS test                                            | Ghodusinejad (2026) — MBE/RMSE/rRMSE standard; NOAA benchmarks in Table 1                   | **Aligned**: same triplet used in irradiance validation literature                                    |
| **Outlier detection**         | Hampel filter (MAD, rolling window)                   | IQR rule or z-score                                                   | Hampel (1974)*; Hamzat (2025) — data reliability challenge in PCM reviews                   | **Improvement**: robust to skewed solar distributions                                                 |
| **Missing data imputation**   | Interpolate → ffill → spatial median → MICE           | Mean imputation or KNN only                                           | Liu (2025) — 90/10 splits, RF/XGBoost for properties; Rubin (1987)*                         | **Aligned**: MICE preserves multivariate correlations                                                 |
| **Climate signature**         | Two-tier (sun-event + daily integral) + PCA           | Monthly mean T and GHI only                                           | Singh (2025) — 40–70°C band, latent heat priority; Odoi-Yorke (2025) — weather as ML input  | **Improvement**: captures monsoon variability monthly means miss                                      |
| **Hot water draw sizing**     | 300 L/day flat volume (Avargani 2021)                 | 30–60 L/day in Taguchi studies or unstated                            | Chen (2025) — ≥30 L/day; Chopra (2023) — 60 L/person × 6 = 360 L/day                        | **Aligned**: domestic-scale draw, consistent with Indian SWH studies                                  |
| **Regime clustering**         | GMM, K=5, BIC selection, `diag` covariance            | K-Means, hierarchical, or admin zones                                 | Liu (2025) — GA/PSO for structure, not climate zones; Singh (2025) — no GMM                 | **Novel (N1)**: discovered regimes vs arbitrary geography                                             |
| **PCM database**              | 25 rows, RF-PMM imputation for gaps                   | Manufacturer datasheet only or ANN-predicted properties               | Martinez (2025) — Rubitherm measured data; Singh (2025) Table 2                             | **Gap**: literature has 200+ PCMs cited but we use auditable 25 only                                  |
| **Feasibility screening**     | 8 hard filters + κ-relaxation                         | Soft penalty or no pre-screen                                         | Singh (2025) — selection priority: L, k, Tm; Abdellatif (2025) — modeling constraints       | **Aligned**: hard Tm/L floor before ranking (Singh priority order)                                    |
| **PCM ranking**               | TOPSIS + GRA + PROMETHEE II + VIKOR + Borda + 5000 MC | Single method: Taguchi+GRA or AHP+TOPSIS alone                        | **Chen (2025)** — Taguchi L36 + GRA only; Chopra (2023) — Monte Carlo for economics         | **Improvement**: 4-method consensus + uncertainty vs Chen's single GRA                                |
| **Physics validation**        | Lumped-enthalpy 3-phase, backward Euler, real weather | TRNSYS Type 840, EnergyPlus, or synthetic sinusoidal GHI              | **Barqawi (2025)** — 3-phase ODE + RK45; Chen (2025) — TRNSYS <5% error                     | **Aligned structure**, different solver (Euler vs RK45) and real vs synthetic climate                 |
| **Seasonal sensitivity**      | Level B: seasonal L_required + TOPSIS re-rank         | Full seasonal re-simulation (TRNSYS per season)                       | Chen (2025) — single annual TRNSYS run; Singh (2025) — monsoon intermittency noted          | **Nearly free** check the plan permits; literature rarely reports seasonal PCM flip                   |
| **Output format**             | Markdown recommendation cards                         | PDF report or TRNSYS output tables                                    | Odoi-Yorke (2025) — reproducibility gap in AI-SWH literature                                | **Improvement**: version-controlled, paste-ready for thesis                                           |


 *Themeßl et al. (2012) and Hampel (1974) are standard methods references added because quantile mapping and MAD outlier detection are not covered in detail in our PCM-specific `sources/` summaries — see References at end.*

### How to read this for your report

- **"Aligned"** = we do what the literature recommends; cite the matching paper.
- **"Improvement"** = we go beyond typical single-site / single-method papers; cite as novelty.
- **"Gap filled"** = literature identifies the problem but few papers implement it; cite as contribution.
- **"Gap"** = honest limitation we still have (e.g., 25 PCM rows vs Singh's 200+ cited studies).

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

**Literature comparison**


| Aspect          | Our pipeline                        | Literature                                                                                                      | Verdict                                |
| --------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| Sampling unit   | Population-weighted ERA5 grid cells | Single city (Chen 2025: one TRNSYS site); bibliometric corpus (Odoi-Yorke 2025: 245 papers, mostly single-site) | **Novel** — N6 population weighting    |
| Grid resolution | 0.25° aligned to ERA5               | TRNSYS uses site coordinates; no grid in most PCM-SWH papers                                                    | **Aligned with reanalysis practice**   |
| Coverage target | 87.5% of state population           | Not reported in Singh (2025) or Chen (2025)                                                                     | **Explicit demand representativeness** |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                           | Key Reference(s)                                                                                                                                                   | Justification                                                                                                                   |
| -------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Population-weighted 0.25° grid (WorldPop + GADM) | Stevens et al. (2015) — WorldPop methodology; Odoi-Yorke (2025) — India 105 AI-SWH pubs, mostly single-site                                                        | Aligns climate samples with domestic hot-water demand geography; grid origin matches ERA5 native cells (no interpolation error) |
| **❌ Rejected** | Uniform 0.25° grid / named-city list             | Odoi-Yorke (2025) — 245 papers, dominant pattern is single-location TRNSYS/ANN case studies; Al-Mamun (2023) — SWH state-of-art reviews cite site-specific designs | Wastes CDS quota on sea/boundary/Ghats cells with negligible demand; biases climate signature toward uninhabited areas          |
| **❌ Rejected** | Administrative-unit (district) centroids         | Chopra (2023) — zone-level solar radiation for Indian HPETC, but one point per climate zone; Liu (2025) — GA/PSO optimises structure, not political boundaries     | Within-district climate gradients (coastal vs inland) are lost; not portable across states with different district counts       |


**Full citations**: Stevens, F.R. et al. (2015). Disaggregating census data for population mapping using random forests. *PLoS ONE*, 10(2), e0101072. Odoi-Yorke, F. (2025). AI for solar water heating systems. *Energy Convers. Manag.: X*, 28, 101378. Chopra, A. et al. (2023). Monte Carlo techno-economic HPETC. See `sources/` summaries.

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

**Literature comparison**


| Aspect            | Our pipeline                    | Literature                                                                                      | Verdict                                      |
| ----------------- | ------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Solar position    | pvlib SPA (Reda & Andreas 2004) | Ghodusinejad (2026): SZA in NWP/TSI pipelines; Chen (2025): TRNSYS uses built-in solar geometry | **Standard** — same accuracy class as TRNSYS |
| Event granularity | Per point × per date × 3 events | Most SWH papers use hourly TRNSYS timesteps at one location                                     | **Finer spatial coverage** (133 points)      |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                       | Key Reference(s)                                                                                                                                   | Justification                                                                                                              |
| -------------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | pvlib SPA (Reda & Andreas 2004)                              | Reda & Andreas (2004) — NREL reference algorithm, sub-0.01° accuracy; Holmgren et al. (2018) — pvlib as standard Python solar library              | Correct refraction, equation of time, and per-point/per-date solar noon shift (20–25 min solstice range in TN)             |
| **❌ Rejected** | Fixed UTC clock hours (06:00, 12:00, 18:00)                  | Ghodusinejad (2026) — hourly GHI aggregation and SZA as standard forecast inputs; Chen (2025) — TRNSYS uses continuous hourly solar geometry       | At 77–80°E, fixed 06:00 UTC misses local noon by up to 48 min → systematic undersampling of peak GHI                       |
| **❌ Rejected** | Simplified astronomical formulas (Spencer 1971; Cooper 1969) | Reda & Andreas (2004) — SPA benchmark vs simplified declination/hour-angle methods; Ineichen & Perez (2002) — clear-sky models assume accurate SZA | ±3–15 min event-time error without refraction; SPA is computationally negligible in pvlib and is the TRNSYS accuracy class |


**Full citations**: Reda, I. & Andreas, A. (2004). Solar position algorithm. *Solar Energy*, 76(5), 577–589. Spencer, J.W. (1971). Fourier series representation of the position of the Sun. *Search*, 2(5), 172. Cooper, P.I. (1969). The absorption of radiation in solar stills. *Solar Energy*, 12(3), 333–346.

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

**Literature comparison**


| Aspect                  | Our pipeline                            | Literature                                                                          | Verdict                                                                |
| ----------------------- | --------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Primary climate archive | ERA5 via CDS API, 10 years, 133 points  | Ghodusinejad (2026): WRF/GFS/ECMWF NWP; Mansouri (2025): NWP + satellite multimodal | **Aligned** — ERA5 is ECMWF reanalysis cited in NWP reviews            |
| Temporal coverage       | 2016–2025 actual history                | Chen (2025): TRNSYS simulation; PVGIS: single TMY year                              | **Improvement** — real inter-annual variability for monsoon years      |
| Download strategy       | Sun-event windows only (~75% reduction) | Full 24h/day in most reanalysis studies                                             | **Pragmatic optimisation** — not in literature but CDS-quota motivated |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                      | Key Reference(s)                                                                                                                                                    | Justification                                                                                                                      |
| -------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | ERA5 via CDS API, sun-event-aligned windows | Hersbach et al. (2020) — ERA5 reanalysis; Ghodusinejad (2026) — ECMWF/NWP cited in irradiance validation reviews; Mansouri (2025) — multimodal NWP+satellite fusion | Only free source combining 10-year hourly history at 0.25°; event windows cut download size ~75% while preserving peak-GHI capture |
| **❌ Rejected** | Google Earth Engine ERA5-Land               | Gorelick et al. (2017) — GEE platform; Muñoz-Sabater et al. (2021) — ERA5-Land (9 km, different variable set)                                                       | ERA5-Land lacks native SSRD export schema matching pipeline; GEE project quotas constrain 133-point × 10-year batch jobs           |
| **❌ Rejected** | MARS direct tape access                     | ECMWF (2019) — MARS archive documentation; Themeßl et al. (2012) — bias correction on regional climate model output                                                 | MARS gives true accumulated-step semantics but requires ECMWF credentials unavailable to external academic users                   |


**Full citations**: Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.*, 146(730), 1999–2049. Gorelick, N. et al. (2017). Google Earth Engine. *Remote Sens. Environ.*, 202, 18–27.

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

**Literature comparison**


| Aspect                        | Our pipeline                                | Literature                                                                                 | Verdict                                                               |
| ----------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| Reference / validation source | NASA POWER hourly (MERRA-2 + CERES + GEWEX) | Ghodusinejad (2026): satellite + ground pyranometer benchmarks; NOAA GHI Model 2 Xcor=0.95 | **Aligned** — independent satellite-assimilated reference             |
| Resolution                    | 0.5° POWER vs 0.25° ERA5                    | Mansouri (2025): multimodal fusion of mismatched resolutions                               | **Standard practice** — use POWER as bias reference, ERA5 as backbone |
| Caching                       | Full 10-year hourly JSON per point          | Most papers use on-demand API or TMY file                                                  | **Reproducibility improvement**                                       |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                  | Key Reference(s)                                                                                                                                                   | Justification                                                                                                                                |
| -------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | NASA POWER REST API, full hourly cache  | Stackhouse et al. (2021) — POWER methodology (MERRA-2 + CERES + GEWEX); Ghodusinejad (2026) — satellite-assimilated GHI as validation reference (NOAA Xcor ≈ 0.95) | Independent of ERA5's model physics; hourly resolution enables exact sun-event matching; JSON cache ensures reproducibility                  |
| **❌ Rejected** | PVGIS TMY (Typical Meteorological Year) | Huld et al. (2012) — PVGIS synthetic year methodology; Chen (2025) — TRNSYS uses actual weather files, not TMY                                                     | TMY is one synthetic representative year — cannot reproduce 2016–2025 inter-annual monsoon/drought variability needed for physics validation |
| **❌ Rejected** | SoDa / raw MERRA-2 download             | Rienecker et al. (2011) — MERRA reanalysis; Gracia-Amillo et al. (2014) — MERRA-2 solar bias over complex terrain                                                  | MERRA-2 native 0.625° × 0.5° grid is coarser; known monsoon-month GHI bias over Indian subcontinent; POWER applies CERES-derived corrections |


**Full citations**: Stackhouse, P.W. Jr. et al. (2021). POWER Project Methodology. NASA Langley. Huld, T. et al. (2012). A new solar radiation database for estimating PV performance. *Prog. Photovolt.*, 20(6), 686–698.

---

## 02 — `02_combine_tamilnadu.py` — ERA5 Combine & Solar Geometry

### Method Chosen: **Nearest-Hour Snap with pvlib SPA Solar Geometry**

Concatenates all monthly NetCDF files per point into a continuous time series, applies `accum_to_flux()` (stateless clip) to radiation fields, computes solar geometry (SZA, azimuth, clearsky GHI via Ineichen model) using `pvlib`, and snaps ERA5 and POWER readings to the exact sun-event timestamps.

> ✅ **CORRECTED (v3.1)**: Replaced diff-based `deaccumulate()` with `accum_to_flux(s) = s.clip(lower=0)`. CDS point downloads already return hourly fluxes; diffing corrupted GHI (noon r ≈ 0.40). See `20_IMPLEMENTATION_ISSUES.md`.

**Why Ineichen clear-sky model chosen:**

- The Ineichen-Perez model (Ineichen & Perez 2002) is the standard in pvlib and requires only location (lat, lon, altitude) plus a Linke turbidity coefficient. It outperforms the Haurwitz and Kasten models at tropical latitudes where aerosol loads and water vapour are high.
- pvlib provides a preloaded Linke turbidity climatology (from Remund et al. 2003) requiring no external data files.

**Alternative 1 rejected for clear-sky: REST2 model**
REST2 (Reference Evaluation of Solar Transmittance, 2 bands) provides higher accuracy but requires five atmospheric input columns (precipitable water, aerosol optical depth at 700 nm, nitrogen dioxide, ozone, surface pressure) that are not in the ERA5 download schema. Rejected because the required inputs are unavailable without a separate atmospheric chemistry download.

**Alternative 2 rejected for clear-sky: Simplified Solis model**
Simplified Solis (Ineichen 2008) is accurate and fast but requires precipitable water as an explicit input. Rejected for the same data-availability reason as REST2.

**Literature comparison**


| Aspect          | Our pipeline                         | Literature                                                                                  | Verdict                                                               |
| --------------- | ------------------------------------ | ------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| GHI from ERA5   | `accum_to_flux()` + /3600 (v3.1 fix) | Ghodusinejad (2026): NWP GHI RMSE 107–125 W/m²; many papers assume correct reanalysis units | **Critical fix** — literature rarely documents CDS vs MARS convention |
| Clear-sky model | Ineichen via pvlib                   | Ghodusinejad (2026): clear-sky index K_c standard input feature                             | **Aligned**                                                           |
| Event matching  | Nearest-hour snap, MAX_MATCH=3 h     | Mansouri (2025): temporal alignment challenge in multimodal fusion                          | **Standard** with documented proxy check via SZA                      |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                         | Key Reference(s)                                                                                                                                                         | Justification                                                                                                                                                    |
| -------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | `accum_to_flux()` + Ineichen clear-sky (pvlib) | Ineichen & Perez (2002) — Ineichen-Perez clear-sky model; Ghodusinejad (2026) — clear-sky index K_c as standard feature; Themeßl et al. (2012) — bias correction context | CDS point downloads return hourly fluxes (not MARS accumulations); Ineichen outperforms Haurwitz/Kasten at tropical aerosol loads                                |
| **❌ Rejected** | REST2 clear-sky model                          | Rigollier et al. (2000) — REST2 two-band transmittance model                                                                                                             | Requires AOD, precipitable water, NO₂, O₃ columns absent from ERA5 download schema — would need separate atmospheric chemistry retrieval                         |
| **❌ Rejected** | Simplified Solis model                         | Ineichen (2008) — simplified Solis formulation                                                                                                                           | Same data-availability constraint: explicit precipitable-water input not in pipeline variables; Ineichen with Remund Linke turbidity needs only lat/lon/altitude |


**Full citations**: Ineichen, P. & Perez, R. (2002). A new airmass independent formulation for the Linke turbidity coefficient. *Solar Energy*, 73(3), 151–157. Rigollier, C. et al. (2000). An automatic method for the determination of the Linke turbidity coefficient. *Solar Energy*, 68(1), 71–80.

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

**Literature comparison**


| Aspect        | Our pipeline                            | Literature                                                                            | Verdict                                      |
| ------------- | --------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------- |
| Daily GHI     | Trapezoidal ∫ of hourly POWER           | Singh (2025): sizing uses daily energy; Chen (2025): TRNSYS hourly → daily efficiency | **More rigorous** than noon-only proxy       |
| Degree-days   | True hourly HDD18/CDD24, annualised     | Chopra (2023): zone-level solar radiation (kWh/m²/day) for India                      | **Aligned** with Indian climate-zone studies |
| Cloud metrics | CCI, cloudy fraction from hourly series | Ghodusinejad (2026): cloud fraction/type as forecast driver                           | **Aligned** — cloud variability captured     |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                  | Key Reference(s)                                                                                                                                                             | Justification                                                                                                                 |
| -------------- | ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Trapezoidal integration of hourly POWER GHI             | Duffie & Beckman (2020) — standard solar engineering integration; Singh (2025) — PCM sizing requires integrated daily energy; Chen (2025) — TRNSYS hourly → daily efficiency | Exact for piecewise-linear hourly curve; robust to isolated missing hours; eliminates annualisation error in HDD/CDD          |
| **❌ Rejected** | Half-sine proxy (2/\pi) \times GHI_{noon} \times daylen | Iqbal (1983) — symmetric clear-sky diurnal model; Barqawi (2025) — synthetic sinusoidal GHI (Eq. 4) for controlled experiments                                               | Assumes single-peaked symmetric curve; RMSE > 1.5 kWh/m²/day on multi-peaked monsoon cloudy days                              |
| **❌ Rejected** | SARAH-3 daily satellite product                         | Pfeifroth et al. (2019) — SARAH-3 Heliosat Edition 3; CM SAF documentation                                                                                                   | Requires EUMETSAT account; incomplete India coverage in 2016–2017; adds third data source when POWER hourly already available |


**Full citations**: Duffie, J.A. & Beckman, W.A. (2020). *Solar Engineering of Thermal Processes*, 5th ed. Wiley. Pfeifroth, U. et al. (2019). Surface solar radiation set Heliosat (SARAH-3). *Earth Syst. Sci. Data*, 11(4), 1929–1946.

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

**Literature comparison**


| Aspect             | Our pipeline                                                         | Literature                                                                     | Verdict                                                      |
| ------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------ |
| Agreement metrics  | MBE, RMSE, Pearson r                                                 | **Ghodusinejad (2026)**: NOAA GHI MBE −23.29 W/m², RMSE 107.41 W/m², Xcor 0.95 | **Identical metric family**                                  |
| Decision gate      | `03b_agreement_analysis.py`: BACKBONE / QUANTILE_MAP / MANUAL_REVIEW | Mansouri (2025): y_{corrected}=f_{ML}(\hat{y}_{NWP}) ML bias correction        | **Aligned intent** — we use QM instead of black-box ML blend |
| Fixed-weight blend | Explicitly rejected                                                  | Not found in our sources with principled derivation                            | **Methodological choice** — documented in bias_decision txt  |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                 | Key Reference(s)                                                                                                                                                   | Justification                                                                                                                                  |
| -------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | MBE + RMSE + Pearson r triplet         | Ghodusinejad (2026) — NOAA GHI MBE −23.29 W/m², RMSE 107.41 W/m², Xcor 0.95; Holmgren et al. (2018) — pvlib validation guidelines; WMO OSCAR reanalysis benchmarks | Decomposes disagreement into systematic bias (fixable by QM) vs scatter vs linear association — standard in irradiance validation literature   |
| **❌ Rejected** | Bland-Altman limits of agreement       | Bland & Altman (1986) — method-comparison for symmetric measurement devices; Ghodusinejad (2026) — asymmetric reference/target framing in NWP validation           | ERA5 (model) vs POWER (satellite) are not symmetric instruments; climate science convention treats POWER as reference and computes signed bias |
| **❌ Rejected** | Kolmogorov-Smirnov distributional test | Massey (1951) — KS two-sample test; Mansouri (2025) — ML bias correction y_{corr}=f_{ML}(\hat{y}_{NWP}) needs pointwise bias diagnosis                             | Detects distributional shift but cannot separate fixable systematic offset from random scatter — less actionable for quantile-mapping gate     |


**Full citations**: Bland, J.M. & Altman, D.G. (1986). Statistical methods for assessing agreement. *Lancet*, 1(8476), 307–310. Massey, F.J. (1951). The Kolmogorov-Smirnov test for goodness of fit. *JASA*, 46(253), 68–78.

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

**Literature comparison**


| Aspect                    | Our pipeline                       | Literature                                                                               | Verdict                                                      |
| ------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Outlier method            | Hampel MAD (rolling)               | Hamzat (2025): data reliability / inconsistent reporting flagged as PCM barrier          | **Standard robust QC** — Hampel (1974)*                      |
| Imputation                | Cascade + MICE                     | Liu (2025): ANN/XGBoost for PCM property prediction; Eldokaishi (2022): ANN for SWH gaps | **Different domain** — we impute weather, not PCM properties |
| Bias correction (Step 2b) | Per-season quantile mapping on GHI | Mansouri (2025): NWP error correction via ML; Themeßl et al. (2012)*: empirical QM       | **Aligned** — QM is established climate downscaling method   |


### Reference Papers — Chosen vs Rejected


| Status         | Method                             | Key Reference(s)                                                                                                                                                                                       | Justification                                                                                                                                    |
| -------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **✅ Chosen**   | Hampel MAD filter + cascade → MICE | Hampel (1974) — robust MAD outlier detection; Rubin (1987) — multiple imputation theory; Themeßl et al. (2012) — empirical quantile mapping for bias; Mansouri (2025) — NWP error correction precedent | Hampel resists masking by right-skewed solar tails; MICE preserves T–RH anti-correlation; per-season QM aligns with climate downscaling practice |
| **❌ Rejected** | IQR (1.5 × IQR) rule               | Tukey (1977) — exploratory data analysis, boxplot outlier rule; Hamzat (2025) — data reliability flagged as PCM deployment barrier                                                                     | Univariate, non-rolling: flags legitimate heat-wave extremes (43°C) as outliers; no temporal context for sensor spikes vs real events            |
| **❌ Rejected** | KNN spatial imputation             | Ghodusinejad (2026) — kNN cited for solar nowcasting, not gap-fill; Liu (2025) — KNN for NEPCM conductivity prediction                                                                                 | Ignores temporal autocorrelation stronger than spatial correlation for short hourly gaps; fills 3 consecutive hours from identical neighbours    |


**Full citations**: Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. Rubin, D.B. (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley. Buuren, S. van & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations. *J. Stat. Softw.*, 45(3), 1–67.

---

## 04b — `04b_climate_signature.py` — Climate Signature Construction

### Method Chosen: **Two-Tier Signature (Sun-Event + Daily Integral) with PCA Compression**

Tier 1 (sun-event statistics): computes per-point means/percentiles of temperature, GHI, humidity, wind at sunrise/noon/sunset. Tier 2 (daily integrals from `02b`): true GHI integral, HDD, CDD, DTR, cloudy fraction, CCI, SAI, seasonality. PCA compresses the correlated temperature/pressure block into 4 components.

> ✅ **CORRECTED (v3.1)**: Uses `DRAW_VOLUME_L = 300.0` (Avargani et al. 2021). See `20_IMPLEMENTATION_ISSUES.md`.

**Why two-tier signature chosen:**

- Sun-event statistics capture the state of the atmosphere at the moments most relevant to solar collector operation (charging at noon, delivery at sunset, overnight retention at sunrise).
- Daily integrals capture integrated energy quantities that determine sizing (total daily GHI, total heating/cooling degree days) — these cannot be derived from 3-event snapshots without the half-sine approximation error.
- PCA removes collinearity: temperature, dewpoint, and pressure are highly correlated, so including all of them raw inflates their combined weight in the GMM distance metric.

**Alternative 1 rejected: Monthly climatological means only**
Using only 12-month mean temperature and GHI values (as in most traditional PCM selection papers, e.g., Sharma et al. 2017) discards all information about intra-month variability (DTR, CCI, cloudy fraction). This is particularly damaging for Tamil Nadu, where the NE monsoon brings concentrated rainfall in Oct–Dec, producing high month-to-month variability that monthly means smooth out. Rejected as insufficient for climate-adaptive PCM selection.

**Alternative 2 rejected: Full-resolution time series clustering (DTW)**
Dynamic Time Warping (DTW) applied to the full 10-year daily GHI series would cluster points by their temporal curve shape, potentially identifying the out-of-phase monsoon pattern. However, DTW on 3,653-day series requires an O(n² × m²) matrix (n = 133 points, m = 3,653 days), which is computationally prohibitive, and the resulting distance metric is not easily interpretable for the PCM target derivation step. Rejected as computationally intractable and not interpretable.

**Literature comparison**


| Aspect                   | Our pipeline                                | Literature                                                                                                                         | Verdict                                          |
| ------------------------ | ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| Climate features         | Two-tier: sun-event stats + daily integrals | **Singh (2025)**: PCM priority = latent heat, k, Tm — needs integrated energy; Odoi-Yorke (2025): ML inputs = irradiance, Ta, flow | **More complete** than monthly-mean papers       |
| Tm target                | Fixed 57°C (50+7°C indirect)                | **Singh (2025)**: optimal SWH PCM band **40–70°C**; Chen (2025): RT35HC for 30°C delivery target                                   | **Within literature band**, higher delivery temp |
| L_required               | 300 L/day ÷ 50 kg PCM                       | Chen (2025): ≥30 L/day, 20% PCM volume; Chopra (2023): 360 L/day (6×60 L)                                                          | **Aligned** with Indian domestic draw studies    |
| Dimensionality reduction | PCA (4 components)                          | Liu (2025): feature selection via GA/ANN; no PCA in PCM-SWH reviews                                                                | **Standard ML** for collinear climate vars       |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                 | Key Reference(s)                                                                                                                                                                                        | Justification                                                                                                                                           |
| -------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Two-tier signature (sun-event + daily integrals) + PCA | Singh (2025) — PCM priority needs integrated energy (L, k, Tm); Durin et al. (2018) — worst-month sizing; Avargani et al. (2021) — 300 L/day draw; Odoi-Yorke (2025) — ML inputs = irradiance, Ta, flow | Captures monsoon intra-month variability (CCI, DTR, cloudy fraction) that monthly means smooth out; PCA removes T–pressure collinearity in GMM distance |
| **❌ Rejected** | Monthly climatological means only                      | Chen (2025) — Taguchi uses hourly TRNSYS, not 12 monthly averages; Al-Mamun (2023) — SWH reviews note seasonal performance variation                                                                    | Discards NE monsoon Oct–Dec concentration and cloudy-day multi-peak GHI structure critical for PCM charge/discharge sizing                              |
| **❌ Rejected** | DTW on full 10-year daily series                       | Berndt & Clifford (1994) — dynamic time warping; Keogh & Ratanamahatana (2005) — DTW complexity survey                                                                                                  | O(n² × m²) for 133 × 3653 days is prohibitive; distance metric not interpretable for Tm_target derivation; Liu (2025) uses GA feature selection instead |


**Full citations**: Durin, A. et al. (2018). Worst Month and Critical Period methods for sizing solar irrigation. *Solar Energy*, 174, 100–112. Berndt, D.J. & Clifford, J. (1994). Using dynamic time warping to find patterns in time series. *KDD Workshop*, 359–370.

---

## 05 — `05_cluster_tamilnadu.py` — Climate Regime Clustering

### Method Chosen: **Gaussian Mixture Model (GMM) with BIC + Silhouette Selection**

Fits GMM for K = 2..10, selects K by BIC minimum and silhouette score within the acceptable band [0.15, 0.40]. Final fit uses K_FINAL = 5.

> ✅ **CORRECTED (v3.1)**: Uses `covariance_type="diag"`. See `20_IMPLEMENTATION_ISSUES.md`.

**Why GMM chosen over hard clustering:**

- Climate in Tamil Nadu is a continuous gradient — the boundary between the coastal humid belt and the interior semi-arid zone is not a sharp line. Points near that boundary have genuine partial membership in both regimes. GMM's soft-membership probabilities capture this uncertainty explicitly and are passed to Phase 5/6 as boundary-awareness weights.
- BIC penalises over-complex models, preventing the algorithm from discovering spurious micro-clusters in sparse data.
- GMM's probabilistic framework allows the model selection (BIC) to be separated from the clustering (hard labels), enabling a more rigorous model comparison.

**Alternative 1 rejected: K-Means**
K-Means assumes spherical, equally-sized clusters and assigns hard membership. These assumptions are violated in Tamil Nadu climate space: the Nilgiris montane climate is a small cluster in feature space surrounded by a large coastal humid cluster. K-Means would either merge the Nilgiris into the coast cluster or split the coast cluster artificially to compensate. The script actually computes K-Means as a reported comparison baseline (see `kmeans_comparison_tamilnadu.csv`), confirming lower silhouette scores.

**Alternative 2 rejected: Hierarchical Agglomerative Clustering (HAC)**
HAC (Ward linkage) builds a dendrogram and does not require specifying K in advance. However, once the tree is cut, membership is hard (no soft probabilities for boundary points), and HAC does not provide a natural probabilistic model that can be queried for new points. HAC also requires O(n²) memory for the distance matrix, which is manageable for 133 points but does not scale to the eventual 4-state joint clustering (n ≈ 800+). Rejected in favour of a method that scales and provides soft membership.

**Literature comparison**


| Aspect            | Our pipeline                          | Literature                                                                                                  | Verdict                                                             |
| ----------------- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Clustering method | GMM, soft membership, BIC K-selection | **Liu (2025)**: GA/PSO for PCM structure, not climate zones; Odoi-Yorke (2025): no GMM in 245 AI-SWH papers | **Novel (N1)** — discovered regimes                                 |
| K selection       | BIC + silhouette, K=5                 | Chen (2025): Taguchi L36 (design factors, not climate clusters)                                             | **Different problem** — we cluster climate, Chen optimises geometry |
| Covariance        | Diagonal (v3.1)                       | McLachlan & Peel (2000)*: full vs diag trade-off for small n                                                | **Corrected** — literature recommends parsimony for n=133           |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                            | Key Reference(s)                                                                                                                                       | Justification                                                                                                                                                                         |
| -------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | GMM, K=2..10, BIC + silhouette, `diag` covariance | McLachlan & Peel (2000) — finite mixture models; Fraley & Raftery (2007) — mclust BIC model selection; Odoi-Yorke (2025) — no GMM in 245 AI-SWH papers | Soft membership captures coastal/interior gradient; BIC penalises over-fitting; diagonal covariance recommended for n=133 (v3.1 correction)                                           |
| **❌ Rejected** | K-Means hard clustering                           | MacQueen (1967) — K-Means; Jain (2010) — clustering review; Arthur & Vassilvitskii (2007) — K-Means++                                                  | Assumes spherical equal-variance clusters — violated by small Nilgiris montane pocket vs large coastal humid cluster; confirmed lower silhouette in `kmeans_comparison_tamilnadu.csv` |
| **❌ Rejected** | Hierarchical Agglomerative (Ward linkage)         | Ward (1963) — minimum variance method; Murtagh & Legendre (2014) — hierarchical clustering review                                                      | Hard labels only (no boundary probabilities); O(n²) memory does not scale to 4-state joint clustering (~800+ points); no generative model for new points                              |


**Full citations**: MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proc. 5th Berkeley Symp.*, 281–297. Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. *JASA*, 58(301), 236–244. Fraley, C. & Raftery, A.E. (2007). Model-based clustering, discriminant analysis, and density estimation. *JASA*, 97(458), 611–631.

---

## 06 — `06_build_pcm_database.py` — PCM Database Construction

### Method Chosen: **Random Forest PMM-like Imputation for Missing Manufacturer Properties**

Builds the current 62-PCM database (55 manufacturer-derived records from the MICE+RF+PMM detailed input + 7 from peer-reviewed literature). Manufacturer property gaps are completed by the upstream MICE+RF+PMM workflow and carried into this database with imputation flags and provenance. Literature rows retain genuinely unreported density, specific heat, conductivity, and cycling values as missing rather than inventing them.

**Why RF imputation chosen:**

- PCM properties are non-linearly correlated (e.g., thermal conductivity correlates with density, which correlates with phase — solid vs. liquid). Random Forest captures these non-linear relationships, unlike linear regression imputation.
- PMM-like approach (predicting the value, then substituting the nearest observed value rather than the raw prediction) avoids imputed values outside the physically plausible range.

**Alternative 1 rejected: Mean/median imputation**
Filling missing properties with column means would destroy the correlation between density and thermal conductivity that exists across PCM families (paraffins have lower density and conductivity than fatty acids, which have lower density and conductivity than salt hydrates). Mean imputation would produce physically implausible combinations. Rejected.

**Alternative 2 rejected: MICE (multivariate chain)**
MICE is the gold standard for tabular imputation and is used in the upstream PCM preprocessing workflow together with Random Forest and Predictive Mean Matching. The current database builder consumes that detailed completed file and preserves its per-property imputation flags; it does not impute the seven literature rows because their missing values have no defensible donor pool.

**Literature comparison**


| Aspect              | Our pipeline                             | Literature                                                                                               | Verdict                                            |
| ------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Database size       | 62 PCMs (55 manufacturer-derived + 7 literature) | **Singh (2025)**: Table 2 lists 200+ cited PCMs; **Martinez (2025)**: Rubitherm industrial measured data | **Auditable expanded subset; further material families remain optional** |
| Property imputation | RF + PMM-like 3-donor blend              | **Liu (2025)**: ANN/XGBoost predict Tm, L with R²≈0.99; **Eldokaishi (2022)**: ANN for SWH               | **Conservative** — we impute gaps, not invent PCMs |
| Tm band             | 42–70°C SWH-specific                     | **Singh (2025)**: **40–70°C** optimal organic PCM band                                                   | **Aligned**                                        |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                             | Key Reference(s)                                                                                                                                     | Justification                                                                                                                                             |
| -------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | RF + PMM-like 3-donor imputation for property gaps | Liu (2025) — RF/XGBoost for PCM property prediction (R²≈0.99); Martinez (2025) — Rubitherm measured industrial data; Breiman (2001) — Random Forests | Captures non-linear density–conductivity–phase correlations; PMM substitution keeps imputed values within observed physical range                         |
| **❌ Rejected** | Column mean/median imputation                      | Little & Rubin (2019) — *Statistical Analysis with Missing Data*; Hamzat (2025) — inconsistent PCM property reporting                                | Destroys family-level correlations (paraffins vs fatty acids vs salt hydrates) → physically implausible density–k combinations                            |
| **❌ Rejected** | MICE multivariate chain (as in Phase 2 weather QC) | Buuren & Groothuis-Oudshoorn (2011) — MICE software; Rubin (1987) — multiple imputation                                                              | Gold standard for large tabular data but unstable with n=18 complete manufacturer rows (< 30 threshold); RF-PMM is single-pass stable for small databases |


**Full citations**: Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32. Little, R.J.A. & Rubin, D.B. (2019). *Statistical Analysis with Missing Data*, 3rd ed. Wiley.

---

## 07 — `07_feasibility_filter.py` — PCM Feasibility Screening

### Method Chosen: **8-Filter Hard-Screen with Step-Down κ-Relaxation**

Applies 8 sequential hard filters (melting window, absolute band, latent heat floor, cycling, supercooling, corrosion, safety, charging). If fewer than 5 candidates survive, the melting window is relaxed by 2K per step (up to 4 steps) — see Section 8's κ-calibration rule in the framework plan.

> ✅ **CORRECTED (v3.1)**: `L_required` now uses 300 L/day draw — latent-heat floor is binding. See `20_IMPLEMENTATION_ISSUES.md`.

**Why hard-screen before MCDM chosen:**

- MCDM methods (TOPSIS, GRA, PROMETHEE II, VIKOR) are compensatory — a PCM with a melting point 15°C below target but excellent latent heat can score well in TOPSIS, even though it cannot physically absorb heat at the right temperature. Hard-screening eliminates this false positive before ranking.
- The 8 filters directly implement the framework plan's Table 12. Deviating from them would require a plan amendment.

**Alternative 1 rejected: Soft-penalty scoring instead of hard-screens**
A penalty function could reduce a PCM's score for being outside the melting window rather than excluding it. This allows compensatory trade-offs (e.g., great latent heat compensates for off-target Tm) that are physically unrealisable. Rejected because a PCM that doesn't melt at the right temperature provides zero latent heat storage, regardless of its rated capacity.

**Alternative 2 rejected: Fuzzy logic membership for melting window**
Fuzzy membership could give partial credit to PCMs near the melting window boundary. This is a valid approach for uncertainty quantification but adds complexity that is not required by the framework plan. The κ-relaxation mechanism already handles the boundary case in a documented, step-wise fashion. Rejected in favour of the plan-compliant step-down approach.

**Literature comparison**


| Aspect               | Our pipeline                                    | Literature                                                                         | Verdict                                             |
| -------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------- | --------------------------------------------------- |
| Selection priority   | Hard screen then rank                           | **Singh (2025)**: (1) L, (2) k, (3) Tm, (4) cp, (5) ρ                              | **Aligned order** — we enforce L and Tm before MCDM |
| Melting window       | [Tm_target−5, Tm_target+8]°C                    | **Singh (2025)**: 40–70°C band; **Abdellatif (2025)**: modeling constraints review | **SWH-specific**, narrower than building PCM        |
| Corrosion veto       | HSI > p75 → exclude salt hydrates               | **Hamzat (2025)**: environmental/corrosion as deployment barrier                   | **Aligned** — humid coastal TN clusters             |
| Charging feasibility | Heuristic in `07b` (5th-percentile not literal) | Barqawi (2025): charging under variable irradiance                                 | **Partial gap** — physics validation supersedes     |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                        | Key Reference(s)                                                                                                                                                       | Justification                                                                                                                                         |
| -------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | 8-filter hard screen + κ-relaxation step-down | Singh (2025) — selection priority: (1) L, (2) k, (3) Tm; Abdellatif (2025) — PCM modeling constraints; Hamzat (2025) — corrosion/deployment barriers in humid climates | MCDM methods are compensatory — hard screen prevents high-L PCM with wrong Tm from ranking; κ-relaxation handles boundary cases per framework plan §8 |
| **❌ Rejected** | Soft-penalty scoring (no exclusion)           | Chen (2025) — Taguchi discrete factor levels enforce feasibility bands; Singh (2025) — Tm band 40–70°C as hard organic PCM constraint                                  | PCM outside melting window provides zero latent heat regardless of rated L — soft penalty allows physically unrealisable top ranks                    |
| **❌ Rejected** | Fuzzy logic melting-window membership         | Zadeh (1965) — fuzzy sets; Odoi-Yorke (2025) — fuzzy logic surveyed for SWH *control*, not material screening                                                          | Valid for control uncertainty but adds complexity not required by framework plan; κ-relaxation already documents boundary handling step-wise          |


**Full citations**: Zadeh, L.A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353. Abdellatif, M. (2025). PCM modeling review. See `sources/Abdellatif2025PCM_Modeling_Review_summary.md`.

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

**Literature comparison**


| Aspect       | Our pipeline                                      | Literature                                                              | Verdict                                              |
| ------------ | ------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- |
| MCDM methods | TOPSIS + GRA + PROMETHEE II + VIKOR               | **Chen (2025)**: Taguchi + **GRA only** (single multi-objective method) | **Strong improvement (N4)** — 4 methods vs 1         |
| Consensus    | Borda + Copeland cross-check                      | Chen (2025): GRG maximisation; no cross-method check                    | **More rigorous**                                    |
| Uncertainty  | 5000-draw MC (Dirichlet weights + Gaussian props) | **Chopra (2023)**: Monte Carlo for LCWH/NPV/PP economics                | **Same technique**, applied to ranking not economics |
| Tm criterion | Gaussian fitness f_Tm                             | Chen (2025): discrete factor levels in Taguchi                          | **More appropriate** for continuous target matching  |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                     | Key Reference(s)                                                                                                                                                                                          | Justification                                                                                                                                                           |
| -------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | TOPSIS + GRA + PROMETHEE II + VIKOR + Borda + 5000-draw MC | Hwang & Yoon (1981) — TOPSIS; Deng (1982) — GRA; Brans & Mareschal (2005) — PROMETHEE II; Opricovic (2004) — VIKOR; Chen (2025) — GRA-only baseline; Chopra (2023) — Monte Carlo for Indian SWH economics | Four methodologically distinct schools; agreement = robust rank; MC propagates weight/property uncertainty (Chopra precedent for Indian SWH uncertainty)                |
| **❌ Rejected** | AHP alone (+ weighted sum)                                 | Saaty (1980) — Analytic Hierarchy Process; Chen (2025) — Taguchi derives weights implicitly via S/N ratios                                                                                                | AHP elicits weights, not rankings; weighted sum is maximally compensatory — allows wrong-Tm PCM to outrank correct-Tm candidate                                         |
| **❌ Rejected** | ELECTRE III outranking                                     | Roy (1996) — ELECTRE methods; Figueira et al. (2005) — MCDM state-of-art survey                                                                                                                           | Produces partial preorder requiring concordance/discordance thresholds hard to elicit without expert panel; not Monte-Carlo-compatible for 5000-draw stability analysis |


**Full citations**: Saaty, T.L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill. Deng, J.L. (1982). Control problems of grey systems. *Syst. Control Lett.*, 1(5), 288–294. Opricovic, S. & Tzeng, G.H. (2004). Compromise solution by MCDM methods: VIKOR. *Eur. J. Oper. Res.*, 156(2), 445–455. Roy, B. (1996). *Multicriteria Methodology for Decision Aiding*. Kluwer.

---

## 10 — `10_physics_validation.py` — Grey-Box Tank Simulation

### Method Chosen: **Lumped-Enthalpy 3-Phase Tank Model (Backward Euler, Hourly)**

Implements a two-node lumped model (tank water Tw + PCM node Tp/melt-fraction f) coupled by a coil heat exchanger, solved with backward (implicit) Euler at hourly time steps over a full representative year of real daily weather for each cluster's medoid point.

> ✅ **CORRECTED (v3.1)**: Ambient tank heat loss `UA_TANK_W_K = 2.0 W/K` active. See `20_IMPLEMENTATION_ISSUES.md`.

**Why backward Euler chosen:**

- The tank water–coil–PCM system has a fast time constant (~minutes for the coil coupling), far shorter than the hourly time step. Forward (explicit) Euler is conditionally stable only when `dt < 2τ`; with τ ≈ 3–5 minutes and dt = 3,600 s, explicit Euler would be wildly unstable. Backward Euler is unconditionally stable for this linear system and requires only a single 2×2 matrix solve per step (closed-form, no iterative solver needed).
- Forward Euler or Runge-Kutta methods are appropriate only when the time step is smaller than the fastest time constant — they are not appropriate here.

**Alternative 1 rejected: EnergyPlus simulation**
EnergyPlus is the U.S. DOE's whole-building energy simulation engine. It does not natively support latent-heat PCM inside a domestic hot water tank node network without a custom EnergyPlusFMU or EMS scripting. The framework plan (Section 10.1) explicitly rejects EnergyPlus for this reason. It would also require a full building geometry input file unrelated to the PCM storage problem. Rejected as specified in plan v3.0.

**Alternative 2 rejected: TRNSYS Type 840 (PCM storage)**
TRNSYS is a transient energy simulation package with a specialised PCM storage component (Type 840). It handles the three-phase enthalpy formulation natively. However, TRNSYS requires a commercial licence (several thousand USD), is not available in the project's computing environment, and cannot be called programmatically from Python for the 5-cluster × 7-PCM × 365-day batch run required here. Rejected due to licensing and integration constraints.

**Literature comparison**


| Aspect            | Our pipeline                                    | Literature                                                                                              | Verdict                                                                   |
| ----------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Model structure   | 3-phase lumped enthalpy (solid → melt → liquid) | **Barqawi (2025)**: identical 3-phase ODE structure (Eqs. 1–16)                                         | **Directly aligned**                                                      |
| Numerical solver  | Backward Euler, hourly dt                       | **Barqawi (2025)**: RK45 via SciPy `solve_ivp`, dt=100 s                                                | **Different solver**, same physics — Euler chosen for hourly climate data |
| Climate input     | Real 10-year daily GHI/Ta from medoid point     | **Barqawi (2025)**: synthetic sinusoidal GHI (Eq. 4); **Chen (2025)**: TRNSYS validated <5% vs physical | **Improvement** — real weather not synthetic                              |
| Heat loss         | Tank UA = 2.0 W/K (v3.1)                        | Barqawi (2025): ambient loss **neglected** (stated assumption)                                          | **Our fix adds realism** Barqawi omitted                                  |
| Validation metric | Spearman ρ vs MCDM rank                         | Barqawi (2025): ML vs fixed-speed % energy gain; Singh (2025): efficiency 54–84% band                   | **Falsifiable rank check (N5)** — not self-referential                    |
| ML control        | Not in Phase 7 (Phase 9+ future)                | Barqawi (2025): FNN flow multiplier; Liu (2025): DRL for LHS                                            | **Scope boundary** — Phase 7 is validation only                           |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                | Key Reference(s)                                                                                                                                                   | Justification                                                                                                                                       |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Lumped-enthalpy 3-phase model, backward Euler, hourly | Barqawi (2025) — identical 3-phase ODE structure (Eqs. 1–16); Chen (2025) — TRNSYS Type validation <5% error; Duffie & Beckman (2020) — lumped tank energy balance | Backward Euler unconditionally stable for coil τ ≈ 3–5 min with dt = 3600 s; real 10-year weather vs Barqawi's synthetic sinusoidal GHI             |
| **❌ Rejected** | EnergyPlus whole-building simulation                  | Crawley et al. (2001) — EnergyPlus; DOE (2023) — no native PCM-in-DHW-tank component                                                                               | Requires custom FMU/EMS scripting; full building geometry unrelated to PCM storage problem; explicitly rejected in framework plan §10.1             |
| **❌ Rejected** | TRNSYS Type 840 PCM storage                           | Klein (2013) — TRNSYS manual; Chen (2025) — primary optimisation platform; Odoi-Yorke (2025) — TRNSYS–ANN hybrid cluster (245 papers)                              | Commercial licence ($1000s); no programmatic Python batch interface for 5 clusters × 7 PCMs × 365 days; Barqawi achieves same physics in open SciPy |


**Full citations**: Crawley, D.B. et al. (2001). EnergyPlus: creating a new-generation building energy simulation program. *Energy Build.*, 33(4), 319–331. Klein, S.A. et al. (2013). *TRNSYS 17: A Transient System Simulation Program*. Univ. of Wisconsin–Madison.

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

**Literature comparison**


| Aspect  | Our pipeline                                   | Literature                                                                   | Verdict                   |
| ------- | ---------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------- |
| Output  | Markdown cards per cluster                     | Chen (2025): TRNSYS tables; Odoi-Yorke (2025): reproducibility gap in AI-SWH | **Reproducible artifact** |
| Content | Top-3, 4 MCDM scores, MC stability, Spearman ρ | Chen (2025): optimal factor levels only; no cross-validation                 | **Richer reporting**      |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                       | Key Reference(s)                                                                                                                   | Justification                                                                                                                 |
| -------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Markdown report via Python string formatting | Odoi-Yorke (2025) — reproducibility gap in AI-SWH literature; Wilson et al. (2017) — good enough practices in scientific computing | Version-controllable, GitHub-renderable, paste-ready for thesis appendix; numbers always consistent with current pipeline run |
| **❌ Rejected** | Jupyter notebook report                      | Kluyver et al. (2016) — Jupyter ecosystem; Rule et al. (2019) — Ten Simple Rules for reproducible notebooks                        | Requires running kernel; poor git diff; cannot be invoked as headless batch pipeline step                                     |
| **❌ Rejected** | LaTeX/PDF generation                         | Lamport (1994) — LaTeX; Chen (2025) — TRNSYS tabular output, not typeset reports                                                   | Requires full TeX installation; over-engineered for framework plan's "quick-reference recommendation cards" deliverable       |


**Full citations**: Wilson, G. et al. (2017). Good enough practices in scientific computing. *PLoS Comput. Biol.*, 13(6), e1005510. Rule, A. et al. (2019). Ten simple rules for writing and sharing computational analyses. *PLoS Comput. Biol.*, 15(7), e1007007.

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

**Literature comparison**


| Aspect         | Our pipeline                             | Literature                                                                                                          | Verdict                                                  |
| -------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| Seasonal check | Recompute L_required per season + TOPSIS | Chen (2025): single annual TRNSYS optimisation; Singh (2025): monsoon intermittency noted but not seasonal PCM rank | **Novel finding potential** — NE monsoon flip test       |
| Method depth   | Single-method TOPSIS (sensitivity only)  | Phase 6 uses full 4-method stack                                                                                    | **Appropriate scope** — "nearly free" per framework plan |
| Draw volume    | 300 L/day (v3.1, matches 04b)            | Chen (2025): 30 L/day minimum stated                                                                                | **Consistent** with Phase 3 signature                    |


### Reference Papers — Chosen vs Rejected


| Status         | Method                                                   | Key Reference(s)                                                                                                                        | Justification                                                                                                                                      |
| -------------- | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **✅ Chosen**   | Seasonal L_required recomputation + single-method TOPSIS | Singh (2025) — monsoon intermittency noted; Chen (2025) — single annual TRNSYS run; Hwang & Yoon (1981) — TOPSIS most cited in PCM-MCDM | Physically motivated: demand temperature and ambient change by season; TOPSIS sufficient to detect #1 PCM flip without full 4-method MC cost       |
| **❌ Rejected** | Full 4-method stack + 5000-draw MC per season            | Chen (2025) — 36 TRNSYS runs for L36 DOE; Chopra (2023) — MC for economics, not seasonal PCM rank                                       | 4 seasons × 5 clusters × 5000 draws ≈ 20 min + 20 result files; marginal value over TOPSIS flip test is low per framework plan "nearly free" scope |
| **❌ Rejected** | MCDM weight perturbation (no L_required recomputation)   | Mansouri (2025) — ML bias correction adjusts model output, not physical sizing inputs                                                   | Not physically motivated — L_required genuinely changes with season; perturbing weights masks whether monsoon demand shift changes optimal PCM     |


**Full citations**: See Chen (2025) and Singh (2025) in `sources/`; Hwang & Yoon (1981) — TOPSIS.

---

## References

### A. Project `sources/` folder (PCM-SWH and climate literature summaries)


| #   | Citation                                                         | Role in pipeline                                                   | Source file                                             |
| --- | ---------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------- |
| 1   | Singh et al. (2025) — PCM-SWH comprehensive review               | Tm band 40–70°C, selection priority L→k→Tm, sizing context         | `Singh2025PCM_SWH_ComprehensiveReview_summary.md`       |
| 2   | Chen et al. (2025) — Taguchi + GRA PCM-nanofluid SWH             | TRNSYS baseline, GRA-only MCDM comparison, rejected as sole method | `Chen2025TaguchiGRA_PCM_Nanofluid_SWH_summary.md`       |
| 3   | Barqawi (2025) — 3-phase PCM tank dynamic simulation             | Grey-box model structure aligned; synthetic GHI rejected           | `Barqawi2025DynamicSimulationPCM_SWH_summary.md`        |
| 4   | Ghodusinejad et al. (2026) — Solar irradiance forecasting review | MBE/RMSE/Xcor validation; kNN/ML cited for rejected alternatives   | `Ghodusinejad2026SolarIrradianceForecasting_summary.md` |
| 5   | Liu et al. (2025) — AI for PCM TES prediction                    | RF/XGBoost property prediction; GA/PSO not used for clustering     | `Liu2025AI_PCM_TES_Prediction_Optimization_summary.md`  |
| 6   | Odoi-Yorke (2025) — AI for SWH systems review                    | Single-site bias; TRNSYS cluster; reproducibility gap              | `OdoiYorke2025AI_SWH_Review_summary.md`                 |
| 7   | Mansouri et al. (2025) — Multimodal renewable forecasting        | Bias correction precedent; rejected ML blend vs QM                 | `Mansouri2025MultimodalRenewableForecasting_summary.md` |
| 8   | Chopra et al. (2023) — Monte Carlo techno-economic HPETC         | MC uncertainty; zone-level sampling rejected                       | `Chopra2023HPETC_MonteCarlo_TechnoEconomic_summary.md`  |
| 9   | Martinez (2025) — Industrial PCM measured properties             | Rubitherm datasheet baseline                                       | `Martinez2025PCM_Industrial_TES_summary.md`             |
| 10  | Hamzat et al. (2025) — PCM solar storage challenges              | Data reliability; corrosion deployment barrier                     | `Hamzat2025PCM_SolarEnergyStorage_summary.md`           |
| 11  | Abdellatif (2025) — PCM modeling review                          | Melting band, cycling, supercooling constraints                    | `Abdellatif2025PCM_Modeling_Review_summary.md`          |
| 12  | Al-Mamun (2023) — SWH state-of-art                               | Site-specific design pattern (rejected uniform grid)               | `AlMamun2023SWH_StateOfArt_summary.md`                  |
| 13  | Eldokaishi (2022) — ANN for water-PCM SWH                        | ANN imputation cited as rejected for weather gaps                  | `Eldokaishi2022WaterPCM_ANN_SWH_summary.md`             |


### B. Chosen-method standard references


| #   | Citation                                                                              | Used in script(s) |
| --- | ------------------------------------------------------------------------------------- | ----------------- |
| 14  | Reda & Andreas (2004). Solar position algorithm. *Solar Energy*, 76(5), 577–589.      | 00b               |
| 15  | Hersbach et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146(730), 1999–2049.     | 01                |
| 16  | Stackhouse et al. (2021). NASA POWER methodology. NASA Langley.                       | 01b               |
| 17  | Ineichen & Perez (2002). Linke turbidity formulation. *Solar Energy*, 73(3), 151–157. | 02                |
| 18  | Themeßl et al. (2012). Empirical-statistical downscaling / QM. *Int. J. Climatol.*    | 02, 04            |
| 19  | Hampel (1974). Robust MAD estimation. *JASA*.                                         | 04                |
| 20  | Rubin (1987). *Multiple Imputation for Nonresponse in Surveys*. Wiley.                | 04, 06            |
| 21  | Avargani et al. (2021). SWH with PCM, 300 L/day. *J. Energy Storage*, 42, 103021.     | 04b               |
| 22  | Durin et al. (2018). Worst-month sizing. *Solar Energy*, 174, 100–112.                | 04b               |
| 23  | McLachlan & Peel (2000). *Finite Mixture Models*. Wiley.                              | 05                |
| 24  | Breiman (2001). Random forests. *Machine Learning*, 45(1), 5–32.                      | 06                |
| 25  | Hwang & Yoon (1981). *Multiple Attribute Decision Making*. Springer — TOPSIS.         | 08, 11            |
| 26  | Deng (1982). Grey relational analysis. *Systems & Control Letters*, 1(5), 288–294.    | 08                |
| 27  | Brans & Mareschal (2005). PROMETHEE methods.                                          | 08                |
| 28  | Opricovic & Tzeng (2004). VIKOR compromise ranking. *EJOR*, 156(2), 445–455.          | 08                |
| 29  | Duffie & Beckman (2020). *Solar Engineering of Thermal Processes*, 5th ed.            | 02b, 10           |
| 30  | Holmgren et al. (2018). pvlib python. *JORS*, 6(1).                                   | 00b, 03           |


### C. Rejected-alternative references (with justification category)


| #   | Citation                                                                              | Rejected method                 | Why rejected (literature basis)                                | Script |
| --- | ------------------------------------------------------------------------------------- | ------------------------------- | -------------------------------------------------------------- | ------ |
| 31  | Stevens et al. (2015). WorldPop random-forest dasymetric mapping. *PLoS ONE*.         | — (supports chosen)             | Population disaggregation standard                             | 00a    |
| 32  | MacQueen (1967). K-Means. *Proc. Berkeley Symp.*                                      | K-Means clustering              | Spherical-cluster assumption; lower silhouette vs GMM          | 05     |
| 33  | Ward (1963). Hierarchical grouping. *JASA*.                                           | HAC/Ward linkage                | Hard labels; O(n²) memory; no soft membership                  | 05     |
| 34  | Berndt & Clifford (1994). Dynamic time warping. *KDD Workshop*.                       | DTW time-series clustering      | O(n²m²) cost; non-interpretable for Tm derivation              | 04b    |
| 35  | Bland & Altman (1986). Limits of agreement. *Lancet*.                                 | Bland-Altman analysis           | Designed for symmetric devices; ERA5/POWER asymmetric          | 03     |
| 36  | Massey (1951). KS test. *JASA*.                                                       | Kolmogorov-Smirnov test         | No bias/scatter decomposition for QM gate                      | 03     |
| 37  | Tukey (1977). *Exploratory Data Analysis*.                                            | IQR outlier rule                | Flags legitimate climate extremes; no temporal window          | 04     |
| 38  | Huld et al. (2012). PVGIS solar database. *Prog. Photovolt.*                          | PVGIS TMY                       | Synthetic single year; no inter-annual variability             | 01b    |
| 39  | Gorelick et al. (2017). Google Earth Engine. *RSE*, 202, 18–27.                       | GEE ERA5-Land                   | Schema mismatch; quota limits                                  | 01     |
| 40  | Rigollier et al. (2000). REST2 clear-sky. *Solar Energy*, 68(1), 71–80.               | REST2 model                     | Requires atmospheric chemistry inputs not in ERA5 download     | 02     |
| 41  | Ineichen (2008). Simplified Solis.                                                    | Simplified Solis                | Requires precipitable water input unavailable in pipeline      | 02     |
| 42  | Pfeifroth et al. (2019). SARAH-3. *ESSD*, 11(4), 1929–1946.                           | SARAH-3 daily product           | EUMETSAT access; incomplete India 2016–2017                    | 02b    |
| 43  | Iqbal (1983). *An Introduction to Solar Radiation*.                                   | Half-sine GHI proxy             | Multi-peaked cloudy-day error                                  | 02b    |
| 44  | Saaty (1980). *Analytic Hierarchy Process*.                                           | AHP + weighted sum              | Weight elicitation only; maximally compensatory ranking        | 08     |
| 45  | Roy (1996). *Multicriteria Methodology for Decision Aiding*.                          | ELECTRE III                     | Partial preorder; threshold elicitation; no MC integration     | 08     |
| 46  | Zadeh (1965). Fuzzy sets. *Information and Control*.                                  | Fuzzy melting-window membership | Control-domain method; κ-relaxation already handles boundaries | 07     |
| 47  | Crawley et al. (2001). EnergyPlus. *Energy Build.*                                    | EnergyPlus simulation           | No native PCM-DHW; plan §10.1 rejection                        | 10     |
| 48  | Klein et al. (2013). *TRNSYS 17* manual.                                              | TRNSYS Type 840                 | Commercial licence; no Python batch API                        | 10     |
| 49  | Spencer (1971). Fourier solar position. *Search*, 2(5), 172.                          | Simplified astronomy            | ±3–15 min error vs SPA                                         | 00b    |
| 50  | Cooper (1969). Solar still absorption. *Solar Energy*, 12(3), 333–346.                | Cooper declination formula      | Same accuracy class as Spencer; inferior to SPA                | 00b    |
| 51  | Little & Rubin (2019). *Statistical Analysis with Missing Data*, 3rd ed.              | Mean/median imputation          | Destroys multivariate PCM property correlations                | 06     |
| 52  | Buuren & Groothuis-Oudshoorn (2011). MICE. *J. Stat. Softw.*                          | MICE on PCM database (n=18)     | Unstable below n≈30; RF-PMM preferred                          | 06     |
| 53  | Kluyver et al. (2016). Jupyter notebooks.                                             | Jupyter report                  | Non-reproducible batch artifact                                | 09     |
| 54  | Rule et al. (2019). Ten simple rules for computational analyses. *PLoS Comput. Biol.* | Notebook-based reporting        | Poor version control vs Markdown                               | 09     |


---

*This document covers all 13 primary scripts in the Tamil Nadu pipeline (Phase 1 through Phase 8). All algorithmic choices are made in conformance with `Objective1_PCM_Climate_Framework_Plan_v3.docx`. Critical bugs corrected in v3.1 (August 2026). Use **Part 0** for supervisor presentation; use per-script **Literature comparison** and **Reference Papers — Chosen vs Rejected** tables for thesis methodology section.*