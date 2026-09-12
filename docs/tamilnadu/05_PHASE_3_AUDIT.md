# 05 — Phase 3 Audit: Climate Signature Construction

Scripts: `signature_lib.py`, `04b_climate_signature.py`.

**Unified with Rajasthan (2026-09-08).** Tamil Nadu's Phase 3 is now a
state-parameterised mirror of `era5-rajasthan/04b_climate_signature.py`:
same shared Tier-1 code (`signature_lib.build_tier1_signature`), same curated
Tier-1 column list, same five sun-event interaction terms, same 8-column PCA
block, same derived-quantity formulas, and — verified against the actual
output CSVs — the **same 88-column output schema** (names, order, dtypes) as
`climate_signature_rajasthan.csv`. The only state-specific piece is the input
plumbing (Tamil Nadu's Tier-2 table uses `_true`/`_mean`-suffixed names that
are renamed to canonical on load; `monsoon_index`, absent from that table, is
computed here as the same Jun–Sep GHI-fraction proxy Rajasthan's `02b` uses).

## Purpose
Collapse each point's 10-year hourly and daily weather data into a single, physically grounded climate signature vector. This vector defines the climatology of each location, maps meteorological stress directly to PCM performance requirements, and computes climate-adaptive PCM thermal targets (`Tm_target`, `L_required`).

---

## Processing Details

### 1. Two-Tier Signature Design
- **Tier 1 (Sun-Event Statistics, `signature_lib.build_tier1_signature()`)**: One shared implementation, called here with `group_keys=["point_id"]` (whole-year, one row per point) and by the seasonal analysis with `group_keys=["point_id", "season"]`. Curated columns (identical to Rajasthan): `T_sunrise_mean/p05`, `T_noon_mean`, `T_sunset_mean/p95`, `diurnal_gradient` (noon−sunrise, an acknowledged underestimate of true DTR), `kt_noon_mean/std`, `GHI_noon_mean`, `GHI_sunset_mean`, `RH_sunrise_mean`, `wind_noon_mean/sunset_mean`, `HSI_sunrise` (Thom's (1959) Discomfort Index), `Ta_mean/p95/p05` (daily-collapsed first), `daylength_mean`, `daylength_amplitude`. The former blanket per-event mean/std/p5/p95 computation is removed.
- **Tier 2 (Daily-Integral Merge)**: True daily integrals from `02b_build_daily_aggregates.py` (`tier2_signature_tamilnadu.csv`), renamed to the canonical names Rajasthan's `daily_aggregates_rajasthan_summary.csv` uses: `GHI_daily_kWh`, `SAI`, `kt_daily_mean/std`, `cloudy_frac`, `CCI`, `HDD18`, `CDD24`, `DTR_true`, `seasonality`. `monsoon_index` is computed in `04b` (Jun–Sep noon GHI / annual noon GHI) — a GHI-fraction proxy, NOT precipitation-based; 3×/day ERA5 sample, coarser than Rajasthan's full-hourly integral. State this as a limitation.

### 2. Climate Feature → PCM Property Mapping
**Design principle:** Every feature must answer "which PCM property does this constrain, and by what physical mechanism?" If that sentence cannot be completed, the feature is removed. Table below is identical to Rajasthan's `05_PHASE_3_AUDIT.md`.

| Feature Group | Represents | PCM Constraint | Target Property |
|---|---|---|---|
| `T_sunrise_mean, RH_sunrise_mean` + `HSI_sunrise` | Pre-dawn condensation risk at storage surface | Corrosion resistance req. | Feeds Phase 5 corrosion veto |
| `T_noon_mean, GHI_noon_mean, kt_noon/std` | Charging-window heat availability & reliability | Melting-window achievability, charging feasibility | `Tm_target_capped_C` (Phase 5 constraint) |
| `T_sunset_mean, wind_sunset_mean` | Evening heat-loss potential during discharge onset | Discharge-window thermal-loss sensitivity | `int_wind_x_TsunsetMinusTdelivery` interaction term |
| `diurnal_gradient, DTR_true` | Daily thermal swing magnitude (Tier 1 underestimates true swing, Tier 2 captures real) | Cycling stress on PCM | `int_DTR_x_cloudyfrac` + Phase 5 cycles constraint |
| `GHI_daily_kWh, kt_daily_mean/std, SAI, CCI` | Total charging energy & day-to-day reliability | Latent-heat sizing & autonomy req. | `L_required_kJ_per_kg` + `int_CCI_x_1minusSAI` interaction |
| `HDD18, CDD24` | Seasonal thermal-load context (degree-days, base 18°C/24°C) | Indirect: feeds PCA temperature block, informs regime characterization | Phase 4 clustering |
| `cloudy_frac, seasonality, monsoon_index` | Charging intermittency & seasonal variability | Cycling stress under intermittent charging | `int_DTR_x_cloudyfrac` + `int_GHI_x_ktstd` interactions |
| `elevation_m` (PCA block only, not standalone) | Atmospheric/airmass context; NE-monsoon vs. hill-station relief within TN | Already baked into pvlib solar-geometry upstream | Indirectly informs regime separation via PC*_z scores |
| `daylength_mean, daylength_amplitude` | Seasonal charging-window-length variation | Charging duration context | Flagged as possibly climatically-tautological (deterministic from latitude/day-of-year) |

**Removed from the former Tamil Nadu list** (not carried over silently): the blanket `RH_mean`/`wind_mean` daily means and the pressure-ratio `elev_proxy` are replaced by the curated sun-event columns above plus real `elevation_m`; the bespoke `HSI = RH_mean·(T_dep<3).mean()` is replaced by Thom's Discomfort Index from `signature_lib`.

### 3. Five Compound Interaction Terms (identical to Rajasthan — sun-event based)
- `int_GHI_x_ktstd` = `GHI_daily_kWh × kt_daily_std` — erratic-but-large resource.
- `int_DTR_x_cloudyfrac` = `DTR_true × cloudy_frac` — cycling stress under intermittency.
- `int_RH_x_TsunriseMinusTm` = `RH_sunrise_mean × (T_sunrise_mean − Tm_target_C)` — condensation risk at the store surface at the condensation-critical instant. **Replaces the former `int_RH_x_TaMinusTm`.**
- `int_wind_x_TsunsetMinusTdelivery` = `wind_sunset_mean × (T_sunset_mean − T_delivery)` — evening convective loss. **Replaces the former `int_wind_x_TaMinusTsoil`** (that term used an undefined `T_soil` proxy — removed, not reintroduced).
- `int_CCI_x_1minusSAI` = `CCI × (1 − SAI)` — combined autonomy requirement.

### 4. Derived Targets & Sizing Methodology
- **Melting Point Target**: $T_{m,\text{target}} = T_{\text{delivery}} + \Delta T_{\text{approach}} = 50.0 + 7.0 = 57.0^\circ\text{C}$, constant across all 133 points. `T_DELIVERY_C` / `DT_APPROACH_C` / `TM_TARGET_C` are imported from the shared `pcm_shared_config.py` (via `config.py`), not hardcoded per state.
- **`Tm_target_capped_C`**: worst-MONTH clearness cap (lowest of the 12 calendar-month mean `kt_daily` values), Hottel–Whillier-style linear collapse toward `Ta_mean` (Durin et al. 2018 worst-month sizing basis). Completed-run range **48.4–54.0 °C** (all 133 points capped below the 57 °C base target). The old single-day `kt_p05` value is retained as `Tm_target_capped_C_p05day` for audit only; nothing downstream reads it.
- **Latent Heat Target ($L_{\text{required}}$)**:
  $$L_{\text{required}} = \frac{\text{SHARE\_PCM} \times m_{\text{water}} \times c_{p,\text{water}} \times \Delta T}{m_{\text{PCM\_assumed}}}$$
  - $m_{\text{water}} = 300\text{ kg}$ (Avargani et al. 2021's 300 L @ 60±2°C over 7 h — a TOTAL volume over the window, not a rate).
  - $c_{p,\text{water}} = 4.186\text{ kJ/(kg}\cdot\text{K)}$; $\Delta T = T_{\text{delivery}} - T_{\text{mains\_est}}$; $m_{\text{PCM\_assumed}} = 50.0\text{ kg}$ (`ASSUMED_PCM_MASS_KG`, shared).
  - **`SHARE_PCM = 0.5`** (in `pcm_shared_config.py`, shared with Rajasthan): literature-anchored combined sensible+latent fractional-share model (Zhao 2022, Huang 2020, Abdelsalam 2020, Koželj 2021; range 0.4–0.78). A CEILING, not an achievability bar.
  - $T_{\text{mains\_est}} = T_{a,\text{mean}} - 2.0$ — a placeholder, NOT a published correlation. Single shared TODO tracked in `pcm_shared_config.T_MAINS_EST_C_TODO` (a Kusuda & Achenbach-style ground-temperature annual-lag model is needed before this is final).
  - *Completed-run range*: per-point $L_{\text{required}}$ **≈ 263–373 kJ/kg**.

### 5. Dimensionality Reduction & Normalization
- **PCA block** (identical to Rajasthan): `[Ta_mean, Ta_p95, Ta_p05, T_sunrise_mean, T_noon_mean, HDD18, CDD24, elevation_m]` — a temperature + **elevation** block (there is no pressure variable; the former "temperature/pressure block" label and the pressure-ratio `elev_proxy` are gone). `elevation_m` is real per-point ERA5 orography from `00c_attach_elevation.py` (range ≈ 0–1283 m across TN), not the former flat 150 m proxy.
- **Component count PINNED to 4** (`PCA_N_COMPONENTS` in `pcm_shared_config.py`), not a data-determined 95%-variance threshold, so `climate_signature_tamilnadu.csv` and `climate_signature_rajasthan.csv` carry the same `PC1..PC4` / `PC1_z..PC4_z` columns and `05_cluster_regions.py` can concatenate them. Tamil Nadu's temp/elevation block reaches 95% variance by PC3 (PC1 alone ≈ 83%), so PC4 here is a low-variance component kept for schema alignment; cumulative variance retained ≈ 0.984.
- **z-Score Normalization**: `NON_CLUSTERING_COLS` (identical to Rajasthan) excludes `lat, lon, population, weight, elevation_m`, the PCA-block raw columns, and `T_mains_est_C, kt_p05, kt_worst_month, Tm_target_capped_C_p05day, tm_target_capped_flag`. Everything else — including `Tm_target_C`, `Tm_target_capped_C`, `L_required_kJ_per_kg`, and all 5 interaction terms — is z-scored.

---

## Outputs

`data/processed/signatures/climate_signature_tamilnadu.csv` — **133 rows × 88 columns** (single `processed/`; the former doubled `data/processed/processed/signatures/` path is fixed). Column schema byte-for-byte name/order/dtype-identical to `climate_signature_rajasthan.csv` (320 rows × 88 columns). Plus QC HTML in `outputs/` (`signature_correlation_heatmap_tamilnadu.html`, `signature_distributions_tamilnadu.html`, `signature_point_map_tamilnadu.html`).

Correlation `|r| > 0.9` flags on the final feature set: **16 pairs** for Tamil Nadu (vs. 37 for Rajasthan) — all in the same families (daylength_mean↔amplitude r=1.0, SAI↔kt_daily_mean r≈1.0, each interaction term vs. its dominant factor, L_required↔PC1). Printed only, not auto-acted-upon.

`04d_signature_interactive.py` (read-only Folium/Plotly explorer, no downstream dependents) was **deleted 2026-09-08** alongside Rajasthan's twin `04f_signature_interactive.py`.

---

## Status
**COMPLETE (unified pipeline, re-run 2026-09-08).** `04b_climate_signature.py` regenerated end-to-end from `tamilnadu_cleaned_physical.csv` + `tier2_signature_tamilnadu.csv` + `daily_aggregates_tamilnadu.csv` + `suntimes.csv` + `population_grid_points.csv` (with real `elevation_m` from `00c_attach_elevation.py`). The NASA POWER hourly cache is not on disk, so `02b` was not re-run — its existing Tier-2 outputs are treated as fixed inputs. Open shared gap: `T_mains_est_C` placeholder (see §4).

---

## Literature Support

| Component | Reference / Method | Source File |
|---|---|---|
| 300 L/day Domestic Draw | Avargani et al. (2021) | `sources/Singh2025PCM_SWH_ComprehensiveReview_summary.md` |
| Fractional PCM Share (0.5) | Zhao (2022); Huang (2020); Abdelsalam (2020) | `13_LITERATURE_MAPPING.md` |
| Discomfort Index (HSI) | Thom (1959) Discomfort Index | Standard meteorological literature |
| Feature-to-Property Mapping | Liu et al. (2025); Singh et al. (2025) Table 2 | `sources/Liu2025AI_PCM_TES_Prediction_Optimization_summary.md` |
