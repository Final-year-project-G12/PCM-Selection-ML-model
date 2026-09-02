# 05 — Phase 3 Audit: Climate Signature Construction

**Scripts**: `04b_climate_signature.py`, `04d_signature_interactive.py`

**Status**: **RUN.** Confirmed indirectly — every Phase 4–6 artefact consumes its output, and the
per-point signature values are visible in `data/plots/verify_clustering/05_cluster_profiles.png`.
The output CSV itself is under the git-ignored `data/processed/` tree and is **not present in this
repository**.

---

## Purpose

Collapse each point's entire 10-year, 3×-daily record into **one row per `point_id`**. That row is
the object Phase 4 actually clusters — not the raw data.

The v3.0 change this script implements is stated in its own docstring:

> The earlier version only used the 3-events/day merged CSV and approximated `GHI_daily_kWh` with a
> half-sine formula, and `DTR` as (noon − sunrise). Those are proxies, not measurements, and the
> plan doc (v3.0 Section 4.3, "Repair 1") is explicit that this is the single highest-value
> remaining data task.

## Hard gate

```python
if not TIER2_FILE.exists():
    raise FileNotFoundError(
        f"{TIER2_FILE} not found. Run 02b_build_daily_aggregates.py first …
         This script cannot proceed without it (plan v3.0 Repair 1).")
```

`04b` will **not** run on Tier-1 proxies alone. This is a real, enforced dependency, not a comment.

## Inputs

- `data/preprocessed/uttarakhand_cleaned_physical.csv` — the **physical-units** file only. `04b`
  never reads the scaled file, because the signature indices (kWh/day, HDD18, CDD24, …) are
  non-linear functions of physical values and would be silently corrupted by pre-scaling.
- `data/processed/tier2_signature_uttarakhand.csv` — `02b`'s output.

## Processing — the six numbered stages

| Stage | What it does |
|---|---|
| [1/6] | Build Tier-1 sun-event signature vectors, one row per `point_id` |
| [2/6] | Left-join Tier-2, report `Points with Tier-2 coverage: n/45`, set canonical columns |
| [3/6] | Derive `Tm_target_C`, `T_mains_est_C`, `L_required_kJ_per_kg` |
| [4/6] | Add 5 interaction terms |
| [5/6] | PCA on the correlated temperature/pressure block; build the clustering column list |
| [6/6] | z-standardise the clustering matrix, join it back, write the output |

### Stage 1 — Tier-1 construction

`daily_frame()` pivots each point's records to one row per date with columns
`{era5_T_amb, era5_GHI, era5_CSI, era5_RHum, era5_precipitation, era5_T_dew} × {sunrise, noon,
sunset}`. The Tier-1 indices are computed from that pivot plus the long-form frame.

The `GHI_daily_kWh_proxy` half-sine formula, which is the one worth recording explicitly:

```python
daylen_hours = (sunset_time_utc − sunrise_time_utc).total_seconds() / 3600
ghi_kw       = noon_GHI / 1000
daily_kwh    = (2.0 / π) · ghi_kw · daylen_hours
```

It uses the **actual** `sunset − sunrise` interval from `suntimes.csv`, not a nominal 12 h, which is
the right choice for a latitude band whose day length swings ~4 h across the year.

Two indices are **explicitly flagged as proxies** in the docstring:

- `DTR_proxy = noon T − sunrise T` — a lower bound on the true diurnal range, because true `Tmax`
  typically lags solar noon by 1–3 h.
- `monsoon_index` — "a JJAS *fraction*, not an absolute rainfall total, since precipitation is only
  sampled 3x/day."

### Stage 2 — the canonical merge

`CANON_MAP` has 13 entries. For each, the canonical column takes the **true Tier-2 value where
present** and falls back to the Tier-1 proxy otherwise:

```python
sig[canon] = sig[true_col].where(sig[true_col].notna(), sig.get(f"{canon}_proxy", np.nan))
```

| Canonical column | Tier-2 source | Tier-1 fallback |
|---|---|---|
| `GHI_daily_kWh` | `GHI_daily_kWh_mean` | `GHI_daily_kWh_proxy` |
| `DTR` | `DTR_true_mean` | `DTR_proxy` |
| `kt_mean`, `kt_std` | `kt_daily_mean`, `kt_daily_std` | `kt_mean_proxy`, `kt_std_proxy` |
| `SAI` | `SAI_true` | `SAI_proxy` |
| `cloudy_frac` | `cloudy_frac_true` | `cloudy_frac_proxy` |
| `CCI` | `CCI_true` | `CCI_proxy` |
| `HDD18`, `CDD24` | `HDD18_true`, `CDD24_true` | `HDD18_proxy`, `CDD24_proxy` |
| `Ta_mean`, `Ta_p95`, `Ta_p05` | `Ta_mean_true`, `Ta_p95_true`, `Ta_p05_true` | `Ta_*_proxy` |
| `seasonality` | `seasonality_true` | `seasonality_proxy` |

Both versions are kept side by side "purely so you can report 'proxy vs. true agreement' in your
methodology," and **both are excluded from the clustering matrix** so only the canonical version
clusters.

**Five signature columns have no Tier-2 counterpart** and remain sun-event/ERA5-derived:
`RH_mean`, `HSI`, `wind_mean`, `monsoon_index`, `elev_proxy` — plus `GHI_mean` (mean noon
`era5_GHI`), which carries no `_proxy` suffix at all and therefore enters the clustering matrix
directly. Note that `02b` *does* compute `RH_mean_true` and `wind_mean_true`, but they have no
`CANON_MAP` entry and are dropped by the `_true` suffix rule — so two already-available Tier-2
values go unused. See `04_PHASE_2_AUDIT.md` Part A.8.

The script prints `Points with Tier-2 coverage: n/45` and warns for any point that fell back to a
proxy. **The actual coverage number is not available in the source files**, but `02b`'s confirmed
45/45-point, 164,385-point-day run implies full Tier-2 coverage.

### Stage 3 — derived PCM targets

```python
T_DELIVERY_C  = 50.0
DT_APPROACH_C =  7.0
TM_TARGET_C   = 57.0                                  # constant for every point, by design

DRAW_RATE_KG_PER_S  = 60.0 / 1000 / 60                # = 0.001 kg/s
CP_WATER            = 4.186                           # kJ/kg·K
ASSUMED_PCM_MASS_KG = 50.0

sig["T_mains_est_C"]        = sig["Ta_mean"] - 2.0
q_night_kw                  = 0.001 × 4.186 × (50 − T_mains_est_C)
sig["L_required_kJ_per_kg"] = (q_night_kw × 3600 × 7) / 50
```

`PREPROCESSING_STEPS.md` explains the sign convention:

> the corrected v2.0 rule: `Tm_target = T_delivery + delta_T_approach` (PCM sits *above* delivery
> temperature so heat flows PCM→water during discharge; the earlier subtract-based rule had the
> sign backwards). Comes out to a constant 57 C here (50 + 7, indirect-system assumption) — held
> constant across all points **by design, not tuned per cluster**.

`04b` prints the resulting `L_required` range, but the values are **not available in the source
files**. They can be bounded from the observed cluster `Ta_mean` medians (≈ 13–25 °C, see
`06_PHASE_4_AUDIT.md`): **`L_required` ≈ 63–82 kJ/kg**, and the Phase 5 floor at 0.7× is
**≈ 44–58 kJ/kg**. The minimum latent heat in the whole 55-row PCM database is 128 kJ/kg, so the
floor is non-binding — which `08_mcdm_ranking.py`'s own diagnostic text confirms independently:
"every candidate's latent heat comfortably clearing L_required in every cluster."

Two things to note about this formula, both material for a write-up:

- **The `− 2.0` K mains-temperature offset is unsourced in-code.** No citation appears anywhere in
  `era5-uttarakhand/`, and it drives `L_required` directly.
- **There is no `SHARE_PCM` fractional-contribution factor.** This `04b` sizes `L_required` from a
  7-hour draw at 0.001 kg/s against the full 50 kg PCM mass — i.e. the PCM alone is assumed to
  supply the whole night load. The resulting values happen to be small enough that the filter never
  binds, so the assumption does not affect this run's outcome, but it should be stated rather than
  left implicit.

### Stage 4 — 5 interaction terms

| Term | Definition |
|---|---|
| `int_GHI_x_ktstd` | `GHI_daily_kWh × kt_std` |
| `int_DTR_x_cloudyfrac` | `DTR × cloudy_frac` |
| `int_RH_x_TaMinusTm` | `RH_mean × (Ta_mean − Tm_target_C)` |
| `int_wind_x_TaMinusTsoil` | `wind_mean × (Ta_mean − Tsoil_proxy_C)`, where `Tsoil_proxy_C = Ta_mean − 3.0` |
| `int_CCI_x_1minusSAI` | `CCI × (1 − SAI)` |

`Tsoil_proxy_C` exists **only** to feed the fourth term and is dropped from the clustering matrix.
Note that `int_wind_x_TaMinusTsoil` therefore reduces algebraically to `3.0 × wind_mean` — it is a
rescaled copy of `wind_mean`, not an independent interaction. Since `wind_mean` is also in the
matrix, this effectively double-weights wind.

### Stage 5 — PCA and clustering-matrix construction

```python
PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elev_proxy"]
StandardScaler → PCA(n_components=0.95, random_state=42)      # retain 95% variance
loadings → pca_loadings.csv
```

**The number of retained components for this run is not available in the source files** —
`pca_loadings.csv` is git-ignored.

Columns removed from the clustering matrix (`DROP_FROM_CLUSTERING`):

- every `PCA_BLOCK` member (now represented by `PC1…PCn`)
- `lat`, `lon` — "never cluster on geography — plan v3.0 Section 6.2"; `05` re-prints this at run
  time
- `population`, `T_mains_est_C`, `Tsoil_proxy_C`
- every column ending `_proxy`
- every column ending `_true` or `_true_mean`

Everything else is z-standardised with `StandardScaler` and appended with a `_z` suffix. The
resulting `_z` set comprises: the non-PCA canonical indices (`GHI_mean`, `kt_mean`, `kt_std`,
`SAI`, `CCI`, `cloudy_frac`, `DTR`, `GHI_daily_kWh`, `seasonality`, `HSI`, `wind_mean`,
`monsoon_index`), `Tm_target_C`, `L_required_kJ_per_kg`, the 5 interaction terms, and `PC1…PCn`.

> **`Tm_target_C` is constant (57.0) across all 45 points**, so its z-score is a zero-variance
> column. It contributes nothing to the clustering but is not excluded.

## Climate Signature Feature-to-PCM-Property Mapping

The design principle the two-tier signature is built on is that every index must earn its place by
constraining a PCM property. The Uttarakhand implementation's mapping:

### Tier 1 — sun-event statistics

| Feature | Physical mechanism | PCM property it constrains |
|---|---|---|
| `GHI_mean` | Mean solar irradiance at the charging instant | Charging-rate feasibility; upper bound on achievable `Tm` |
| `RH_mean` | Annual mean relative humidity → condensation risk at the PCM container | Corrosion-resistance requirement; encapsulation choice |
| `HSI` | `RH_mean × fraction(T_amb − T_dew < 3 K)` — combined humidity + near-saturation signal | Intended as the corrosion-veto trigger. **In this run it triggers nothing** — `07`'s corrosion veto is not implemented, and all 55 database candidates are organic. |
| `wind_mean` | Mean wind speed → convective loss from collector and tank | Tank/collector loss coefficient; indirectly the required storage margin |
| `monsoon_index` | JJAS share of annual precipitation → seasonal charging gap | Storage sizing for the monsoon under-charging window (descriptive, not a ranking criterion) |
| `elev_proxy` | `mean(P_atm)/1013.25` → atmospheric column mass | Air mass into the Ineichen clear-sky model; PCA thermodynamic block |

### Tier 2 — true daily-integral indices

| Feature | Physical mechanism | PCM property it constrains |
|---|---|---|
| `GHI_daily_kWh` | True daily charging energy available | `L_required` sizing — the latent-heat floor |
| `kt_mean` | Annual mean clearness index → solar resource quality | Charging reliability; the `07b` regime cap uses it directly |
| `kt_std` | Day-to-day clearness variability | Charging intermittency; feeds `int_GHI_x_ktstd` |
| `SAI` | `Σ GHI / Σ GHI_clearsky` → fraction of the clear-sky resource actually delivered | Latent-heat margin requirement |
| `cloudy_frac` | Fraction of days with `kt < 0.35` | Autonomy sizing — how often the PCM must carry the load alone |
| `CCI` | Longest consecutive cloudy-day run (days) | Worst-case autonomy; the binding case for storage capacity |
| `DTR` | True `Tmax − Tmin` → daily thermal cycling magnitude | Cycling-stability requirement (`cycles ≥ 300` in Phase 5) |
| `Ta_mean` | Annual mean ambient | `T_mains_est_C` → `L_required`; PCA block |
| `Ta_p95` | Hot design percentile | Upper end of the melting window; safety at extreme heat |
| `Ta_p05` | Cold design percentile | Night-discharge environment; low-temperature cycling stress |
| `HDD18` | Heating degree-days, base 18 °C | Seasonal demand context; PCA block |
| `CDD24` | Cooling degree-days, base 24 °C | Seasonal demand context; PCA block |
| `seasonality` | `std/mean` of monthly-mean daily GHI | Seasonal resource swing → sizing for the worst month |

### Derived targets (not in the clustering matrix as discriminators)

| Quantity | Role |
|---|---|
| `Tm_target_C` = 57 °C | Drives the Phase 5 melting window `[52, 65]` °C and the Phase 6 Gaussian `f_Tm` criterion. **Constant across all points**, so it discriminates nothing. |
| `L_required_kJ_per_kg` | Drives the Phase 5 latent-heat floor `L ≥ 0.7 × L_required`. Varies with `Ta_mean` but lands well below every candidate's latent heat, so it also discriminates nothing. |

### Why the two-tier design is necessary

Neither tier alone is sufficient, and the Uttarakhand run demonstrates exactly why:

- **Tier 1 alone underestimates.** `DTR_proxy = noon − sunrise` is a lower bound on the true
  diurnal range. `GHI_daily_kWh_proxy` is a half-sine reconstruction from a single instantaneous
  sample. Degree-days from a 3-point daily mean are not degree-days from a true daily mean.
- **Tier 2 alone loses the charge/discharge instants.** The sun-event samples are the only place
  the pipeline observes conditions *at* the moments that matter thermally.
- **Tier 2 also rescued this run.** Because the canonical solar and temperature columns come from
  NASA POWER via `02b`, the clustering matrix's entire solar block was insulated from the ERA5 GHI
  magnitude anomaly documented in `04_PHASE_2_AUDIT.md` Part A.3. The `_proxy` variants carry the
  anomaly but are excluded by the suffix rule. **This is the single largest practical payoff of the
  Repair-1 design and should be reported as such.**

### PCA scope — and why the solar block is kept out

PCA is applied to `Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elev_proxy` only — the mutually
correlated thermodynamic block. The solar and variability indices (`GHI_daily_kWh`, `kt_mean`,
`kt_std`, `SAI`, `CCI`, `cloudy_frac`, `DTR`, `seasonality`, `monsoon_index`, `HSI`, `wind_mean`)
are deliberately **kept out**, because they carry the discriminating signal for regime separation
and for PCM target derivation. Compressing them would reduce exactly the information the downstream
recommendation depends on.

### Indices that carry a known problem into the clustering matrix

| Index | Problem | Severity |
|---|---|---|
| `GHI_mean` | ERA5 noon GHI, no Tier-2 override — carries the −211 W/m² anomaly | High |
| `elev_proxy` | Built from `era5_P_atm`, 37.1 % of which was NaN'd one-sidedly by the 850 hPa bound and imputed | High for a montane state |
| `RH_mean` | ERA5-side, +11.4 % MBE vs POWER, unused `RH_mean_true` available | Moderate |
| `wind_mean` | ERA5-side, −1.14 m/s MBE vs POWER, unused `wind_mean_true` available | Moderate |
| `HSI` | Built on `RH_mean`, so inherits its offset | Moderate |
| `monsoon_index` | Permanently a 3×/day ERA5 precipitation *fraction*; JJAS here vs JJA in `SEASON_MAP` | Low (a ratio; descriptive only) |
| `int_wind_x_TaMinusTsoil` | Algebraically `3.0 × wind_mean` — a rescaled duplicate, not an interaction | Low |
| `Tm_target_C` | Zero-variance column | Cosmetic |

## `04d_signature_interactive.py` — explorer

Reads `climate_signature_uttarakhand.csv` and writes Folium/Plotly HTML to
`data/processed/signatures/interactive/`. Produces a multi-layer map with one toggleable layer per
index (`MAP_LAYERS = GHI_daily_kWh, Ta_mean, DTR, kt_mean, cloudy_frac, CCI, HDD18, CDD24, RH_mean,
HSI, monsoon_index, L_required_kJ_per_kg`), an interactive correlation heatmap, index-distribution
histograms, and a scatter matrix of the key PCM-facing indices "to eyeball the clustering structure
before `05` finds it formally."

**Its output directory is under the git-ignored `data/processed/` tree, so none of it is present in
this repository.**

## Outputs

| File | Contents | Committed? |
|---|---|---|
| `data/processed/signatures/climate_signature_uttarakhand.csv` | 45 rows: raw indices + `_z` columns | No |
| `data/processed/signatures/pca_loadings.csv` | PCA component loadings | No |
| `signature_correlation_heatmap.png` | 18-index correlation | No |
| `signature_distributions.png` | per-index histograms, with a constant-value special case | No |
| `point_signature_map.png` | lon/lat scatter coloured by `GHI_daily_kWh` and `monsoon_index` | No |
| `data/processed/signatures/interactive/*.html` | `04d` output | No |

None of Phase 3's own outputs are committed. The only surviving evidence of the signature values is
`data/plots/verify_clustering/05_cluster_profiles.png`, which plots six of them by cluster.

## Dependencies

`pandas`, `numpy`, `scikit-learn` (`PCA`, `StandardScaler`), `matplotlib`, `seaborn`;
`plotly` + `folium` + `branca` for `04d`.

## Validation

| Check | Result |
|---|---|
| Tier-2 file exists before running | **Enforced** — hard `FileNotFoundError` |
| Tier-2 coverage per point reported | Implemented; value not available in the source files |
| Reads the physical (unscaled) file only | **Confirmed** — `PHYSICAL_FILE` is the only climate input |
| lat/lon excluded from clustering | **Confirmed** — dropped in `DROP_FROM_CLUSTERING`, re-announced by `05` |
| PCA retains 95 % variance | Implemented (`n_components=0.95`); component count not available |
| Diagnostic plots handle degenerate columns | **Yes** — `signature_distributions.png` has an explicit constant-value branch, which is what `Tm_target_C` triggers |

## Problems / risks

1. **`Tm_target` is constant at 57 °C for every point.** A stated design decision, and the direct
   cause of the identical survivor sets and identical #1 PCM in Phases 5 and 6. It means Phase 3
   contributes no climate-driven differentiation to the PCM target itself — all differentiation
   would have to come from `L_required`, which is non-binding.
2. **`T_mains_est_C = Ta_mean − 2.0` is unsourced in-code** and drives `L_required` directly.
3. **`L_required` has no `SHARE_PCM` fractional-contribution factor** — the PCM alone is implicitly
   assumed to supply the whole night load. Non-binding in this run, but it should be stated.
4. **`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly** — the one solar column
   the Tier-2 repair does not cover.
5. **`RH_mean` and `wind_mean` are taken from the ERA5 side despite Tier-2 equivalents existing**
   (`RH_mean_true`, `wind_mean_true` are computed by `02b` and discarded). A two-entry `CANON_MAP`
   addition would fix it.
6. **`int_wind_x_TaMinusTsoil` is a rescaled duplicate of `wind_mean`** (`= 3.0 × wind_mean`), so
   wind is effectively double-weighted in the clustering matrix.
7. **`monsoon_index` uses JJAS while `SEASON_MAP` uses JJA** — unreconciled, and `monsoon_index` is
   in the clustering matrix.
8. **`Tm_target_C` is a zero-variance column in the clustering matrix.** Harmless but untidy.
9. **`elev_proxy` is built from the column most damaged by Phase 2's physical bounds** (37.1 % of
   `era5_P_atm` NaN'd one-sidedly and imputed) — see `04_PHASE_2_AUDIT.md` Part B.8. For a state
   whose central methodological weakness is elevation, this is the most consequential inherited
   defect in the signature.
10. **No Phase 3 output is committed**, so `pca_loadings.csv` — which `NEXT_STEPS.md` specifically
    asks the student to inspect ("check how much weight `elev_proxy` carries") — cannot be examined
    from this repository.

## Status

**COMPLETE.** The two-tier merge works as designed and demonstrably protected the clustering matrix
from the pipeline's largest data defect. The open items are the constant `Tm_target` (a design
choice with large downstream consequences), the unsourced mains-temperature offset, and the four
ERA5-side columns that could have used already-computed Tier-2 values.
