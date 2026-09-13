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

**Three signature columns have no Tier-2 counterpart** and remain sun-event/ERA5-derived: `HSI`,
`monsoon_index`, `elevation_m` (real per-point elevation, static per point — see the elevation note
below) — plus `GHI_mean` (mean noon `era5_GHI`), which carries no `_proxy` suffix at all and
therefore enters the clustering matrix directly. **RESOLVED (2026-09):** this used to also list
`RH_mean`/`wind_mean` here, on the claim that `02b`'s `RH_mean_true`/`wind_mean_true` had no
`CANON_MAP` entry. That was inaccurate — `CANON_MAP` already had both entries; the actual bug was
that the Tier-1 fallback columns were named `RH_mean`/`wind_mean` instead of the `_proxy`-suffixed
names the fallback logic expected, so the Tier-2 override silently resolved to NaN instead of the
Tier-1 value whenever Tier-2 coverage was missing. Fixed by renaming the Tier-1 columns. Currently
latent (100% Tier-2 coverage in this dataset) but fixed for robustness. See `04_PHASE_2_AUDIT.md`
Part A.8.

The script prints `Points with Tier-2 coverage: n/45` and warns for any point that fell back to a
proxy. **The actual coverage number is not available in the source files**, but `02b`'s confirmed
45/45-point, 164,385-point-day run implies full Tier-2 coverage.

### Stage 3 — derived PCM targets

**RESOLVED — this block described a formula version predating the 2026-09 fixes** (the code's own
comments record: "Previous code: DRAW_RATE_KG_PER_S = 60.0/1000/60 -> 0.001 kg/s (WRONG)" — that
rate/volume formula was missing water's density factor, making `L_required` ~1000x too small). The
CURRENT formula, verified against `04b_climate_signature.py` lines 63-93 and 262-276:

```python
T_DELIVERY_C  = 50.0
DT_APPROACH_C =  7.0
TM_TARGET_C   = 57.0                                  # constant baseline for every point, by design
                                                       # (regime-capped for some clusters by
                                                       # 07b_charging_feasibility.py — see
                                                       # 07_PHASE_5_AUDIT.md)

DRAW_VOLUME_L       = 300.0                           # litres/day (domestic household)
DRAW_MASS_KG        = DRAW_VOLUME_L * 1.0             # kg (water density ~1 kg/L)
CP_WATER            = 4.186                           # kJ/(kg*K)
ASSUMED_PCM_MASS_KG = 150.0                           # raised from 50kg after the draw-sizing fix;
                                                       # SHARE_PCM=0.5 imported from config.py

sig["T_mains_est_C"]        = sig["Ta_mean"] - 2.0
q_total_kJ                  = DRAW_MASS_KG * CP_WATER * (T_DELIVERY_C - sig["T_mains_est_C"])
sig["L_required_kJ_per_kg"] = (q_total_kJ * SHARE_PCM) / ASSUMED_PCM_MASS_KG
```

`PREPROCESSING_STEPS.md` explains the sign convention:

> the corrected v2.0 rule: `Tm_target = T_delivery + delta_T_approach` (PCM sits *above* delivery
> temperature so heat flows PCM→water during discharge; the earlier subtract-based rule had the
> sign backwards). Comes out to a constant 57 C here (50 + 7, indirect-system assumption) — held
> constant across all points **by design, not tuned per cluster** (though regime-capped downward
> for Clusters 1/2 — see `07_PHASE_5_AUDIT.md`).

`04b`'s current run prints an `L_required` range of **approximately 113-190 kJ/kg** across the 45
points. The minimum latent heat in the whole 55-row PCM database is 128 kJ/kg, so the 0.7x floor
(~79-133 kJ/kg) is largely non-binding — which `08_mcdm_ranking.py`'s own diagnostic text confirms
independently: "every candidate's latent heat comfortably clearing L_required in every cluster."

One thing still worth noting for a write-up:

- **The `− 2.0` K mains-temperature offset is still unsourced in-code.** No citation appears
  anywhere in `era5-uttarakhand/`, and it drives `L_required` directly — a stated assumption, not a
  bug.
- **There IS now a `SHARE_PCM` fractional-contribution factor (0.5, from `config.py`)** — the
  earlier claim that no such factor existed described the pre-fix formula. The PCM now supplies
  ~50% of overnight delivery, with tank sensible heat covering the rest.

### Stage 4 — 4 interaction terms (RESOLVED — a 5th was removed for exactly the reason described below)

| Term | Definition |
|---|---|
| `int_GHI_x_ktstd` | `GHI_daily_kWh × kt_std` |
| `int_DTR_x_cloudyfrac` | `DTR × cloudy_frac` |
| `int_RH_x_TaMinusTm` | `RH_mean × (Ta_mean − Tm_target_C)` |
| `int_CCI_x_1minusSAI` | `CCI × (1 − SAI)` |

A fifth term, `int_wind_x_TaMinusTsoil = wind_mean × (Ta_mean − Tsoil_proxy_C)` with
`Tsoil_proxy_C = Ta_mean − 3.0`, used to exist here — this section originally flagged that it
reduces algebraically to `3.0 × wind_mean` (a rescaled copy of `wind_mean`, not an independent
interaction) and, since `wind_mean` is also in the matrix, effectively double-weighted wind. **This
has since been fixed** (the code's own comment confirms: "REMOVED: int_wind_x_TaMinusTsoil" — only
`Tsoil_proxy_C` itself was dropped from the clustering matrix before, not this whole term). The
script now prints "Added 4 interaction terms (int_wind_x_TaMinusTsoil removed — see comment)."

### Stage 5 — PCA and clustering-matrix construction

```python
PCA_BLOCK = ["Ta_mean", "Ta_p95", "Ta_p05", "HDD18", "CDD24", "RH_mean", "elevation_m"]  # was elev_proxy
StandardScaler → PCA(n_components=0.95, random_state=42)      # retain 95% variance
loadings → pca_loadings.csv
```

The current run retains **2 components** (PC1 explains 90.7% of variance, PC2 6.8%). Loadings on
PC1: `Ta_mean`(-0.394), `Ta_p95`(-0.395), `Ta_p05`(-0.385), `HDD18`(0.360), `CDD24`(-0.358),
`RH_mean`(0.378), `elevation_m`(0.373) — a balanced contribution, not the outsized -0.33/0.59 the
old `elev_proxy` carried.

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
| `elevation_m` | Real per-point elevation (ERA5 geopotential, 196-2510m) — **was** `mean(P_atm)/1013.25`, a pressure-ratio proxy, fixed 2026-09 | Air mass into the Ineichen clear-sky model (via `02`'s per-point altitude, also fixed); PCA thermodynamic block |

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

PCA is applied to `Ta_mean, Ta_p95, Ta_p05, HDD18, CDD24, RH_mean, elevation_m` (was `elev_proxy`) only — the mutually
correlated thermodynamic block. The solar and variability indices (`GHI_daily_kWh`, `kt_mean`,
`kt_std`, `SAI`, `CCI`, `cloudy_frac`, `DTR`, `seasonality`, `monsoon_index`, `HSI`, `wind_mean`)
are deliberately **kept out**, because they carry the discriminating signal for regime separation
and for PCM target derivation. Compressing them would reduce exactly the information the downstream
recommendation depends on.

### Indices that carry a known problem into the clustering matrix

| Index | Problem | Severity |
|---|---|---|
| `GHI_mean` | ERA5 noon GHI, no Tier-2 override — **RESOLVED (2026-09)**, the deaccumulation bug that deflated it ~10x is fixed | Was High, now resolved |
| `elevation_m` | **RESOLVED (2026-09)** — no longer built from `era5_P_atm`; now real elevation from ERA5 geopotential, unaffected by the 850 hPa bound issue | Was High for a montane state, now resolved |
| `RH_mean` | ERA5-side proxy fallback, used only when Tier-2 (`RH_mean_true`) is missing — **RESOLVED (2026-09)**, the fallback naming bug that could silently null this is fixed; currently 100% Tier-2 coverage so this row is canonical NASA POWER data, not ERA5, for every point in this run | Was Moderate, now resolved |
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

1. **`Tm_target` is constant at 57 °C for every point at the Phase-3 (`04b`) signature stage** — a
   stated design decision. This section used to say this was "the direct cause of the identical
   survivor sets and identical #1 PCM in Phases 5 and 6" — **RESOLVED (2026-09): that was actually a
   downstream bug**, not an inevitable consequence of a constant Phase-3 `Tm_target`.
   `07b_charging_feasibility.py`'s regime-dependent Tm cap (applied per-cluster, after clustering)
   was supposed to differentiate `Tm_target` per cluster but had a normalization bug that made it a
   no-op; fixed, and Clusters 1/2 now get a genuinely lower `Tm_target` (55.16C/56.51C). See
   `07_PHASE_5_AUDIT.md`.
2. **`T_mains_est_C = Ta_mean − 2.0` is unsourced in-code** and drives `L_required` directly. Still
   an open, stated-but-uncited assumption, not a bug.
3. **~~`L_required` has no `SHARE_PCM` fractional-contribution factor~~ RESOLVED** — this described
   the pre-fix formula. The current formula imports `SHARE_PCM=0.5` from `config.py`; the PCM now
   supplies ~50% of overnight delivery, tank sensible heat the rest. See Stage 3 above.
4. **~~`GHI_mean` enters the clustering matrix carrying the ERA5 GHI anomaly~~ RESOLVED** — the
   deaccumulation bug that caused the anomaly is fixed (see `04_PHASE_2_AUDIT.md` Part A.3).
5. **~~`RH_mean` and `wind_mean` are taken from the ERA5 side despite Tier-2 equivalents existing~~
   RESOLVED, but not by adding `CANON_MAP` entries (they already existed)** — the real bug was a
   `_proxy`-suffix naming mismatch in the Tier-1 fallback columns, now fixed. Both are canonical
   NASA POWER Tier-2 values for every point in this run (100% Tier-2 coverage).
6. **~~`int_wind_x_TaMinusTsoil` is a rescaled duplicate of `wind_mean`~~ RESOLVED** — this term has
   been removed entirely; only 4 interaction terms remain (see Stage 4 above).
7. **`monsoon_index` uses JJAS while `SEASON_MAP` uses JJA** — unreconciled, and `monsoon_index` is
   in the clustering matrix.
8. **`Tm_target_C` is a zero-variance column in the clustering matrix.** Harmless but untidy.
9. **~~`elev_proxy` is built from the column most damaged by Phase 2's physical bounds~~ RESOLVED
   (2026-09).** `elevation_m` (real, from ERA5 geopotential) has replaced `elev_proxy`
   (pressure-derived) entirely — this signature index is no longer built from `era5_P_atm` at all,
   so the 850 hPa bound issue documented in `04_PHASE_2_AUDIT.md` Part B.8 (still open for the raw
   `era5_P_atm` column itself) no longer contaminates it.
10. **No Phase 3 output is committed**, so `pca_loadings.csv` cannot be examined from this
    repository. `NEXT_STEPS.md` used to ask the reader to "check how much weight `elev_proxy`
    carries" — that check has since been done and the result recorded (a balanced ~0.37 PC1 loading,
    not an outsized one) in this file and in `NEXT_STEPS.md` itself.

## Status

**COMPLETE.** The two-tier merge works as designed and demonstrably protected the clustering matrix
from the pipeline's largest data defect (the ERA5 GHI deaccumulation bug, since fixed). The
remaining open item from this phase is the unsourced `T_mains_est_C` mains-temperature offset — the
constant-`Tm_target`-causes-identical-results concern and the four "ERA5-side instead of Tier-2"
columns are both resolved (2026-09), per the numbered list above.
