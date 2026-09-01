# 05 — Phase 3 Audit: Climate Signature Construction

Scripts: `signature_lib.py`, `04_climate_signature_rajasthan.py`.

## Purpose

Reduce each point's 10-year climate record to one compact "climate signature" vector suitable for
clustering and for deriving PCM performance targets. The framework doc's own design principle
(§6.1, quoted): *"Every index must answer the question 'which PCM property does this constrain, and
by what physical mechanism?'. If that sentence cannot be completed, the index is removed."*

## Inputs

**Corrected 2026-08-11**: `climate_rajasthan_points_CLEAN.csv` (Phase 2.5's output —
`03b_quality_check_rajasthan.py`'s Hampel-filtered/imputed clean file, NOT `02_combine_rajasthan.py`'s
raw output directly, since 2026-08-11 — see `15_QUALITY_CONTROL.md` Part 2 and `04_PHASE_2_AUDIT.md`'s
corrected Dependencies section), `daily_aggregates_rajasthan{,_summary}.csv`, `suntimes.csv`,
`population_grid_points.csv` (Phases 1–2/2.5).

## Processing — Tier 1 (sun-event statistics, `signature_lib.build_tier1_signature()`)

Shared between this script's Level-A (whole-year) call and `05_cluster_rajasthan.py`'s Level-B
(per-season) call — one implementation, two `group_keys`. Produces, per group:
`T_sunrise_mean/p05`, `T_noon_mean`, `T_sunset_mean/p95`, `diurnal_gradient` (noon−sunrise, an
acknowledged *underestimate* of true DTR since peak air temp lags solar noon by 2–3h — this is why
Tier 2's `DTR_true` exists as a companion), `kt_noon_mean/std`, `GHI_noon_mean`, `GHI_sunset_mean`,
`RH_sunrise_mean`, `wind_noon_mean/sunset_mean`, `HSI_sunrise`, `Ta_mean/p95/p05` (daily-collapsed
first, then aggregated), `daylength_mean`, `daylength_amplitude` (half the seasonal swing, standard
oscillation-amplitude convention).

**`HSI_sunrise` is literally Thom's (1959) Temperature-Humidity Index (THI)**, not a bespoke
"humidity stress index":
```
HSI_sunrise = T_sunrise_mean − 0.55·(1 − RH_sunrise_mean/100)·(T_sunrise_mean − 14.5)
```
Cited in-code to Thom, E.C., "The Discomfort Index," *Weatherwise* 12(2), 1959 — the name in the
project's own variable naming ("humidity stress index") is a relabeling of an established index, not
an original derivation; this should be cited as Thom's THI in any write-up, not presented as novel.

## Processing — Tier 2 (daily-integral join)

Left-joined from `daily_aggregates_rajasthan_summary.csv`: `GHI_daily_kWh, SAI, kt_daily_mean/std,
cloudy_frac, CCI, HDD18, CDD24, DTR_true, seasonality, monsoon_index`.

## Processing — derived PCM-facing quantities

**`Tm_target_C = T_delivery + ΔT_approach = 50 + 7 = 57.0°C`**, constant across all 320 points by
design (indirect-system assumption; T_delivery is the Indian-domestic SWH delivery target per
framework doc §6.3, ΔT_approach is the midpoint of the doc's stated 5–8 K heat-exchanger approach
range).

**`Tm_target_capped_C`** — the per-point regime-adjusted upper bound, capturing "Tm must lie below
the collector delivery temperature achievable on a poor-insolation period":
```
kt_worst_month = min over 12 calendar months of (mean kt_daily for that month, pooled 2016-2025)
kt_ratio = clip(kt_worst_month / kt_daily_mean, upper=1.0)
Tm_target_capped_C = min(57.0, Ta_mean + kt_ratio·(57.0 − Ta_mean))
```
**This formula was revised on 2026-08-11**, replacing an original `kt_p05` (5th-percentile single
*day*) basis with `kt_worst_month` (lowest of 12 calendar-month *means*), after the single-day basis
was checked against independent field evidence (Nahar 2003, tested at Jodhpur — inside this state's
arid-west cluster — reporting 100 L delivered at average 50–70°C across the year) and found to
produce implausibly low caps (40.8–49.5°C, at/below the low end of what a real Jodhpur system
delivered even in its weakest season). The `kt_p05`-based value is retained as
`Tm_target_capped_C_p05day` for audit/comparison only — **no downstream script reads it.** The
worst-month basis is anchored to Durin et al. (2018), "'Worst Month' and 'Critical Period' Methods
for the Sizing of Solar Irrigation Systems" — a genuine, appropriately-applied sizing-methodology
citation for this kind of cap.

**`L_required_kJ_per_kg`** — the latent-heat floor:
```
T_mains_est_C = Ta_mean − 2.0          [documented as NOT a published correlation — see below]
Q_night_kJ = 300.0 · 4.186 · (50.0 − T_mains_est_C)     [Avargani et al. 2021: 300 L @ 60±2°C, 7h]
L_required_kJ_per_kg = Q_night_kJ / 50.0                [ASSUMED_PCM_MASS_KG = 50 kg placeholder]
```
This formula was itself corrected in-place: an earlier version fed a **60 L/min sustained rate for
7 hours** (25,200 L total) into the same formula — traced to a units confusion where Avargani et
al.'s cited figure ("300 L of hot water at 60±2°C for 7 h of operation") is a **total volume over
the discharge window**, not a per-minute rate; the corrected code uses the literal 300 L total. The
script's own docstring is explicit that **`L_required` is a ceiling, not an achievability bar** — it
assumes the PCM bed alone supplies the entire load with zero contribution from tank sensible heat or
overlapping collector charge, an assumption that does not hold even in Avargani's own experimental
rig — and flags forward, correctly, that Phase 5's fixed κ=0.7 latent-heat constraint will zero out
every candidate given this ceiling (confirmed true — see `07_PHASE_5_AUDIT.md`).

**✅ VALIDATED (2026-08-31 re-run complete):** The all-latent assumption was corrected to use
SHARE_PCM=0.5 (literature-anchored). Avargani et al.'s 300 L benchmark is delivered by a combined
PCM-tank architecture; literature on combined sensible-latent SWH reports PCM contributing 40–78%
of total delivery (Zhao 2022, Huang 2020, Abdelsalam 2020, Kowhitney 2021). The corrected formula:
```
L_required = (SHARE_PCM * Q_night) / ASSUMED_PCM_MASS_KG    [SHARE_PCM = 0.5, literature-anchored]
```
**Validation results (2026-08-31 re-run):**
- L_required halved: 608–641 kJ/kg (old all-latent) → 285–344 kJ/kg (new, literature-anchored)
- Output message: "L_required_kJ_per_kg  : 285 - 344 kJ/kg  (literature-anchored, PCM 50% of total night delivery, with tank sensible heat + concurrent charging supplying the rest)"
- Clustering stability: bootstrap-ARI improved from 0.8137 to 0.8272 (robust to methodology change)
- Downstream impact: Phase 5 κ-calibrated survivors increased from 20 to 39 candidates (9/14/16 per cluster)

## The five interaction terms (exact, with in-code physical justification)

```
int_GHI_x_ktstd              = GHI_daily_kWh × kt_daily_std            (erratic-but-large resource)
int_DTR_x_cloudyfrac         = DTR_true × cloudy_frac                  (cycling stress under intermittency)
int_RH_x_TsunriseMinusTm     = RH_sunrise_mean × (T_sunrise_mean − Tm_target_C)   (condensation risk)
int_wind_x_TsunsetMinusTdelivery = wind_sunset_mean × (T_sunset_mean − T_delivery) (evening convective loss)
int_CCI_x_1minusSAI           = CCI × (1 − SAI)                        (combined autonomy requirement)
```

## PCA block

`PCA_BLOCK = [Ta_mean, Ta_p95, Ta_p05, T_sunrise_mean, T_noon_mean, HDD18, CDD24, elevation_m]` (8
columns — note the section's own internal label calling this "the correlated temperature/*pressure*
block" is inaccurate; there is no pressure variable in the actual list, likely a leftover label from
a template). `StandardScaler` → `PCA(n_components=0.95, random_state=42)` — retains **4 components**
for the Rajasthan run (not a fixed integer; data-determined). Loadings and explained-variance ratio
are printed for interpretation, not silently discarded.

**Elevation is a resolved design ambiguity, not excluded**: the brief's "STATIC ATTRIBUTES...NOT
included in the clustering feature matrix" instruction and its "elevation_m" PCA-block membership
initially read as contradictory; the code's own resolution (documented as "Correction 2") is that
elevation's *raw* value is reporting-only, but it *does* feed the PCA block, and the resulting PC*_z
scores (which subsume it) are what actually enters the clustering matrix.

## Standardization

`NON_CLUSTERING_COLS` explicitly excludes `lat, lon, population, weight, T_mains_est_C, kt_p05,
kt_worst_month, Tm_target_capped_C_p05day, tm_target_capped_flag`, plus the raw PCA-block columns
(replaced by `PC1..PC4`). Everything else — including `Tm_target_C`, `Tm_target_capped_C`,
`L_required_kJ_per_kg`, and all 5 interaction terms — is z-scored. Verified directly against the
actual output CSV header: no `lat_z`/`lon_z` exist, confirming the exclusion claim in data, not just
in code.

## Literature support

Thom (1959) THI — direct, correctly attributable citation for `HSI_sunrise`. Avargani et al. (2021),
*J. Energy Storage* — direct citation for the 300 L/60±2°C/7h night-discharge basis. Durin et al.
(2018) — direct citation for the worst-month sizing method. Nahar (2003) — cited as field-evidence
justification for the `kt_worst_month` correction, present in `.claude/references.md` as a bare
citation note (not a full BibTeX entry — worth completing before a formal write-up).
`T_mains_est_C = Ta_mean − 2.0` is **explicitly documented in-code as not derived from any published
correlation** — kept only for cross-state consistency with an identically-unsourced Tamil Nadu
precedent. This is a genuine literature gap, not a citation the audit failed to find: **a real
ground-temperature lag correlation (e.g., Kusuda & Achenbach-style annual-lag models) should be
substituted before this number is presented as anything more than a placeholder.**

## Validation

Correlation heatmap + `|r|>0.9` flagging on the final (pre-standardization, post-PCA-block-removal)
feature set — printed only, not persisted, not auto-acted-upon. No flagged pairs are reported as
"already handled by PCA" without independent verification; the check explicitly distinguishes new
collinearity from PCA-absorbed collinearity.

## Outputs

`climate_signature_rajasthan.csv` — 320 rows × 86 columns. Plus, added 2026-08-11: `outputs/
signature_distributions_rajasthan.html` (histogram of every clustering-input column across all 320
points — a bimodal column here previews a possible Level-A cluster split on that feature alone) and
`outputs/signature_point_map_rajasthan.html` (geographic view of `GHI_daily_kWh` and `monsoon_index`)
— both pure visualization of data this script already computes, in addition to the pre-existing
`outputs/signature_correlation_heatmap_rajasthan.html`.

## Dependencies

Requires Phase 2's `climate_rajasthan_points.csv` and `daily_aggregates_rajasthan_summary.csv`.
Feeds Phase 4 (clustering) and Phase 5 (feasibility targets `Tm_target_C`, `Tm_target_capped_C`,
`L_required_kJ_per_kg`).

## Problems / risks

- **Dangling citation**: the module docstring references
  `Objective1_Section5_Methodology_Update.docx` for the draw-rate correction provenance, and
  **explicitly self-flags that this file was not found in the project tree** — a real, honestly-logged
  gap, not a fabricated citation. Resolve before final write-up (either locate the file or update the
  in-code pointer).
- **RESOLVED — "forward-dated docstring" concern.** A previous version of this audit flagged
  "Correction 5" (the `kt_worst_month` fix)'s 2026-08-11 date as a likely clock/environment artifact.
  It is not: 2026-08-11 is a real date with many independently-verified, mutually-consistent same-day
  fixes across this codebase (the GMM canonical-relabeling fix, `provenance_lib.py`, `physics_lib.py`'s
  two solver bugs — see `20_IMPLEMENTATION_ISSUES.md` items 8-10). This fix is settled history.
- **`T_mains_est_C` is unsourced** — flagged above; this feeds directly into `L_required_kJ_per_kg`,
  which is the constraint that currently zeros out the entire feasibility filter (Phase 5), so this
  is not a low-priority gap.
- **`monsoon_index` proxy status is a structural, not incidental, limitation** — confirmed
  unconditionally true (PRECTOTCORR never downloaded), correctly self-flagged in a printed warning,
  but should be stated as a limitation in any methodology write-up that reports `monsoon_index`.

## Status

**COMPLETE — with two open citation gaps** (dangling methodology-update reference, unsourced
T_mains lag correlation) that should be resolved before this phase's derived quantities
(`Tm_target_capped_C`, `L_required_kJ_per_kg`) are presented as fully literature-grounded.
