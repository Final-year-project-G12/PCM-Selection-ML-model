# 14 — ERA5 vs NASA POWER Validation: The Full Story

This is the audit checkpoint explicitly called out as mandatory: the cross-source validation that
caught the deaccumulation bug before it contaminated the climate signature, and the decision it
produced. Script: `03b_agreement_analysis.py`.

## Variable pairs compared

```
era5_GHI      ↔ power_ALLSKY_SFC_SW_DWN   (GHI)
era5_T_amb    ↔ power_T2M                 (T_amb)
era5_RHum     ↔ power_RH2M                (RHum)
era5_W_spd    ↔ power_WS10M               (W_spd)
```

## Matching

Reuses `02_combine_rajasthan.py`'s row-level merge — same point, same `(date, event)`, each source
independently nearest-in-time-matched within `MAX_MATCH_HOURS=3` of the true sun-event instant (see
`10_TEMPORAL_PROCESSING.md` for the caveat that ERA5 and POWER can in principle match to different
actual instants within that window, since the matched timestamp itself is never persisted).

## Statistics — exact formulas

```
MBE  = mean(ERA5 − POWER)          [positive ⇒ ERA5 overestimates relative to POWER]
RMSE = sqrt(mean((ERA5 − POWER)²))
r    = Pearson correlation (pandas .corr()), requires n>1 and nonzero std in both series
```
Stratified at four granularities per variable: overall, by season alone, by event alone, and by
season×event — **20 rows per variable × 4 variables = 80 rows total**, confirmed matching
`era5_power_agreement_rajasthan.csv`'s actual row count.

## Decision rule — exact thresholds, evaluated at solar noon only

```
CORR_GOOD = 0.90        CORR_SEVERE = 0.70
MBE_SMALL_FRAC = 0.05   SEASON_SPREAD_FRAC = 0.05
```
- **BACKBONE** (no correction) iff simultaneously: `r_noon ≥ 0.90` AND `|MBE_noon|/mean(POWER noon
  GHI) ≤ 0.05` AND `(max−min season noon MBE)/mean(POWER noon GHI) ≤ 0.05`.
- **QUANTILE_MAP** iff `r_noon ≥ 0.70` but BACKBONE's stricter conditions fail.
- **MANUAL_REVIEW** (no averaging/blending, diagnostics only) iff `r_noon < 0.70` or undefined.
- **Fixed-weight blending (e.g. 0.6·ERA5+0.4·POWER) is never implemented and is explicitly rejected
  by design** — "no principled derivation for a fixed weight between two independent
  reanalysis/satellite-derived products," stated both in the module docstring and in the persisted
  decision text.

Sunrise/sunset are deliberately excluded from the decision (only noon drives it) because both
sources sit near the solar-elevation zero-crossing at those events, where relative disagreement is
naturally large but physically inconsequential — a correct, explicitly-reasoned scoping choice, not
an oversight.

## Actual Rajasthan result — the numbers that matter for the write-up

**Overall (all events, all seasons)**: n=3,506,880, MBE=6.94 W/m², RMSE=83.34 W/m², r=0.9727.

**Solar noon, all seasons (the decision-driving row)**: n=1,168,960, **MBE=10.95 W/m², RMSE=113.79
W/m², r=0.8102.**

**Per-season noon MBE**: Winter=+38.10, Summer=+15.26, Monsoon=−35.78, Retreat=+26.88 W/m² (spread =
73.88 W/m², 10.0% of mean daytime GHI — exceeds the 5% `SEASON_SPREAD_FRAC` gate).

**Decision: `QUANTILE_MAP`** — `r_noon=0.8102` fails BACKBONE's `≥0.90` gate (and the season-spread
gate also fails independently), but clears the `≥0.70` `CORR_SEVERE` floor for QUANTILE_MAP. Decision
text (verbatim, from `outputs/bias_decision_rajasthan.txt`):

> "ERA5 and NASA POWER GHI show a systematic, season-dependent bias at solar noon (Pearson r = 0.810;
> per-season noon MBE spread = 73.88 W/m², 10.0% of mean daytime GHI) — consistent with ERA5
> overestimating GHI in specific seasons... Rather than average or blend the two sources with an
> undocumented fixed weight, daytime ERA5 GHI is empirically quantile-mapped onto the NASA POWER
> distribution, fit separately for each of the four seasons. RMSE improved in 4/4 seasons and Pearson
> r improved in 3/4 seasons after correction."

## Before/after quantile-mapping table (verbatim from the actual output)

| Season | n | MBE before | RMSE before | r before | MBE after | RMSE after | r after |
|---|---|---|---|---|---|---|---|
| Winter | 811,441 | 12.99 | 83.97 | 0.9661 | 0.09 | 82.19 | 0.9647 |
| Summer | 673,063 | 24.58 | 83.60 | 0.9839 | 0.15 | 75.95 | 0.9856 |
| Monsoon | 665,947 | −11.47 | 109.93 | 0.9541 | −0.11 | 103.57 | 0.9597 |
| Retreat | 699,263 | 26.68 | 88.42 | 0.9701 | −0.16 | 77.26 | 0.9749 |

Note: Winter's `r` after correction (0.9647) is marginally *lower* than before (0.9661) — this is the
one of four seasons where the prose's "3/4 seasons" r-improvement claim excludes; RMSE improved in
all four regardless.

**Important scope note**: these before/after numbers are computed on **all-event daytime rows**
(`ERA5 GHI>0`, all three sun-events, n≈2.85M combined), not the noon-only n=1,168,960 used for the
*decision itself* — the decision is made on noon-only statistics, but the fitted correction covers
all daytime rows per season.

## The mandatory pre/post-fix comparison the user specifically asked to preserve

Before the `accum_to_flux()` fix (see `09_ERA5_DATA_PIPELINE.md`): noon GHI Pearson r ≈ 0.01, near-zero
median ERA5 GHI (~2 W/m² vs POWER's ~37 W/m²) — a physically implausible, near-total mismatch. **After**
the fix: r=0.8102, MBE=10.95 W/m² at solar noon. This is not a marginal statistical improvement — it
is the difference between a dataset that would have silently corrupted every downstream climate
signature, cluster, and PCM target derived from GHI, and one that passes a defensible cross-source
validation. **This should be presented in the thesis/paper as direct evidence that the validation
pipeline is doing real work** — it caught a genuine, high-impact preprocessing fault before it
propagated, which is exactly what a cross-source validation step is for.

## Is ERA5-backbone/POWER-cross-check scientifically justified here?

Given the actual result (QUANTILE_MAP, not BACKBONE), the correct framing is: **ERA5 (quantile-corrected)
is the climate backbone; POWER is both the correction target and the reported independent cross-check** —
not a simple "ERA5 alone, no correction needed" story. As noted in `04_PHASE_2_AUDIT.md`, the fitted
quantile-mapping correction is **not currently applied back into `climate_rajasthan_points.csv`** —
Phase 3 onward consumes the raw (deaccumulation-fixed but not bias-corrected) ERA5 GHI. This is the
single most important open methodological decision flagged across this entire audit: either apply
the correction upstream before Phase 3, or explicitly document that Phase 3+ uses uncorrected GHI and
justify why the residual bias (MBE≈11 W/m² at noon, ~10% seasonal spread) is acceptable at the level
of aggregation Phase 3 actually operates at (10-year point averages, not instantaneous values).

## Literature support

Framework doc §5.1–5.2 directly specifies this methodology (MBE/RMSE/r, season×event stratification,
three-branch decision rule, explicit rejection of fixed-weight blending) — the implementation matches
the specification closely and was checked line-by-line against it during this audit, not merely
assumed to match. No additional external citation is needed for MBE/RMSE/Pearson-r themselves
(standard statistics); quantile mapping as a bias-correction technique is a well-established climate
data-processing method (e.g., Cannon et al. 2015 for the general empirical-quantile-mapping family) —
not independently found cited in this project's `references.bib`/`.claude/references.md` during this
audit; adding a QM methodology citation would strengthen the write-up's Phase-2 methodology section.
