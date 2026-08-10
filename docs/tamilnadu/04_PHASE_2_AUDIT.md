# 04 — Phase 2 Audit: Combine, Daily Aggregates, Preprocessing & QC

True scripts: `02_combine_tamilnadu.py` (`03b_interactive_raw_qa (3).py`),
`02b_build_daily_aggregates.py` (`02_combine_tamilnadu (3).py`), `04_preprocess_tamilnadu.py`
(`05b_cluster_interactive (3).py`), plus 6 read-only QA/plot scripts.

## Status: code complete, never executed

## The combine script — deaccumulation, confirmed genuinely correct

Unlike Rajasthan's history (where an earlier `deaccumulate()` was found to be *wrong* for that
pipeline's specific CDS download behavior and replaced with a no-diff `accum_to_flux()`), Tamil
Nadu's `deaccumulate()` implements a **true diff-based deaccumulation with an explicit reset-hour
override**, and nothing in this audit found evidence it is wrong for this pipeline:

```python
def deaccumulate(s):
    s = pd.Series(np.asarray(s, dtype=float), index=s.index).copy()
    diff = s.diff()
    reset_mask = s.index.hour.isin([1, 13])
    diff[reset_mask] = s[reset_mask]
    return diff.clip(lower=0)
```

Consecutive-hour differencing is the default path; hours 1 and 13 UTC (the first fully-accumulated
hour after each 00Z/12Z forecast-cycle reset) use the raw value directly since it already equals
exactly one hour's flux; a `clip(lower=0)` floor guards against floating-point noise. **This is
exactly the pattern the audit checked for and did not find a bug in** — it is neither the "treat
every accumulated hour as already-a-flux" pattern (which would be wrong if TN's CDS download behaves
like the classic MARS convention) nor a naive uniform diff that would corrupt the reset-hour values.
**Important caveat**: this function's correctness for *this specific pipeline's actual CDS download
behavior* has not been empirically re-verified the way Rajasthan's was (no raw-vs-diffed comparison
script or diagnostic exists for Tamil Nadu, and no execution has occurred to check whether observed
raw values behave as a genuine cumulative-since-reset field). Given Rajasthan's own discovery that
*its* CDS download did not follow the assumed convention, **this should be independently verified for
Tamil Nadu the first time the pipeline is actually run**, using the same diagnostic Rajasthan used
(check whether a meaningful fraction of consecutive-hour raw values are lower than their predecessor
within the same nominal cycle — if so, TN's download also returns already-flux values and this
`diff()`-based function would need the same fix Rajasthan applied).

## Solar geometry and derived variables — identical logic to Rajasthan

`pvlib.location.Location(...).get_solarposition()` (method unpinned, same latent risk as Rajasthan),
`.get_clearsky(model="ineichen")` (no explicit Linke turbidity override, same as Rajasthan). DNI:
primary branch uses the direct-radiation ERA5 field if present, fallback `GHI/cos(SZA)` where
`cos_z>0.05`, both clipped `[0,1400]`. DHI: closure residual `GHI−DNI·cos(SZA)`, clip≥0. CSI: forced
0 below `GHI_clearsky=10` W/m² threshold, else clip `[0,1.5]`. All formulas verified identical to
Rajasthan's `compute_solar()`/`apply_unit_conversions()` by direct code comparison.

## Nearest-in-time matching

`MAX_MATCH_HOURS=3`, applied independently to ERA5 and NASA POWER against the same target sun-event
time — same design (and same unrecorded-matched-timestamp gap) as Rajasthan.

## Daily aggregates builder ("Phase 2 Repair 1")

Reads the full cached NASA POWER hourly series (not the 3-sample/day combine output) to build true
daily integrals. Exact formulas: `GHI_daily_kWh = Σ(hourly W/m²)/1000` (implicit 1-reading-per-hour
assumption, stated in-code); `kt_daily = GHI_daily_kWh/GHIcs_daily_kWh`, clip `[0,1.5]`, requires
`GHIcs_daily_kWh>0.05`; `DTR_true = Ta_max_true − Ta_min_true`; day-quality gate
`MIN_HOURS_PER_DAY=20` (days with fewer valid hourly readings are dropped, not averaged short —
explicit design choice). Tier-2 summary formulas: `SAI_true = ΣGHI_daily_kWh/ΣGHIcs_daily_kWh` (ratio
of sums, not mean of ratios); `cloudy_frac_true` = fraction of days with `kt_daily<KT_CLOUDY_THRESHOLD
(0.35)`; `CCI_true` = longest consecutive cloudy-day run length (run-length-encoded); `HDD18_true`/
`CDD24_true` = cumulative sum (not annualized) of degree-days at base 18°C/24°C;
`seasonality_true` = coefficient of variation of 12 monthly-mean GHI values. Self-documented
limitation, quoted verbatim: *"monsoon_index therefore still comes from the ERA5 3x/day
precipitation-fraction proxy... it is NOT recomputed here as a true Tier-2 index"* — matches the same
gap Rajasthan has.

## The 13-step preprocessing pipeline — verified faithful to its own documented spec

`04_preprocess_tamilnadu.py` is the largest, most complex file in the TN pipeline (no equivalent
exists as a distinct step in Rajasthan's pipeline, which folds comparable QC into the combine step
and `03_verify_climate_csv.py` instead). Every one of the 13 documented steps was independently
confirmed present with matching thresholds:

| Step | Confirmed implementation |
|---|---|
| 1 — Dedup | `duplicated(subset=["point_id","date","event"])` |
| 2 — Physical bounds | 18-variable bound table (wider than Rajasthan's — e.g. `T_amb∈[-30,55]` vs `[-5,60]`); out-of-range → **NaN, not clipped**, except RH which is clipped `[0,100]` (an inconsistency with the stated "never clip" rule); below-horizon (`SZA≥90`) forces solar fields to 0 |
| 3 — Hampel/MAD outlier filter | window=15 occurrences each side (31 total), `HAMPEL_N_SIGMA=3.0`, MAD→σ constant `1.4826`, requires `roll_mad>1e-6` to avoid flagging flat stretches; flagged → NaN, never deleted |
| 3b — Yeo-Johnson skew diagnostic | report-only, `yeo_johnson_skew.csv`, no columns transformed — confirmed |
| 4 — Hierarchical imputation | (a) linear interp `limit=3` interior gaps → (b) ffill/bfill `limit=3` → (c) point→zone→global median (zone built via ad hoc `KMeans(n_clusters=min(8,...))` on lat/lon, explicitly **not** the Phase-4 climate clustering) → (d) `IterativeImputer(max_iter=10, random_state=42)` MICE fallback, triggered only if still missing |
| 5 — Temporal coverage validation | warns if `<99%` of expected days present per point×event |
| 6 — Feature engineering | wind vector decomposition, cloud opacity (`1−CSI`), dew-point depression, IST solar hour angle, daytime flag |
| 7 — Lag features | `LAG_OCCURRENCES=[1,7,30]`, grouped by `(point_id,event)`, sorted by date — "occurrences," not hours |
| 8 — Rolling means/stds | `ROLL_OCCURRENCES=[7,30]`, `min_periods=3`; std NaN silently filled to 0 |
| 9 — Delta features | 1-occurrence diff, first-row NaN silently filled to 0 |
| 9b — Savitzky-Golay diagnostic | visual QA only, one sample point/year, data untouched |
| 9c — Drop lag-warmup rows | drops first 30 occurrences per `(point_id,event)` group (no valid 30-days-prior lag) — explicitly done **before** imputation sees them, to avoid imputation "papering over" a structural warmup gap |
| 10 — Correlation analysis | Pearson + Spearman, daytime-only, sampled ≤50,000 rows |
| 11 — VIF | same sampling, confirmed **nothing dropped** based on VIF — pure reporting |
| 12 — MinMax scaling | fit **only on the first 70% chronologically-sorted rows** — confirmed leakage-safe design |
| 13 — Final hard-gate validation | zero NaN/Inf in physical file, scaled train-portion within `[0,1]±1e-6` (val/test rows may legitimately fall outside — documented, not a bug), zero duplicates, required columns present |

**Verdict**: the 13-step pipeline is faithfully, precisely implemented exactly as its own docstring
and the project README describe. The two minor deviations found (RH clipped rather than NaN'd;
rolling-std/delta NaNs silently zero-filled) are low-impact and worth a one-line mention in a
methodology write-up, not a correctness concern.

## Outputs (expected schemas)

`climate_tamilnadu_points.csv`, `daily_aggregates_tamilnadu.csv`, `tier2_signature_tamilnadu.csv`,
`tamilnadu_cleaned_physical.csv` (Phase-3 input), `tamilnadu_cleaned_scaled.csv`, `scalers.pkl`,
`qc_report.txt`, plus correlation/VIF/skew diagnostic files.

## Dependencies

Requires Phase 1's complete point/time/NetCDF/JSON set (none of which exists yet, since Phase 1 has
never run). Phase 3 reads `tamilnadu_cleaned_physical.csv` and `tier2_signature_tamilnadu.csv`.

## Problems / risks

- **The deaccumulation function's correctness for this pipeline's actual CDS download behavior is
  unverified** — code-plausible, matches the intended pattern, but Rajasthan's own history shows this
  exact category of assumption can be wrong for a specific CDS request configuration. **This should
  be the first thing checked** when Tamil Nadu's pipeline is run for the first time.
- Everything else in this phase is code-complete and closely, faithfully mirrors either Rajasthan's
  design (combine/daily-aggregates) or its own detailed internal specification (the 13-step
  preprocessing pipeline) — the main risk across this phase is simply "never been run," not a
  specific known defect.

## Status

**COMPLETE AS CODE, NEVER RUN.** The 13-step preprocessing pipeline in particular is a genuinely
strong, carefully-designed piece of work — its leakage-safe scaling split and structural
lag-warmup-row handling are both more rigorous than what Rajasthan's simpler pipeline needed to do
(Rajasthan has no equivalent ML-feature-engineering step at this stage). The one open scientific
question is whether the deaccumulation assumption holds for TN's actual CDS response, which cannot
be answered without running the pipeline once.
