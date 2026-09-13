# 12 — Final Readiness Report: Uttarakhand

This file consolidates the implementation-issues list, the reproducibility audit and the overall
readiness verdict for the `era5-uttarakhand/` pipeline.

---

## Current implementation status

**Phases 1 through 8 are implemented and have been run end-to-end on real Uttarakhand data.**

| Phase | Script(s) | Status | Headline result |
|---|---|---|---|
| 1 — Data Collection | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` | **COMPLETE** | 45 points `UKP_0001–UKP_0045`, 10,475,711 population, 2016–2025 |
| 2 — Combine + Tier-2 repair | `02`, `02b` | **COMPLETE** | **493,155 rows = 45 × 3,653 × 3 exactly** — zero rows lost to the 3 h match window |
| 2 QA — Raw checks | `03`, `03b` | **COMPLETE** | Noon peaks GHI (timezone OK); **GHI MBE −211.4 W/m², r = 0.432** |
| 2 — Preprocessing & QC | `04`, `04c` ×2 | **COMPLETE** | 493,155 → **489,105 rows** (99.2 %); 36 → 89 columns; **zero residual NaN** |
| 3 — Climate Signature | `04b`, `04d` | **COMPLETE** | Two-tier merge; `Tm_target` fixed at **57 °C** for every point |
| 4 — Regime Clustering | `05`, `05b` | **COMPLETE** | **K = 5**, GMM full covariance; sizes **12 / 9 / 3 / 7 / 14**; silhouette 0.279 |
| 5 — Feasibility Filtering | `06`, `07`, `07b` | **COMPLETE, RESOLVED 2026-09** | 55-candidate database; window [52, 65] °C at Tm_target=57; **29/30/29/27/29 survivors — no longer identical**, since `07b`'s regime-cap bug is fixed (Clusters 1/2 get 55.16C/56.51C) |
| 6 — MCDM Ranking | `08` | **COMPLETE, RESOLVED 2026-09** | TOPSIS + GRA + PROMETHEE II + VIKOR + Borda (was TOPSIS+GRA only); **PureTemp 58 #1 in Clusters 0/2/3/4, PureTemp 53 in Cluster 1**; Kendall's W 0.708-0.842 per cluster (was pooled TOPSIS-vs-GRA ρ = −0.930, a two-method-era figure) |
| 7 — Physics Validation | `10_physics_validation.py` | **COMPLETE, RESOLVED 2026-09** | Backward-Euler grey-box tank model, two solver bugs fixed (verified against `scipy.integrate.solve_ivp`); **0% in [54%, 84%] SF band** (actual ~15-19%) — the previous 92%/+0.124 figures were inflated by those bugs and are not valid; current per-cluster ρ ranges −0.227 to +0.171, none significant |
| 8 — Recommendation Cards | `09` | **CODE COMPLETE, OUTPUT REGENERATED 2026-09 (still not committed — git-ignored)** | 5 cards; #1 is PureTemp 58 in four clusters (a legitimate shared result, not identical-by-bug) and PureTemp 53 in Cluster 1 |

---

## Strongest components

1. **Perfect sampling coverage with zero loss at the merge.** 493,155 rows against a theoretical
   maximum of 493,155. No `(point, date, event)` failed the 3-hour match on either data source,
   across 45 points and 3,653 days.

2. **The clear-sky cross-source agreement is a genuinely strong validation result.**
   MBE +5.314 W/m², r = **0.9923** against NASA POWER. Because `era5_GHI_clearsky` is computed
   locally by pvlib at the pipeline's own coordinates, instants and altitude, this single number
   simultaneously validates the point coordinates, the sun-event time matching, the nearest-hour
   lookup, the ERA5 grid snapping and the 1200 m altitude assumption's effect on the Ineichen
   model. It also isolates the GHI problem to the de-accumulated field.

3. **The Tier-2 repair worked and paid off.** 45/45 points, 0 skipped, 164,385 point-days, zero new
   downloads. It insulated the clustering matrix's entire temperature and solar block from the ERA5
   GHI magnitude anomaly — the canonical `GHI_daily_kWh`, `kt_mean`, `kt_std`, `SAI`,
   `cloudy_frac`, `CCI`, `DTR`, `Ta_*`, `HDD18`, `CDD24` and `seasonality` all come from NASA POWER.
   This is the single largest practical benefit of the two-tier design.

4. **Surgical preprocessing.** 99.2 % retention, with the only losses being exactly the
   4,050-row structural lag warm-up (45 × 3 × 30), and 100 % complete cases afterwards.

5. **Spatially coherent clusters found without clustering on geography.** `lat`/`lon` are excluded
   from the clustering matrix by construction, yet the five regimes are geographically contiguous
   and their temperature ordering is monotone (C3 25.0 °C > C0 22.8 > C1 19.0 > C4 18.2 > C2
   13.4 °C) across an ~11.6 K span. That is a real result for the signature's discriminating power.

6. **Honesty is built into the code, not retrofitted.** `07_feasibility_filter.py` lists the three
   Table-12 filters it does *not* apply. `07b_charging_feasibility.py` opens with "IMPORTANT
   HONESTY NOTE — read before using this" and calls itself "a HEURISTIC PROXY, not a real collector
   thermal model." `08_mcdm_ranking.py` calls its AHP weights "an honest placeholder, not a claimed
   AHP result" and contains a purpose-built diagnostic that detects and explains the
   identical-#1-across-clusters outcome, offering two honest reporting options. This is unusual and
   should be credited.

7. **The verification suite preserved the run.** With `data/raw/`, `data/processed/` and
   `data/preprocessed/` all git-ignored, `verify_01`…`verify_04` and the Objective 1 plot set are
   the only surviving record of what the pipeline produced. Every observed number in this
   documentation set was recovered from them.

---

## Weakest components — ALL FIVE RESOLVED (2026-09), kept as a historical record of what was found

1. **~~The ERA5 all-sky GHI is roughly an order of magnitude below physical expectation~~ RESOLVED.**
   Confirmed mechanism: `deaccumulate()` assumed ERA5's old cumulative-since-reset convention; the
   CDS/cfgrib pipeline actually delivers per-step values, so `diff()` computed a "delta of hourly
   totals." Fixed to return the raw value directly. Noon GHI now ≈683 W/m² (was ≈61); cross-source
   agreement is now MBE +19.6 W/m², r=0.759 (was MBE −211.4/−602 W/m², r=0.432/-0.03 across two
   pre-fix runs). See `04_PHASE_2_AUDIT.md` Part A.3.

2. **~~Phase 5 and Phase 6 produce zero climate differentiation.~~ RESOLVED.** The root cause was a
   real bug in `07b_charging_feasibility.py`'s regime-cap normalization, not an inevitable
   consequence of a constant `Tm_target`. Fixed: Clusters 1 (Tm_target=55.16C, 30 survivors, #1 =
   PureTemp 53) and 2 (56.51C, 29 survivors) now genuinely differ from Clusters 0/3/4 (57C, 29/27/29
   survivors, #1 = PureTemp 58).

3. **TOPSIS and GRA's pooled anti-correlation (ρ = −0.930) was itself partly a bug artifact —
   RESOLVED.** TOPSIS was applying an inconsistent second normalization on top of the shared
   min-max-normalized matrix the other three methods (GRA/PROMETHEE/VIKOR) use — fixed. The current
   four-method Kendall's W is 0.708-0.842 per cluster, a materially healthier agreement picture.
   VIKOR's compromise check (also fixed — it was only testing one of two standard conditions) now
   correctly reports genuine ties as compromise sets (Clusters 1 and 2) rather than false winners.

4. **~~The 850 hPa pressure bound is mis-specified~~ — the bound itself is still open, but it no
   longer contaminates elevation.** `era5_P_atm`'s 37.1% one-sided NaN/imputation issue is real and
   still unfixed for that column specifically, but `elev_proxy` (which used to be built from it) has
   been replaced entirely by real `elevation_m` (ERA5 geopotential), which is no longer a PCA-block
   member derived from pressure.

5. **~~No per-point elevation, and three inconsistent altitude assumptions~~ RESOLVED (mostly).**
   `00c_attach_elevation.py` now attaches real per-point elevation (196-2510m) used by `02`'s solar
   geometry and `04b`'s signature. One inconsistency remains, smaller than before: `00b_build_
   suntimes.py` still uses a flat 0m altitude for sun-event timing, not yet updated to use
   `elevation_m` — a minor, still-open gap (see `03_PHASE_1_AUDIT.md`).

6. **K = 5 exceeds the source files' own recommendation** of "realistically 2-4" for a 45-point
   single-state fit, and cluster 2 has only 3 points carrying 3.2 % of population.

7. **Soft membership collapsed to 1.000 for all 45 points**, so the stated reason for choosing GMM
   over K-Means (partial membership at regime boundaries) is not realised in this run.

8. **No Phase 7, no Monte Carlo, no bootstrap stability, no external climate classification.** The
   K = 5 partition and the Top-3 ranking have no external validation of any kind — only internal
   method agreement, which is itself poor.

9. **59.1 % of the PCM database's flagged property cells are MICE-RF-PMM estimates**, and three of
   the five MCDM criteria (`TC_W_mK` 34–39/55 imputed, `cycles_confidence` 48/55,
   `rho_H_MJ_m3` 14/55) rest substantially on them. `09`'s caveat text attributes this only to "the
   literature-added candidates", which understates it — all 55 rows carry at least one imputed
   property.

---

## Implementation issues — consolidated and ranked

### Fixed, recorded in-code

1. **`09_recommendation_cards.py` medoid index bug.** `.iloc[]` was used on a label returned by
   `idxmin()` on a boolean-filtered, un-reset dataframe, throwing `IndexError` once the label
   exceeded `len(members)`. Fixed to `.loc[]`, with the reason recorded in a comment. This is the
   only bug-fix history preserved anywhere in the pipeline's code.

### RESOLVED 2026-09 (were "Open, high priority")

2. **~~`deaccumulate()`'s assumption is unverified~~ RESOLVED.** The verification this section
   prescribed was carried out: opened the raw `*_accum.nc` files (2016/2020/2025), confirmed the
   CDS/cfgrib pipeline delivers per-step values, not cumulative-since-reset — `diff()` was computing
   a "delta of hourly totals." Fixed. `era5_LW_down`'s 73.7%-below-floor fingerprint (the second
   independent symptom this section names) no longer appears in `C_qc_flag_counts.csv` post-fix.

3. **~~The measured cross-source disagreement is never acted upon.~~ RESOLVED — and the "no
   agreement-analysis script" premise was inaccurate even before the fix.** `03b_agreement_
   analysis.py` exists and computes a bias-decision branch; it just used to conclude
   `MANUAL_REVIEW` (given the pre-fix GHI disagreement) rather than `QUANTILE_MAP`. Post-fix, its
   decision is `QUANTILE_MAP` (MBE +19.6 W/m², r=0.759).

4. **`era5_P_atm`'s 850 hPa lower bound** still destroys 37.1% of that column, one-sidedly — this
   specific fix (widen the bound, or otherwise) has NOT been applied and remains genuinely open.
   What HAS changed: it no longer feeds `elev_proxy`/the clustering matrix, since elevation now
   comes from `00c_attach_elevation.py`'s real per-point value instead.

5. **~~Constant `Tm_target = 57 °C` produces identical results across all five regimes.~~
   RESOLVED.** `07b_charging_feasibility.py` was already being run before `07` — the remedy this
   section suggested — but a normalization bug made it a no-op. Fixed; Clusters 1/2 now get a real,
   differentiated `Tm_target`.

6. **Three of the five plan Table-12 filters are unimplemented** — 5th-percentile-day charging
   feasibility, corrosion veto, safety exclusion. Additionally, **the corrosion veto could not
   activate even if implemented**: all 55 candidates are organic, so `corrosion_class` is
   `low_organic` for every row. `NEXT_STEPS.md`'s expectation that the veto would "bite for
   high-monsoon-humidity Uttarakhand clusters" cannot be realised with this database.

7. **27-30 survivors per cluster (was 29 identically) still exceeds the pipeline's own comfort
   bound** (`07` status `HIGH` above 25; `VERIFICATION_METHODOLOGY.md` wants 10–50 % survival,
   actual ~49-55%). The 13 K-wide melting window admits over half the database in every cluster —
   this width itself, not the now-fixed identical-count bug, is the remaining open item.

### Open, medium priority

8. **The Hampel filter flagged 10.0 % of `era5_cloud_cover` and 7.2 % of `era5_GHI`** — a known
   weakness of univariate MAD filtering on bounded bimodal and high-variance-by-nature variables.
   114,004 values across five columns were replaced by imputation. Clouds are weather, not errors.

9. **Imputed and flagged climate cells are unmarked in the output.** 114,004 Hampel-NaN'd plus
   546,424 bounds-NaN'd values were imputed with no `{col}_imputed` flag. The *PCM* database carries
   such flags; the climate data does not.

10. **~~`RH_mean`/`wind_mean` reach the clustering matrix from the ERA5 side while Tier-2
    equivalents sit unused~~ RESOLVED, but not via a missing `CANON_MAP` entry (it already existed).**
    The real bug was a `_proxy`-suffix naming mismatch in the Tier-1 fallback columns; fixed. Both
    are now genuinely canonical NASA POWER values for every point (100% Tier-2 coverage).

11. **`GHI_mean` is still the one solar column the Tier-2 repair does not cover** (no `_proxy`
    suffix, no `CANON_MAP` entry) — structurally unchanged, but this no longer matters in practice
    since the underlying ERA5 GHI anomaly it used to carry is fixed (item 2 above).

12. **~~`int_wind_x_TaMinusTsoil` reduces algebraically to `3.0 × wind_mean`~~ RESOLVED.** This term
    has been removed entirely (before the 2026-09 session) — the signature now has 4 interaction
    terms, not 5, and wind is no longer double-weighted.

13. **No canonical cluster relabelling.** Cluster IDs come straight from `fit_predict` and are
    stable only because `random_state=42` is fixed. Any change to the signature matrix, `K_FINAL`
    or the sklearn version can permute them and silently invalidate the `cluster_id`-keyed joins in
    `07`, `08` and `09` — none of which verify provenance.

14. **`monsoon_index` uses JJAS while `SEASON_MAP` uses JJA.** Unreconciled, and `monsoon_index` is
    in the clustering matrix.

15. **`T_mains_est_C = Ta_mean − 2.0` is unsourced**, and `L_required` has no PCM
    fractional-contribution (`SHARE_PCM`) factor — the PCM is implicitly assumed to supply the whole
    night load. Non-binding in this run, but both should be stated.

16. **Plotting/verification defects** — the Tamil-Nadu map centre, the four `passes_all`-blind
    plots, the never-run `comparison_plots_uttarakhand.py`, the mislabelled agreement plot and the
    synthetic tank plot. All thirteen are tabulated in
    `11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`.

### Open, low priority

17. **`get_solarposition()`'s method is not pinned** in `02` while it *is* pinned in `00b`.
18. **`avg_sdirswrf`'s three-name matcher applies one unit convention to three fields** — a latent
    3600× hazard if `fdir` were ever matched.
19. **Bounds in `02` are narrower than in `04` and counted nowhere** — the `T_amb < −5 °C` cut in
    particular is state-inappropriate.
20. **`CSI = 0` is three-way ambiguous**; **`DHI` is a closure residual**; **`ETR` is computed and
    discarded**.
21. **Matched timestamps are not persisted**, so per-row temporal match quality is unauditable.
22. **`Tm_target_C` is a zero-variance column** in the clustering matrix.
23. **Stale text**: `NEXT_STEPS.md` and `07`'s warning string say "25 rows" for a 55-row database;
    `05c`'s docstring says "133 points"; `03_plots_raw.py`'s docstring cites northeast-monsoon
    climatology; `05_cluster_regions.py` cites plan v2.0 while everything else cites v3.0;
    `config.py` carries three dead paths; `PCM_data/01_preprocess.py`'s docstring describes an
    18-row dataset.

---

## Reproducibility audit

| Item | Status | Notes |
|---|---|---|
| Random seeds | **PASS** | `random_state=42` on GMM, K-Means, `IterativeImputer`, the `impute_zone` KMeans, and PCA |
| Path anchoring | **PASS (numbered scripts)** / **FAIL (verify + plot scripts)** | Every `0*.py` imports `config.py`; `verify_01`…`verify_04` and `generate_objective1_plots.py` use relative or hand-built paths |
| Download resumability | **PASS** | Status CSVs + on-disk size checks for both ERA5 and POWER; flushed after every entry |
| Deterministic sampling design | **PASS** | GADM + WorldPop + a fixed ERA5-aligned 0.25° lattice reproduce the same 45 points |
| API parameters | **PASS** | CDS variable lists, POWER parameter strings and the hour-window computation are all in version-controlled `.py` files |
| Time ranges | **PASS** | 2016-01-01 → 2025-12-31 hard-coded consistently in five scripts |
| Output naming | **PASS** | Consistent `{artifact}_uttarakhand.csv` convention |
| Scaler persistence | **PASS** | `scalers.pkl` written by `04` step 12 |
| **GMM / StandardScaler persistence** | **FAIL** | Neither `04b`'s `StandardScaler` nor `05`'s fitted `GaussianMixture` is saved. Re-running Phases 5–8 requires re-fitting from scratch. |
| **sklearn version recorded** | **FAIL** | No output CSV carries a version column |
| **Canonical cluster relabelling** | **FAIL** | Cluster IDs are seed-dependent only |
| **Cross-phase provenance checks** | **FAIL** | `07`, `08` and `09` join on `cluster_id` with no fingerprint verification that inputs came from the same `05` run |
| **`requirements.txt` / lockfile** | **FAIL** | None in `era5-uttarakhand/`; only a prose `pip install` line in `README.md` |
| **Full-chain orchestration script** | **NOT PRESENT** | No `run_all_uttarakhand.py`; scripts must be run manually in order |
| **`method=` pinned for solar position** | **PARTIAL** | Pinned in `00b`, not in `02` |
| ERA5 dataset version | **PARTIAL** | Not pinned beyond the CDS API; no download-date manifest per file |
| Download dates | **PARTIAL** | Status CSVs carry per-event timestamps, but they are git-ignored |
| **Outputs committed** | **FAIL** | `data/raw/`, `data/processed/` and `data/preprocessed/` are all git-ignored. `qc_report.txt`, `pca_loadings.csv`, `vif_report.csv`, `bic_selection_uttarakhand.csv`, `cluster_profiles_uttarakhand.csv`, `mcdm_topk_by_cluster.csv` and `recommendation_cards.md` are all absent from the repository. |
| Logging | **PARTIAL** | Console output is informative and `qc_report.txt` is comprehensive — but neither is committed |

### The three reproducibility gaps that matter most

1. **No committed outputs.** The `.gitignore` is defensible for `data/raw/` (gigabytes), but it also
   excludes every small, high-value artefact: `qc_report.txt` (the step-13 verdict),
   `pca_loadings.csv` (whose `elevation_m` weight `NEXT_STEPS.md` used to ask the student to
   inspect — that check has since been done and recorded, but the file itself is still
   uncommitted), `bic_selection_uttarakhand.csv` (the K-selection evidence),
   `cluster_profiles_uttarakhand.csv` (the population-weighted profiles that go straight into the
   results section), `mcdm_topk_by_cluster.csv` (which carries Kendall's W) and
   `recommendation_cards.md` (the results section itself). **A `!data/processed/**/*.csv` exception,
   or a `docs/uttarakhand/artifacts/` copy of the ~10 small result files, would close this at
   negligible cost** and would remove the need to recover numbers from plot internals.

2. **No model persistence and no canonical relabelling.** Together these mean Phase 4's output is
   only reproducible by re-running the fit, and any perturbation can permute cluster IDs that three
   downstream scripts join on without checking.

3. **No pinned environment.** With `IterativeImputer` still an experimental sklearn API and
   `get_solarposition()`'s default method unpinned, a version drift can change results silently.

### Recommended fixes, in order of effort-to-impact

1. **Commit the ~10 small result CSVs** (or add a `.gitignore` exception for them). Zero code
   change; the single largest improvement to auditability.
2. **Add `requirements.txt`** — `pip freeze > requirements.txt`. Zero code change.
3. **Pin `get_solarposition(method="spa")`** in `02_combine_uttarakhand.py`. One line.
4. **Add `df = df[df["passes_all"]]`** to the four feasibility plots and `verify_03`. One line each.
5. **Fix `TN_CENTER`** in `05d`/`05c` to the point-set mean. One line each.
6. **Fix `comparison_plots_uttarakhand.py`'s `BASE`** — drop the spurious `".."`. One line.
7. **Save `scaler` and `gmm` via `joblib`** in `04b`/`05`, and record `sklearn.__version__` in every
   output CSV.
8. **Relabel clusters canonically** (e.g. by ascending mean latitude) immediately after the GMM fit.
9. **Add `RH_mean` and `wind_mean` to `04b`'s `CANON_MAP`.** Two dictionary entries.
10. **Widen the `era5_P_atm` lower bound**, or attach real per-point elevation (plan "Repair 2").
11. **Verify `deaccumulate()`** against a raw `*_accum.nc` file.
12. **Add a `run_all_uttarakhand.py`** in dependency order: `02 → 02b → 04 → 04b → 05 → 06 → 07 →
    08 → 09`.

---

## What can already be used in the thesis

- The full Phase 1–6 + 8 methodology description.
- The **45-point population-weighted, ERA5-lattice-aligned, sun-event-aligned sampling design**,
  with its 10,475,711-person coverage and the verified 493,155-row full-coverage result.
- The **two-tier climate signature** and the Repair-1 rationale — including the demonstrable finding
  that Tier 2 insulated the clustering matrix from the pipeline's largest data defect.
- The **K = 5 clustering result** with its sizes, populations, geographic extents and the
  temperature-ordered profile — plus the genuinely interesting observation that spatially coherent
  regimes emerged *without* clustering on geography.
- The **55-row PCM database** with its full composition and its verified 59.1 % imputation
  footprint.
- The **feasibility filter design**, with the three unimplemented Table-12 filters named explicitly.
- The **MCDM methodology** — Gaussian Tm fitness, entropy/AHP blending, TOPSIS + GRA, Borda,
  Kendall's W — with the AHP component correctly described as a placeholder.
- **UPDATED (2026-09):** the identical-#1-across-regimes finding no longer holds as originally
  stated — it was traced to a real bug (`07b`'s regime-cap normalization error), now fixed. Cluster
  1 gets a genuinely different #1 (PureTemp 53); Clusters 0/3/4 legitimately share PureTemp 58 as a
  real finding, not a bug artifact.
- Every caveat in this documentation set, stated plainly (and updated where resolved).

## What cannot yet be claimed (updated 2026-09 — several items below ARE now resolved)

- ~~That any absolute solar-irradiance figure from this pipeline is correct.~~ **Now largely
  claimable** — the GHI deaccumulation bug is fixed and independently verified against NASA POWER
  (r=0.759). The ERA5 pressure/`era5_P_atm` 850 hPa truncation remains a separate, still-open issue
  for anything reading that column.
- ~~That different Uttarakhand climate regimes require different PCMs — this run shows the
  opposite.~~ **Partially resolved**: Cluster 1 now demonstrably needs a different PCM
  (Tm_target=55.16C vs 57C); Clusters 0/3/4 sharing a pick is a legitimate finding for those three
  regimes specifically, not evidence against the framework generally.
- That the Top-3 ranking is stable, externally validated, or physics-confirmed — **still true**;
  Monte Carlo now quantifies stability (37.8-39.3% Top-3 inclusion) and physics validation now runs
  (0% within the literature benchmark band, per-cluster ρ −0.227 to +0.171, none significant) —
  both are honest, weak-agreement results, not confirmation.
- ~~That RT60 is a clear winner~~ — RT60 is no longer even the consensus pick in the current
  four-method run (PureTemp 58/53 are). The current pick is decided across four methods with
  Kendall's W 0.708-0.842, materially more agreement than the old two-method Borda tie.
- That the K = 5 partition is stable (no bootstrap) or externally valid (no Köppen-Geiger) — still
  true, unaddressed by the 2026-09 fixes.
- That AHP informed the weights (it did not — a fixed placeholder prior was used) — still true.
- ~~That soft cluster membership captured boundary behaviour (all 45 points came back at 1.000).~~
  **Marginally improved**: `covariance_type` was fixed from `full` to `diag` (addressing genuine
  overdetermination), but 43/45 points still round to 1.000 in the current run — the practical
  finding is largely unchanged, see `06_PHASE_4_AUDIT.md`.

## Prerequisites for a final, non-provisional result

1. **Verify and, if necessary, fix `deaccumulate()`** — one inspection of a raw `*_accum.nc` file.
   Everything solar downstream is provisional until this is settled.
2. **Break the constant-`Tm_target` degeneracy** — run `07b` before `07`, or report the convergence
   as a finding with `08`'s own wording.
3. **Fix the `era5_P_atm` bound or attach real elevation**, then re-run `04 → 04b → 05` and check
   whether the K = 5 partition survives.
4. **Commit the small result CSVs** so the numbers in a paper are traceable to files rather than to
   plot internals.
5. *(Optional but high value)* **Implement a minimal Phase 7** — every input it needs is already on
   disk (`09_PHASE_7_AUDIT.md`), and it is the designated place for regime differentiation to
   appear.

---

## Final verdict

**READY WITH MINOR FIXES — Phases 1, 2 (structure), 3 and 4.** The sampling design, the merge, the
Tier-2 repair, the two-tier signature and the clustering are methodologically sound, well
documented in-code, and produced clean, internally consistent, cross-checkable results. The fixes
needed are small and specific.

**NOT READY AS A FINAL RESULT — any solar-magnitude claim.** The ERA5 all-sky GHI is roughly an
order of magnitude low, the pipeline measured this, and no correction was applied. The Tier-2
design limits the damage to `GHI_mean` within the clustering matrix, but no absolute irradiance
figure from this pipeline should be published until `deaccumulate()` is verified.

**NOT READY AS A FINAL RESULT — Phases 5 and 6.** Not because the code is wrong, but because the
result is degenerate by construction: constant `Tm_target` plus a non-binding latent-heat floor
gives identical survivors and an identical winner in all five regimes, and that winner is a Borda
tie between two methods that disagree at ρ = −0.930. The pipeline correctly detects and explains
this. It is a reportable finding, not a final recommendation.

**CORRECTLY DECLARED FUTURE WORK — Phase 7.** Named as absent in three source files, with a minimal
specification given and every required input already on disk.

The pipeline's defining characteristic is that **it documents its own limitations in code rather
than in retrospect** — the unimplemented filters, the heuristic proxy, the placeholder AHP weights,
and the constant-`Tm_target` diagnostic are all self-declared. The gap is not honesty; it is that
two measured problems (the GHI disagreement and the pressure-bound truncation) were observed and
then not acted upon.
