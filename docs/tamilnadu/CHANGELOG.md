# CHANGELOG — Objective 1, Tamil Nadu Pipeline

Everything changed in response to the two review documents
(`Objective1_TamilNadu_STATUS_AND_TODO (2).md` and `FIXES.md`). Grouped by
file. "Was" = what the earlier version did; "Now" = what changed.

---

## 2026-09-17 — Orphaned scripts/data cleanup + a real reproducibility gap fixed

Full audit of every root `.py` file against what `run_all_tamilnadu.py` and the docs actually
reference as current, prompted by "are there unused plots/data/code still lying around."

### Deleted — orphaned/retired scripts (none referenced by any current script or doc, verified with `grep` before removal)
- **`07b_charging_feasibility.py`** — retired 2026-09-08 (superseded by Constraint 6 in
  `07_feasibility_filter.py`), but never actually deleted. Worth flagging why this one mattered
  more than a dead file: it wrote to `data/processed/clustering/cluster_profiles_tamilnadu.csv`
  — the **same file** `05_cluster_tamilnadu.py` (current Phase 4) owns — so leaving it in place
  was a live risk of silently clobbering current cluster profiles if ever run by mistake, not
  just clutter.
- **`11_level_b_seasonal_analysis.py`** — renamed to `11_seasonal_pcm_sensitivity.py` 2026-09-08;
  this was the pre-rename file, never deleted. Its orphaned output
  (`data/processed/pcm/level_b_seasonal_{summary.md,topk.csv}`, dated 2026-09-14) deleted too.
- **`12_mcdm_interactive_plots.py`** — read/wrote pre-unification dead filenames
  (`mcdm_final_results_complete.csv`, `mcdm_full_scores_by_cluster.csv`), wrote to the
  already-deleted `data/plots/mcdm/`, not referenced in `11_PLOTS_GUIDE.md` or anywhere current.
- **`04d_signature_interactive.py`** — `CHANGELOG.md` already claimed this was "deleted
  2026-09-08" in an earlier entry; it physically still existed. Actually deleted now.
- **`03e_interactive_raw_plotly.py`, `03f_interactive_raw_folium.py`,
  `04e_interactive_preprocessed_plotly.py`, `04f_interactive_preprocessed_folium.py`** — not
  referenced in `11_PLOTS_GUIDE.md`'s "How to Run" list or `run_all_tamilnadu.py`; `04e`/`04f`
  specifically generated the already-deleted orphaned `data/plots/preprocessed_interactive/`
  (3.6GB, see the first 2026-09-16 plot-cleanup entry). All four confirmed byte-identical
  (`diff -q`) to copies already preserved in `plots_tamilnadu_ppt/interactive_plots/` — deleted
  only the root duplicates, left the presentation folder's copies untouched.

### Deleted — orphaned data
- `data/processed/pcm/mcdm_final_results_complete.csv` (Sep 5) and
  `data/processed/pcm/mcdm_full_scores_by_cluster.csv` (Sep 14) — dead pre-unification MCDM
  output files. `plots/generate_tamilnadu_plots.py` was already fixed to stop reading the latter
  (2026-09-16 plot cleanup entry); the stale files themselves were left behind until now.

### Fixed — a real bug, not just cleanup
- **`plots/verify_04_ranking_tamilnadu.py`** was still functionally reading
  `mcdm_full_scores_by_cluster.csv` (not just a comment, unlike the other scripts checked) —
  harmless in practice (the loaded `full` dataframe is never used beyond a shape print, all 6
  plots use `topk` only) but was silently reading a stale, now-deleted file. Pointed at
  `mcdm_full_rankings.csv`.
- **`run_all_tamilnadu.py`'s `SETUP_SCRIPTS` never included `00c_attach_elevation.py`** — added
  2026-09-16 but never wired into the runner. A fresh `--include-setup` run would have silently
  skipped elevation attachment entirely, leaving `02_combine_tamilnadu.py` to fall back to its
  flat `DEFAULT_ALT_M` for every point — exactly the bug elevation was added to fix, reintroduced
  by omission. Inserted right after `00a_build_population_grid.py` (its only dependency).
  Verified with `--dry-run --include-setup`.
- **`docs/era5_tamilnadu/RUN_TAMILNADU_PIPELINE.md`** — a doc kept as "current" in the earlier
  root-.md consolidation was still telling readers to run the retired `07b_charging_feasibility.py`
  and the renamed-away `11_level_b_seasonal_analysis.py`, and didn't mention `00c` at all. Fully
  corrected (run order, script-purpose table, and the "important notes" section).

---

## 2026-09-16 (third pass) — `thermal_margin` criterion added, physics-agreement investigation closed

Follow-up to the "OPEN ISSUE" from the first 2026-09-16 pass (negative Cluster-0 Spearman ρ
between MCDM consensus rank and physics-simulated performance). Full narrative in
`09_PHASE_7_AUDIT.md`; this entry is the terse changelog version.

### `08_mcdm_ranking.py` — 9th criterion added: `thermal_margin`
- **Was**: 8 Table-13 criteria (see the first 2026-09-16 entry below for the `Tm_target_capped_C`
  and asymmetric-σ fixes already tried and confirmed inert on rank order).
- **Now**: `thermal_margin = Tm_target_capped_C − Tm` (benefit criterion, headroom below the
  achievability ceiling) added to `CRITERIA`/`CRITERIA_TYPE`/`LITERATURE_WEIGHTS_TABLE13` and
  computed in `build_criteria_matrix()`. Its 0.10 AHP prior was carved out of `Tm_fitness`'s
  original 0.24 (now 0.14 + 0.10 = 0.24 combined), not added on top, so the "Tm-related" share of
  the weight pie is unchanged relative to the other 7 criteria — a deliberate, stated design
  choice, not an arbitrary weight injection. Blended through the same entropy+AHP mechanism as
  every other criterion, so its actual influence per cluster is data-driven, not hardcoded.

### Result: fixed Cluster 0, did not fix Clusters 1/2 — and that's the final, closed finding
- **Cluster 0**: Spearman ρ (MCDM rank vs. simulated solar fraction) **-0.595 → +0.381**. Its new
  consensus #1 (`RT57HC`) now lands inside the 54-84% benchmark band (55.4% simulated). Real,
  traceable improvement — Cluster 0's problem (its old #1, `n-Octacosane`, sat right at the
  achievability ceiling and stalled on below-average solar days) is exactly what this criterion
  measures.
- **Cluster 1**: ρ +0.176 → +0.103 (no real improvement). **Cluster 2**: ρ +0.048 → +0.024 (no
  real improvement). Diagnosed why with a clean test: `CrodaTherm 60` has *identical* Tm/latent
  heat/thermal conductivity in every cluster, yet its simulated performance is 38.3% (worst) in
  Cluster 0, 79.2% (best) in Cluster 1, 65.9% (best) in Cluster 2 — proof the real driver is a
  dynamic interaction between a PCM's Tm and each cluster's specific 10-year day-by-day weather
  trajectory, not any static per-candidate property. No static MCDM criterion — margin-based,
  delivery-distance-based (also tested, also inconsistent: ρ of `|Tm-60°C|` vs. simulated SF was
  +0.558/-0.104/-0.158 across the three clusters, wrong sign in two of three), or otherwise — can
  capture that.
- **Statistical power caveat, should have been raised earlier**: none of these correlations, in
  either the before or after state, are statistically significant (p=0.12-0.96 throughout,
  n=8-10 per cluster). Every number in this whole investigation is a descriptive point estimate,
  not a significance-tested claim.
- **Decision: stop here.** Iterating the MCDM criteria further to force a positive correlation in
  Clusters 1/2 would mean overfitting the ranking to this one physics simulation's specific
  parameterization — which defeats the purpose of having two independent checks (MCDM ranking and
  physics validation) in the first place. The disagreement in Clusters 1/2 is kept and reported as
  a genuine finding: static MCDM criteria and dynamic physics simulation can legitimately
  disagree, and that is exactly the kind of thing Phase 7 exists to catch.

### Side effects of the new criterion (reported honestly, not just the improvement)
- Consensus Top-1 is now `RT57HC` in **all three clusters** (was `n-Octacosane (C28)` / `PureTemp
  60` / `PureTemp 60`) — flagged in `08_PHASE_6_AUDIT.md` as worth an independent sanity check,
  not silently accepted, since it reproduces the pre-2026-09-08-unification "same PCM everywhere"
  pattern (for a different underlying reason this time).
- Kendall's W (4-method agreement) moved 0.786→0.848 in Cluster 0 (up) but 0.839→0.477 in
  Cluster 1 (down sharply, crossing into "ambiguous") and 0.815→0.780 in Cluster 2 (down
  slightly) — a real trade-off, not a free win.

### Docs updated to match
`08_PHASE_6_AUDIT.md` and `09_PHASE_7_AUDIT.md` rewritten with the full final state (9 criteria,
final Top-1 picks, closed investigation). `recommendation_cards.md` and
`physics_validation_spearman.csv` already reflect this run (regenerated as part of it, no
separate action needed).

---

## 2026-09-16 (second pass) — Plot directory cleanup + root .md consolidation

### Plots: `data/plots/` 5.7GB → 139MB
- **Deleted, orphaned** (generators not referenced in `run_all_tamilnadu.py`
  or `docs/era5_tamilnadu/11_PLOTS_GUIDE.md`): `data/plots/preprocessed_interactive/`
  (3.6GB — its generators, `04e_interactive_preprocessed_plotly.py` and
  `04f_interactive_preprocessed_folium.py`, are undocumented anywhere in the
  maintained pipeline); `data/plots/mcdm/` (5.7MB — its generator,
  `12_mcdm_interactive_plots.py`, reads/writes pre-unification dead
  filenames — `mcdm_final_results_complete.csv`, `mcdm_full_scores_by_cluster.csv`
  — that the current engine no longer produces).
- **Deleted, exact duplicates**: `plots/verify_clustering/`,
  `plots/verify_feasibility/`, `plots/verify_preprocessing/`,
  `plots/verify_ranking/`, `plots/comparison/` — each `plots/verify_0N_*.py`
  and `plots/comparison_plots_tamilnadu.py` writes to `data/plots/...` per
  their own `OUT_DIR`; these were stale leftover copies sitting inside the
  `plots/` scripts folder, confirmed identical/near-identical via `diff -rq`.
- **Regenerated** (documented/required per `11_PLOTS_GUIDE.md`, were stale —
  built before this session's elevation/k=3/`TM_TARGET_C` corrections):
  `03_plots_raw.py`, `03b_agreement_analysis.py`, `04c_postprocess_plots.py`,
  `04c_interactive_postprocess_qc.py`, `05b_cluster_interactive.py`,
  `05d_plots_comprehensive.py`, `plots/comparison_plots_tamilnadu.py`,
  `plots/verify_01..04_*.py`, and `03b_interactive_raw_qa.py` (this last one
  specifically: its stale output was 2.0GB; the fresh regeneration is
  **21MB** — a ~100x reduction, apparently a pre-existing bloat issue in the
  old output, not anything this session's changes caused).

### Plots, round 2 — two more orphaned/stale directories found and removed
- `data/plots/interactive_explorer/` (Sep 2, stale) — `05c_explore_interactive.py`'s
  cached Streamlit map; regenerates automatically next time that app runs.
- `data/processed/signatures/interactive/` (Sep 12, stale, genuinely
  orphaned) — `04b_climate_signature.py` does not write to this path at
  all; it was dead output left behind by `04d_signature_interactive.py`
  (already deleted 2026-09-08). Current signature interactive plots are
  written directly by `04b` to `outputs/signature_*_tamilnadu.html`.
  `docs/era5_tamilnadu/11_PLOTS_GUIDE.md` corrected to match.
- **Deliberately NOT touched**: `plots_tamilnadu_ppt/` (6.4MB, Sep 5) —
  looked stale/orphaned by the same pattern as the two above, but is
  actively referenced by `../presentation_1.tex` and
  `../progress_review_slides_explained.md` (one directory up, outside this
  pipeline folder) — a real presentation asset, not a pipeline byproduct.
  Left alone.

### Root `.md` files: 15 → 8
- **Deleted, empty**: `12_FINAL_READINESS_REPORT.md` at repo root (0 bytes —
  a stray duplicate; the real content is `docs/era5_tamilnadu/12_FINAL_READINESS_REPORT.md`).
- **Deleted, fully subsumed**: `PREPROCESSING_STEPS.md` (`README_PREPROCESSING.md`
  covers everything it did — 03/04/04b/05 — plus `02b`/`04c`, in more
  current detail; verified line-by-line before deleting, no unique content).
- **Deleted, completed/superseded**: `NEXT_STEPS.md` (a 4-day sprint plan;
  every item is now done — its "still open" items live on in
  `docs/era5_tamilnadu/00_MASTER_OVERVIEW.md`, which is the correct home
  going forward — references to it updated in `PIPELINE_FILE_GUIDE.md` and
  `05_cluster_regions.py`).
- **Deleted, redundant**: `PLOT_INTEGRATION_GUIDE.md` (same ground as
  `docs/era5_tamilnadu/11_PLOTS_GUIDE.md`, and its "84 files across 12
  directories" count is now wrong after the plot cleanup above).
- **Deleted, superseded, in order of dependency**: `FINALIZATION_SUMMARY.md`
  (a manifest describing the next two files) → `FYP_Tamil_Nadu_Enhanced_Audit.md`
  (a "with real outputs" report whose embedded numbers predate 3+ major
  pipeline revisions since Sep 5) → `FYP_Tamil_Nadu_Phase_Audits_Consolidated.md`
  (its own header says it's a literal concatenation of the `docs/era5_tamilnadu/`
  files — 100% redundant with the actively-maintained folder those came
  from, now also stale on top of that).
- **Updated, not deleted**: `README_PREPROCESSING.md` (stale `Tm_target=57°C`
  and flat-150m-elevation claims corrected to the current 67°C / real
  elevation state; `00c_attach_elevation.py` added to its run order) and
  `PIPELINE_FILE_GUIDE.md` (added the missing `00c_attach_elevation.py`
  entry; fixed its dangling `NEXT_STEPS.md` reference).
- **Kept as-is**: `Objective1_PCM_Climate_Framework_Plan_v3.md` and
  `Objective2_PCM_Design_Optimization_Workflow.md` (foundational specs, not
  progress reports — out of scope for this cleanup), `README.md`,
  `RUN_TAMILNADU_PIPELINE.md`, `CHANGELOG.md` (this file),
  `CLEANUP_CANDIDATES.md` (the prior cleanup's own audit trail).

## 2026-09-16 — Elevation integration, k-selection reverted to auto, two path-resolution bugs, Phase 6 Tm-target fix, physics-disagreement diagnosis

This pipeline copy (`new_obj/tamilnadu_pipeline/`) was moved one directory
level deeper than the original `PCM-Selection-ML-model/era5-tamilnadu/`
layout several scripts' relative-path logic still assumed. Fixing that
surfaced a second, more consequential issue: the reachable
`pcm_shared_config.py` had a physics-corrected `TM_TARGET_C` that wasn't
wired up before. Full re-run of the CORE chain end-to-end.

### `00c_attach_elevation.py` — RUN FOR THE FIRST TIME on this pipeline
- **Was**: Tamil Nadu had no elevation script at all — `02_combine_tamilnadu.py`
  used a flat `DEFAULT_ALT_M = 150` for every point's solar geometry /
  clear-sky irradiance, unlike Rajasthan which already had this script.
  (`03_PHASE_1_AUDIT.md` documented this as a known Tamil-Nadu-specific gap.)
- **Now**: downloads ERA5's time-invariant geopotential (one CDS request,
  cached under `data/raw/era5/invariant/`), attaches real per-point
  `elevation_m` to `population_grid_points.csv` (range -0.0–1,283.4 m, mean
  282.4 m across the 133 points — the ERA5 ~28 km grid smooths the true
  Nilgiris peak, an accepted, documented limitation). `02_combine_tamilnadu.py`
  already had the `elevation_m`-reading code path in place (just never had
  data to read); no change needed there.

### `config.py`, `06_build_pcm_database.py`, `08_mcdm_ranking.py` — path-resolution bug fixed
- **Was**: all three assumed `pcm_shared_config.py` / `PCM_data/` live
  directly under this pipeline's PARENT directory (`BASE_DIR.parent`) — true
  under the original `PCM-Selection-ML-model/era5-tamilnadu/` layout, false
  here (`new_obj/tamilnadu_pipeline/`, parent is `new_obj/`, which has
  neither). Every script that imports `config` was crashing immediately with
  `ModuleNotFoundError: No module named 'pcm_shared_config'` — this blocked
  the ENTIRE pipeline, unrelated to elevation.
- **Now**: each does a small candidate-path search (original layout first,
  then `BASE_DIR.parent.parent / "PCM-Selection-ML-model"`), raising a clear
  error only if neither resolves. Non-destructive, portable to either layout.

### `pcm_shared_config.py` resolution — `TM_TARGET_C` correction now actually in effect
- **Was**: two on-disk copies of `pcm_shared_config.py` had diverged —
  `jammalamadugu/PCM-Selection-ML-model/pcm_shared_config.py` (updated
  2026-09-14, `T_DELIVERY_C` corrected 50→60°C, since the night-discharge
  formula's 300L/7h capability figure (Avargani et al. 2021) is only
  validated AT 60±2°C delivery, not 50°C) vs.
  `jammalamadugu/All_objective_all/PCM-Selection-ML-model/pcm_shared_config.py`
  (uncorrected, still 50°C). Whichever was reachable before this pipeline
  moved into `new_obj/` is what produced every `Tm_target_C=57.0` result
  documented earlier in this project's history (including `14_PHASE_6_
  MCDM_RANKING_EXPLAINED.md` / `15_CLUSTER0_BUMP_CHART_EXPLAINED.md` — now
  superseded, see their headers).
- **Now**: the path fix above resolves to the corrected copy.
  **`TM_TARGET_C = 67.0`** (was 57.0) drives every cluster's feasibility
  window and MCDM `Tm_fitness`/`f_Tm` criterion from this run onward.

### `05_cluster_tamilnadu.py` — `LEVEL_A_K_OVERRIDE` reverted to auto (`None`)
- **Was**: `LEVEL_A_K_OVERRIDE = 5`, forced 2026-09-14 ("student decision"),
  overriding the documented 3-tier `suggest_k()` cascade's own suggestion.
- **Now**: reverted to `None`. Re-checked the cascade against the fresh,
  elevation-corrected signature data — same conclusion as before: **k=3**
  is the only k in the expected single-state range (2-4) that lands in the
  realistic silhouette band, with the highest bootstrap-ARI (0.630) of any
  in-range k. The forced k=5 was the *worst* of {2,3,4,5} on both GMM
  silhouette (0.203, lowest) and bootstrap-ARI (0.557, lowest) — not just
  "not the top pick," measurably the least internally-cohesive, least
  resample-stable option among the plausible candidates. See
  `06_PHASE_4_AUDIT.md`.

### `08_mcdm_ranking.py` — `Tm_fitness` criterion now scored against the achievable ceiling, not the raw delivery target
- **Was**: `tm_target = prof.Tm_target_C` (line 991) — the raw, uncapped
  delivery-temperature target (67.0°C). `07_feasibility_filter.py`'s own
  Constraint 6 already restricts every survivor to `Tm ≤
  Tm_target_capped_C` (the kt_worst_month-derived, literature-anchored
  ceiling on what each cluster's real solar charging can actually reach —
  61.94/61.09/61.37°C for clusters 0/1/2, all well below the raw 67°C
  target). Scoring the Gaussian target-fitness criterion — which dominates
  the ranking at 66–79% entropy weight, well past the script's own 40%
  domination flag — against the wrong, unreachable target meant "closest to
  target" was mathematically identical to "highest Tm in the survivor pool"
  in every cluster, since every survivor sits below both numbers regardless.
- **Now**: `tm_target = prof.Tm_target_capped_C`. More physically grounded
  (raw `f_Tm` values shifted substantially, e.g. n-Octacosane 0.402→0.996 in
  Cluster 0), but **did not change the rank order or the Cluster 0 Spearman
  ρ** (still -0.595) — see the open issue below for why, and don't assume
  this fix alone resolves the physics-MCDM disagreement.
- **Also added, then found to be inert on this dataset**: an asymmetric
  Gaussian (`SIGMA_TM_UPPER_K=2.0` for `Tm > tm_target`, tighter than the
  unchanged `SIGMA_TM_LOWER_K=4.0` below it) — the plan doc's own stated
  "physically better motivated" but previously unimplemented extension
  ("Tm too high is worse than too low"). Had **zero effect**: Constraint 6
  already excludes every candidate above `Tm_target_capped_C` by
  construction, so no survivor ever falls into the `sigma_upper` branch.
  Left in place (harmless, theoretically correct for any future cluster
  with above-target survivors) but does not address the current finding.

### OPEN ISSUE (not fixed, root-caused) — MCDM consensus rank vs. simulated physics performance
- `10_physics_validation.py`'s Spearman check: **ρ = -0.595 (Cluster 0),
  +0.176 (Cluster 1), +0.048 (Cluster 2)**, mean -0.124. Only 6/26 simulated
  candidates land in the 54-84% benchmark band. The negative Cluster-0
  result survived BOTH fixes above.
- **Diagnosed mechanism**: no criterion in the current 8-criterion set
  measures thermal MARGIN below the achievability ceiling — only proximity
  TO it. A candidate sitting right at the ceiling (e.g. n-Octacosane,
  61.6°C vs. a 61.94°C ceiling) scores best on `f_Tm` but stalls
  (incomplete melt) on a below-average solar day, hurting real annual solar
  fraction (45.7%, not in-band). A candidate with real margin (e.g.
  n-Heptacosane, 59.0°C, MCDM rank 4) melts fully and reliably and
  outperforms it in simulation (60.2%, in-band). This is a genuine gap in
  the plan doc's Table 13 criteria set, not a coding error — flagged here
  for a future methodology decision (e.g. an explicit margin/reliability
  criterion), not resolved in this pass.

### Level B re-run (`05a_level_b_regime_shift_tamilnadu.py`) — fresh numbers, was NOT stale-carried
- k=3 (was k=4 in the 2026-09-08 doc), bootstrap-ARI 0.8095. Regime-shift
  fraction 114/133 points (85.7%, was 90.2%). Season-tautology ARI=0.371,
  NMI=0.447 (was 0.501/0.602) — still "moderate agreement," same reading as
  before, numbers shifted because Level A's k changed underneath it and the
  elevation correction shifted the underlying signature slightly.

### `README.md` — updated to match current pipeline scope
- **Was**: documented only Phase 1-2 (data collection through `02_combine_tamilnadu.py`),
  still described the flat 150m elevation approximation, and ended with an early,
  since-superseded "Next: cross-region clustering" plan referencing a `03_cluster_regions.py`
  that isn't the actual Phase 4 script.
- **Now**: documents the full Phase 1-8 pipeline, points to `run_all_tamilnadu.py` /
  `RUN_TAMILNADU_PIPELINE.md` for the run guide, corrects the elevation note to describe
  `00c_attach_elevation.py`, and adds a "Current status" section summarizing the 2026-09-16
  results (k=3, Top-1 picks, the open MCDM-vs-physics disagreement) with a pointer to
  `docs/era5_tamilnadu/00_MASTER_OVERVIEW.md`.

### Repo cleanup — `CLEANUP_CANDIDATES.md`'s reviewed list acted on
- Deleted (none were git-tracked, so this is not recoverable via `git checkout` —
  each item was re-verified against current disk state first, and the two "diff first"
  items were independently re-diffed and confirmed exact/no-difference duplicates before
  removal): `Objective 1/` (860 KB, older pipeline snapshot missing Phase 7/Level B),
  `data - Copy/` (134 MB, pre-Phase-7/8 data backup), `docs/era5_tamilnadu_1.zip` (~64 KB,
  older zip duplicate), `docs/era5_tamilnadu.zip` (~76 KB, now-stale zip of the actively-edited
  `docs/era5_tamilnadu/` folder), `sources_extracted/sources/` (~418 KB, partial older
  extraction, re-diffed: zero content differences from `sources/` on common files),
  `sources/references - Copy.bib` (20 KB, re-diffed: byte-identical to `references.bib`),
  `__pycache__/` (root + `plots/__pycache__/`, regenerates automatically). ~135 MB freed total.
  See `CLEANUP_CANDIDATES.md` for the full per-item record (kept, updated with outcomes rather
  than deleted, since it's the audit trail for this cleanup).
- **Not touched**: `PCM_data/PCM_data/` (source-of-truth generator, not redundant),
  `docs/era5_tamilnadu/` (the live docs), `plots/` vs `plots_tamilnadu_ppt/` (not reviewed
  further — still an open item).

### `11_seasonal_pcm_sensitivity.py` — now degenerate under the corrected 67°C target
- **Was** (2026-09-08, Tm_target=57°C): 3/12 (cluster, season) combinations
  flipped the #1 PCM.
- **Now** (Tm_target=67°C): every season in every cluster has fewer than 2
  survivors once `L_required` is recomputed per-season — the seasonal
  breakdown produces no meaningful comparison under the corrected target.
  Script still exits 0 (this is a legitimate finding given the input, not a
  crash); flagged here so the old 3/12 number isn't quoted as current.

---

## 2026-09-08 — Phase 5 unification with Rajasthan

Phase 5 (feasibility filtering) was unified with `era5-rajasthan/07_feasibility_filter.py`.
Rajasthan is the reference for the LOGIC; Tamil Nadu's filenames
(`feasibility_survivors_by_cluster{,_kappa_calibrated}.csv`) are canonical
for BOTH states (Rajasthan's `feasibility_survivors_rajasthan*.csv` were
renamed to match).

### `07_feasibility_filter.py` — rewritten to the unified architecture
- **Was**: 7 filters, no charging-feasibility constraint, no κ-calibration,
  read `Tm_target_C_regime_capped` from `07b` when present, emitted one
  output with `passes_all`.
- **Now**: 8 constraints in Rajasthan's exact order. **Constraint 6 =
  charging feasibility**, `Tm ≤ Tm_target_capped_C`, taken directly from
  Phase 3's `kt_worst_month`-derived ceiling in
  `cluster_profiles_tamilnadu.csv` (not re-derived). `flag_unreported` /
  `flag_unknown` semantics on C3/C4/C5 (C4, C5 never exclude). Constraint 7
  (corrosion veto) and 8 (safety) ported verbatim — both structurally inert
  on the current DB (0 salt hydrates; flammability is Yes/No not a grade).
  `calibrate_kappa_for_cluster()` ported verbatim (κ 0.7→0.0 by 0.1, target
  8-20 survivors/cluster, evaluated at the primary run's final
  melting-window relaxation round). Emits BOTH
  `feasibility_survivors_by_cluster.csv` (fixed κ=0.7) and
  `feasibility_survivors_by_cluster_kappa_calibrated.csv`. Both stamped with
  `upstream_cluster_profile_fingerprint`. Emits `survives_all` plus
  `passes_all` (alias) for downstream compatibility.

### `07b_charging_feasibility.py` — DELETED
- The `REFERENCE_GOOD_DAY_TEMP` / `MIN_ACHIEVABLE_TEMP` heuristic that wrote
  `Tm_target_C_regime_capped` is superseded by Constraint 6. Removed from
  `run_all_tamilnadu.py`. There is now one charging-feasibility path.

### `06_build_pcm_database.py`
- **Was**: `family` column fell back to `np.where(df.get("is_rt_line", 0) == 1, …)`.
- **Now**: `family = df["manufacturer"]` directly (matches Rajasthan;
  `is_rt_line` is not in the canonical preprocessing output). Confirmed this
  script is a thin builder over `PCM_Properties_cleaned_mice_pmm_detailed.csv`
  — it has no independent imputation loop.

### `08_mcdm_ranking.py`, `09_recommendation_cards.py`
- Now read `feasibility_survivors_by_cluster_kappa_calibrated.csv` (mirrors
  Rajasthan's `08`). Survivor-boolean reads tolerate `survives_all` /
  `passes_all`. `08`'s "#1 identical statewide" note no longer points at the
  retired `07b`.

### `11_seasonal_pcm_sensitivity.py`
- `tm_target` now prefers `Tm_target_capped_C`, then legacy
  `Tm_target_C_regime_capped`, then `Tm_target_C`.

### Data layout
- The `data/processed/processed/` path-duplication bug was already fixed in
  `config.py` / `04b_climate_signature.py`; its stale mirror tree
  (`era5-tamilnadu/data/processed/processed/`, 35 files) was **deleted**.

### `05_cluster_tamilnadu.py` (cross-region Phase 4, standalone — not in run_all)
- Fixed `REGION_FILES["Rajasthan"]`: was `SIGNATURE_DIR.parent.parent /
  "era5-rajasthan" / … / "signatures" / …` which resolved to
  `era5-tamilnadu/data/era5-rajasthan/…` and used a `signatures/` segment
  Rajasthan doesn't have. Now `BASE_DIR.parent / "era5-rajasthan" / "data"
  / "processed" / "climate_signature_rajasthan.csv"`.

---

## v3.2 Bug Fixes (Phase 7 physics solver — critical correctness)

Found during a cross-check of the Tamil Nadu pipeline against the
Rajasthan pipeline's documented, already-fixed bug history. Both bugs
below are the *same bug classes* Rajasthan's `physics_lib.py` audit
already names and fixes — this pipeline had independently reintroduced
them. v3.1 fixed the "no ambient loss" symptom, but the tank still
never actually cooled overnight because of these two solver bugs, so
Phase 7's output was still stuck at the pre-v3.1 failure signature
(85–100% solar fraction, 0–1 cycles/year, 0% in the 54–84% benchmark
band) even after v3.1 shipped.

### `10_physics_validation.py` — backward-Euler closed-form solve bug
- **Was**: `Tw_new` in the pre-melt (phase 1) and post-melt (phase 3)
  sensible branches was solved with numerator
  `(Tw + dt*a*tc + loss_coeff*tamb)*(1+dt*c) + dt*b*(Tp + dt*c*Tw)` —
  the trailing `dt*c*Tw` inside the PCM-coupling term is spurious; it
  does not appear when the 2×2 implicit system is solved algebraically
  (the correct numerator uses the *old* `Tp` alone: `dt*b*Tp`). This is
  the identical bug class the Rajasthan audit documents as "a wrong
  closed-form backward-Euler solve... caused unbounded temperature
  blow-up." Verified numerically with the script's own default
  parameters: the buggy formula pushed `Tw_new` to 69.2°C in a single
  step from a 45°C collector with no other heat source — thermodynamically
  impossible for this passive linear coupling. The corrected formula
  gives 44.5°C for the same inputs.
- **Fix applied**: numerator corrected to use `dt*b*Tp` (old `Tp` only)
  in both the phase-1 and phase-3 branches.
- **Observed effect**: every simulated PCM before this fix landed at
  85–100% annual solar fraction (0% within the 54–84% benchmark band,
  0–1 complete cycles/year) — the same "tank never actually discharges"
  signature the v3.1 ambient-loss fix was supposed to prevent. After
  this fix alone (before the night-isolation fix below), 10% of runs
  fell in-band.

### `10_physics_validation.py` — missing night/idle collector-coupling isolation
- **Was**: the collector-tank coupling coefficient `a` was applied
  identically day and night. At night the collector temperature `Tc`
  collapses to ambient (`isolar = 0`), so an un-isolated `a*(Tc-Tw)`
  term drains the tank back out through the idle collector loop at
  essentially the same rate it charges during the day — on top of the
  separate `UA_TANK_W_K` ambient-loss term, double-counting overnight
  losses. This is the second bug class from the same Rajasthan audit
  ("Barqawi's bidirectional a·(Tc−Tw) term let the tank drain heat
  through an idle collector overnight nearly as fast as it charged
  during the day").
- **Fix applied**: added `NIGHT_ISOLATION_FRACTION = 0.05`; the
  collector-coupling coefficient is gated to 5% of its daytime value
  whenever `Tc < Tw` (collector colder than tank), matching Rajasthan's
  fix exactly. Only the collector coupling is gated — the PCM-tank
  coupling `b` is an internal exchange, not a valved external loop, and
  is left untouched.
- **Observed effect (both fixes together)**: solar fractions now spread
  physically across roughly 20–80% (not pinned at 85–100%), complete
  cycles/year moved from 0–1 to tens/hundreds (physically plausible PCM
  freeze-melt cycling), and 41% of simulations now fall within the
  54–84% benchmark band (up from 0%). Mean Spearman ρ across clusters
  moved from **-0.151** to **+0.177** — still a weak-agreement, honestly
  reportable finding (not a data-fabrication target), but no longer an
  artifact of a broken solver. **Re-run required**: `10_physics_validation.py`
  → `09_recommendation_cards.py` (both already re-run to produce the
  current on-disk artifacts as of this fix).
- Remaining gap, explicitly not fixed here (a parameter-calibration
  question, not a bug): 59% of simulations still fall outside the
  54–84% band, split between above and below it depending on cluster —
  the tank/collector parameters (`M_W_KG`, `A_C_M2`, `COLLECTOR_EFF`,
  draw schedule) are stated literature-anchored assumptions, not
  empirically fit to this pipeline's own points, and calibrating them
  further would need real deployment data or a decision to match
  Rajasthan's own calibrated values — do not further hand-tune them
  just to force more runs into the benchmark band.

---

## v3.1 Bug Fixes (August 2026 — critical correctness)

### `02_combine_tamilnadu.py`
- **Deaccumulation bug fixed.** Was: `deaccumulate()` with `pd.Series.diff()` corrupted GHI (noon r ≈ 0.40). Now: `accum_to_flux(s) = s.clip(lower=0)` — matches Rajasthan fix.

### `04_preprocess_tamilnadu.py`
- **Quantile mapping added (Step 2b).** Per-season empirical quantile mapping of daytime `era5_GHI` onto NASA POWER distribution. Saves `ghi_quantile_mapping_report.csv`.

### `03b_agreement_analysis.py` (NEW)
- Cross-source validation decision gate (BACKBONE / QUANTILE_MAP / MANUAL_REVIEW). Outputs `era5_power_agreement_tamilnadu.csv`, scatter HTML, `bias_decision_tamilnadu.txt`.

### `04b_climate_signature.py` (already fixed in prior round)
- **1000× flow rate bug fixed.** Now uses `DRAW_VOLUME_L = 300` (Avargani et al. 2021).

### `11_seasonal_pcm_sensitivity.py` (renamed 2026-09-08 from `11_level_b_seasonal_analysis.py`)
- **Draw volume aligned with 04b.** Seasonal `L_required` now uses 300 L/day formula (was still using buggy `DRAW_RATE_KG_PER_S`).
- **Renamed** — it is a *post-Phase-6* re-ranking, not a Phase-4 clustering step. The actual Phase-4 Level B (per-point-per-season GMM re-clustering) is the new `05a_level_b_regime_shift_tamilnadu.py`.
- **New**: `outputs/qc_seasonal_pcm_flip_heatmap_tamilnadu.html` — (cluster × season) grid coloured by #1-PCM identity, flipped cells red-outlined against the annual baseline.

### `05a_level_b_regime_shift_tamilnadu.py` (NEW 2026-09-08)
- Phase 4 **Level B — Regime Shift**, ported from Rajasthan's `05a`. Per-point-per-season Tier-1 signature (via shared `signature_lib.build_tier1_signature`), fresh GMM (k-scan 2–8), regime-shift fraction + season-tautology check, `outputs/qc_level_b_regime_shift_sankey_tamilnadu.html` alluvial plot.

### `cluster_lib.py` (NEW 2026-09-08)
- Shared Phase-4 machinery (`bootstrap_ari_stability`, 3-tier `suggest_k`, `fit_k_range`, `canonical_relabel_by_latitude`) — one implementation for both states and both clustering levels. `05_cluster_tamilnadu.py`'s hardcoded `K_FINAL=5` replaced by the cascade (now selects **k=3**); k-scan widened 2–10 → 2–12; Köppen-Geiger external validation + canonical latitude relabel + `provenance_lib` hard-fail checks wired in.

### `05_cluster_tamilnadu.py` (already fixed in prior round)
- **GMM covariance fixed.** `covariance_type="diag"` (was `"full"`).

### `10_physics_validation.py` (already fixed in prior round)
- **Tank ambient heat loss added.** `UA_TANK_W_K = 2.0 W/K`.

### `config.py`
- Added `OUTPUTS_DIR` for agreement analysis outputs.

### All `docs/era5_tamilnadu/*.md` files
- Updated from "known issues" to "corrected (v3.1)" status.
- Added Literature Support sections referencing `sources/` summaries from `sources.zip`.

**Re-run required**: `02_combine` → full downstream chain for scientifically valid outputs.

---

## Bug fixes (correctness, not new features)

### `02b_build_daily_aggregates.py`
- **HDD18/CDD24 annualization.** Was: summed over the full 10-year record
  (so "HDD18" was ~10x a real annual figure). Now: divided by the number
  of distinct years actually present in each point's usable-day set.
- **CCI (consecutive-cloudy-day run) gap bridging.** Was: a dropped day
  (< 20/24 hours of NASA POWER coverage) could silently let two separate
  cloudy runs be counted as one continuous run. Now: any calendar-date
  gap > 1 day forces a run break before the max-run calculation. Also now
  reports `n_date_gaps_gt1day` per point and flags points with >20 gaps.

### `04b_climate_signature.py`
- **Same HDD18/CDD24 annualization bug**, same fix, applied to the Tier-1
  sun-event proxy version (`HDD18_proxy`/`CDD24_proxy`).

---

## Feature completions (things that were honestly flagged as not-yet-done, now done)

### `07_feasibility_filter.py`
- **Corrosion veto** (Table 12, filter 6) — now implemented: excludes any
  `corrosion_class == "check_manually"` PCM in a cluster whose HSI sits
  above the 75th percentile across all clusters. Currently a near-no-op
  given your mostly-organic 25-row database (only one inorganic
  candidate is flagged `"check_manually"`) — becomes load-bearing once
  you add real salt hydrates or extend to a more humid state.
- **Safety exclusion** (Table 12, filter 7) — now implemented: keyword
  veto against the flammability field (`"highly/extremely flammable"`,
  `"toxic"`). Also currently a no-op given your data (paraffins/fatty
  acids are "combustible," not "highly flammable" in standard hazard
  classification) — the mechanism is real, just unused by current rows.
- Docstring updated to reflect 7/8 Table 12 filters now implemented (only
  the true 5th-percentile-insolation charging-feasibility filter remains
  unimplemented in its literal form — `07b_charging_feasibility.py`'s
  heuristic is the closest available substitute, and Phase 7's simulated
  performance now supersedes the need for it in practice).

### `08_mcdm_ranking.py` (v2 — substantial rewrite)
- **Added PROMETHEE II** — net outranking flow with V-shape preference
  function, q=0.10/p=0.30 (fraction of the normalized [0,1] criterion
  range — documented simplification, see script docstring).
- **Added VIKOR** — compromise ranking Q_i (v=0.5), plus the standard
  acceptable-advantage/acceptable-stability check (flags when a single
  "winner" isn't statistically distinct and a compromise set should be
  reported instead).
- **Added 5,000-draw Monte Carlo stability analysis** (plan v3.0 Section
  9.6) — Dirichlet-perturbed weights + Gaussian-perturbed PCM properties
  (Tm ±1K, latent heat ±5%, conductivity ±10%), reporting per-PCM Top-3
  inclusion probability, Top-1 retention rate, and mean Spearman rho vs.
  the unperturbed baseline ranking.
- **Consensus upgraded** from 2-method Borda to 4-method Borda, with
  Copeland pairwise-majority computed as an explicit cross-check —
  disagreement between the two is now flagged in output rather than
  silently resolved.
- Kendall's W now computed across all 4 methods (was 2).

### `09_recommendation_cards.py` (v2)
- Now includes a **Phase 7 physics validation** table per cluster
  (simulated annual solar fraction, benchmark-band flag, complete
  cycles/year) and the per-cluster Spearman rho, when
  `10_physics_validation.py` has been run. Falls back gracefully (with a
  clear note) if it hasn't.
- Top-3 table now shows all 4 methods' scores + Monte Carlo Top-3%, not
  just TOPSIS/GRA.

---

## New scripts

### `10_physics_validation.py` — Phase 7, NOT deferred to future work
Grey-box lumped-enthalpy PCM tank model (3-phase: pre-melt sensible,
isothermal melting, post-melt sensible — adapted from Barqawi2025's ODE
structure, already in your literature summaries), solved with backward
Euler (implicit, unconditionally stable — needed because the tank's
thermal time constant here is short relative to an hourly step). Driven
by each cluster's medoid point's **real** 10-year daily GHI/temperature
data (`daily_aggregates_tamilnadu.csv` from `02b`) for one representative
year, not synthetic weather. Simulates every feasibility survivor per
cluster, computes annual solar fraction, checks it against the plan's
54-84% published benchmark band (Table 16), and computes Spearman rho
between the MCDM consensus rank and simulated performance per cluster
(Table 17's three-outcome interpretation — all three are publishable if
diagnosed, which the script's output does for you).

All tank/collector parameters (mass, coil area, HTC, collector
efficiency, draw schedule) are stated assumptions with literature
citations in the docstring — exactly the same honesty standard as every
other assumption already in this pipeline. This is a genuine grey-box
model, not a toy — but it is still a simplified lumped model, per the
plan's own explicit permission ("a crude model honestly described beats
an elaborate one that is wrong").

### `11_seasonal_pcm_sensitivity.py` — post-Phase-6 seasonal PCM sensitivity
*(Renamed 2026-09-08 from `11_level_b_seasonal_analysis.py` — see the
2026-09-08 section above. The Phase-4 "Level B" name now belongs to
`05a_level_b_regime_shift_tamilnadu.py`.)* The "nearly free" addition the
plan calls out specifically for Tamil Nadu's out-of-phase north-east
monsoon. For each existing Level-A cluster, recomputes L_required per
season (Ta_mean varies seasonally; Tm_target stays constant per the plan's
rule) and re-ranks with a single-method TOPSIS (using the SAME weights as
the annual ranking, for a fair comparison) per (cluster, season). Reports
whether the #1 PCM flips
between seasons — a flip is direct empirical motivation for Objective 3's
adaptive controller, generated from your own data; no flip is also a
valid, reportable finding (the Tm_target rule is robust to seasonal
swings). This is the "nearly free" version explicitly permitted by the
plan, not the full independent-per-season-GMM-clustering version — say
so in your methodology.

---

## Still open (unchanged from the review — not addressed this round, by design)

1. **PCM database at ~25/40-60 rows.** Real gap, self-flagged since the
   first version of `06_build_pcm_database.py`. Add RT58/RT60/RT62HC,
   PLUSS OM55/OM65, and a properly-cited salt hydrate if you have time —
   nothing fabricated in this pipeline, so this stays a coverage gap
   until you source real values.
2. **External cluster validation** (ARI vs. Köppen-Geiger / NBC-ECBC
   zones, plan v3.0 Section 7.5) — not implemented. Needs an external
   Köppen/NBC shapefile or lookup table joined to your 133 points, which
   this pipeline doesn't currently have. Lower priority for a TN-only
   scope per both reviews, but the step the plan calls out as "what earns
   credibility" for the clustering — add if you extend to more states.
3. **Elevation** — still the flat 150m proxy. Both reviews agree this is
   fine for Tamil Nadu's gentle terrain; only becomes non-optional for
   Uttarakhand's 200m-7000m range.
4. **`monsoon_index` stays proxy-only** — NASA POWER precipitation was
   never downloaded (see `02b`'s docstring). Unchanged, documented.
5. **K_FINAL=5 is hand-set, not selected by the Rajasthan-style tiered
   rule.** `05_cluster_tamilnadu.py` reports BIC/silhouette/Davies-Bouldin/
   Calinski-Harabasz for k=2..10 but does not compute bootstrap-ARI
   stability, so there is no data-driven tie-break when several k values
   sit in the accepted silhouette band. In the current run, k=6
   (silhouette 0.305) and k=9 (0.312) both score higher than the
   hard-coded k=5 (0.262) within that band. This is flagged, not
   changed, here — re-clustering at a different k would cascade through
   every downstream phase (feasibility, MCDM, physics, cards) and change
   the headline per-regime recommendations, which is a scientific
   decision for the project owner, not something to silently redo.
   Add a bootstrap-ARI pass (resample the 133 points with replacement,
   refit GMM, compare via Adjusted Rand Index against the full-data
   labels, repeat ~50x) if you want the same rigor Rajasthan's audit
   applied before finalizing k.
6. **No cross-phase provenance/fingerprint check.** Rajasthan's pipeline
   hard-fails (`SystemExit`) if Phase 6/7/8's input `cluster_profiles`
   doesn't match what's currently on disk, because sklearn's
   `GaussianMixture` cluster-index order is not guaranteed stable across
   separate re-runs. The Tamil Nadu scripts have no equivalent check —
   low risk today (nothing here indicates it has actually caused a
   mismatch), but a real gap if `05_cluster_tamilnadu.py` is ever re-run
   with different data/parameters without also re-running 07→10 in the
   same pass.
7. **MCDM criteria set is reduced to 5, not the framework doc's 8.**
   `08_mcdm_ranking.py` ranks only on Tm-fitness, latent heat (climate-
   relative), volumetric latent heat, thermal conductivity, and cycling
   confidence — `cost`, `corrosion`, and `supercooling` are dropped
   entirely rather than carried as always-near-zero-weight criteria the
   way Rajasthan does. This is a documented, deliberate scope reduction
   (the database has no real cost data and only one corrosion-relevant
   candidate), not an error, but it also means Rajasthan's dominant
   "supercooling drives 48–64% of the entropy weight and the physics
   model can't simulate it" finding cannot recur here — a different,
   narrower set of caveats applies to this pipeline's MCDM/physics
   disagreement instead.
8. **Absolute latent-heat floor (`LATENT_HEAT_ABSOLUTE_MIN_KJ_KG = 100`)
   is a Tamil-Nadu-only addition beyond the framework doc's Table 12**,
   which specifies only the relative `L ≥ 0.7 × L_required` rule. Given
   `L_required` is ~301–326 kJ/kg here, `0.7 × L_required` (≈211–228
   kJ/kg) already exceeds the 100 kJ/kg absolute floor, so this addition
   is currently a no-op — but it should be named explicitly as a
   deviation from the literal spec if the write-up quotes Table 12
   verbatim.

---

## What to actually run, in order, for a complete Objective 1

```
python 06_build_pcm_database.py        # (only if you haven't — confirm INPUT_CSV)
python 07_feasibility_filter.py         # now with corrosion + safety filters
python 08_mcdm_ranking.py               # now full 4-method + Monte Carlo (~5000 draws — allow a minute or two)
python 09_recommendation_cards.py       # now includes physics validation section
python 10_physics_validation.py         # Phase 7, run BEFORE the 09 above if you want it in the cards
python 05a_level_b_regime_shift_tamilnadu.py  # Phase 4 Level B — regime-shift re-clustering
python 11_seasonal_pcm_sensitivity.py  # post-Phase-6 — seasonal PCM flip check (TN's monsoon story)
```

Note the ordering nuance: run `10` before the final `09` if you want
physics results baked into `recommendation_cards.md` — `09` checks for
`10`'s output files and includes them automatically if present, so you
can also just run `09` again after `10` to regenerate the cards with the
physics section added.
