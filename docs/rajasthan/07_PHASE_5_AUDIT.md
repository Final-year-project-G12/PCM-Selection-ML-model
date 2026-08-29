# 07 — Phase 5 Audit: Feasibility Filtering (+ PCM Property Database)

Scripts: `PCM_data/01_preprocess.py` (shared database imputation), `07_feasibility_filter_rajasthan.py`.

## A note on file provenance in this folder

`until phase 4/` contains ~15 files whose **filenames do not match their content** — independently
confirmed via byte-level magic-number checks (three files named `*.csv` are actually PNG images;
files named like Python scripts are markdown, etc.), consistent with the project's own README in
that folder documenting the same problem and attributing it to browser download auto-suffixing. This
audit used the **correctly-named canonical copies** in `PCM-Selection-ML-model/PCM_data/`, which is
also what the live Rajasthan pipeline actually imports from (traced directly via
`07_feasibility_filter_rajasthan.py` line 146:
`PCM_MANUFACTURER_CSV = BASE_DIR.parent / "PCM_data" / "data" / "PCM_Properties_cleaned_mice_pmm_detailed.csv"`).
The `until phase 4/06_build_pcm_database.py`-labeled script is a **Tamil-Nadu-scoped, vestigial**
component — Rajasthan's feasibility filter does not call it at all; it re-implements its own
manufacturer-row loading inline. See `21_REPRODUCIBILITY.md` for the file-mislabeling hazard itself.

## Purpose

(1) Maintain a real, cited PCM property database in the corrected 42–70°C band. (2) Filter that
database against every cluster's physical/safety/economic requirements before any ranking happens,
so the MCDM stage never has to implicitly discover an infeasible candidate through its scores.

## PCM database status — current state (RE-RUN COMPLETE 2026-08-14 — numbers below are live, not historical)

**55 rows** in `PCM_Properties_55records_42_70C_dense.csv` (the current `IN_PATH` in
`PCM_data/01_preprocess.py`), up from the prior 18-row canonical file (8 Pluss savE + 10 Rubitherm
RT). Composition: 14 Rubitherm RT-line, 7 Pluss savE, 4 PCM Products Ltd (PlusICE), 5 PureTemp,
1 CrodaTherm, and 24 literature-sourced rows (n-alkanes, fatty acids, paraffin/composite blends).
This is **inside** the framework doc's 40–60-row target for the 42–70°C band (Table 5). The melting-
point band itself is densely covered, including the previously-named 55–63°C gap (RT54HC=54, RT55=55,
RT57HC=57, PureTemp 58, CrodaTherm 60/RT60/PureTemp 60, RT62HC=62, PureTemp 63). **Note**: the script's
own `literature_rows()` function (unchanged) still unconditionally appends its own 7 Singh2025
literature rows on top of the 55-row manufacturer database — the actual candidate pool this script
evaluates is **62 rows** (55 + 7), not 55. This was true before the expansion too (18+7=25) and is
not itself a bug, just worth stating precisely.

**What is still true**: every one of the 55 expanded manufacturer rows is an organic/composite PCM —
zero salt-hydrate or other inorganic rows are present (not even the old out-of-band `savE® HS36`,
which does not appear in the new dense file) — so the corrosion-veto constraint (constraint 7 below)
remains structurally inert regardless of the expansion.

**What happened in the 2026-08-14 re-run**: `PCM_Properties_cleaned_mice_pmm_detailed.csv` was
regenerated and both `07_feasibility_filter_rajasthan.py` and `08_mcdm_ranking_rajasthan.py` (Phase 6)
were successfully re-run end-to-end against the expanded database. **Two real, previously-undocumented
bugs were found and fixed to make this possible** — see the new section immediately below — before any
of the results in this file could be regenerated.

### Two blocking bugs found and fixed during the 2026-08-14 re-run

1. **Path-nesting mismatch, `PCM_data/` vs `PCM_data/PCM_data/`.** `01_preprocess.py` (and its `data/`
   output folder) live inside a doubly-nested `PCM_data/PCM_data/` directory on disk — almost
   certainly the same class of zip-extraction artifact this project's docs already flag for the
   `until phase 4/` folder (see the file-provenance note above). `07_feasibility_filter_rajasthan.py`'s
   `PCM_MANUFACTURER_CSV` path (`BASE_DIR.parent / "PCM_data" / "data" / ...`), and its own inline
   comment ("matching where `PCM_data/` actually sits alongside `era5-rajasthan/`"), both assume the
   *non*-nested layout (`PCM_data/data/...`, `01_preprocess.py` directly in `PCM_data/`). This means
   `PCM_Properties_cleaned_mice_pmm_detailed.csv`, once regenerated, would land at
   `PCM_data/PCM_data/data/...` — one level away from where the feasibility filter (and the MCDM
   script) actually looks. **This is why the detailed file was "missing"** even after
   `01_preprocess.py` ran successfully: it was never missing, it was in the wrong place relative to
   what the consuming scripts expect. Fixed non-destructively (repo layout left as-is): the
   regenerated detailed CSV is copied to the `PCM_data/data/` path the consuming scripts read from,
   rather than restructuring the folder tree.
2. **`is_rt_line` column removed by the new `01_preprocess.py`, still referenced by both
   `07_feasibility_filter_rajasthan.py`'s `load_manufacturer_rows()` and
   `08_mcdm_ranking_rajasthan.py`'s `load_rich_pcm_properties()`.** The updated preprocessing script
   (rewritten for the 55-row, 6-manufacturer database) deliberately keeps the full `pcm_type` text
   instead of collapsing it to a binary Rubitherm/Pluss product-line flag (its own docstring: "Unlike
   the earlier script, [Type] is used as-is... preserving that extra chemical-family signal"). Neither
   of the two consuming scripts was updated to match, so both raised `KeyError: 'is_rt_line'` and
   could not run at all against the new detailed CSV, regardless of the path issue above. Fixed
   minimally in both files: the dropped `is_rt_line` binary flag is replaced with the real
   `manufacturer` column the new preprocessing script already provides (6 distinct values instead of
   2), which is used only for a descriptive `family` label in Phase 5 (not read by any constraint
   logic) but *is* load-bearing in Phase 6's Monte Carlo same-family donor-fallback logic — see
   `08_PHASE_6_AUDIT.md` for that distinction and an open judgment-call note (whether `pcm_type`,
   the plan doc's literal "type-class" language, would be a more faithful grouping than
   `manufacturer` for that specific fallback, deferred rather than resolved here).

Neither bug is specific to the database expansion itself — both would have blocked *any* re-run of
Phase 5/6 against a regenerated detailed file, expanded or not. They were simply never triggered
before now because the detailed file had never been regenerated since the preprocessing script itself
was rewritten.

### Imputation method (exact, not what the docstring implies)

Hand-rolled MICE-style chained-equations loop (`N_ITER=8`), `RandomForestRegressor(n_estimators=300,
max_depth=4, min_samples_leaf=2, random_state=42)` refit per numeric column per iteration, columns
processed fewest-missing-first (standard MICE heuristic). A custom PMM-*like* step follows the
forest prediction: nearest **3** real donors by prediction-space distance
(`N_DONORS=3`), combined via **inverse-distance-weighted average**, not classic single-donor PMM.
**This is a documented-vs-implemented discrepancy worth flagging**: the script's own docstring calls
the result "a REAL, previously-measured value donated from the most physically-similar PCM," but the
code produces a weighted *blend* of three real values, not a single donated real value — the output
is not itself a value any real PCM ever measured. Categorical columns (`flammability`, `appearance`)
use a directly-predicting `RandomForestClassifier`, no donor-blend step.

### Cross-series donor pool — the specific question this audit was asked to verify

**Confirmed empirically, not just from the docstring claim**: donor eligibility is governed solely
by "has a real value or not" (`train_idx = ~miss_mask`), global across the whole table (18 rows at
the time of this audit; 55 rows now) — there is no product-line filter. For properties missing across
*all* Rubitherm RT rows (e.g. `TC_liquid`, `TC_solid`, `Cp_solid`), the only *possible* real donors
were Pluss savE rows, and the actual provenance table confirms **100% of logged donors for these
properties are Pluss savE products** — e.g. RT35's `Cp_solid` donors are all savE OM/HS products.
This is the "Rubitherm-only-imputes-from-Rubitherm" problem the project's own docstrings describe.
RT60 and RT62HC have since been added in the 55-row expansion (RT58 itself was not; the closest new
entries in that gap are PureTemp 58 and RT57HC). **Re-checked directly against the regenerated
`PCM_Properties_cleaned_mice_pmm_detailed.csv` and the raw dense CSV: the pattern persists unchanged.**
All 14 Rubitherm RT-line rows — RT60 and RT62HC included — report only a single combined
`Thermal Conductivity - Both Phases = 0.2 W/mK` figure in their manufacturer datasheet; none report
`TC_liquid`/`TC_solid` separately. RT60's and RT62HC's own imputed `TC_liquid`/`TC_solid` donors
(per `05_imputation_provenance.csv`) are Literature/Pluss/CrodaTherm/PCM-Products-Ltd rows — never
another Rubitherm row. The hoped-for "adding RT60/RT62HC might have independently-reported values"
did not pan out; it was a reasonable hypothesis that this re-run disproves with data rather than
resolves in the database's favor.

## `07_feasibility_filter_rajasthan.py` — all 8 constraints, exact as implemented

| # | Constraint | Exact rule | Behavior |
|---|---|---|---|
| 1 | Melting window (relaxable) | `Tm ∈ [Tm_target−5, Tm_target+8]` (K), widened ±2K per round, up to 4 rounds | pass/fail |
| 2 | Absolute band | `Tm ∈ [42, 70]°C` | pass/fail |
| 3 | Latent heat floor | `L ≥ κ·L_required`, κ=0.7 fixed | pass / fail / flag_unreported |
| 4 | Cycling stability | `cycles ≥ 300` | pass / fail / **flag_unreported** (never excludes) |
| 5 | Supercooling | `Tm − Tm_freezing ≤ 8K` | pass / fail / **flag_unknown** (never excludes) |
| 6 | Charging feasibility (new) | `Tm ≤ Tm_target_capped_C` (from Phase 3, not re-derived) | pass/fail |
| 7 | Corrosion veto (new) | bare salt hydrate + cluster `HSI_sunrise` > 75th percentile, unless encapsulated | pass / not_applicable / excluded_bare_high_hsi / excluded_unverified_encapsulation |
| 8 | Safety exclusion (new) | toxic/flammable field | **flag-only in practice** — never actually excludes, since the source field is an unqualified yes/no, not a severity grade |

## The headline finding, re-verified 2026-08-14: still 0 survivors at nominal thresholds

Re-confirmed directly from the regenerated `feasibility_survivors_rajasthan.csv` (186 rows = 3
clusters × 62 candidates): **every single row still has `survives_all = False`** at the fixed κ=0.7
latent-heat floor. This is not a bug and the expansion does not change it — it remains the predicted,
self-flagged consequence of Phase 3's corrected `L_required` derivation (626/608/640 kJ/kg ceiling for
clusters 0/1/2 respectively) against the *expanded* database's own best-case candidate. **The
best-case candidate improved but the gap is still enormous**: the single highest latent-heat value in
the 62-candidate pool is now `RT70HC` at **260 kJ/kg** (Tm=70°C), up from the pre-expansion best of
~252 kJ/kg (`C30H62`, a literature row) — runners-up are Stearic acid (259), n-Hexacosane (256),
n-Tetracosane (255), n-Octacosane (253). `0.7 × 608 ≈ 426 kJ/kg` (using the lowest of the three
clusters' ceilings) still exceeds even this improved best case by more than 1.6×. The database
expansion added real breadth and depth but did not — and structurally could not have been expected to
— close a gap this large; Phase 3's own docstring prediction holds exactly as before.

### The companion κ-calibration pass — re-run 2026-08-14, materially better result

`calibrate_kappa_for_cluster()` steps κ down from 0.7 to 0.0 in 0.1 increments (at the primary run's
already-relaxed melting window), targeting 8–20 survivors per cluster. **New result, all three
clusters now healthy:**

| Cluster | Old (pre-expansion) | New (55-row database) | Status |
|---|---|---|---|
| 0 | n=5, `insufficient_even_at_kappa_0` | **κ=0.2, n=9, `in_band`** | Was undersized/unreachable → now clears the 8-survivor floor |
| 1 | κ=0.2, n=8 | **κ=0.3, n=14, `in_band`** | |
| 2 | (n=7, implied undersized) | **κ=0.2, n=16, `in_band`** | |

Total survivors at each cluster's calibrated κ: **39** (9+14+16), up from 20 (5+8+7) — nearly double,
and — the headline change — **Cluster 0 is no longer stuck at "insufficient even at κ=0."** This is
the actual input Phase 6's MCDM ranking now consumes, and every row in the regenerated
`mcdm_rankings_rajasthan.csv` carries an updated `pcm_database_status` tag reflecting the 55-row
database (no longer `"PROVISIONAL — ~25-row..."`) — see `08_PHASE_6_AUDIT.md`.

A separate, dated bug fix remains documented in-code from before this re-run: the kappa-calibration
inequality direction was inverted in an earlier version ("FIXED 2026-08-11: an earlier version of this
loop had the inequality backwards, which counted candidates as 'admitted' at kappa values far above
their actual breakeven — caught by a direct contradiction in the output"). That fix was already in
place going into this re-run and required no further changes.

## Cross-phase provenance stamping (added 2026-08-11)

Both output files now carry an `upstream_cluster_profile_fingerprint` column (constant per file),
computed by `provenance_lib.file_fingerprint()`/`fingerprint_id()` (mtime+size+row_count of
`cluster_profiles_rajasthan.csv` at the moment this script reads it). This exists because Phase 7
caught a real bug: Phase 5's and Phase 6's outputs had been generated from two different on-disk
states of `cluster_profiles_rajasthan.csv` (different runs of `05_cluster_rajasthan.py`), causing
them to disagree cluster-by-cluster on which PCMs belonged to which `cluster_id` despite matching in
total row count. Phase 6 now reads this stamp and hard-fails (`SystemExit`, not a warning) if it
doesn't match the `cluster_profiles_rajasthan.csv` currently on disk — see `provenance_lib.py` and
`19_PHASE_7_ONWARD.md` for the full incident writeup, and `06_PHASE_4_AUDIT.md` for the companion fix
(canonical cluster relabeling) in the script that actually produces the labels.

## Corrosion veto — structurally inert on this run's data

In the pre-expansion 18/25-row database, only one row (`savE® HS36`) was salt-hydrate-typed, and it
was already excluded by constraints 1/2 regardless. **The 55-row expanded database does not change
this, confirmed by the 2026-08-14 re-run**: `Salt-hydrate-typed candidates in the database: 0` per the
script's own printed diagnostic — constraint 7 fired `not_applicable` for all 62 candidates in every
cluster (`excluded_c7_corrosion_veto=0` in all three clusters' summary rows). This constraint, while
correctly implemented, still cannot fire on Rajasthan's data. It will become meaningful only once the
database gains real salt-hydrate candidates (sodium acetate trihydrate, sodium thiosulfate
pentahydrate — a gap the 2026-08-12 expansion did not close) and/or once run against a more humid
state (Assam).

## Literature support

Framework doc §8 (Table 12) directly specifies all 8 constraints, including the three "new" ones —
the implementation matches the spec's structure closely. Avargani et al. (2021) underlies
`L_required`'s provenance (see Phase 3 audit). No independent literature source was found for the
specific κ=0.7 / cycling≥300 / supercooling≤8K numeric thresholds themselves beyond the framework
doc's own Table 12 — these read as engineering judgment calls documented in the project's own
methodology document, not independently peer-reviewed thresholds; state them as such in a write-up.

## Validation

Per-cluster audit trail (pass/fail/flag/relax status per constraint per candidate) is fully
persisted, which is itself the validation mechanism — nothing is silently dropped. The
"insufficient even at κ=0" cluster is explicitly flagged rather than silently omitted.

## Outputs

`feasibility_survivors_rajasthan.csv`, `feasibility_survivors_rajasthan_kappa_calibrated.csv`,
`cluster_profiles_rajasthan.csv` (consumed, not produced, here).

## Dependencies

Requires Phase 4's cluster profiles (`Tm_target_C`, `Tm_target_capped_C`, `L_required_kJ_per_kg`,
`HSI_sunrise`) and the shared PCM database. Feeds Phase 6 directly and exclusively via the
κ-calibrated survivor set, and — since 2026-08-11 — Phase 6 verifies this handoff via the provenance
fingerprint stamp described above before trusting it.

## Problems / risks

- **The database-size gap is closed (18/25 → 55 rows, inside the 40–60 target) and Phase 5/6/7/8 have
  now all been re-run against it (2026-08-14).** The latent-heat feasibility floor is still
  structurally unreachable at the nominal κ=0.7 (best case 260 kJ/kg vs. a ~608–626 kJ/kg ceiling),
  but the κ-calibrated companion pass now clears the healthy 8–20-survivor band in **all three**
  clusters (9/14/16), where Cluster 0 previously couldn't reach 8 survivors even at κ=0. The Phase 7
  physics-validation re-run against this new, healthier candidate pool is also now done and the
  negative-rho result persists — see `19_PHASE_7_ONWARD.md`.
- Constraint 8 (safety) never excludes anything in practice given current data sparsity — flagged
  correctly by the code itself, but worth stating plainly in a write-up rather than implying safety
  screening is currently doing real work.
- No `encapsulation` column exists anywhere in the database yet — constraint 7's
  "unless encapsulated" branch is untestable until that field is populated.
- Two blocking bugs (path-nesting, missing `is_rt_line` column) had to be fixed before this re-run
  could execute at all — see the dedicated section above. Both are now fixed in
  `07_feasibility_filter_rajasthan.py` and `08_mcdm_ranking_rajasthan.py`; the underlying
  `PCM_data/PCM_data/` nested-folder layout on disk was left as-is (fixed via a non-destructive file
  copy instead), so a future contributor regenerating the detailed CSV from scratch must remember to
  copy it from `PCM_data/PCM_data/data/` to `PCM_data/data/` (or fix the path properly) before
  re-running Phase 5.

## Status

**COMPLETE as implemented, PCM database prerequisite COMPLETE (55 rows, inside the 40–60 target), AND
Phase 5 has now been re-run against it (2026-08-14).** The 0-survivors-at-κ=0.7 result persists (as
predicted) but the κ-calibrated companion pass now produces a healthy, non-undersized survivor pool in
every cluster (39 total, vs. 20 before), including the previously-blocked Cluster 0. **This file's
numbers are current.** **Update, 2026-08-14 (later same day): Phase 7's physics validation and Phase
8's recommendation cards have both now also been re-run against this candidate pool** — the negative
Spearman-rho validation result persists (all 3 clusters still ≤0.4, though two of three moved less
negative and Cluster 1 flipped sign) — see `19_PHASE_7_ONWARD.md` and `08_PHASE_6_AUDIT.md` for the
full current numbers. Every phase from 5 through 8 is now current as of 2026-08-14; nothing in this
chain is pending re-run.
