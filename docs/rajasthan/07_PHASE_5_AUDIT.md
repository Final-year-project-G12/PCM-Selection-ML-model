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

## PCM database status — current state

**18 rows** in the canonical `PCM_Properties.csv` (8 Pluss savE + 10 Rubitherm RT), **25 rows**
counting the vestigial TN-branch script's 7 appended literature rows (Singh 2025 Table 2 fatty
acids/eutectics/paraffin). This is well short of the framework doc's 40–60-row target for the
42–70°C band (Table 5) — **the same gap the parallel PCM-database-expansion task targets.** No
salt hydrate beyond one already-out-of-band `savE® HS36` (Tm=35°C, Inorganic type) is present; the
55–63°C melting-point coverage gap the framework doc names is not yet closed.

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
by "has a real value or not" (`train_idx = ~miss_mask`), global across the whole 18-row table — there
is no product-line filter. For properties missing across *all* Rubitherm RT rows (e.g. `TC_liquid`,
`TC_solid`, `Cp_solid`), the only *possible* real donors are Pluss savE rows, and the actual
provenance table confirms **100% of logged donors for these properties are Pluss savE products** —
e.g. RT35's `Cp_solid` donors are all savE OM/HS products. This is the "Rubitherm-only-imputes-from-
Rubitherm" problem the project's own docstrings describe, and it is exactly what adding RT58/RT60/
RT62HC (which may have independently-reported values for these properties) would help correct,
consistent with the rationale in the parallel database-expansion task.

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

## The headline finding: 0 survivors at nominal thresholds

Confirmed directly from `feasibility_survivors_rajasthan.csv` (75 rows = 3 clusters × 25 candidates):
**every single row has `survives_all = False`** at the fixed κ=0.7 latent-heat floor. This is not a
bug — it is the predicted, self-flagged consequence of Phase 3's corrected `L_required` derivation
(~610–643 kJ/kg ceiling per cluster) against a best-case database candidate (~252 kJ/kg latent heat):
`0.7 × 610 ≈ 427 kJ/kg`, unreachable by any current candidate. Phase 3's own docstring predicted this
exactly, in advance, before Phase 5 was even built.

### The companion κ-calibration pass — what actually produces usable output today

`calibrate_kappa_for_cluster()` steps κ down from 0.7 to 0.0 in 0.1 increments (at the primary run's
already-relaxed melting window), targeting 8–20 survivors per cluster. Result: Cluster 1 reaches
`calibrated_kappa=0.2` ("in_band"); **Cluster 0 cannot reach the 8-survivor target even at κ=0.0**
(`status = "insufficient_even_at_kappa_0"`) — the melting-window and charging-feasibility constraints
alone cap it below 8. **This is the actual input Phase 6's MCDM ranking consumes** — every row in
`mcdm_rankings_rajasthan.csv` is explicitly tagged
`pcm_database_status = "PROVISIONAL — ~25-row database, not yet expanded to 40-60"`.

A separate, dated bug fix is documented in-code: the kappa-calibration inequality direction was
inverted in an earlier version ("FIXED 2026-08-11: an earlier version of this loop had the
inequality backwards, which counted candidates as 'admitted' at kappa values far above their actual
breakeven — caught by a direct contradiction in the output").

## Corrosion veto — structurally inert on this run's data

Only one database row (`savE® HS36`) is salt-hydrate-typed, and it is already excluded by
constraints 1/2 regardless — so this constraint, while correctly implemented, has never actually
fired on Rajasthan's data. It will become meaningful once the database gains real salt-hydrate
candidates (sodium acetate trihydrate, sodium thiosulfate pentahydrate — exactly what the parallel
database-expansion task targets) and/or once run against a more humid state (Assam).

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
κ-calibrated survivor set.

## Problems / risks

- **The database-size gap is the single highest-leverage open item in the entire pipeline** — every
  downstream ranking (Phase 6) and every future physics validation (Phase 7) inherits whatever
  candidate pool Phase 5 produces, and that pool is currently both too small (18–25 of 40–60 target
  rows) and structurally unable to satisfy its own nominal latent-heat constraint.
- Constraint 8 (safety) never excludes anything in practice given current data sparsity — flagged
  correctly by the code itself, but worth stating plainly in a write-up rather than implying safety
  screening is currently doing real work.
- No `encapsulation` column exists anywhere in the database yet — constraint 7's
  "unless encapsulated" branch is untestable until that field is populated.

## Status

**COMPLETE as implemented — but the PCM database prerequisite is NOT complete**, and Phase 5's own
nominal-threshold result (0 survivors) is a direct, correctly-flagged consequence of that gap, not a
filter-logic defect. **This is the single blocking prerequisite for treating Phase 6 output as final.**
