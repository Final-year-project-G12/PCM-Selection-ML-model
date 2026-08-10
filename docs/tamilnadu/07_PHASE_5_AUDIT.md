# 07 — Phase 5 Audit: PCM Database & Feasibility Filtering

True scripts: `01_preprocess.py` (disk: `02_cross_series_donor_audit (1).png`, mislabeled as an
image), `06_build_pcm_database.py` (disk: `07b_charging_feasibility (2).py`),
`07_feasibility_filter.py` (disk: `08_mcdm_ranking (2).py`), `07b_charging_feasibility.py` (disk:
`07_feasibility_filter (2).py`, optional).

## Status: code complete, never executed against a live cluster-profile file (though the PCM
database itself is a state-independent, shared resource and its imputation logic is directly
inspectable regardless)

## PCM property imputation — near-copy of Rajasthan's approach, not identical

Same hand-rolled MICE-style chained-equations design: `RandomForestRegressor(n_estimators=300,
max_depth=4, min_samples_leaf=2, random_state=42)` per numeric column per iteration, fewest-missing-
column-first processing order, inverse-distance-weighted blend of the **3 nearest real donors in
prediction space** (`N_DONORS=3`) — confirmed exact match to Rajasthan's `N_DONORS=3`.
`N_ITER=8` refinement rounds (not independently confirmed identical to Rajasthan's value in this
pass, but explicitly named as a constant, not inferred). Same nucleation-temperature special case
(imputed as a subcooling *offset* from freezing point, not an absolute value, then translated back).
Same categorical handling (`RandomForestClassifier`, donors logged as evidence only, not blended).
Same `parse_messy_numeric()` regex logic for parsing manufacturer range/peak text
(e.g. `"32-38 (peak: 35)"` → prefers the stated peak over the range midpoint).

**Confirmed via the actual provenance table** (`05_imputation_provenance.csv`, real content found
under `PCM_Properties_cleaned_mice_pmm_detailed (2).csv` on disk — 116 rows): every Rubitherm
RT-line row's `Tm_nucleation` (and other RT-line-wide-missing properties) is donated from Pluss savE
products, confirming the same cross-manufacturer donor-pool behavior independently confirmed for
Rajasthan's identical-pattern script.

## Raw PCM database — same 18 manufacturer rows as Rajasthan (same underlying source table)

`PREPROCESSING_STEPS (3).md` (true content: raw `PCM_Properties.csv`) contains the same 18 rows (8
Pluss savE + 10 Rubitherm RT) with the same structural gap pattern (RT-line entirely missing
`Cp_solid`, `TC_liquid`, `TC_solid`, `flammability`, `flash_point`) confirmed in both the Rajasthan
and Tamil Nadu audits — this is very likely the **same underlying manufacturer-scraped source table**
shared across both state pipelines (consistent with the project's own design intent that the PCM
database is state-independent, built once).

## PCM database builder v2 — 7 literature rows, identical to what was found in the Rajasthan audit

```
Myristic acid          Tm=53.0°C   L=190.0 kJ/kg
Palmitic acid           Tm=63.0°C   L=185.4 kJ/kg
Myristic-Palmitic eutectic (58/42)   Tm=42.6°C   L=169.7 kJ/kg
Palmitic-Stearic eutectic (64.2/35.8) Tm=52.3°C  L=181.7 kJ/kg
Paraffin wax (generic)  Tm=64.0°C   L=173.6 kJ/kg
C22H46                  Tm=44.5°C   L=249.0 kJ/kg
C30H62                  Tm=65.5°C   L=252.0 kJ/kg
```
Source: `Singh2025PCM_SWH_ComprehensiveReview_summary.md`, Table 2 — cited directly, matching
exactly what this audit's Rajasthan-side research independently found for the same "vestigial
TN-branch" script. **~25 total candidates** (18 manufacturer + 7 literature), self-documented as
short of the 40–60-row target, same missing-family list as Rajasthan (RT58/RT60/RT62HC, OM55/OM65,
a sourced salt hydrate).

`any_property_imputed` computed over a 12-property `IMPUTABLE_PROPS` subset (excludes
`heat_storage_Wh_kg`, `volume_expansion`, `max_op_temp`, `flash_point`) — `True` for all 10 RT-line
rows, `False` by construction for all 7 literature rows (genuinely unmeasured, left NaN, not imputed
against a donor pool that doesn't exist for them).

**Note**: `INPUT_CSV` in this script requires **manual path editing** before it can run (comment:
"EDIT THIS PATH to wherever PCM_Properties_cleaned_mice_pmm_detailed.csv actually sits") — a small,
explicit, self-documented manual step, not a silent failure risk.

## Feasibility filter — same core 5 constraints as Rajasthan's original (pre-expansion) filter

```
ABSOLUTE_TM_MIN, ABSOLUTE_TM_MAX = 42.0, 70.0
WINDOW_LOWER_OFFSET, WINDOW_UPPER_OFFSET = 5.0, 8.0
LATENT_HEAT_FRACTION = 0.7
CYCLES_FLOOR = 300
SUPERCOOLING_MAX_K = 8.0
MIN_SURVIVORS, MAX_RELAX_STEPS, RELAX_STEP_K = 5, 4, 2.0
```
Exact same 5 base constraints, exact same thresholds and auto-relax logic (2K/step, up to 4 steps,
triggered below 5 survivors) as Rajasthan's filter before its 3 additional constraints were added.
Cycling/supercooling: unknown values pass (flagged, not excluded) — same design as Rajasthan.

**Because `L_required` is currently understated by roughly an order of magnitude (see
`05_PHASE_3_AUDIT.md`), this filter's latent-heat constraint (`L ≥ 0.7×L_required`) will pass far more
candidates than the scientifically-corrected basis would allow — the opposite risk from Rajasthan's
(which currently fails everything).** Fixing the upstream `L_required` formula will very likely
change which candidates survive this filter, possibly substantially — this filter's logic itself is
sound, but its numeric input is not yet trustworthy.

### The 3 missing constraints — confirmed genuinely absent from code, not disabled

Grepped the full file for any reference to charging feasibility, corrosion, or safety/toxicity:
**zero code paths, zero dead branches, zero commented-out blocks** — they exist only as a docstring
bullet list explaining *why* they're missing (needs a per-cluster 5th-percentile GHI figure not yet
computed; needs a real per-PCM corrosion class the database doesn't have; needs toxicity data that
doesn't exist). This is **more transparent than a silent gap** — the absence is documented as a
known limitation with a specific data dependency named for each, matching the README's and
`FIXES.md`'s characterization exactly.

## Optional charging-feasibility heuristic — a labeled, honest proxy

```python
REFERENCE_GOOD_DAY_TEMP_C = 70.0
MIN_ACHIEVABLE_TEMP_C = 42.0
POOR_DAY_Z = 1.28   # ~5th percentile, normal approximation
poor_day_kt = (kt_mean - POOR_DAY_Z * kt_std).clip(lower=0.05)
reliability_ratio = (poor_day_kt / kt_mean).clip(0, 1)
achievable_temp = MIN_ACHIEVABLE_TEMP_C + reliability_ratio * (REFERENCE_GOOD_DAY_TEMP_C - MIN_ACHIEVABLE_TEMP_C)
Tm_target_C_regime_capped = min(Tm_target_C, achievable_temp)
```
Explicitly labeled in its own docstring as "a HEURISTIC PROXY, not a real collector thermal model,"
with the honest disclosure instruction: *"The scaling constants... are stated assumptions, not
measured values — say so explicitly if you use this in your paper."* This can only ever *lower*
`Tm_target`, never raise it — consistent with its purpose (capping the target on unreliable-solar
regimes). Off by default; `07_feasibility_filter.py` only uses the capped value if this script has
already been run and the column exists, else falls back to the constant `Tm_target_C=57`.

## Literature support

Framework doc Table 5/12 for the constraint structure and thresholds (same as Rajasthan). Singh et
al. (2025) Table 2 for the 7 literature PCM rows (directly, correctly cited). Al-Mamun (2023) cited
in the charging-feasibility heuristic's comment for the "~70°C plausible FPC delivery" reference
point — a real, appropriately-scoped citation for a stated assumption, not a rigorous derivation.

## Validation

None possible yet — no `feasibility_survivors_by_cluster.csv` exists.

## Outputs (expected)

`pcm_database_tamilnadu.csv` (~25 rows), `feasibility_survivors_by_cluster.csv` (one row per
cluster×PCM, all filter columns retained, not just survivors).

## Dependencies

Requires Phase 4's `cluster_profiles_tamilnadu.csv` (`Tm_target_C`, `L_required_kJ_per_kg`) and the
PCM database. Feeds Phase 6 directly.

## Problems / risks

- **The `L_required` bug (Phase 3) is the dominant risk for this phase** — everything else here is
  sound, but its numeric output is currently built on an understated latent-heat ceiling.
- Database size gap (~25 of 40–60 target) — identical, already-self-flagged issue to Rajasthan.
- `INPUT_CSV` requiring manual path editing is a minor first-run friction point, not a defect.

## Status

**CODE COMPLETE, NEVER RUN — blocked on the Phase 3 `L_required` fix for any of its numeric output to
be trustworthy.** The filter logic itself, the imputation methodology, and the literature-PCM
additions are all sound and closely comparable in quality to Rajasthan's equivalent components.
