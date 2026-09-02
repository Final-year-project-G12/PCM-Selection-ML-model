# 01 — Project Context

## Identity

"OBJECTIVE 1 — Climate-Region-Aware PCM Recommendation Framework," Group 12, B.Tech CSE Final
Year, Amrita School of Engineering. This documentation covers the **Uttarakhand** state pipeline,
implemented in `PCM-Selection-ML-model/era5-uttarakhand/`.

The Uttarakhand scripts cite the governing plan document as **v3.0** in every Phase 2–8 docstring
(`02b`, `04b`, `04`, `06`, `07`, `07b`, `08`, `09`, `05_cluster_uttarakhand`). The single
exception is `05_cluster_regions.py`, which still cites **v2.0 §7** — this is the unrun
multi-state script, and its version lag is visible in its own docstring.

The plan document itself is **not present inside `era5-uttarakhand/`**. Every plan reference in
this documentation set is therefore recorded as "cited by the script", not verified against the
document.

## Scope decision recorded in the source files

`NEXT_STEPS.md`, line 3:

> Scope decision for this sprint: **finish Objective 1 on Uttarakhand alone.** Cross-state
> clustering (the original 4-state plan v3.0 design) is real future work, already documented and
> state-parameterised, but it is not required to defend Objective 1 as a working framework. Don't
> spend time trying to onboard another state's data yet.

`README_PREPROCESSING.md` restates it:

> **You do not need to cluster across other states to finish Objective 1.** The objective
> statement is "cluster meteorological data and identify Top-2/Top-3 PCM candidates per climatic
> regime" — nothing requires those regimes to span state boundaries.

`05_cluster_uttarakhand.py`'s docstring gives the same justification and adds the expected
within-state structure: "the high-altitude Himalayan belt around Chamoli/Pithoragarh vs. the Doon
Valley around Dehradun vs. the Terai plains around Udham Singh Nagar/Haridwar are very plausibly
different regimes — elevation alone spans roughly 200-2000m of populated terrain here."

## Explicit "do not do this now" list

`NEXT_STEPS.md`'s "What to explicitly not do right now" section is unusually specific and is part
of the audit trail:

| Item | Instruction in the source file |
|---|---|
| Other states' data | "Don't onboard Rajasthan/Assam/Tamil Nadu data. `05_cluster_regions.py` stays untouched and ready for later." |
| TabTransformer/VAE encoder ablation | "Don't build" — "explicitly optional-only in the plan doc and adds nothing to Objective 1's core claim" |
| Per-point real elevation | **"Do"** think about it — "unlike the Tamil Nadu build …, Uttarakhand's 200m-2000m populated elevation range is exactly the case this repair was written for" |
| 5,000-draw Monte Carlo | "Don't run … unless Phase 5/6 finishes with time spare … it is genuinely optional" |
| Fixing `monsoon_index` via `PRECTOTCORR` | "Don't try" — "flag the proxy limitation in text instead, it costs no correctness in the ranking (monsoon_index isn't a ranking criterion, it's descriptive of the regime)" |

## Phase numbering — as used by the Uttarakhand scripts

| Phase | Name in the Uttarakhand docstrings | Script(s) |
|---|---|---|
| 0/1 | Sampling design + raw download | `00a`, `00b`, `01`, `01b`, `00_unzip_accum` |
| 1 | Combine (ERA5 + NASA POWER merge) | `02_combine_uttarakhand.py` |
| 2 (Repair 1) | Daily-integral aggregates | `02b_build_daily_aggregates.py` |
| 2 | Preprocessing and Quality Control (13 steps) | `04_preprocess_uttarakhand.py` |
| 3 | Climate Signature Construction (Tier 1 + Tier 2) | `04b_climate_signature.py` |
| 4 | Climate Regime Clustering | `05_cluster_uttarakhand.py` (`05_cluster_regions.py` = multi-state, unrun) |
| 5 | PCM database + Feasibility Filtering | `06`, `07`, `07b` |
| 6 | Multi-Criteria Ranking Engine | `08_mcdm_ranking.py` |
| 7 | Physics-Based Validation | **no script in `era5-uttarakhand/`** |
| 8 | Explanation and Final Output | `09_recommendation_cards.py` |

Note the numbering quirk: the Uttarakhand `README.md` labels `02_combine_uttarakhand.py` as
"PHASE 2 — COMBINE" in its pipeline diagram, but `README_PREPROCESSING.md` and
`PREPROCESSING_STEPS.md` label the same script "Phase 1". Both labellings appear in the source
files; neither is corrected here.

## Sprint status recorded in `NEXT_STEPS.md`

The status table in `NEXT_STEPS.md` was written mid-sprint and is **older than the artefacts in
`data/plots/`**. Reproduced in substance, with this audit's finding alongside:

| Phase | `NEXT_STEPS.md` status | What the committed artefacts show |
|---|---|---|
| 1. Data Collection | "**Done.** Points confirmed …, ~87.5% population coverage" | Confirmed — 45 points, 10,475,711 population |
| 2. Preprocessing & QC | "`02b` confirmed run (45/45 points, 0 skipped, 164,385 point-days). `04` code delivered — confirm it's actually been run" | `04` **has** been run: 489,105 output rows, 89 columns |
| 3. Climate Signature | "Code delivered …, **not yet confirmed run**" | Has been run — Phase 4–6 artefacts downstream of it exist |
| 4. Clustering | "Code delivered …, **not yet confirmed run**" | Has been run at **K = 5**; sizes 12/9/3/7/14 |
| 5. Feasibility | "Code delivered, **not yet run**" | Has been run — 275-row survivors CSV |
| 6. MCDM Ranking | "Code delivered, **not yet run**" | Has been run — 15-row Top-3 CSV |
| 7. Physics Validation | "**Not written.**" | Still not written — no script exists |
| 8. Recommendation Cards | "Code delivered, **not yet run**" | Cannot be confirmed — output is git-ignored |

`NEXT_STEPS.md` should be treated as a **plan document that has been overtaken by the run**, not
as a current status report.

## Known internal inconsistency: PCM database size

Three source files in `era5-uttarakhand/` disagree about the PCM database:

- `06_build_pcm_database.py` (docstring, lines 1–21): **55 rows** — 24 Literature, 14 Rubitherm
  Technologies, 7 Pluss Advanced Technologies, 5 PureTemp, 4 PCM Products Ltd., 1 CrodaTherm.
- `NEXT_STEPS.md` (line 17 and line 176): "**~25 candidates total**" and "PCM database is ~25
  rows, not 40-60."
- `07_feasibility_filter.py` (line 158) prints "your database (25 rows) is thin for this" in its
  low-survivor warning message.

**Resolution from the artefacts:** the committed
`PCM_data/PCM_data/data/PCM_Properties_cleaned_mice_pmm_detailed.csv` has exactly **55 rows** with
exactly the manufacturer breakdown `06` claims, and the committed plot
`data/plots/verify_feasibility/06_summary.png` reports 55 PCM rows per cluster. The 25-row figures
in `NEXT_STEPS.md` and `07`'s warning string are stale text from an earlier database generation.

An earlier 25-row generation is independently evidenced: `data/plots/verify_feasibility/` and
`data/plots/verify_ranking/` each contain a second summary PNG under a different filename, from a
run with a 25-row database and a completely different Top-3 (RT54HC / RT55 / RT64HC). See
`11_OBJECTIVE1_PLOTTING_AND_VERIFICATION_AUDIT.md`.

## Other stale text carried in the source files

| Item | Where | Correct value |
|---|---|---|
| "133 points" | `05c_explore_interactive.py` docstring | 45 |
| "hot dry Apr-Jun, NE monsoon Oct-Dec" | `03_plots_raw.py` docstring, check E | `PREPROCESSING_STEPS.md` has the right one: "hot foothill/Terai summer Apr–Jun, southwest monsoon Jun–Sep, cold high-altitude winter Dec–Feb" |
| `TN_CENTER = [10.9, 78.5]` | `05d_plots_comprehensive.py` line 72; `05c` line 399 | Uttarakhand's point-set mean, approximately [29.7, 79.0] |
| "10/10 Rubitherm RT rows and 8/8 Pluss savE rows" (an 18-row dataset) | `PCM_data/PCM_data/01_preprocess.py` docstring | The `IN_PATH` it reads has 55 records |
| plan **v2.0** §7 | `05_cluster_regions.py` | Every other Phase 2–8 script cites v3.0 |
| `CLIMATE_COMBINED_FILE`, `PROCESSED_NAMED_DIR`, `PROCESSED_GRID_DIR` | `config.py` | Dead paths — no current script writes to them |

None of these affects a computed result. They are recorded because a reader of the source will
otherwise trip over them.

## Uttarakhand-specific contextual notes from the source files

**Terrain is the defining constraint.** `README_PREPROCESSING.md` states it directly:

> `02_combine_uttarakhand.py` uses a flat **1200m** proxy for every point's solar-geometry
> calculations, not real per-point elevation. … Uttarakhand's populated terrain genuinely spans
> roughly 200m (Terai plains near Udham Singh Nagar/Haridwar) to 2000m (hill towns), and elevation
> drives both solar-geometry inputs (air mass, clear-sky irradiance) and the temperature-based
> indices (HDD18/CDD24, Ta_mean) directly. This is plan v3.0's "Repair 2," written with
> Uttarakhand specifically in mind.

**Small N.** With only 45 points, `README_PREPROCESSING.md` flags two QC steps for extra
scepticism: step 4's spatial-zone imputation fallback ("noticeably coarser zones with 45 points to
group") and step 11's VIF ("computed over fewer independent spatial samples"). It also warns that
a high silhouette is "more likely to mean an over-simple signature than a genuinely crisp regime
split" at this N, and that K should realistically be 2–4 rather than higher.

**Corrosion mechanism.** `NEXT_STEPS.md` anticipates that "the corrosion veto [will] bite for
high-monsoon-humidity Uttarakhand clusters (Terai/valley points during Jun-Sep) … same veto,
different physical mechanism, worth noting in text." **This did not happen** — the corrosion veto
is not implemented in `07_feasibility_filter.py` at all (its docstring lists it under "NOT
applied"), and every one of the 55 database candidates is organic, so the veto could not have
activated even if it had been implemented.

**Constant `Tm_target`.** `04b_climate_signature.py` sets `Tm_target_C = 57` for every point by
design (`T_DELIVERY_C = 50` + `DT_APPROACH_C = 7`, "indirect-system assumption"). Because the
melting-window filter and the Gaussian Tm-fitness criterion are both driven by `Tm_target`, this
is the single largest reason the five regimes return identical survivor sets and an identical #1
PCM. `08_mcdm_ranking.py` detects and prints this explicitly rather than letting it pass silently.

## What this documentation set does not claim

- It does **not** import any number, PCM name, cluster count, methodology detail, or conclusion
  from the Rajasthan, Tamil Nadu, or Assam pipelines.
- Where a value could not be verified inside `era5-uttarakhand/` it is marked **"not available in
  the source files."**
- Row counts recovered from committed plot artefacts are labelled *(observed)*; counts derived
  arithmetically from script constants are labelled *(expected)*.
- Approximate values read off a rendered chart (rather than parsed from a CSV) are marked as
  approximate at the point of use.
