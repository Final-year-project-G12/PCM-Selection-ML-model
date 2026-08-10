# 00 — Master Overview: Tamil Nadu Climate → PCM Selection Pipeline (Objective 1)

## Scope and how this differs from the Rajasthan audit

This documents `D:\Final Year Project\tamilnadu\` — the Objective-1 (climate-region-aware PCM
recommendation) implementation for Tamil Nadu. It is a sibling of the fully-audited
`era5-rajasthan/` pipeline (see `docs/era5_rajasthan/`), explicitly written to mirror its method
("Identical method to the Rajasthan sibling pipeline... so both regions produce directly comparable
outputs," quoted from multiple TN script docstrings). **Two facts make this audit structurally
different from the Rajasthan one, and both are stated up front rather than discovered gradually:**

1. **Every file in this folder has a filename that does not match its content.** This is not a
   finding of this audit — it is self-documented by the project's own `README.md`, which contains a
   35-row correspondence table, and independently confirmed here by reading every file's actual
   content. See `01_FILENAME_CORRESPONDENCE.md`. All descriptions below use **correct/intended**
   names throughout, per that table.
2. **This pipeline has never been executed end-to-end.** No `data/` folder exists anywhere under
   `tamilnadu/`. Every finding in this documentation set is a **static code audit** — there are no
   output CSVs, cluster assignments, MCDM rankings, or recommendation cards to ground-truth against,
   unlike Rajasthan (which has a complete, executed, 320-point dataset). Treat every quantitative
   claim below as "what the code would produce if run," not "what it produced."

Also present, and explicitly separate from this Objective-1 pipeline: `era5-tamilnadu-pipeline/`
(duplicated as `intlo_unna/`) — a **different project**, built around Mansouri et al. (2025)'s
multimodal-learning solar-forecasting paper, covering 222 locations for 2024–2025 only, targeting
GHI *forecasting* (not PCM selection). It shares no code or data path with the Objective-1 pipeline
documented here and is not covered further in this set.

## Pipeline map (using corrected script names throughout)

```
Phase 1 — DATA ACQUISITION
  00a_build_population_grid.py   → population_grid_points.csv (~133 points, 0.25° ERA5-aligned grid)
  00b_build_suntimes.py          → suntimes.csv (sunrise/noon/sunset UTC, 2016-2025, pvlib SPA)
  01_download_era5_tamilnadu.py  → data/raw/era5/points/*.nc (240 files expected, sun-event hours)
  01b_download_nasapower.py      → data/raw/nasapower/*.json (1330 files expected, 133 pts × 10 yrs)
  00_unzip_accum.py              → (CDS zip-disguised-.nc fixer)
        ↓
Phase 2 — COMBINE & PREPROCESSING & QC
  02_combine_tamilnadu.py        → climate_tamilnadu_points.csv (deaccumulation, solar geometry, merge)
  02b_build_daily_aggregates.py  → daily_aggregates_tamilnadu.csv, tier2_signature_tamilnadu.csv
  04_preprocess_tamilnadu.py     → tamilnadu_cleaned_{physical,scaled}.csv (13-step QC pipeline)
  (+ 6 read-only QA/plot scripts, raw and post-clean, static + interactive)
        ↓
Phase 3 — CLIMATE SIGNATURE CONSTRUCTION
  04b_climate_signature.py       → climate_signature_tamilnadu.csv
        ↓  [FINDING: L_required uses the pre-correction buggy formula — see 05_PHASE_3_AUDIT.md]
Phase 4 — CLIMATE REGIME CLUSTERING
  05_cluster_tamilnadu.py        → cluster_assignments/profiles_tamilnadu.csv (GMM, K_FINAL=5, hardcoded)
  05_cluster_regions.py          → (multi-state, confirmed not runnable — needs ≥2 states)
        ↓
Phase 5 — PCM DATABASE & FEASIBILITY FILTERING
  01_preprocess.py (PCM imputation) → PCM_Properties_cleaned_mice_pmm{,_detailed}.csv
  06_build_pcm_database.py       → pcm_database_tamilnadu.csv (~25 candidates)
  07_feasibility_filter.py       → feasibility_survivors_by_cluster.csv
  07b_charging_feasibility.py    → (optional heuristic Tm cap, not applied by default)
        ↓
Phase 6 — MCDM RANKING  ◄── CURRENT IMPLEMENTATION FRONTIER
  08_mcdm_ranking.py             → mcdm_topk_by_cluster.csv, mcdm_full_scores_by_cluster.csv
  (TOPSIS + GRA only — no PROMETHEE/VIKOR/Monte Carlo, confirmed absent even as stubs)
        ↓
Phase 7 — PHYSICS VALIDATION      [NOT IMPLEMENTED — explicitly deferred as future work]
Phase 8 — RECOMMENDATION CARDS
  09_recommendation_cards.py     → recommendation_cards.md (pure aggregation, "computes nothing new")
```

## Phase status at a glance

| Phase | True script | Status | Headline finding |
|---|---|---|---|
| 1 — Data Acquisition | grid/suntimes/ERA5/POWER/zip-fix | **CODE COMPLETE, NEVER RUN** | Same design as Rajasthan (population-weighted, sun-event-aligned); 133 points expected, not confirmed |
| 2 — Combine/Preprocess/QC | combine, daily aggregates, 13-step QC | **CODE COMPLETE, NEVER RUN** | Deaccumulation logic is genuinely correct (diff + reset-hour override); 13-step preprocessing matches its own spec closely |
| 3 — Climate Signature | `04b_climate_signature.py` | **CODE COMPLETE — contains an unfixed bug Rajasthan already found and fixed** | `L_required_kJ_per_kg` uses the pre-correction `DRAW_RATE_KG_PER_S=60/1000/60` formula — understates the latent-heat design target by roughly an order of magnitude |
| 4 — Regime Clustering | `05_cluster_tamilnadu.py` | **CODE COMPLETE, NEVER RUN** | `covariance_type="full"` throughout (no diag-covariance issue Rajasthan had); `K_FINAL=5` genuinely hardcoded, not derived |
| 5 — Feasibility Filtering | PCM database + `07_feasibility_filter.py` | **CODE COMPLETE, NEVER RUN** | Same core 5 constraints as Rajasthan's original (pre-expansion) filter; charging-feasibility/corrosion/safety are honestly absent (not stubbed) |
| 6 — MCDM Ranking | `08_mcdm_ranking.py` | **CODE COMPLETE, NEVER RUN** | TOPSIS + GRA only, 5 criteria, no Monte Carlo, no PROMETHEE/VIKOR (confirmed zero code, not disabled code) |
| 7 — Physics Validation | — | **NOT IMPLEMENTED** | Explicitly named as accepted future work in the project's own status doc |
| 8 — Recommendation Cards | `09_recommendation_cards.py` | **CODE COMPLETE, NEVER RUN** | Pure aggregation, well-designed schema, nothing to critique methodologically |

## What's already self-documented (and independently confirmed accurate here)

This folder's own `README.md` and `FIXES.md` already contain a thorough self-audit — a
35-row filename correspondence table and a gap analysis against the governing framework doc
(`Objective1_PCM_Climate_Framework_Plan_v3.docx`). Both were independently verified during this
audit (spot-checks and full reads across 14 scripts) and found accurate. This documentation set adds
three things beyond what already existed: (1) exact formulas/constants pulled from full reads of
every script, not summarized from memory, (2) the `L_required` bug discovery (not previously flagged
in `README.md`/`FIXES.md`), and (3) a structured, phase-by-phase audit format matching the Rajasthan
documentation set for direct comparability.

## Known issues (see `11_IMPLEMENTATION_ISSUES.md` for full detail)

1. **[NEW FINDING] `L_required_kJ_per_kg` bug, unfixed.** `04b_climate_signature.py` computes
   `DRAW_RATE_KG_PER_S = 60.0/1000/60` (0.001 kg/s) and derives `L_required` from a 7-hour rate
   projection — the exact formula Rajasthan's own `04_climate_signature_rajasthan.py` docstring
   diagnoses as a units-confusion bug and replaces with a 300 L *total* (Avargani et al. 2021)
   basis. TN's version produces `L_required` values roughly an order of magnitude smaller than the
   corrected basis, with no in-code acknowledgment. This directly weakens Phase 5's feasibility
   filter (makes the latent-heat constraint too easy to pass, not too hard, unlike Rajasthan's
   opposite problem).
2. **MCDM stack is TOPSIS+GRA only** — 2 methods, not 4, no Monte Carlo uncertainty layer. Confirmed
   by full-file read: zero PROMETHEE/VIKOR/Monte Carlo code exists, not even disabled/stubbed.
3. **Phase 7 (physics validation) does not exist.**
4. **GMM `covariance_type="full"`, `K_FINAL` hardcoded** to 5 (single-state) / 6 (multi-region), not
   derived from the BIC/silhouette scan the code itself computes.
5. **Elevation is a flat 150 m proxy** for the population grid (vs. Rajasthan's real per-point ERA5
   geopotential elevation) — self-documented as acceptable for TN's gentle terrain.
6. **Level B (seasonal) clustering does not exist** in this pipeline (Rajasthan has it).
7. **External classification validation does not exist** (same gap as Rajasthan pre-fix, but TN has
   no stub/TODO structure for it at all — it's simply absent).
8. **PCM database is ~25 rows**, same 40–60-row target gap as Rajasthan, same missing families
   (RT58/RT60/RT62HC, OM55/OM65, a sourced salt hydrate).
9. **Charging feasibility, corrosion veto, and safety exclusion are honestly absent** from the
   default feasibility filter (not silently skipped — documented in the docstring as "need data this
   project doesn't have yet"). An optional heuristic (`07b_charging_feasibility.py`) partially covers
   charging feasibility only, off by default.
10. **The whole pipeline has never been run** — every number above is a code-level prediction, not a
    measured result.

## Recommended next step

Before running this pipeline for the first time: (1) rename every file per the correspondence table
in `01_FILENAME_CORRESPONDENCE.md` — the project's own README already recommends this and states
it's safe (nothing has been executed under the wrong name); (2) fix the `L_required` bug in
`04b_climate_signature.py` using Rajasthan's already-corrected formula as the template; (3) then run
the pipeline end-to-end to get the first real, ground-truthed Tamil Nadu results. Only after that is
done does it make sense to prioritize adding PROMETHEE/VIKOR/Monte Carlo to match Rajasthan's Phase 6
maturity — a code-complete-but-never-run pipeline with a known scientific bug should not be extended
before it's fixed and executed once.
