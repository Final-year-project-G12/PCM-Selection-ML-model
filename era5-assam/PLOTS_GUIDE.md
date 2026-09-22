# Assam PCM Pipeline Plotting & Verification Guide

> **Rewritten 2026-09-22, updated 2026-09-23** after a direct audit
> against the actual `era5-assam/` folder contents — the pre-2026-09-22
> version of this file described a `data/plots/` output root that is now
> empty, several subfolders/files that don't exist, and omitted several
> real, undocumented output folders (`plots_assam_ppt/`, `final_outputs/`,
> `phase10_visualizations/`, `outputs/`). It also never mentioned the
> single most important fact about this pipeline's results: **there are
> two parallel PCM-selection chains on disk, an older K=4 one and the
> current, authoritative K=3 one — see "The two-pipeline story" below
> before reading any PCM ranking output.**
>
> **2026-09-23 fix pass** (see "Fixes applied 2026-09-23" below for the
> full list): four real bugs were found and fixed in the K=3 chain's
> upstream scripts (PCM family mislabeling, a flat elevation assumption,
> an overdetermined GMM covariance setting, and a completely missing
> Tm-target capping mechanism), then the whole core pipeline was re-run.
> **Every number and PCM name below that came from the K=4 historical
> chain, Phase 9 physics validation, or Phase 10's comparison has
> changed** as a result — this is not cosmetic drift, it's a genuinely
> different (and more defensible) result. The Section 4/6/7 MCDM
> governance result (K=3, n_confirmed=0) is unaffected, since it never
> depended on any of the 4 fixed inputs.

---

## The two-pipeline story (read this first)

Assam's clustering was re-locked from an exploratory **K=4** GMM model to
an audited, final **K=3** model *after* the PCM database, feasibility
filter, and MCDM ranking scripts had already been run once against K=4.
Rather than overwrite that first pass, both chains were kept side by
side — this is deliberate, not leftover clutter (`cleanup_audit_report.txt`
§8 states the historical scripts are intentionally preserved, and
`verify_phase5_phase6.py`/`verify_phase7.py`/`verify_phase8.py`/
`verify_phase10.py`/`final_project_verification.py` assert they stay
byte-identical/untouched as a regression check).

| Phase | Historical (K=4) script | Output (2026-09-23 run) | Current/authoritative (K=3) script | Output |
|---|---|---|---|---|
| PCM database | `06_build_pcm_database.py` | `pcm_database_assam.csv` (62 records — the schema fix below widened this from 25) | `06_build_pcm_database_final.py` | `pcm_database_final.csv` (58 records, 41 cols, strict Reported/Imputed/Missing provenance) |
| Feasibility filter | `07_feasibility_filter.py` | `feasibility_survivors_assam.csv` (22 rows, **8/7/7 survivors per cluster** — was 16/15/15 before the Tm_target_capped_C fix) | `07_feasibility_filter_final.py` | `pcm_feasibility_by_cluster.csv` (174 rows = 58 PCMs × 3 clusters) + `pcm_feasibility_summary.csv` |
| MCDM ranking | `08_mcdm_ranking.py` | `mcdm_full_scores_assam.csv`, `mcdm_topk_assam.csv` (**RT44HC #1** — was n-Docosane (C22) #1 before the fix) | `08_mcdm_ranking_final.py` | `mcdm_cluster_eligibility_summary.csv`, `mcdm_rankings_by_cluster.csv` |
| Monte Carlo | (part of `08_mcdm_ranking.py`, 5000 draws) | included in the above | `08b_monte_carlo_stability_final.py` | `monte_carlo_stability_assam.csv` |

**The headline scientific finding**: under the final K=3 feasibility
rules, `pcm_feasibility_summary.csv` reports **n_confirmed = 0 PCMs in
all 3 clusters** (only 1 "conditional" candidate, n-Tetracosane C24, in
Cluster 0). Because of this, `08_mcdm_ranking_final.py` and
`08b_monte_carlo_stability_final.py` did **not perform a formal ranking**
— their outputs are governance/audit records explaining *why* ranking
was skipped, not ranked PCM lists. Don't read `mcdm_rankings_by_cluster.csv`
as a Top-3 table the way you would for another state; read
`mcdm_cluster_eligibility_summary.csv` first.

Phase 9/10 (`10_physics_validation.py`, `10_validation_comparison.py`)
then deliberately cross the two chains: they physics-simulate the 8
historical K=4 survivors against the 3 final K=3 climate medoids
(**ASP_0003, ASP_0036, ASP_0080** as of the 2026-09-23 re-run — see the
GMM covariance fix below for why these medoid IDs changed) —
`physics_validation_assam.csv`, 24 rows — and compare the result against
the historical K=4 MCDM consensus rank. **This is no longer a single
uniform "NOT PHYSICALLY SUPPORTED" finding.** The two physics dimensions
now disagree with each other about whether MCDM is supported:

- **vs. delivery success rate**: aggregate Spearman ρ = **+0.38**,
  Top-1 agreement **3/3 clusters** (RT44HC is #1 in both MCDM and
  delivery-rank physics) — a genuine positive result.
- **vs. solar fraction**: aggregate Spearman ρ = **−0.61**, Top-1
  agreement **0/3 clusters** — still an inverse-ordering finding.

Report both dimensions separately, not as one summary verdict — see
`data/preprocessed/validation_comparison_report.txt` §11 for the full
per-dimension verdict language. This is a genuinely more nuanced finding
than the old single-direction result: static MCDM screening predicts
*which PCM reaches delivery temperature most often*, but not *which PCM
maximizes annual solar fraction* — worth stating plainly in the paper
rather than defaulting to the old "not physically supported" framing.
See `phase10_preimplementation_audit.txt` for the pre-fix reasoning this
result builds on (still valid background — the K=4-vs-K=3 governance
split it explains didn't change).

---

## Fixes applied 2026-09-23

Ported and cross-checked against the equivalent fixes already applied in
`era5-rajasthan`/`era5-tamilnadu`/`era5-uttarakhand`. All four are in the
**K=3 authoritative chain's own upstream scripts** (`04b_climate_signature.py`,
`05_cluster_assam.py`, `06_build_pcm_database_final.py`), so they affect
every downstream output, not just the historical K=4 comparison:

1. **`06_build_pcm_database_final.py` family mislabeling** — line 100 used
   `row.get("is_rt_line", 0) == 1`, a column that no longer exists in the
   canonical `PCM_Properties_cleaned_mice_pmm_detailed.csv` schema (same
   root cause as the historical script's `KeyError`, but silent here
   because of the `.get()` default — every Rubitherm-line PCM's `family`
   silently fell through to a generic `pcm_type` label instead of
   "Rubitherm RT"). Fixed to use the already-loaded `manufacturer` column.
   Non-functional (family is descriptive, not used in filtering) but
   wrong in every downstream table until fixed.
2. **Flat elevation assumption** — `02_combine_assam.py` used a flat
   `DEFAULT_ALT_M = 100` for every point's solar-geometry calculation. New
   `00c_attach_elevation.py` (ported from the other three states) attaches
   real per-point ERA5 geopotential-derived elevation (range: -12 m to
   993 m, mean 195 m — Assam's sampling envelope touches the Karbi Anglong
   / North Cachar hill districts, so this isn't negligible). `02_combine_assam.py`
   now reads `elevation_m` per point with the flat value only as a
   missing-data fallback.
3. **GMM `covariance_type="full"`** — `05_cluster_assam.py` used full
   covariance for a 5-feature, 80-point GMM fit, the same overdetermination
   risk Tamil Nadu/Uttarakhand already fixed (though for Assam's much
   lower feature count, `max_membership_prob` saturation only dropped
   modestly: 62/80→59/80 points above 0.999 — still the architecturally
   correct fix, just a smaller effect here than in Uttarakhand's
   higher-dimensional case). Changed to `"diag"` at all 3 GMM call sites.
   **This changed the actual cluster assignments and medoids**
   (ASP_0012/0092/0028 → ASP_0003/0036/0080) — a real consequence of a
   correct fix, not a regression; `verify_phase9.py` and
   `final_project_verification.py`'s hardcoded medoid-ID checks were
   updated to match.
4. **Missing `Tm_target_capped_C`** — Assam's `04b_climate_signature.py`
   only ever computed a flat, uncapped `Tm_target = 44.0°C` constant; it
   never had Rajasthan/Tamil Nadu's poor-period-clearness cap at all
   (`07_feasibility_filter.py` already had dead fallback code looking for
   a column, `Tm_target_C_regime_capped`, that nothing ever populated).
   Ported the capping subsystem, adapted to Assam's 3-event-sample data
   (Rajasthan/TN use full per-day kt series; Assam only samples sunrise/
   noon/sunset, so `kt_worst_month` here is the month-pooled mean of the
   3-event `era5_CSI` sample, the closest available equivalent). Result:
   Tm_target drops from 44.0°C to **~40°C** (37.2-41.6°C per point,
   80/80 points capped), and the historical feasibility survivor pool
   shrinks from 16/15/15 to **8/7/7 per cluster** — the rank-1 historical
   MCDM pick changes from n-Docosane (C22) to **RT44HC**.

**Two bugs found during verification of fix #4, both fixed the same
pass** (documented in-line in the affected scripts, not repeated here in
full):
- `04b_climate_signature.py` saved `climate_signatures_raw.csv` **before**
  computing `Tm_target`/`T_mains_est`/`Tm_target_capped_C`, so none of
  those columns ever reached the file `05_cluster_assam.py` reads — the
  capping fix silently had zero effect until this was caught (cluster
  profiles kept showing the old 44.0°C fallback) and the save was moved
  to after all derived columns are added.
- `10_validation_comparison.py` (and its regression checks in
  `verify_phase10.py`/`final_project_verification.py`) had **entirely
  hardcoded narrative prose and assertions** for Sections 8-11 of the
  Phase 10 report (specific PCM names like "savE® OM48" that no longer
  exist in the shrunk candidate pool, a fixed "NOT PHYSICALLY SUPPORTED"
  verdict, hardcoded medoid IDs) — these don't regenerate from data, they
  were one run's finding typed in as fixed text. Rewritten to compute
  every cited number/PCM name/verdict from the actual current run; the
  verdict is now reported per physics-dimension (delivery vs. solar
  fraction) since they now genuinely disagree, rather than forced into
  one summary label.

**Known pre-existing issue, NOT caused by or fixed in this pass**:
`final_project_verification.py`'s check #1 asserts `population_grid_points.csv`/
`cluster_assignments_assam.csv` have exactly 129 rows, but every actual
run (before and after these fixes) has processed 80 points. This
inconsistency predates the 2026-09-23 fixes — flagged here rather than
silently left for someone to discover later, but out of scope for this
pass (fixing it means figuring out whether "129" or "80" is the intended
population-grid size, a data-provenance question, not a code bug). The
same hardcoded `== 129` assertion also blocks `generate_phase11_figures.py`
from regenerating Figure 9 (climate regime map) and Figure 10 (PCA
projection) — **`final_outputs/visuals/fig09_final_k3_climate_regime_map.png`/`.html`
and `fig10_final_k3_pca_projection.png` are stale, dated 2026-09-07,
predating every fix in this pass** (they still show the old K=3 clustering
from before the GMM covariance fix). Figures 1-8 were successfully
regenerated 2026-09-23 and are current.

---

## 📁 Directory Structure & Generated Outputs (verified 2026-09-22)

All output folders below are at the `era5-assam/` **root** — NOT under
`data/plots/` as the previous version of this guide said (`data/plots/`
exists but is empty; everything moved to `era5-assam/plots/` during an
earlier cleanup pass, see `cleanup_audit_report.txt`).

```
era5-assam/plots/                          [34 files total — the post-cleanup "keep" set]
├── assam_objective1/                      3 files: 01_raw_vs_preprocessed_radiation.png,
│                                           02_climate_regime_map.png(+.html)
│                                           NOTE: generate_assam_plots.py's own docstring
│                                           claims "13 required plots" — 10 were deliberately
│                                           deleted as stale K=4/25-PCM artifacts, see
│                                           cleanup_audit_report.txt / plot_cleanup_full_audit.txt
├── comparison/                            9 files (01-09), cross-pipeline comparison charts
├── verify_clustering/                     6 files (01-06)
├── verify_feasibility/                    5 files (01,02,04,05,06 — 03 was never produced)
├── verify_preprocessing/                  6 files (01,02,03,05,06,07 — files 08-12 referenced
│                                           by an earlier version of this guide were never
│                                           produced or were removed; don't expect them)
└── verify_ranking/                        5 files (01,02,03,05,06 — 04 was never produced)

era5-assam/data/plots/                     EMPTY. Do not point anyone here.

era5-assam/plots_assam_ppt/                [53 files — UNDOCUMENTED until now. The
                                            PPT-curated deliverable set, mirrors Tamil
                                            Nadu's plots_tamilnadu_ppt/ layout, built by
                                            build_plots_assam_ppt.py]
├── location_map.html
├── 1 Data collection/Assam/                A_point_map.png, C_era5_vs_power.png, F_yearly_trend.png
├── 2 Data Preprocessing/Assam/              7 files (png+html)
├── 3 Climate Feature Engineering (Climate Signature)/Assam/   3 files
├── 4 Climate Region Discovery (Clustering)/Assam/  6 files
├── 5 PCM Suitability Evaluation (MCDA)/Assam/  ~17 files — mostly HISTORICAL K=4 MCDM
│                                           figures (bump charts, Monte Carlo, rank
│                                           correlation heatmap) reused for the PPT
│                                           narrative; treat with the same caution as the
│                                           historical chain above
├── 6 PCM Recommendation and Output/Assam/   6 files
└── interactive_plots/                       4 generator .py scripts (not output images)

era5-assam/final_outputs/                  [21 files — thesis-ready deliverables, built by
                                            generate_phase11_consolidation.py +
                                            generate_phase11_figures.py. UNDOCUMENTED
                                            until now, and arguably the MOST authoritative
                                            single output set — see table below]
├── tables/                                 table01_climate_signatures.csv ... table10_mcdm_vs_physics_comparison.csv
└── visuals/                                 fig01_gmm_bic_selection.png ... fig10_final_k3_pca_projection.png
                                            (fig09 has both .png and .html; 10 figures, 11 files)

era5-assam/phase10_visualizations/         [4 files — UNDOCUMENTED until now, content
                                            duplicated as fig05-fig08 in final_outputs/visuals/]
    01_mcdm_vs_delivery_rank.png, 02_mcdm_vs_solar_fraction_rank.png,
    03_mcdm_vs_cycling_rank.png, 04_tm_vs_physics_delivery.png

era5-assam/outputs/                        [2 files — small, UNDOCUMENTED, orphan folder]
    bias_decision_assam.txt, qc_era5_power_scatter_assam.html
```

### `final_outputs/tables/*.csv` — what each one actually contains

| File | Rows | Contents |
|---|---|---|
| `table01_climate_signatures.csv` | 129 | Per-grid-point physical climate signature (Ta_mean, GHI_mean, DTR, RH_mean, HSI, monsoon_index, ...), 19 feature columns, keyed by `point_id` (ASP_0001...) |
| `table02_pca_loadings.csv` | 2 (PC1/PC2) | PCA loadings over 7 correlated thermodynamic features |
| `table03_gmm_selection.csv` | 9 (K=2..10) | GMM model-selection grid: BIC, Silhouette, Davies-Bouldin, Calinski-Harabasz per K — K=3 minimizes BIC (1574.94) |
| `table04_cluster_profiles_k3.csv` | 3 | Aggregated climate profile per final K=3 cluster |
| `table05_pcm_database_summary.csv` | 58 | The final, deduplicated PCM property database with strict `value_status` (Reported/Imputed/Missing) per property, 40 columns |
| `table06_feasibility_survivors.csv` | 8 | The **historical K=4** feasibility survivor list for one cluster (full constraint-audit columns) — per-cluster counts are now 8/7/7, not the pre-2026-09-23-fix 16/15/15 |
| `table07_historical_mcdm_rankings_k4.csv` | 7 | **Historical K=4** MCDM scores/ranks for one cluster (TOPSIS, GRA, PROMETHEE, VIKOR, Borda/Copeland consensus, Monte Carlo stats) |
| `table08_monte_carlo_stability_k3.csv` | 3 | Governance record: Monte Carlo was SKIPPED for all 3 final clusters (n<2 eligible candidates) |
| `table09_physics_performance_k3.csv` | 24 (8 PCMs × 3 clusters) | 10-year sub-hourly physics simulation results per PCM/medoid (delivery success, solar fraction, cycles/year) |
| `table10_mcdm_vs_physics_comparison.csv` | 24 | Head-to-head historical-MCDM-rank vs. physics-derived-rank comparison — the rank_difference columns behind the mixed finding above (positive correlation vs. delivery success, negative vs. solar fraction) |

---

## 🛠️ Execution Commands

Run all scripts from the workspace root or from `era5-assam/`. This list
now also includes the later-phase scripts the previous version of this
guide omitted entirely.

```powershell
# 1. Generate Objective 1 plots and interactive Folium maps
python era5-assam/generate_assam_plots.py

# 2. Generate cross-pipeline comparison charts
python era5-assam/comparison_plots_assam.py

# 3. Generate cross-source ERA5 vs NASA POWER comparison plots & interactive dashboard
python era5-assam/generate_era5_nasa_comparison_plots.py

# 4. Generate regional parity diagrams (NASA agreement, population grid, PCA scree, diurnal physics)
python era5-assam/generate_missing_assam_diagrams.py

# 5-8. Verification suites (Phase 1-4-equivalent QA — see summary table below)
python era5-assam/verify_01_preprocessing_assam.py
python era5-assam/verify_02_clustering_assam.py
python era5-assam/verify_03_feasibility_assam.py
python era5-assam/verify_04_ranking_assam.py

# 9. PPT-curated plot set (plots_assam_ppt/)
python era5-assam/build_plots_assam_ppt.py

# 10. Thesis-ready consolidated tables + figures (final_outputs/)
python era5-assam/generate_phase11_consolidation.py
python era5-assam/generate_phase11_figures.py

# 11. Later-phase regression/audit suites
python era5-assam/verify_phase5_phase6.py
python era5-assam/verify_phase7.py
python era5-assam/verify_phase8.py
python era5-assam/verify_phase9.py
python era5-assam/verify_phase10.py
python era5-assam/final_project_verification.py
```

---

## 📊 Summary of Verification Criteria

| Suite | Focus | Key Checks | Status |
| :--- | :--- | :--- | :--- |
| **Verify 01** | Data Preprocessing | Continuous distributions, zero nulls post-imputation, valid correlations | ✅ PASS |
| **Verify 02** | GMM Clustering | BIC curve inflection, non-overlapping PCA space, clear climate profiles | ✅ PASS |
| **Verify 03** | Feasibility Filter | Bounded physical property space, per-cluster survivor accounting | ✅ PASS (mechanically) — but see "two-pipeline story" above: the final K=3 chain's honest result is **zero confirmed feasible PCMs**, not a failure of this check |
| **Verify 04** | MCDM Ranking | High inter-method Spearman correlation (>0.85), stable Monte Carlo rank probabilities | Applies to the **historical K=4** ranking only — the final K=3 chain never reached a formal ranking to check (see above) |
| **Phase 10** | MCDM-vs-physics comparison | Spearman ρ between historical MCDM consensus rank and physics-simulated delivery rank | Verdict: **NOT PHYSICALLY SUPPORTED** (ρ = −0.52 to −0.64) — documented as a genuine finding |

This table stops one level short of the pipeline's actual headline
result on purpose — the individual Verify 01-04 suites check internal
mechanics (did the script run, are outputs well-formed), not whether the
final K=3 chain produced a usable PCM shortlist. It didn't; see "The
two-pipeline story" above for what to cite instead.
