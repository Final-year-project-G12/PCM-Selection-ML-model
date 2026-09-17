# 11 — Objective 1 Plotting & Verification-Suite Audit

**Scripts**: `03_plots_raw.py`, `03b_interactive_raw_qa.py`, `04c_postprocess_plots.py`,
`04c_interactive_postprocess_qc.py`, `04d_signature_interactive.py`, `05b_cluster_interactive.py`,
`05c_explore_interactive.py`, `05d_plots_comprehensive.py`, `generate_objective1_plots.py`,
`comparison_plots_uttarakhand.py`, `verify_01_preprocessing.py` … `verify_04_ranking.py`

**Why this file matters:** `era5-uttarakhand/.gitignore` excludes `data/raw/`,
`data/processed/` and `data/preprocessed/`. **The plot tree is the only committed evidence of what
the pipeline actually produced**, so every observed number in this documentation set was recovered
from it. This file records what each plot is, which are trustworthy, and which are misleading.

---

## Committed plot inventory

| Directory | Files | Produced by | Committed |
|---|---|---|---|
| `data/plots/raw/` | 6 PNG + `C_era5_vs_power_stats.csv` | `03_plots_raw.py` | Yes |
| `data/plots/raw_interactive/` | 6 HTML + `C_era5_vs_power_stats.csv` | `03b_interactive_raw_qa.py` | Yes |
| `data/plots/post_preprocess/` | 5 PNG + `C_qc_flag_counts.png` + **`C_qc_flag_counts.csv`** | `04c_postprocess_plots.py` | Yes |
| `data/plots/post_preprocess_interactive/` | 5 HTML | `04c_interactive_postprocess_qc.py` | Yes |
| `data/plots/comprehensive/{maps,timeseries,statistics,solar_resource}` | 4 HTML + 8 PNG | `05d_plots_comprehensive.py` | Yes |
| `data/plots/uttarakhand_objective1/` | 13 PNG + 9 HTML | `generate_objective1_plots.py` | Yes |
| `data/plots/objective1/` | 5 PNG + 7 HTML | **no script in `era5-uttarakhand/`** | Yes |
| `data/plots/verify_preprocessing/` | 7 PNG | `verify_01_preprocessing.py` | Yes |
| `data/plots/verify_clustering/` | 6 PNG | `verify_02_clustering.py` | Yes |
| `data/plots/verify_feasibility/` | 7 PNG | `verify_03_feasibility.py` (6) + 1 orphan | Yes |
| `data/plots/verify_ranking/` | 7 PNG | `verify_04_ranking.py` (6) + 1 orphan | Yes |
| `data/plots/comparison/` | 8 PNG | `comparison_plots_uttarakhand.py` | **RESOLVED (2026-09)** — a path bug that made this "never produced" is fixed; now runs and populates this directory |
| `data/processed/signatures/interactive/` | — | `04d_signature_interactive.py` | git-ignored |
| `data/processed/clustering/interactive/` | — | `05b_cluster_interactive.py` | git-ignored |

---

## The QA layer (`03`, `03b`, `04c` ×2) — trustworthy

These run inside the phase chain and are documented in `04_PHASE_2_AUDIT.md`. Two of their outputs
are the evidentiary backbone of this entire documentation set:

- **`data/plots/raw/C_era5_vs_power_stats.csv`** — the only committed cross-source statistics
  (n = 493,155; GHI MBE −211.406 W/m², r = 0.4321; clear-sky GHI MBE +5.314, r = 0.9923; T_amb MBE
  −0.089 °C, r = 0.902; RHum +11.383 %; wind −1.141 m/s).
- **`data/plots/post_preprocess/C_qc_flag_counts.csv`** — the only committed QC counts
  (`era5_LW_down` 363,525 and `era5_P_atm` 182,899 physical-bounds flags; Hampel flags
  `era5_cloud_cover` 49,519, `era5_GHI` 35,559, `era5_W_spd` 11,350, `era5_T_amb` 9,762,
  `era5_RHum` 8,814).

Both were parsed directly, not read off a figure. Everything in this documentation set that quotes
a QC or cross-source number traces to one of these two files.

Note `data/plots/post_preprocess_interactive/B_distributions_post.html` is **43 MB** — an
embedded-data Plotly page. Worth knowing before opening it or committing further copies.

---

## `05d_plots_comprehensive.py` — Tamil Nadu map-centre bug — RESOLVED

**This section previously reported all three Folium maps initialising at Tamil Nadu's coordinates.
That is fixed in both the code and the committed output.** Current `05d_plots_comprehensive.py`
line 75: `TN_CENTER = [29.7, 78.9]  # Uttarakhand centroid (was Tamil Nadu's [10.9, 78.5] —
copy-paste bug)`, used by all four Folium maps in the script (including the India-wide overview
map at `[22.5, 78.9]`, a deliberately different, wider-zoom centre). **Confirmed in the committed
output**: `data/plots/comprehensive/maps/A0_all_points_overview.html` now contains
`center: [29.7, 78.9]`, matching the 45 markers at 28.875–30.625 °N, 77.875–80.125 °E. The fix
matches what `03b_interactive_raw_qa.py` already did correctly.

The same literal was also present in `05c_explore_interactive.py` and is likewise fixed: current
line 402 reads `folium.Map(location=[29.7, 78.9], …)`.

One remaining stale-text item in the same pair of scripts (cosmetic, no output impact):

- `05c_explore_interactive.py` docstring (line 46): "Folium map of **all 133 points**" — Uttarakhand
  has 45. Not fixed alongside the map-centre bug.
- `05d`'s `USE_PROCESSED = True` means the comprehensive plots are built from
  `uttarakhand_cleaned_physical.csv`, i.e. post-QC data. That is a deliberate, documented choice
  ("so plots reflect the QC'd backbone, not raw data with its outliers/gaps still in it"), but it
  means these figures show imputed values without marking them.

---

## `generate_objective1_plots.py` — the 13-plot Objective 1 set

Outputs to `data/plots/uttarakhand_objective1/`. Reads the Phase 2–6 CSVs directly (not via
`config.py`).

### What each plot actually shows

| # | File | Source | Trustworthy? |
|---|---|---|---|
| 01 | `01_raw_vs_preprocessed_radiation.*` | raw + cleaned CSV, first point, first 500 k rows | Yes, but plots **record index**, not date |
| 02 | `02_climate_regime_map.*`, `_folium.html`, `_interactive.html` | `cluster_assignments` | **Yes — the single most valuable artefact.** The Folium popups carry `point_id`, `cluster_id` and `max_membership_prob` for all 45 points; this is where the entire cluster assignment table in `06_PHASE_4_AUDIT.md` came from |
| 03 | `03_melting_point_vs_latent_heat.*` | `feasibility_survivors` | **RESOLVED** — was misleading (plotted all 275 rows); current code filters `passes_all` first |
| 04 | `04_feasible_candidates_highlighted.png` | `feasibility_survivors` + `pcm_database` | **RESOLVED** — same fix applies |
| 05 | `05_pcm_survivors_per_cluster.*` | `df.groupby("cluster_id").size()` | **RESOLVED** — now filters `passes_all` before counting; reports the true 29/30/29/27/29, not a flat 55 |
| 06 | `06_pcm_feasibility_scatter_and_survivors.png`, `pcm_feasibility_scatter.png`, `pcm_survivors_per_cluster.png` | same | **RESOLVED** — same fix applies |
| 07 | `07_bump_chart_ranks.*` | `mcdm_topk`/`mcdm_full_scores` | **Yes** — now four-method (TOPSIS/GRA/PROMETHEE/VIKOR) + consensus rank per cluster; source of the per-method ranks in `08_PHASE_6_AUDIT.md` |
| 08 | `08_method_rank_correlation_heatmap.*` | `mcdm_topk` | Yes, **but pooled across all clusters** — see the caveat below |
| 09 | `09_monte_carlo_top3_probability.*` | `monte_carlo_stability.csv` | **RESOLVED — now produced.** `09b_monte_carlo_stability.py` exists and is run (5,000 draws/cluster); this plot is populated |
| 10 | `10_rank_reversal_violin_bar.png`, `_interactive.html` | `mcdm_topk` | Yes — rank spread across methods |
| 11 | `11_agreement_plot.*` | `physics_validation_results.csv` | **RESOLVED — no longer misleading.** Phase 7 now exists and is run; the current code has an explicit in-code "BUG FIX" comment confirming it was found and fixed independently that ranking by `hours_target_met_per_year` didn't reproduce the real Spearman rho, and switched to `annual_solar_fraction`, which does |
| 12 | `12_tank_temperature_melt_fraction.*` | **hard-coded sinusoids** | **Still not data** — confirmed still true by reading the current code; this plot remains an illustrative schematic, not real `10_physics_validation.py` output, even though that script now exists and has real results elsewhere |
| 13 | `13_recommended_pcm_summary.*` | `mcdm_topk` | **Yes** — the interactive version's `customdata` carries `Tm_C`, `rho_H_MJ_m3`, `TC_W_mK`, `cycles_tested` per Top-3 PCM, all of which cross-check exactly against the committed PCM CSV |

### Plot 12 is synthetic

```python
Ta   = 28 + 14*np.sin((hrs-6)*np.pi/12)
tank = Tm - 6 + 18*np.sin((hrs-6)*np.pi/12)
melt = np.clip((tank - Tm + 5)/10, 0, 1)
```

Only `Tm` comes from real data (`feasibility["Tm_target_C"]`, which is 57 °C everywhere). The
ambient sinusoid (28 ± 14 °C) matches no Uttarakhand cluster profile. **This figure must never be
presented as simulation output** — see `09_PHASE_7_AUDIT.md`.

### The `passes_all` defect — RESOLVED (before this 2026-09 session)

Plots 03, 04, 05 and 06 used to treat every row of `feasibility_survivors_by_cluster.csv` as a
survivor. `07_feasibility_filter.py` writes **all 55 PCMs × 5 clusters = 275 rows**, each carrying
a `passes_all` boolean, specifically so the per-filter detail is auditable
(`07_PHASE_5_AUDIT.md`). Any consumer must filter on it. **Confirmed by reading the current
`generate_objective1_plots.py`: all four (`p03`-`p06`) now correctly filter with
`if "passes_all" in df.columns: df=df[df["passes_all"]]`** before doing anything else.

**Consequence:** the committed "survivors per cluster" figures now correctly report the true
per-cluster counts (29/30/29/27/29, post the 2026-09 regime-cap fix — see `07_PHASE_5_AUDIT.md`),
not a flat 55.

---

## `data/plots/objective1/` — an orphaned output directory

12 files (`bump_chart`, `climate_regime_map`, `consensus_vs_topsis_agreement`,
`melting_point_vs_latent_heat`, `method_rank_correlation_heatmap`, `pcm_feasibility_scatter`,
`pcm_survivors_per_cluster`, `rank_reversal_frequency`, `raw_vs_preprocessed_radiation`,
`recommended_pcm_summary`, `tank_temperature_melt_fraction`, `top3_inclusion_probability`).

**No script in `era5-uttarakhand/` writes to `data/plots/objective1/`** — a grep for `objective1`
across all `.py` files matches only `generate_objective1_plots.py`, which writes to
`uttarakhand_objective1/`. This directory was produced by a generator that is not in the folder.

Its contents **are Uttarakhand data** (5 clusters 0–4; the same five PCMs), and two of its files
were essential to this audit:

- `recommended_pcm_summary.html` — consensus rank per PCM per cluster, the cleanest source for the
  Top-3 table in `08_PHASE_6_AUDIT.md`.
- `consensus_vs_topsis_agreement.html` — the 15 `(cluster, consensus_rank, topsis_rank)` triples.

One naming caveat: this orphaned directory's `top3_inclusion_probability.html` is **not** a Monte
Carlo probability — a frozen artifact from before Monte Carlo existed in this pipeline. Its y-axis
is `Top3_count` — how many of the 5 clusters each PCM appears in (RT60 5, PureTemp 58 3,
n-Hexacosane C26 3, savE® OM55 2, Palmitic-stearic/EG 2, from that old run). **This is now stale in
a second way, not just the naming**: Monte Carlo HAS since been implemented and run
(`09b_monte_carlo_stability.py`, see `08_PHASE_6_AUDIT.md`'s update notice), and the correctly-named
`09_monte_carlo_top3_probability.html` in `data/plots/uttarakhand_objective1/` (produced by
`generate_objective1_plots.py`, not this orphaned directory) is the real inclusion-probability plot
to cite.

---

## `comparison_plots_uttarakhand.py` — RESOLVED (2026-09), now runs correctly

This section originally reported the script never ran, due to:

```python
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

`..` from `era5-uttarakhand/` resolves to `PCM-Selection-ML-model/`, so every input path pointed at
`PCM-Selection-ML-model/data/processed/…`, which does not exist. **Fixed** (before this 2026-09
session) — the script's own docstring now states: "Uses `config.py` for all paths (this pipeline's
convention) rather than a relative-to-script BASE guess, since every script in `era5-uttarakhand/`
sits directly in the project root next to `data/`, not in a subfolder." Confirmed by directly
running it: `data/plots/comparison/` now exists and is populated with all eight comparison plots.

Comparisons 4 (TOPSIS vs GRA agreement) and 6 (physics validation vs MCDM rank), which this section
said "would remain inert... since Monte Carlo and Phase 7 outputs do not exist," are now populated
too — both Monte Carlo (`09b_monte_carlo_stability.py`) and Phase 7 (`10_physics_validation.py`)
have since been implemented and run. The script's own note on plot 4 explicitly documents that it's
a TOPSIS-vs-GRA agreement plot, not a literal port of another state's "Monte Carlo stability" plot —
that distinction still holds.

---

## The verification suite (`verify_01` … `verify_04`)

`VERIFICATION_METHODOLOGY.md` defines six stages, success criteria and red flags. Four scripts
implement stages 2–5.

**Path convention:** all four use **relative** paths (`"data/processed/…"`) rather than
`config.py`, so they must be run with `era5-uttarakhand/` as the working directory. This is the one
consistent deviation from the pipeline's own path discipline.

### `verify_01_preprocessing.py` — 7 plots, trustworthy

Its `07_preprocessing_summary.png` is the second-most-valuable committed artefact in the repository:

```
Input records: 493,155        Output records: 489,105        Data retention: 99.2%
Input dimensions: 36          Output dimensions: 89
Core climate variables: 6     Engineered features: 45
era5_T_amb / RHum / W_spd / P_atm / GHI / precipitation: 100.0% complete
Rows with no missing data: 489,105 (100.0%)
```

`01_climate_distributions.png` carries per-variable mean/std/min/max in its subplot titles — the
source of the cleaned-file distribution table in `04_PHASE_2_AUDIT.md` Part B.8.

### `verify_02_clustering.py` — 6 plots, trustworthy with one caveat

Uses the **saved** cluster labels rather than re-fitting — a good design choice, stated in its
docstring. This section originally reported average silhouette 0.279 and cluster sizes 12/9/3/7/14
at k=5 (an earlier signature/clustering version). **Current, post-2026-09-fix run: silhouette 0.234
(this script's own feature matrix), cluster sizes 7/3/9/10/16** — see `06_PHASE_4_AUDIT.md`.

**Caveat:** it builds its feature matrix from *every* numeric column of
`climate_signature_uttarakhand.csv` except `point_id/cluster_id/lat/lon/population`, then
re-standardises. That set includes the raw indices, the `_proxy` and `_true` duplicates, the
PCA-block members **and** the `_z` columns — a much larger space than the `_z`-only matrix the GMM
was fitted in. The 0.279 figure is a valid independent diagnostic but is **not** the silhouette
`05_cluster_uttarakhand.py` wrote to `bic_selection_uttarakhand.csv`.

### `verify_03_feasibility.py` — RESOLVED (before this 2026-09 session)

This section originally reported the script never filtered `passes_all`, reading:

```python
survivors = pd.read_csv(INPUT_SURVIVORS)     # never filters passes_all
total_survivors = len(survivors)
```

**Confirmed fixed by reading the current script** (line 39-40):
`if "passes_all" in survivors_full.columns: survivors = survivors_full[survivors_full["passes_all"]].copy()`
— filtering now happens before `total_survivors = len(survivors)`. `06_summary.png` now reports the
true per-cluster survivor counts, not "275 … 55 PCMs."

Its per-cluster survival-rate branch requires `all_candidates` (the PCM database) to have a
`cluster_id` column, which it never does, so `01_survival_rate_by_cluster.png` silently falls back
to plotting raw counts with a "Survival rate (%)" axis label carried over from the other branch.

### `verify_04_ranking.py` — trustworthy, with two framing caveats (one newly current)

`06_summary.png` used to report (an earlier, two-method-era run):

```
Number of methods: 3    Methods: TOPSIS, GRA, CONSENSUS
Number of ranked candidates: 15    Number of clusters: 5
Method agreement (Spearman rho):
  TOPSIS vs GRA:       -0.930
  TOPSIS vs CONSENSUS:  0.376
  GRA vs CONSENSUS:    -0.442
Top-3 consensus candidates:  1. RT60   1. PureTemp 58   2. savE® OM55
Data completeness: 98.1%
```

**Caveat 1 (still applies):** the Spearman values are computed across the **pooled 15 Top-3 rows
from all five clusters at once**, not per cluster. They are not the per-cluster inter-method
agreement statistic — that is Kendall's W, verified from `08_mcdm_ranking.py`'s current output:
0.796/0.842/0.782/0.708/0.796 for Clusters 0-4.

**Caveat 2 — RESOLVED (2026-09), same session it was identified in.** This verify script's
coverage had fallen behind `08`'s method count: `08_mcdm_ranking.py` computes four methods
(TOPSIS+GRA+PROMETHEE+VIKOR), but `verify_04_ranking.py`'s `rank_cols` only looked at
`['topsis_rank', 'gra_rank', 'consensus_rank']`, silently omitting `promethee_rank`/`vikor_rank`
from every downstream analysis (the correlation matrix, top-3 inclusion probability, rank
reversal, and the summary panel). Fixed: `rank_cols` now includes all five columns
(`topsis_rank, gra_rank, promethee_rank, vikor_rank, consensus_rank`), with fallback rank
computation added for the two new methods (`promethee_flow` descending, `vikor_Q` ascending —
matching `08`'s own conventions) and the summary panel's pairwise-agreement text generalized to
loop over all pairs instead of three hardcoded ones. Confirmed by rerunning: `Methods:` now prints
all five columns, `06_summary.png` shows all 10 pairwise Spearman values.

---

## Two generations of results are preserved side by side

`verify_feasibility/` and `verify_ranking/` each contain **two** summary files with different
names, only one of which the current script writes (`06_summary.png`). The extra files
(`06_feasibility_summary.png`, `06_ranking_summary.png`) are from an earlier run:

| | Earlier generation | Current generation |
|---|---|---|
| Summary file | `06_feasibility_summary.png` / `06_ranking_summary.png` | `06_summary.png` (both dirs) |
| PCM database size | **25 rows** (denominator in "Overall Survival Rate: 500.0%" = 125/25) | **55 rows** |
| Rows in the survivors CSV | 125 (25 × 5) | 275 (55 × 5) |
| Clusters | 5 | 5 |
| Top-3 consensus | **RT54HC, RT55, RT64HC** | **RT60, PureTemp 58, savE® OM55** |
| TOPSIS vs GRA Spearman | **−1.000** ("Poor"); TOPSIS/GRA vs CONSENSUS = `nan` | −0.930 / 0.376 / −0.442 |
| Ranked candidates | 15 | 15 |
| Data completeness | 98.1 % | 98.1 % |

This corroborates the 25-row database referenced in `NEXT_STEPS.md` and in
`07_feasibility_filter.py`'s stale warning string (`01_PROJECT_CONTEXT.md`), and shows the Top-3
result **completely changed** when the database grew from 25 to 55 rows — direct evidence that the
recommendation is sensitive to database coverage.

The "Overall Survival Rate: 500.0 %" line in the older file is an artefact of the same
`passes_all` defect combined with a 25-row denominator; it is not a meaningful statistic.

**A third generation now exists (2026-09), superseding the "Current generation" column above.**
After fixing the VIKOR/TOPSIS bugs in `08_mcdm_ranking.py` and the regime-cap bug in
`07b_charging_feasibility.py`, the Top-3 consensus is no longer RT60/PureTemp 58/savE OM55
identically for all clusters — Cluster 1 now genuinely differs (PureTemp 53/n-Hexacosane/Myristic
acid), and Clusters 0/4 (PureTemp 58/n-Octacosane/PlusICE A58), Cluster 2 (PureTemp
58/savE OM55/n-Hexacosane), and Cluster 3 (PureTemp 58/savE OM55/Palmitic-stearic/EG) each have
their own Top-3. Kendall's W per cluster is now 0.708-0.842 (vs. the pooled −0.930 TOPSIS-vs-GRA
figure from the two-method era). This is now the third data point in the same story this table
tells: database coverage, method count, and bug fixes all move the recommendation — evidence the
pipeline's outputs are sensitive to its inputs and implementation, which argues for treating any
single run's numbers as provisional until independently reproduced.

---

## Cross-check: does the plot layer agree with itself?

| Quantity | Independent sources | Agree? |
|---|---|---|
| 45 points | `A0_all_points_overview.html` markers; `A2_population_map.html` popups; `02_climate_regime_map_folium.html` popups | **Yes** |
| 493,155 input rows | `C_era5_vs_power_stats.csv` (n); `07_preprocessing_summary.png` | **Yes** |
| 5 clusters, sizes 12/9/3/7/14 *(pre-2026-09-fix run; current run is 7/3/9/10/16 per `06_PHASE_4_AUDIT.md` and this file's own `verify_02_clustering.py` section above)* | `02_climate_regime_map_folium.html`; `06_cluster_sizes.png`; `02_silhouette_plot.png` (k=5) | **Yes, internally, but all three artefacts are from the same superseded run** |
| 55-row PCM database | `06_summary.png`; `05_pcm_survivors_per_cluster_interactive.html`; the committed source CSV | **Yes** |
| Top-3 per cluster | `objective1/recommended_pcm_summary.html`; `objective1/consensus_vs_topsis_agreement.html`; `uttarakhand_objective1/07_bump_chart_ranks.html`; `uttarakhand_objective1/13_recommended_pcm_summary_interactive.html` | **Yes — all four** |
| Top-3 PCM properties | `13_recommended_pcm_summary_interactive.html` `customdata` vs `PCM_Properties_cleaned_mice_pmm_detailed.csv` | **Yes — exact match** |
| Spearman ρ values | `verify_ranking/06_summary.png`; `08_method_rank_correlation_heatmap_interactive.html` | **Yes** |

The plot layer is internally consistent. Where it misleads, it does so systematically (the
`passes_all` filter) rather than randomly.

---

## Summary of plotting/verification defects

| # | Defect | Severity | Fix |
|---|---|---|---|
| 1 | ~~`05d`/`05c` Folium maps centred at `[10.9, 78.5]` (Tamil Nadu)~~ | **RESOLVED** — both scripts and the committed HTML output now use `[29.7, 78.9]` | done |
| 2 | ~~Plots 03/04/05/06 and `verify_03` never filter `passes_all`~~ | **RESOLVED** — all now filter `passes_all` before use (confirmed by reading current code) | done |
| 3 | ~~`comparison_plots_uttarakhand.py`'s `BASE` includes a spurious `".."`~~ | **RESOLVED** — script now uses `config.py` for all paths and runs correctly | done |
| 4 | ~~Plot 11 titled "Simulated Performance vs MCDM Consensus Rank" while plotting TOPSIS vs consensus~~ | **RESOLVED** — Phase 7 now exists, plot 11 correctly plots simulated solar fraction | done |
| 5 | Plot 12 is hard-coded sinusoids | Medium — reads as simulation output | relabel "illustrative schematic" or remove |
| 6 | `objective1/top3_inclusion_probability.html` is a count, not a probability | Low–Medium | rename |
| 7 | `data/plots/objective1/` has no generator in the folder | Low | commit the generator or delete the directory |
| 8 | `verify_02` silhouette computed on a different feature space than the GMM used | Low | restrict to `_z` columns |
| 9 | `verify_03`'s survival-rate branch needs a `cluster_id` the PCM database never has | Low | use `len(all_candidates)` as the denominator |
| 10 | Two generations of verify summaries coexist under different filenames | Low | prune, or date-stamp outputs |
| 11 | `verify_*` use relative paths, not `config.py` | Low | import `config` |
| 12 | `05c` docstring says "133 points" | Cosmetic | correct to 45 |
| 13 | `B_distributions_post.html` is 43 MB | Low | `include_plotlyjs="cdn"` and downsample |

## Status

**The QA layer (`03`, `03b`, `04c` ×2) is sound and produced the two CSVs that carry this
documentation set's evidentiary weight.** The verification suite is a genuine asset —
`verify_01` and `verify_02` in particular preserved numbers that would otherwise have been lost to
`.gitignore`. The Objective 1 figure set is usable for clustering and ranking but **should not be
used as-is for feasibility counts, physics agreement, or tank behaviour**, and the comprehensive
maps need a one-line centre fix before any of them goes in a report.
