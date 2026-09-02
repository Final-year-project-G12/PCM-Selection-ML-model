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
| `data/plots/comparison/` | — | `comparison_plots_uttarakhand.py` | **Never produced** |
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

## `05d_plots_comprehensive.py` — a real, verifiable defect

The script initialises **all three Folium maps at Tamil Nadu's coordinates**:

```python
TN_CENTER = [10.9, 78.5]          # line 72
...
m0 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")   # line 115
m1 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB positron")   # line 146
m2 = folium.Map(location=TN_CENTER, zoom_start=7, tiles="CartoDB dark_matter")# line 170
```

**Confirmed in the committed output.** `data/plots/comprehensive/maps/A0_all_points_overview.html`
contains:

```javascript
L.map("map_f201b1045ef73e9acb348711b5718335", {
    center: [10.9, 78.5],
    ...
    "zoom": 7,
```

while all 45 markers are at 28.875–30.625 °N, 77.875–80.125 °E. **Every map in
`data/plots/comprehensive/maps/` opens roughly 2,200 km south of the data.** The markers are
correct; only the initial viewport is wrong. Fix: `location=[point_meta["lat"].mean(),
point_meta["lon"].mean()]`, which is what `03b_interactive_raw_qa.py` already does correctly.

The same literal appears in `05c_explore_interactive.py` line 399
(`folium.Map(location=[10.9, 78.5], …)`).

Two further stale-text items in the same pair of scripts (cosmetic, no output impact):

- `05c_explore_interactive.py` docstring: "Folium map of **all 133 points**" — Uttarakhand has 45.
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
| 03 | `03_melting_point_vs_latent_heat.*` | `feasibility_survivors` | **Misleading** — plots all 275 rows, not `passes_all` survivors |
| 04 | `04_feasible_candidates_highlighted.png` | `feasibility_survivors` + `pcm_database` | **Misleading** — same, labelled "Feasible-C{cid}" |
| 05 | `05_pcm_survivors_per_cluster.*` | `df.groupby("cluster_id").size()` | **Wrong** — counts **all** rows per cluster. Reports 55 per cluster; the real `passes_all` count is 29 |
| 06 | `06_pcm_feasibility_scatter_and_survivors.png`, `pcm_feasibility_scatter.png`, `pcm_survivors_per_cluster.png` | same | **Same defect** |
| 07 | `07_bump_chart_ranks.*` | `mcdm_topk` | **Yes** — TOPSIS / GRA / consensus rank per cluster; source of the per-method ranks in `08_PHASE_6_AUDIT.md`. Note `head(12)` truncates 3 of the 15 rows |
| 08 | `08_method_rank_correlation_heatmap.*` | `mcdm_topk` | Yes, **but pooled across all clusters** — see the caveat below |
| 09 | *(absent)* | `monte_carlo_stability.csv` / `top3_inclusion_probability` | **Never produced** — neither input exists, `p09()` prints "top3_inclusion_probability not found" |
| 10 | `10_rank_reversal_violin_bar.png`, `_interactive.html` | `mcdm_topk` | Yes — rank spread across methods |
| 11 | `11_agreement_plot.*` | `physics_validation_results.csv` **(absent)** | **Misleading** — falls through to plotting TOPSIS rank vs consensus rank while the title still reads "Simulated Performance vs MCDM Consensus Rank" |
| 12 | `12_tank_temperature_melt_fraction.*` | **hard-coded sinusoids** | **Not data.** See below |
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

### The `passes_all` defect

Plots 03, 04, 05 and 06 all treat every row of `feasibility_survivors_by_cluster.csv` as a
survivor. `07_feasibility_filter.py` writes **all 55 PCMs × 5 clusters = 275 rows**, each carrying
a `passes_all` boolean, specifically so the per-filter detail is auditable
(`07_PHASE_5_AUDIT.md`). Any consumer must filter on it. `08_mcdm_ranking.py` and
`09_recommendation_cards.py` do; these four plots do not.

**Consequence:** the committed "survivors per cluster" figures report **55**, the size of the whole
database. The reproduced true figure is **29** per cluster.

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

One naming caveat: `top3_inclusion_probability.html` is **not** a Monte Carlo probability. Its
y-axis is `Top3_count` — how many of the 5 clusters each PCM appears in (RT60 5, PureTemp 58 3,
n-Hexacosane C26 3, savE® OM55 2, Palmitic-stearic/EG 2). No Monte Carlo was ever run
(`08_PHASE_6_AUDIT.md`). The filename is misleading and should not be cited as an inclusion
probability.

---

## `comparison_plots_uttarakhand.py` — never ran

```python
BASE = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
```

`..` from `era5-uttarakhand/` resolves to `PCM-Selection-ML-model/`, so every input path points at
`PCM-Selection-ML-model/data/processed/…`, which does not exist. The output directory
`data/plots/comparison/` is **absent from the repository**, confirming the script produced nothing.

Fix: drop the `".."` so `BASE` is the script's own folder — the pattern every other script uses via
`config.py`.

Its eight comparisons (cluster GHI profiles, Tm_target vs cluster temperature, MCDM methods
side-by-side, Monte Carlo stability, latent-heat distributions, physics validation, cross-cluster
top-PCM properties, weight-sensitivity) would be genuinely useful; comparisons 4 and 6 would remain
inert regardless, since Monte Carlo and Phase 7 outputs do not exist.

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
docstring. `02_silhouette_plot.png` reports **average silhouette 0.279** at k = 5, and
`06_cluster_sizes.png` confirms 12 / 9 / 3 / 7 / 14.

**Caveat:** it builds its feature matrix from *every* numeric column of
`climate_signature_uttarakhand.csv` except `point_id/cluster_id/lat/lon/population`, then
re-standardises. That set includes the raw indices, the `_proxy` and `_true` duplicates, the
PCA-block members **and** the `_z` columns — a much larger space than the `_z`-only matrix the GMM
was fitted in. The 0.279 figure is a valid independent diagnostic but is **not** the silhouette
`05_cluster_uttarakhand.py` wrote to `bic_selection_uttarakhand.csv`.

### `verify_03_feasibility.py` — has the `passes_all` defect

```python
survivors = pd.read_csv(INPUT_SURVIVORS)     # never filters passes_all
total_survivors = len(survivors)
```

Its `06_summary.png` reports "Total Survivors: 275 … 55 PCMs" per cluster — the whole database.

Its per-cluster survival-rate branch requires `all_candidates` (the PCM database) to have a
`cluster_id` column, which it never does, so `01_survival_rate_by_cluster.png` silently falls back
to plotting raw counts with a "Survival rate (%)" axis label carried over from the other branch.

### `verify_04_ranking.py` — trustworthy, with a framing caveat

`06_summary.png`:

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

**Caveat:** the Spearman values are computed across the **pooled 15 Top-3 rows from all five
clusters at once**, not per cluster. They are not the per-cluster inter-method agreement statistic —
that is Kendall's W, which `08` computes and which is not recoverable from any committed artefact.

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

---

## Cross-check: does the plot layer agree with itself?

| Quantity | Independent sources | Agree? |
|---|---|---|
| 45 points | `A0_all_points_overview.html` markers; `A2_population_map.html` popups; `02_climate_regime_map_folium.html` popups | **Yes** |
| 493,155 input rows | `C_era5_vs_power_stats.csv` (n); `07_preprocessing_summary.png` | **Yes** |
| 5 clusters, sizes 12/9/3/7/14 | `02_climate_regime_map_folium.html`; `06_cluster_sizes.png`; `02_silhouette_plot.png` (k=5) | **Yes** |
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
| 1 | `05d`/`05c` Folium maps centred at `[10.9, 78.5]` (Tamil Nadu) | Medium — every comprehensive map opens 2,200 km off | `location=[lat.mean(), lon.mean()]` |
| 2 | Plots 03/04/05/06 and `verify_03` never filter `passes_all` | Medium — "survivors per cluster" reads 55 instead of 29 | add `df = df[df["passes_all"]]` |
| 3 | `comparison_plots_uttarakhand.py`'s `BASE` includes a spurious `".."` | Medium — script has never produced output | drop the `".."` |
| 4 | Plot 11 titled "Simulated Performance vs MCDM Consensus Rank" while plotting TOPSIS vs consensus | Medium — misleading in a paper | skip the plot when `PHYS_VAL` is absent |
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
