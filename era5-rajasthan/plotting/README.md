# `plotting/` is retired — use `../PLOTSV2/`

This folder's scripts have been removed. Everything it did now lives in
**[`../PLOTSV2/`](../PLOTSV2/)**, documented in
**[`../PLOTSV2/PLOTS_GUIDE.md`](../PLOTSV2/PLOTS_GUIDE.md)**.

```
cd ../PLOTSV2
python run_all_plots_v2.py              # everything, then assembles Plots/
python run_all_plots_v2.py objective1   # the 13 objective-1 plots
python run_all_plots_v2.py verify       # the four verification suites
python run_all_plots_v2.py phases       # the phase-1 / phase-3 figures
python run_all_plots_v2.py comparison   # the cross-step comparison set
python run_all_plots_v2.py plots        # re-assemble the curated Plots/ folder
```

## Why it was retired

PLOTSV2 was written as a corrected re-implementation of this folder, and both
copies were being maintained in parallel. Several PLOTSV2 scripts carry docstrings
naming the specific bug they fix in their `plotting/` counterpart — these were not
cosmetic differences:

| Retired script | Bug PLOTSV2 corrects |
| :--- | :--- |
| `verify_01_preprocessing_rajasthan.py` | Read `climate_rajasthan_points_clean.csv`, which lacks the engineered lag/rolling/delta columns, so the feature-engineering plot silently skipped |
| `verify_02_clustering_rajasthan.py` | Used Level B (seasonal) cluster assignments where the downstream PCM chain uses Level A |
| `verify_03_feasibility_rajasthan.py` | Counted all 186 evaluation rows as survivors — reported "62 survivors" per cluster instead of 9 / 14 / 16; also looked for `pass_*` constraint columns Rajasthan does not have |
| `verify_04_ranking_rajasthan.py` | Left `borda_score` in the method list, producing a spurious −1.00 row in the correlation heatmap |
| `05_bump_chart.py` | Derived consensus with `borda_score.rank()` (ascending) — but higher Borda is better, so the axis was inverted and the best candidate drawn last |
| `comparison_plots_rajasthan.py` | Read Level B clusters; never produced comparison plot 7 at all; did not filter on `survives_all` |

## Where each script went

**Deleted — PLOTSV2 supersedes them:**

| Retired | Replacement |
| :--- | :--- |
| `01_raw_vs_preprocessed.py` | `generate_rajasthan_plots.py` → `01_raw_vs_preprocessed_radiation.*` |
| `02_climate_regime_map_copy.py` | → `02_climate_regime_map.*` (+ folium) |
| `03_pcm_feasibility_scatter.py` | → `06_pcm_feasibility_scatter_and_survivors.png` |
| `04_pcm_survivors_per_cluster.py` | → `05_pcm_survivors_per_cluster.*` |
| `05_bump_chart.py` | → `07_bump_chart_ranks_cluster_*.*` |
| `06_method_correlation_heatmap.py` | → `08_method_rank_correlation_heatmap.*` |
| `08_rank_reversal_frequency.py` | → `10_rank_reversal_violin_bar.*` |
| `11_summary_cards.py` | → `13_recommended_pcm_summary.*` |
| `comparison_plots_rajasthan.py` | `../PLOTSV2/comparison_plots_rajasthan.py` |
| `verify_01`–`04_*.py` | `../PLOTSV2/verify_01`–`04_*.py` |
| `run_all_plots.py` | `../PLOTSV2/run_all_plots_v2.py` |

**Moved into PLOTSV2 unchanged** — these had no PLOTSV2 equivalent, so deleting
them would have lost capability rather than removed duplication:

- `09_mcdm_vs_physics_agreement.py` — per-cluster Spearman ρ with p-values,
  cross-checked against the audit-documented values. Richer than PLOTSV2's
  plot 11, which is a single scatter.
- `comparison_phase3_tmcap_old_vs_new.py`
- `comparison_phase5_lrequired_before_after.py` — the `L_required` methodology
  correction figure (see CLAUDE.md §3.1)
- `fix_unicode_issues.py`

## Two caveats

1. **The old figures under `../outputs/objective1_plots_rajasthan/` are orphaned.**
   Nothing can regenerate them — their scripts are gone. Everything still
   reachable has been regenerated inside `PLOTSV2/`. The one thing with no PLOTSV2
   equivalent is the *per-cluster* method-correlation heatmaps
   (`04_mcdm_agreement/method_correlation_heatmap_cluster_{0,1,2}.html`); PLOTSV2
   produces a single pooled heatmap instead.

2. **`../outputs/` is not a plotting folder.** Only `objective1_plots_rajasthan/`
   (~211 MB) came from here. The other ~399 MB is main-pipeline QC output
   (`qc_clean_distributions_rajasthan.html` alone is 235 MB). Do not delete
   `outputs/` wholesale.

The other `.md` files in this folder are stubs pointing into
`../PLOTSV2/PLOTS_GUIDE.md`; this whole folder can be deleted once you are
satisfied nothing references it.
