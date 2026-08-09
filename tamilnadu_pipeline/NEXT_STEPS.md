# Objective 1 — 4-Day Sprint Plan (Tamil Nadu Only)

Scope decision for this sprint: **finish Objective 1 on Tamil Nadu alone.**
Cross-state clustering (the original 4-state plan v3.0 design) is real
future work, already documented and state-parameterised, but it is not
required to defend Objective 1 as a working framework. Don't spend any of
your 4 days trying to onboard another state's data.

## Where things actually stand

| Phase | What's needed | Status |
|---|---|---|
| 1. Data Collection | ERA5 + NASA POWER, 133 TN points, 3 sun-events/day, 10 years | **Done.** Confirmed correct (~650MB, ~1.46M rows matches the math). |
| 2. Preprocessing & QC | 13-step sequence + Tier-2 daily-integral repair | **Done.** `02b_build_daily_aggregates.py` + `04_preprocess_tamilnadu.py` both ran clean, all hard-gate checks passed. |
| 3. Climate Signature | 2-tier ~18-index vector per point, Tm_target/L_required, PCA, standardization | **Done.** Updated `04b_climate_signature.py` (Tier1+Tier2 merge) ran. |
| 4. Climate Regime Clustering | GMM, BIC-selected K, silhouette sanity | **Done.** `05_cluster_tamilnadu.py` ran — check `bic_selection_tamilnadu.csv` landed in a sane K before treating it as final. |
| 5. Feasibility Filtering | Hard-filter PCM database per cluster | **Code delivered, not yet run.** `06_build_pcm_database.py` v2 (now sources from your MICE+RF+PMM cleaned manufacturer data — all 18 rows fully populated with traceable, donor-logged imputation, not blindly zeroed — plus 7 literature rows, ~25 candidates total) + `07_feasibility_filter.py`. Run these next. |
| 6. Multi-Criteria Ranking | TOPSIS + GRA minimum, entropy+AHP weights, Gaussian Tm fitness transform, Borda consensus | **Code delivered, not yet run.** `08_mcdm_ranking.py`. This is the headline deliverable. |
| 7. Physics-Based Validation | Grey-box lumped enthalpy tank model, Spearman rho vs. MCDM rank | **Stretch goal, not written.** Do a minimal single-PCM sanity version per cluster if time allows, otherwise record as future work — an accepted, publishable outcome per the plan doc. |
| 8. Explanation & Output | Recommendation card per cluster | **Code delivered, not yet run.** `09_recommendation_cards.py` — turns 4-6's output directly into your results section. |

---

## Day-by-day (4 days)

### Day 1 — close out Phase 2 and 3 properly
1. `python 02b_build_daily_aggregates.py` — reads the NASA POWER hourly
   cache already on disk, no new downloads, a few minutes to run.
2. `python 03_plots_raw.py` — confirm noon still peaks GHI (plot B) and
   ERA5-vs-POWER agreement (plot C) looks sane before trusting anything
   downstream.
3. `python 04_preprocess_tamilnadu.py` — Phase 2. Check `qc_report.txt`
   ends with all checks PASS.
4. `python 04c_postprocess_plots.py` — confirm cleaning didn't flatten
   the seasonal GHI shape (plot E) and residual missing % is ~0 (plot A).

### Day 2 — Phase 3 + Phase 4 + PCM database (parallel if you have a teammate)
5. `python 04b_climate_signature.py` — Phase 3, merges Tier1+Tier2. Check
   `pca_loadings.csv` reads sensibly (should look like "heat"/"humidity"
   components) and `signature_distributions.png` doesn't show anything
   degenerate.
6. `python 05_cluster_tamilnadu.py` — Phase 4. Look at
   `bic_selection_tamilnadu.csv`, pick K where silhouette lands in the
   0.15-0.40 band (not higher — see the script's own comments on why),
   set `K_FINAL` at the top of the script, re-run once.
7. In parallel: build the PCM property database. Target 30-40 rows in the
   **42-70 C melting band** (this is the v3.0-corrected band; the older
   presentation slide said 35-65 C — use 42-70 as primary and mention the
   discrepancy was corrected, since 42-70 is derived from the corrected
   Tm_target rule, not the earlier passive-cooling-derived range). Pull
   directly from what's already in your `Sources/` folder:
   - Rubitherm RT-series: RT42, RT44HC, RT50, RT54HC, RT55, RT58, RT64HC
     (`Martinez2025PCM_Industrial_TES_summary.md` has *measured* not just
     datasheet values for RT54HC/RT55/RT64HC — use those, they're more
     defensible than vendor sheets alone).
   - PLUSS OM-series: OM42, OM45, OM48, OM55, OM65
     (`Singh2025PCM_SWH_ComprehensiveReview_summary.md` Table 2 has
     several eutectic blends with exact Tm/latent-heat pairs).
   - Fatty acids/eutectics from the same Singh table.
   - Salt hydrates (sodium acetate trihydrate ~58C) — include but expect
     the corrosion veto to remove them in high-humidity TN clusters
     (coastal points).
   Columns needed per row: name, type, Tm (C), latent heat (kJ/kg),
   thermal conductivity (W/m.K), density, specific heat, cycling
   stability (cycles, or "not reported"), corrosion class, supercooling
   (K, or "not reported"), approximate cost, safety flag.

### Day 3 — Phase 5 + Phase 6 core
8. Feasibility filter: for each cluster's `Tm_target`/`L_required` from
   `cluster_profiles_tamilnadu.csv`, apply the hard filters (melting
   window `[Tm_target-5, Tm_target+8]`, absolute 42-70C band, latent heat
   >= 0.7x L_required, corrosion veto if HSI above that cluster's 75th
   percentile, supercooling veto >8K, safety exclusion). Report survivor
   counts per cluster (target 8-20; relax the window by 2K if under 5).
9. Implement the **Gaussian Tm fitness transform** before anything else
   touches melting temperature:
   `f_Tm = exp(-(Tm - Tm_target)^2 / (2*sigma^2))`, sigma ~4K. This is
   the step every PCM-MCDM paper gets wrong if skipped — do it first.
10. Implement TOPSIS and GRA (minimum viable pair — both already have
    worked equations in your `Chen2025TaguchiGRA...` and
    `OdoiYorke2025...` summary files to check your implementation
    against). Entropy weights computed per cluster from that cluster's
    own filtered matrix; blend 0.5/0.5 with AHP priors if you can get 10
    minutes with your guide, otherwise use entropy-only (lambda=1) and
    say so.
11. Add PROMETHEE II if time remains — it's the method that handles the
    target-based Tm criterion most naturally and is worth having as a
    second independent method even if you drop VIKOR/CoCoSo.

### Day 4 — aggregation, minimal validation, writeup
12. Aggregate to consensus Top-3 per cluster via Borda count; report
    Kendall's W across your 2-3 methods per cluster (low W is itself a
    valid, reportable finding — it means that regime's PCM choice is
    genuinely ambiguous).
13. If time allows: a **minimal** physics check — one grey-box lumped
    PCM tank simulation (reuse the ODE structure from
    `Barqawi2025DynamicSimulationPCM_SWH_summary.md`'s Eqs. 1-16, they're
    already extracted for you) for just the Top-1 PCM in 2-3 clusters,
    compared against the Table 16-style benchmarks (annual solar fraction
    54-84%). This does not need to be the full 5,000-point Monte Carlo
    validation — a single calibration run per cluster is enough to write
    "consistent with published benchmarks" honestly. If you can't fit
    this in, say explicitly in the paper that physics validation is
    future work — that's an accepted outcome per the plan doc, not a
    weakness you need to hide.
14. Write the recommendation cards (one per cluster: identity, climate
    signature summary, Tm_target/L_required, survivor count, Top-3 with
    per-method ranks, caveats) — this is your results section.
15. Reproducibility pass: clean rerun of `02b -> 04 -> 04b -> 05 ->`
    your Phase 5/6 script, confirm every number you're about to write in
    the paper regenerates.

---

## What to explicitly NOT do in these 4 days

- Don't onboard Rajasthan/Assam/Uttarakhand data. `05_cluster_regions.py`
  stays untouched and ready for later.
- Don't build the TabTransformer/VAE encoder ablation — it's explicitly
  optional-only in the plan doc and adds nothing to Objective 1's core
  claim.
- Don't chase per-point real elevation (plan v3.0's "Repair 2") — that
  repair matters for Uttarakhand's 200m-7000m range, not for Tamil Nadu's
  much gentler terrain. Flat 150m proxy + one sentence in limitations is
  fine here.
- Don't run the full 5,000-draw Monte Carlo stability analysis unless
  Phase 5/6 finishes with a full day spare — a smaller draw count (even
  500) with the method reported honestly beats skipping it silently, but
  it is genuinely optional against the 4-day clock.
- Don't try to add PRECTOTCORR to the NASA POWER download and fix
  `monsoon_index` — flag the proxy limitation in text instead, it costs
  no correctness in the ranking (monsoon_index isn't a ranking criterion,
  it's descriptive of the regime).

---

## Files this sprint adds to your repo

```
02b_build_daily_aggregates.py    (Phase 2 Repair 1)             — RUN, done
04c_postprocess_plots.py         (post-clean QA, PNG)           — RUN, done
04c_interactive_postprocess_qc.py(post-clean QA, interactive)
03b_interactive_raw_qa.py        (raw QA, interactive)
04d_signature_interactive.py     (Phase 3 explorer, interactive)
05_cluster_tamilnadu.py          (Phase 4, single-state)         — RUN, done
05b_cluster_interactive.py       (Phase 4 explorer, interactive)
04b_climate_signature.py         (Tier1+Tier2 merge)             — RUN, done
06_build_pcm_database.py         (Phase 5 prep — PCM database)   — RUN NEXT
07_feasibility_filter.py         (Phase 5 — feasibility filter)  — RUN NEXT
08_mcdm_ranking.py               (Phase 6 — TOPSIS+GRA ranking)  — RUN NEXT
09_recommendation_cards.py       (Phase 8 — results section)     — RUN LAST
README_PREPROCESSING.md          (documents every Phase 2-4 step)
```

Run order for what's left:
```
python 06_build_pcm_database.py     # edit INPUT_CSV path at the top first
python 07_feasibility_filter.py
python 08_mcdm_ranking.py
python 09_recommendation_cards.py
```

`03_plots_raw.py` and `04_preprocess_tamilnadu.py` are unchanged from the
originals — they were already correct.

## What's genuinely still open after 06-09 run

- **PCM database is ~25 rows, not 40-60.** `06`'s docstring lists exactly
  what's missing (RT58/RT60/RT62HC, PLUSS OM55/OM65, a properly-sourced
  salt hydrate). Add real datasheet rows if time allows; the pipeline
  works correctly either way, it's a coverage question, not a
  correctness one.
- **Corrosion veto and 5th-percentile-day charging feasibility** are not
  applied in `07` — the database and cluster profiles don't carry the
  data those two specific filters from Table 12 need yet. Documented,
  not silently skipped.
- **Physics validation (Phase 7)** is not written. If you have a spare
  half-day, a single-PCM grey-box run per cluster (reuse Barqawi 2025's
  ODE structure, already extracted in your Sources/ summary) against the
  Table 16 benchmark ranges (annual solar fraction 54-84%) is enough to
  write "consistent with published benchmarks" honestly — full 5,000-PCM,
  every-cluster validation is not required.
