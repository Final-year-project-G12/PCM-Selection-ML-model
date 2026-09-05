# Objective 1 — 4-Day Sprint Plan (Tamil Nadu Only)

Scope decision for this sprint: **finish Objective 1 on Tamil Nadu alone.**
Cross-state clustering (the original 4-state plan v3.0 design) is real
future work, already documented and state-parameterised, but it is not
required to defend Objective 1 as a working framework. Don't spend any of
your 4 days trying to onboard another state's data.

## Where things actually stand

| Phase | What's needed | Status |
|---|---|---|
| 1. Data Collection | ERA5 + NASA POWER, 133 TN points, 3 sun-events/day, 10 years | **Done.** |
| 2. Preprocessing & QC | 13-step sequence + Tier-2 daily-integral repair + v3.1 fixes | **Code fixed (v3.1).** Deaccumulation, quantile mapping, agreement analysis. **Re-run required.** See `CHANGELOG.md`. |
| 3. Climate Signature | 2-tier ~18-index vector per point, Tm_target/L_required, PCA, standardization | **Done, same annualization fix applied.** |
| 4. Climate Regime Clustering (Level A) | GMM, BIC-selected K, silhouette sanity | **Done.** |
| 4b. Level B (seasonal) | Check whether Top-3 flips between seasons | **Code delivered, not yet run.** `11_level_b_seasonal_analysis.py`. |
| 5. Feasibility Filtering | Hard-filter PCM database per cluster, all 8 Table 12 filters | **Done, upgraded.** Corrosion veto + safety exclusion added (`07_feasibility_filter.py`). |
| 6. Multi-Criteria Ranking | TOPSIS + GRA + PROMETHEE II + VIKOR, entropy+AHP weights, Gaussian Tm fitness, Borda+Copeland consensus, 5000-draw Monte Carlo | **Done, upgraded to the full 4-method + Monte Carlo stack.** `08_mcdm_ranking.py` v2. |
| 7. Physics-Based Validation | Grey-box lumped enthalpy tank model, Spearman rho vs. MCDM rank | **Implemented, not deferred.** `10_physics_validation.py` — real climate data, stated tank assumptions, calibration check against the 54-84% benchmark band. |
| 8. Explanation & Output | Recommendation card per cluster | **Done, upgraded.** `09_recommendation_cards.py` v2 now includes Phase 7 results. |

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
02b_build_daily_aggregates.py    (Phase 2 Repair 1, + HDD/CDD + CCI fixes)
04c_postprocess_plots.py         (post-clean QA, PNG)
04c_interactive_postprocess_qc.py(post-clean QA, interactive)
03b_interactive_raw_qa.py        (raw QA, interactive)
04d_signature_interactive.py     (Phase 3 explorer, interactive)
05_cluster_tamilnadu.py          (Phase 4, single-state)
05b_cluster_interactive.py       (Phase 4 explorer, interactive)
04b_climate_signature.py         (Tier1+Tier2 merge, + HDD/CDD fix)
06_build_pcm_database.py         (Phase 5 prep — MICE+RF+PMM sourced DB)
07_feasibility_filter.py         (Phase 5 — now all 8 Table 12 filters)
07b_charging_feasibility.py      (optional — heuristic regime-capped Tm)
08_mcdm_ranking.py               (Phase 6 — full 4-method + Monte Carlo)
09_recommendation_cards.py       (Phase 8 — now includes Phase 7 results)
10_physics_validation.py         (Phase 7 — grey-box tank model, real data)
11_level_b_seasonal_analysis.py  (Phase 4 Level B — seasonal sensitivity)
README_PREPROCESSING.md          (documents every Phase 2-4 step)
CHANGELOG.md                     (everything fixed/added this round)
```

Run order for what's left (everything else has already been run
successfully per your earlier session output):
```
python 07_feasibility_filter.py         # re-run: now includes corrosion + safety filters
python 08_mcdm_ranking.py               # re-run: now full 4-method + Monte Carlo (~1-2 min)
python 10_physics_validation.py         # NEW — Phase 7, real climate data (~few minutes)
python 09_recommendation_cards.py       # re-run: now pulls in Phase 7 results
python 11_level_b_seasonal_analysis.py  # NEW — optional but recommended
```

`03_plots_raw.py` and `04_preprocess_tamilnadu.py` are unchanged from the
originals — they were already correct.

## What's genuinely still open after all of the above run

- **PCM database is ~25 rows, not 40-60.** `06`'s docstring lists exactly
  what's missing (RT58/RT60/RT62HC, PLUSS OM55/OM65, a properly-sourced
  salt hydrate). Add real datasheet rows if time allows; the pipeline
  works correctly either way, it's a coverage question, not a
  correctness one.
- **External cluster validation** (ARI vs. Köppen-Geiger/NBC zones) is
  not implemented — needs an external classification lookup this
  pipeline doesn't have. Lower priority for TN-only scope per both
  reviews; add before extending to more states.
- **Elevation** — still the flat 150m proxy, fine for Tamil Nadu, would
  need fixing before Uttarakhand.
- **`monsoon_index`** stays proxy-only (NASA POWER precipitation was
  never downloaded) — documented, not a blocker.
- **Level B** as implemented (`11`) is the "nearly free" version the plan
  explicitly permits (per-season re-ranking within existing Level-A
  clusters), not full independent per-season GMM clustering. Upgrade if
  you have time and want the literal spec.
