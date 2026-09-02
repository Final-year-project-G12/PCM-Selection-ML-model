# Objective 1 — Sprint Plan (Uttarakhand Only)

Scope decision for this sprint: **finish Objective 1 on Uttarakhand alone.**
Cross-state clustering (the original 4-state plan v3.0 design) is real
future work, already documented and state-parameterised, but it is not
required to defend Objective 1 as a working framework. Don't spend time
trying to onboard another state's data yet.

## Where things actually stand

| Phase | What's needed | Status |
|---|---|---|
| 1. Data Collection | ERA5 + NASA POWER, 45 population-weighted Uttarakhand points, 3 sun-events/day, 10 years | **Done.** Points confirmed (`00a_build_population_grid.py`, ~87.5% population coverage); `02_combine_uttarakhand.py` produces `climate_uttarakhand_points.csv`. |
| 2. Preprocessing & QC | 13-step sequence + Tier-2 daily-integral repair | **`02b_build_daily_aggregates.py` confirmed run** (45/45 points, 0 skipped, 164,385 point-days). `04_preprocess_uttarakhand.py` code delivered — confirm it's actually been run and `qc_report.txt` ends in all-PASS before moving on. |
| 3. Climate Signature | 2-tier ~18-index vector per point, Tm_target/L_required, PCA, standardization | **Code delivered (`04b_climate_signature.py`, Tier1+Tier2 merge), not yet confirmed run.** |
| 4. Climate Regime Clustering | GMM, BIC-selected K, silhouette sanity | **Code delivered (`05_cluster_uttarakhand.py`), not yet confirmed run.** With only 45 points, expect a smaller K than a 133-point state would support — see `README_PREPROCESSING.md` for why. |
| 5. Feasibility Filtering | Hard-filter PCM database per cluster | **Code delivered, not yet run.** `06_build_pcm_database.py` (sources from the MICE+RF+PMM-cleaned manufacturer data in `PCM_data/`, ~25 candidates total) + `07_feasibility_filter.py`. Run these next, after Phases 3-4 are confirmed. |
| 6. Multi-Criteria Ranking | TOPSIS + GRA minimum, entropy+AHP weights, Gaussian Tm fitness transform, Borda consensus | **Code delivered, not yet run.** `08_mcdm_ranking.py`. This is the headline deliverable. |
| 7. Physics-Based Validation | Grey-box lumped enthalpy tank model, Spearman rho vs. MCDM rank | **Not written.** Do a minimal single-PCM sanity version per cluster if time allows, otherwise record as future work — an accepted, publishable outcome per the plan doc. |
| 8. Explanation & Output | Recommendation card per cluster | **Code delivered, not yet run.** `09_recommendation_cards.py` — turns 4-6's output directly into your results section. |

---

## Suggested order of operations

### Step 1 — confirm Phase 2 and 3 are actually clean
1. `python 02b_build_daily_aggregates.py` — **already confirmed run**, safe to
   re-run (always overwrites fresh, doesn't resume).
2. `python 03_plots_raw.py` — confirm noon still peaks GHI (plot B) and
   ERA5-vs-POWER agreement (plot C) looks sane before trusting anything
   downstream.
3. `python 04_preprocess_uttarakhand.py` — Phase 2. Check `qc_report.txt`
   ends with all checks PASS. With only 45 points, pay particular
   attention to how many rows step 4's zone-median imputation fallback
   actually had to use — with fewer points, each spatial "zone" is
   coarser than it would be for a bigger state.
4. `python 04c_postprocess_plots.py` — confirm cleaning didn't flatten
   the seasonal GHI shape (plot E) and residual missing % is ~0 (plot A).

### Step 2 — Phase 3 + Phase 4 + PCM database (parallel if you have a teammate)
5. `python 04b_climate_signature.py` — Phase 3, merges Tier1+Tier2. Check
   `pca_loadings.csv` reads sensibly (should look like "heat"/"humidity"
   components) and `signature_distributions.png` doesn't show anything
   degenerate. Check how much weight `elev_proxy` carries — see the
   elevation note in `README_PREPROCESSING.md`; with Uttarakhand's real
   elevation spread (~200-2000m populated), this is worth a genuine look,
   not just a caveat sentence.
6. `python 05_cluster_uttarakhand.py` — Phase 4. Look at
   `bic_selection_uttarakhand.csv`, pick K where silhouette lands in the
   0.15-0.40 band (not higher — see the script's own comments on why;
   with only 45 points a high silhouette is *more* likely to mean an
   over-simple signature than a genuinely crisp regime split). Set
   `K_FINAL` at the top of the script, re-run once.
7. In parallel: build/extend the PCM property database. Target the
   **42-70 C melting band** (this is the v3.0-corrected band, derived
   from the corrected Tm_target rule — not a state-specific number, it
   applies the same way here as anywhere). Pull directly from whatever's
   already in your `Sources/` folder — the same literature base the
   Tamil Nadu build used (Rubitherm RT-series, PLUSS OM-series, fatty
   acids/eutectics, salt hydrates) is general PCM materials research, not
   region-specific, so it carries over directly. One thing that *does*
   differ from the Tamil Nadu version: expect the corrosion veto to bite
   for high-monsoon-humidity Uttarakhand clusters (Terai/valley points
   during Jun-Sep) rather than the coastal-salinity framing that applied
   to Tamil Nadu's coastal points — same veto, different physical
   mechanism, worth noting in text.

### Step 3 — Phase 5 + Phase 6 core
8. Feasibility filter: for each cluster's `Tm_target`/`L_required` from
   `cluster_profiles_uttarakhand.csv`, apply the hard filters (melting
   window `[Tm_target-5, Tm_target+8]`, absolute 42-70C band, latent heat
   >= 0.7x L_required, corrosion veto if HSI above that cluster's 75th
   percentile, supercooling veto >8K, safety exclusion). Report survivor
   counts per cluster. With fewer, more elevation-differentiated clusters
   than a 133-point state, don't be surprised if per-cluster survivor
   counts are on the lower end — relax the window by 2K if under 5.
9. Implement the **Gaussian Tm fitness transform** before anything else
   touches melting temperature:
   `f_Tm = exp(-(Tm - Tm_target)^2 / (2*sigma^2))`, sigma ~4K. This is
   the step every PCM-MCDM paper gets wrong if skipped — do it first.
10. Implement TOPSIS and GRA (minimum viable pair). Entropy weights
    computed per cluster from that cluster's own filtered matrix; blend
    0.5/0.5 with AHP priors if you can get time with your guide,
    otherwise use entropy-only (lambda=1) and say so.
11. Add PROMETHEE II if time remains — it's the method that handles the
    target-based Tm criterion most naturally and is worth having as a
    second independent method even if you drop VIKOR/CoCoSo.

### Step 4 — aggregation, minimal validation, writeup
12. Aggregate to consensus Top-3 per cluster via Borda count; report
    Kendall's W across your 2-3 methods per cluster (low W is itself a
    valid, reportable finding — it means that regime's PCM choice is
    genuinely ambiguous, and with fewer points per cluster here than in a
    bigger state, this is a real possibility worth watching for).
13. If time allows: a **minimal** physics check — one grey-box lumped
    PCM tank simulation for just the Top-1 PCM in 1-2 clusters, compared
    against published Table-16-style benchmarks (annual solar fraction
    54-84%). A single calibration run per cluster is enough to write
    "consistent with published benchmarks" honestly. If you can't fit
    this in, say explicitly in the paper that physics validation is
    future work — an accepted outcome per the plan doc, not a weakness
    you need to hide.
14. Write the recommendation cards (one per cluster: identity, climate
    signature summary, Tm_target/L_required, survivor count, Top-3 with
    per-method ranks, caveats) — this is your results section.
15. Reproducibility pass: clean rerun of `02b -> 04 -> 04b -> 05 ->`
    your Phase 5/6 scripts, confirm every number you're about to write in
    the paper regenerates.

---

## What to explicitly not do right now

- Don't onboard Rajasthan/Assam/Tamil Nadu data. `05_cluster_regions.py`
  stays untouched and ready for later.
- Don't build the TabTransformer/VAE encoder ablation — it's explicitly
  optional-only in the plan doc and adds nothing to Objective 1's core
  claim.
- **Do** think about per-point real elevation (plan v3.0's "Repair 2")
  before you finalize clusters — unlike the Tamil Nadu build (where this
  was reasonably deprioritized), Uttarakhand's 200m-2000m populated
  elevation range is exactly the case this repair was written for. It
  doesn't have to happen before Phase 4's first pass, but check whether
  `elev_proxy` is carrying real weight in `04b`'s PCA/correlation output
  before treating a first clustering run as final.
- Don't run the full 5,000-draw Monte Carlo stability analysis unless
  Phase 5/6 finishes with time spare — a smaller draw count (even 500)
  with the method reported honestly beats skipping it silently, but it
  is genuinely optional.
- Don't try to add PRECTOTCORR to the NASA POWER download and fix
  `monsoon_index` — flag the proxy limitation in text instead, it costs
  no correctness in the ranking (monsoon_index isn't a ranking criterion,
  it's descriptive of the regime).

---

## Files this phase adds to your repo

```
02b_build_daily_aggregates.py    (Phase 2 Repair 1)             — RUN, confirmed done
04c_postprocess_plots.py         (post-clean QA, PNG)
04c_interactive_postprocess_qc.py(post-clean QA, interactive)
03b_interactive_raw_qa.py        (raw QA, interactive)
04d_signature_interactive.py     (Phase 3 explorer, interactive)
05_cluster_uttarakhand.py        (Phase 4, single-state)
05b_cluster_interactive.py       (Phase 4 explorer, interactive)
05c_explore_interactive.py       (Streamlit explorer, raw/processed/comparison)
05d_plots_comprehensive.py       (batch maps/timeseries/stats plots)
04b_climate_signature.py         (Tier1+Tier2 merge)
06_build_pcm_database.py         (Phase 5 prep — PCM database)   — RUN NEXT
07_feasibility_filter.py         (Phase 5 — feasibility filter)  — RUN NEXT
07b_charging_feasibility.py      (optional — regime-dependent Tm cap)
08_mcdm_ranking.py               (Phase 6 — TOPSIS+GRA ranking)  — RUN NEXT
09_recommendation_cards.py       (Phase 8 — results section)     — RUN LAST
README_PREPROCESSING.md          (documents every Phase 2-4 step)
```

Run order for what's left:
```
python 04_preprocess_uttarakhand.py   # if not already confirmed run
python 04c_postprocess_plots.py
python 04b_climate_signature.py
python 05_cluster_uttarakhand.py      # set K_FINAL after reviewing bic_selection_uttarakhand.csv, re-run
python 06_build_pcm_database.py       # edit INPUT_CSV path at the top first
python 07_feasibility_filter.py
python 08_mcdm_ranking.py
python 09_recommendation_cards.py
```

`03_plots_raw.py` and `04_preprocess_uttarakhand.py` are schema-identical
to their Tamil Nadu counterparts — nothing in their logic needed to change
for Uttarakhand, only filenames.

## What's genuinely still open after 06-09 run

- **PCM database is ~25 rows, not 40-60.** `06`'s docstring lists exactly
  what's missing. Add real datasheet rows if time allows; the pipeline
  works correctly either way, it's a coverage question, not a
  correctness one.
- **Corrosion veto and 5th-percentile-day charging feasibility** are not
  applied in `07` — the database and cluster profiles don't carry the
  data those two specific filters from Table 12 need yet. Documented,
  not silently skipped. `07b_charging_feasibility.py` covers the
  regime-dependent Tm cap piece of this if you want it before `07`.
- **Physics validation (Phase 7)** is not written. If you have a spare
  half-day, a single-PCM grey-box run per cluster against the Table 16
  benchmark ranges (annual solar fraction 54-84%) is enough to write
  "consistent with published benchmarks" honestly — full validation
  across every cluster is not required.
- **Elevation proxy** — flagged above, worth resolving before you treat
  Phase 4's clusters as final if `elev_proxy` shows real weight.
