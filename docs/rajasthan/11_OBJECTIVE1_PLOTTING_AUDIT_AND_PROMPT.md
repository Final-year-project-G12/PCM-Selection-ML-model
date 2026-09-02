# 11 — Objective 1 Plotting Audit & Generation Prompt (Rajasthan)

## 0. Read this first — the staleness banner

⚠️ **As of 2026-08-31, Phase 3's `L_required` methodology was corrected (SHARE_PCM=0.5), and this
cascades through Phases 5–9.** Every plot below that touches feasibility survivors, MCDM rankings,
physics validation, or recommendation cards is either (a) not yet regenerated against the corrected
numbers, or (b) will be, once Phases 5–9 are re-run. The prompt in §6 builds this staleness check in
directly (via `provenance_lib.py`'s fingerprinting, already used by Phases 6/7/9) so every affected
plot is watermarked rather than silently presented as final. Plots 2 (regime map) and anything
upstream of Phase 4 are unaffected by this correction.

---

## 1. Status at a glance

| # | Plot | Status | Location if it exists |
|---|---|---|---|
| 1 | Raw vs. preprocessed radiation | **MISSING** (close cousins exist, not this exact chart) | see §2.1 |
| 2 | Climate-regime map | **EXISTS** | `outputs/qc_cluster_map_rajasthan.html` |
| 3 | Melting point vs. latent heat, feasible highlighted | **MISSING** | — |
| 4 | Number of feasible PCM candidates per regime | **MISSING** | — |
| 5 | *(→ #3 and #4, exact filenames)* `pcm_feasibility_scatter.png`, `pcm_survivors_per_cluster.png` | **MISSING** | — |
| 6 | Bump chart — rank per method + consensus | **MISSING** | — |
| 7 | Heatmap — Spearman/Kendall correlation between the 4 methods | **MISSING** (underlying numbers exist) | data in `mcdm_method_agreement_rajasthan.csv` |
| 8 | Histogram/bar — Monte Carlo Top-3 inclusion probability | **EXISTS** | `outputs/qc_montecarlo_inclusion_rajasthan.html` |
| 9 | Violin/bar — rank-reversal frequency across draws | **MISSING** (underlying numbers exist) | data persisted per-candidate in Phase 6 output, no plot |
| 10 | Agreement plot — simulated rank vs. MCDM consensus rank, per cluster | **MISSING** | — |
| 11 | Tank temperature / melt-fraction profile, representative day–night cycle | **MISSING**, requires a small simulator instrumentation change | — |
| 12 | Summary figure — recommended PCM + key properties, per cluster | **MISSING** (text version exists) | text: `outputs/recommendation_cards_rajasthan.md` |

(Numbering above matches your list with items 3/4/5 merged since #5 just names the files for #3/#4.)

---

## 2. Per-plot detail — why it's plotted, and how to tell if it's *right*

### 2.1 Raw vs. preprocessed radiation

**Why plotted:** GHI is the pipeline's single most consequential variable — it's what the
deaccumulation bug corrupted (Phase 2) and what Phase 2.5's Hampel filter almost over-corrected. A
raw-vs-clean overlay is the direct visual evidence that (a) the deaccumulation fix worked and (b)
the cleaning step did **not** smooth away real weather.

**How to verify:** Per Phase 2.5's own documented finding, GHI and CSI are *deliberately excluded*
from outlier winsorizing (clouds are real, not noise). So the correct-looking result is: raw and
cleaned GHI distributions should look **nearly identical** — if cleaning visibly flattens GHI's
variance or clips its cloudy-day dips, that's the same over-correction bug the project already found
and fixed once, resurfacing. By contrast, T_amb/RHum/W_spd **should** show visible tail-trimming
post-clean — those three are the ones actually Hampel-filtered.

**Closest existing material:** Phase 2.5's `outputs/qc_raw_*.html` and `outputs/qc_clean_*.html` (8
files each) contain the raw and clean distributions separately, but not side-by-side in one figure.

### 2.2 Climate-regime map

**Status: EXISTS** — `outputs/qc_cluster_map_rajasthan.html` (Phase 4, folium).

**Why plotted:** This is the visual proof of novelty claim N1 — that climate regimes were
*discovered* (GMM) rather than hand-drawn.

**How to verify:** Confirm k=3, confirm cluster 0 is the southernmost group (clusters are
canonically relabeled by ascending mean latitude, per the 2026-08-11 fix), and don't expect regime
boundaries to match Köppen-Geiger zones cleanly — ARI=0.19/NMI=0.32 against Köppen is a real,
reported, and *expected* low-to-moderate agreement (the GMM finds finer structure than Köppen's
broad classes), not an error to chase down.

### 2.3 / 2.4 Melting point vs. latent heat (feasible highlighted) + survivor count per regime

**Why plotted:** This is the single figure that makes Phase 5's feasibility filtering legible — it
shows, at a glance, why so few candidates survive the corrected 42–70°C / latent-heat-floor
constraints, and which specific candidates do.

**How to verify — this is where the staleness banner matters most.** Plot the reference band/line
for `L_required` from the CSV actually being read, and title the chart with that number. Two known,
very different numbers exist depending on which run produced the CSV:
- **Pre-correction (stale):** L_required ≈ 608–641 kJ/kg, 0 survivors at nominal κ=0.7 everywhere.
- **Post-correction (2026-08-31, current methodology):** L_required ≈ 285–344 kJ/kg, primary run
  (κ=0.7) survivors 4/7/5, κ-calibrated survivors 9/14/16 (39 total).

If the plot shows 0 survivors at κ=0.7, you are reading the stale file — check `feasibility_
survivors_rajasthan.csv`'s embedded `upstream_cluster_profile_fingerprint` against the current
`cluster_profiles_rajasthan.csv` before trusting the number.

**Survivor-count bar chart specifically:** verify the three cluster bars sum to the number in the
CSV filename tag (`pcm_database_status`) — should read the 55-row-database tag, not "PROVISIONAL —
~25-row."

### 2.5 Bump chart — rank per method + consensus

**Why plotted:** Makes visible, per candidate, whether the four MCDM methods (TOPSIS, PROMETHEE II,
VIKOR, GRA) broadly agree or diverge — the human-readable companion to Kendall's W.

**How to verify:** Check specifically for a **VIKOR sign inversion** — this was a real, documented,
fixed bug (compromise index computed with reversed best/worst terms), caught precisely because it
made VIKOR's line look near-perfectly *inverted* against TOPSIS/PROMETHEE (rho as low as −0.86) in
an earlier run. If a freshly generated bump chart shows VIKOR's line running opposite to the other
three, the bug has resurfaced — that is a real red flag, not a stylistic quirk to shrug off.

### 2.6 Heatmap — Spearman/Kendall correlation between the 4 methods

**Why plotted:** Numeric companion to the bump chart; this is literally the diagnostic the project
used to catch the VIKOR bug and to identify GRA as a "structural outlier."

**How to verify:** Expect **GRA** to show the lowest mean pairwise correlation with the other three
methods (Phase 6's own diagnostic names GRA the structural outlier in all three clusters). Expect
Cluster 0's block to look visibly weaker/patchier than Clusters 1 and 2's (Kendall's W: Cluster 0 =
0.388 vs. Clusters 1/2 = 0.634–0.635) — Cluster 0's low agreement is a genuine, still-open finding,
not a data-sparsity artifact (n=9, a healthy sample size).

### 2.7 Monte Carlo Top-3 inclusion probability

**Status: EXISTS** — `outputs/qc_montecarlo_inclusion_rajasthan.html`.

**Why plotted:** Quantifies how robust each candidate's Top-3 status is to ±weight and ±property
perturbation (Dirichlet/Gaussian draws), so the final Top-3 isn't presented as more certain than it
is.

**How to verify:** Candidates flagged `any_property_imputed` should generally show *wider* variance
/ lower peak inclusion probability than candidates with fully measured properties — imputation
uncertainty should visibly propagate, not vanish.

### 2.8 Rank-reversal frequency (violin/bar)

**Why plotted:** A single retained-Top-3 percentage can hide a lot of churn below the cutoff line;
rank-reversal frequency shows how often *any* two candidates swap order across the 1,000 draws
(N_DRAWS=1000, not the framework doc's 5000 — a documented, defensible deviation, not a shortcut to
hide).

**How to verify:** Cluster 0 (Kendall's W=0.388, the weakest cross-method agreement) should show
higher rank-reversal frequency than Clusters 1/2 (W=0.634–0.635) — this is a specific, testable
prediction from the project's own numbers; if Cluster 0 doesn't show elevated churn, that's worth a
second look at either plot.

### 2.9 Agreement plot — simulated performance rank vs. MCDM consensus rank

**Why plotted:** This is Phase 7's headline result made visual — does a higher Borda/Copeland rank
actually deliver better simulated solar fraction? The framework doc frames this validation as "what
makes the result publishable, not skippable."

**How to verify against the known, already-computed numbers:** the trend per cluster should visually
match:
- Cluster 0: ρ = −0.385 (downward trend — higher MCDM rank, *worse* simulated performance)
- Cluster 1: ρ = +0.125 (weak upward trend — best of the three, still weak)
- Cluster 2: ρ = −0.097 (flat/weak downward)

If a regenerated scatter shows any cluster trending strongly positive, the join between
`mcdm_rankings_rajasthan.csv` and `physics_validation_rajasthan.csv` is probably wrong (check you're
joining on cluster_id **and** PCM candidate, using the canonically-relabeled cluster_id, not a raw
GMM label — see §2.2).

### 2.10 Tank temperature / melt-fraction profile, day–night cycle

**Why plotted:** Makes the lumped-enthalpy physics model's behavior legible for a reader who won't
read `physics_lib.py` — shows charging (daytime, solar-driven Tw rise), the melt plateau (latent-heat
absorption near Tm), and overnight discharge.

**Note — this one needs a small code change, not just a plotting script.** Per the Phase 7/8 audits,
the simulator currently persists only aggregate metrics (annual solar fraction, hours-in-band, cycle
counts) — no hourly Tw/Tp/melt-fraction time series is saved to disk anywhere. The prompt in §6 asks
Claude Code to add an optional `save_timeseries=True` hook to the existing simulation call (not
rewrite the physics), for one representative day per cluster medoid.

**How to verify:** Energy conservation should be visually obvious — no unphysical temperature jumps
(this is exactly the class of bug the Backward-Euler solver fix and night-loss fix corrected,
verified in-code to ~1.6e-13 J residual). The plateau during melting should be visible and roughly
flat (near Tm_target≈57°C). Overnight discharge should sustain 58–62°C for a while before decaying,
consistent with the Avargani benchmark night-delivery test already passing in Phase 7.

### 2.11 Summary figure — recommended PCM + key properties per cluster

**Why plotted:** A graphical, thesis/viva-ready version of `recommendation_cards_rajasthan.md` —
one glance per cluster: top pick, Tm, latent heat, MCDM confidence, physics-validation caveat.

**How to verify:** Cross-check the Top-1 name/properties shown against
`recommendation_cards_rajasthan.md` programmatically (parse the md or, better, read the same CSVs it
was built from) — never hand-retype the numbers, since **the current recommendation cards are also
tagged stale pending the Phase 5–9 re-run** (2026-08-31 correction). Include the staleness watermark
on this figure specifically, since it's the one most likely to get screenshotted into a slide without
context.

---

## 3. New: comparison plots per pipeline step (your item beyond the 13)

You asked for step-by-step comparison plots as **separate code, separate output folder**. These are
not in your numbered list but follow naturally from what each phase's audit already documents as a
"before/after" or "old vs. new" moment:

| Phase | Comparison | Why |
|---|---|---|
| Phase 2.5 | Raw vs. cleaned, per variable (T_amb/RHum/W_spd/GHI/CSI), 5-panel figure | Same rationale as §2.1, generalized to all Hampel-filtered variables, not just GHI |
| Phase 3 | `diurnal_gradient` (Tier 1, sun-event) vs. `DTR_true` (Tier 2, daily-integral) | Tier 1 is a documented *underestimate* of true diurnal range — this plot makes that gap visible, which is the entire justification for keeping both tiers (N2) |
| Phase 3 | `Tm_target_capped_C` (worst-month basis) vs. the retained-for-audit `Tm_target_capped_C_p05day` (old single-day basis) | Shows exactly why the 2026-08-11 methodology revision was needed — the p05-day basis produces implausibly low caps (40.8–49.5°C) vs. field evidence |
| Phase 4 | Level A (spatial, whole-year) vs. Level B (temporal, per-season) cluster assignment | Shows the regime-shift fraction and season-tautology check visually, not just as a printed table |
| Phase 5 | L_required: pre-correction (608–641 kJ/kg) vs. post-correction (285–344 kJ/kg) survivor counts | This is the single most important before/after in the whole pipeline right now — makes the 2026-08-31 fix's impact undeniable |
| Phase 6 | Pre-bugfix vs. post-bugfix VIKOR ranks (historical, from commit history / saved intermediate CSVs if available) | Documents the caught-and-fixed sign-inversion bug visually, for the methodology write-up's "self-audit" narrative |
| Phase 7 | PCM tank vs. plain sensible-only tank, same weather | Reproduces the honest ~0.0% difference finding (at 50 kg PCM vs. 300 kg tank) — the framework doc's cited +30%/+4–8% literature gain did *not* reproduce here, and that's reported, not hidden |
| Phase 8 | k=0.0 (no supercooling penalty) vs. k=0.3 (max penalty) Spearman ρ per cluster | Visualizes the counter-intuitive finding that the penalty *worsens* Clusters 1–2 agreement |

---

## 4. Proposed output structure

```
outputs/objective1_plots_rajasthan/
├── 01_raw_vs_preprocessed/
├── 02_climate_regime_map/          (points to existing qc_cluster_map_rajasthan.html, not regenerated)
├── 03_feasibility/
│   ├── pcm_feasibility_scatter.png
│   └── pcm_survivors_per_cluster.png
├── 04_mcdm_agreement/
│   ├── bump_chart_rajasthan.html
│   └── method_correlation_heatmap_rajasthan.html
├── 05_montecarlo/
│   └── rank_reversal_frequency_rajasthan.html
├── 06_physics_validation/
│   ├── mcdm_vs_physics_agreement_rajasthan.html
│   └── tank_profile_{cluster0,cluster1,cluster2}_rajasthan.html
├── 07_recommendation_summary/
│   └── summary_cards_rajasthan.png
└── comparison_plots/
    ├── phase2_5_raw_vs_clean/
    ├── phase3_tier1_vs_tier2/
    ├── phase3_tmcap_old_vs_new/
    ├── phase4_levelA_vs_levelB/
    ├── phase5_lrequired_before_after/
    ├── phase6_vikor_bugfix_before_after/
    ├── phase7_pcm_vs_plaintank/
    └── phase8_penalty_k0_vs_k3/
```

---

## 5. Prompt to give Claude Code

```
Context: This is the Rajasthan branch of an Objective-1 PCM-selection pipeline (Phases 1–9, see
00_MASTER_OVERVIEW.md and the per-phase audit docs in this repo). Read the relevant phase audit
doc(s) and the actual CSV/HTML outputs on disk before writing any plotting code — do not assume
column names, just verify them directly against the files.

⚠️ CRITICAL FIRST STEP: Phase 3's L_required methodology was corrected 2026-08-31 (SHARE_PCM=0.5),
which cascades through Phases 5–9. Before writing any plot that reads a Phase 5+ output file
(feasibility_survivors_rajasthan*.csv, mcdm_rankings_rajasthan.csv, physics_validation_rajasthan.csv,
spearman_rho_by_cluster_rajasthan.csv, recommendation_cards_rajasthan.md), import
`provenance_lib.py` and check whether each file's embedded `upstream_cluster_profile_fingerprint`
matches the current on-disk `cluster_profiles_rajasthan.csv`. If Phases 5–9 have not yet been
re-run against the corrected signatures, DO NOT block — generate the plots anyway against whatever
is currently on disk, but stamp every affected figure with a visible "STALE — pending Phase 5–9
re-run (2026-08-31 L_required correction)" watermark/annotation, and print a one-line console
warning per script. Re-running Phases 5–9 first (`python run_all_rajasthan.py --from
07_feasibility_filter_rajasthan.py`) is preferred if time allows, but the plotting code itself must
work either way.

Build the following as SEPARATE scripts, one per plot or tightly-related plot group (do not merge
unrelated plots into one file), each with its own file, so any one can be re-run independently.
Output folder structure exactly as below. Use plotly for anything with more than ~10 data points,
per-point hover detail, or where a reader would want to zoom/filter (bump chart, correlation
heatmap, rank-reversal, agreement scatter, tank profile); use folium only where already established
(the existing regime map). Static matplotlib/seaborn PNG is fine only for the two explicitly-named
PNG outputs (pcm_feasibility_scatter.png, pcm_survivors_per_cluster.png) and the summary card figure.

For every plot, include a short in-code VERIFICATION block (printed to console, not just the figure)
that checks the plotted data against the specific known result documented in the audit files, and
prints PASS/WARN accordingly. Do not skip this — it's the actual point of this task, not an
afterthought.

============================================================
PART A — the 13 requested plots
============================================================

1. Raw vs. preprocessed radiation
   - Read climate_rajasthan_points.csv (raw, Phase 2) and climate_rajasthan_points_clean.csv
     (Phase 2.5). Plot overlaid GHI distributions (histogram or KDE) before/after cleaning, plus a
     second panel showing the same for T_amb (or RHum) so the contrast is visible: GHI should look
     nearly unchanged (deliberately excluded from Hampel filtering — verify programmatically: compute
     a distributional distance metric, e.g. KS statistic, between raw and clean GHI; it should be
     small). T_amb should show visible tail-trimming (larger KS statistic). Print both KS statistics
     as the verification block. Output: outputs/objective1_plots_rajasthan/01_raw_vs_preprocessed/

2. Climate-regime map
   - Already exists at outputs/qc_cluster_map_rajasthan.html (Phase 4). Do not regenerate; instead
     write a short script that copies/symlinks it into
     outputs/objective1_plots_rajasthan/02_climate_regime_map/ for a single consolidated delivery
     folder, and prints the k, cluster sizes, and canonical-relabeling confirmation (cluster 0 =
     lowest mean latitude) as its verification block.

3. Melting point vs. latent heat scatter, feasible candidates highlighted
   → outputs/objective1_plots_rajasthan/03_feasibility/pcm_feasibility_scatter.png
   - Read the full PCM candidate pool (PCM_Properties_cleaned_mice_pmm_detailed.csv +
     literature_rows(), or however 07_feasibility_filter_rajasthan.py assembles its candidate set —
     import and call its own loader function rather than re-implementing it) and
     feasibility_survivors_rajasthan_kappa_calibrated.csv. Scatter Tm (x) vs. latent_heat (y), all
     candidates in light grey, survivors colored by cluster_id, non-survivors left grey. Draw a
     shaded vertical band for the 42–70°C target range and a horizontal reference line at the
     CURRENT L_required value read directly from cluster_profiles_rajasthan.csv (not hardcoded — it
     differs pre/post the 2026-08-31 fix). Title must state which L_required value is being used.
     Verification block: print survivor count per cluster and compare against the
     pcm_database_status tag in the CSV; warn if 0 survivors are found at nominal kappa (this would
     indicate the stale pre-correction file is being read).

4. Number of feasible PCM candidates per climate regime
   → outputs/objective1_plots_rajasthan/03_feasibility/pcm_survivors_per_cluster.png
   - Simple grouped bar: primary run (kappa=0.7 fixed) survivor count vs. kappa-calibrated survivor
     count, per cluster, from feasibility_survivors_rajasthan.csv vs.
     feasibility_survivors_rajasthan_kappa_calibrated.csv. Annotate each cluster's calibrated kappa
     value on its bar. Verification block: print totals and compare against the audit-documented
     numbers if the fingerprint matches a known run (39 total post-correction calibrated, or 20 total
     pre-correction) — print which one it matches, or flag as a new/unrecognized run.

5. Bump chart — rank per method + consensus
   → outputs/objective1_plots_rajasthan/04_mcdm_agreement/bump_chart_rajasthan.html (plotly)
   - Read mcdm_rankings_rajasthan.csv. For each cluster (separate chart or faceted), plot each
     candidate's rank under TOPSIS, PROMETHEE, VIKOR, GRA, and Borda-consensus as connected points
     across 5 x-positions, one line per candidate, hover shows candidate name + all 5 ranks.
     Verification block: compute Spearman rho between VIKOR's ranks and TOPSIS's ranks per cluster;
     print a WARN if rho < -0.5 (this is the specific signature of the historical VIKOR sign-inversion
     bug re-appearing — expected/healthy is a positive or near-zero correlation, not strongly
     negative).

6. Heatmap — Spearman/Kendall correlation between the 4 methods
   → outputs/objective1_plots_rajasthan/04_mcdm_agreement/method_correlation_heatmap_rajasthan.html
     (plotly, one heatmap per cluster or faceted)
   - Read mcdm_method_agreement_rajasthan.csv if it already contains pairwise method correlations; if
     it only has partial data, compute the 4x4 Spearman correlation matrix directly from
     mcdm_rankings_rajasthan.csv's per-method rank columns. Verification block: identify which method
     has the lowest mean pairwise correlation with the other three in each cluster, and print whether
     it matches the audit's own finding (GRA, all three clusters).

7. Histogram/bar — Monte Carlo Top-3 inclusion probability per candidate
   - Already exists at outputs/qc_montecarlo_inclusion_rajasthan.html (Phase 6). Copy into
     outputs/objective1_plots_rajasthan/05_montecarlo/ alongside the new plot below rather than
     regenerating. Verification block: print correlation between inclusion probability and
     any_property_imputed flag (should be negative — imputed-property candidates should show lower/
     wider inclusion probability).

8. Violin or bar — rank-reversal frequency across the Monte Carlo draws
   → outputs/objective1_plots_rajasthan/05_montecarlo/rank_reversal_frequency_rajasthan.html (plotly)
   - Read the per-candidate rank-reversal-frequency column already persisted in Phase 6's output
     (check mcdm_rankings_rajasthan.csv's columns directly — the Phase 6 audit confirms this is
     computed and saved, verify the exact column name on disk rather than guessing). One violin/bar
     per cluster. Note: N_DRAWS=1000 in this pipeline (not literature-cited 5000) — label the axis/
     caption accordingly, don't claim 5000 draws. Verification block: print mean rank-reversal
     frequency per cluster and confirm Cluster 0 (Kendall's W=0.388) is higher than Clusters 1/2
     (W=0.634-0.635); print WARN if not, since this is a specific documented prediction.

9. Agreement plot — simulated performance rank vs. MCDM consensus rank, per cluster
   → outputs/objective1_plots_rajasthan/06_physics_validation/mcdm_vs_physics_agreement_rajasthan.html
     (plotly, one panel per cluster)
   - Join mcdm_rankings_rajasthan.csv (Borda/Copeland rank) with physics_validation_rajasthan.csv
     (simulated annual solar fraction) on cluster_id + candidate identity — confirm the join key by
     inspecting both files' columns directly, do not assume. Scatter MCDM rank (x) vs. simulated
     solar fraction (y), with a fitted trend line, per cluster. Verification block: compute Spearman
     rho per cluster and compare against the audit-documented values (Cluster 0: -0.385, Cluster 1:
     +0.125, Cluster 2: -0.097 — pre-correction numbers; if Phases 5-9 have been re-run, these
     reference numbers will differ, so compare against spearman_rho_by_cluster_rajasthan.csv's own
     current contents rather than the hardcoded historical numbers). Flag clearly if a re-run has
     happened.

10. Tank temperature / melt-fraction profile over a representative day-night cycle
    → outputs/objective1_plots_rajasthan/06_physics_validation/tank_profile_{cluster_label}_rajasthan.html
      (plotly, one file per cluster medoid, dual-axis: Tw/Tp on left, melt_fraction 0-1 on right)
    - This requires a small instrumentation change to physics_lib.py (or a wrapper script that calls
      its simulation function with an added save_timeseries=True / return_hourly=True option) rather
      than a rewrite. Add a hook that returns/saves the hourly Tw, Tp (or Tc), and melt_fraction
      arrays for ONE representative simulated day per cluster medoid (pick a clear-sky day near the
      cluster's kt_daily_mean, using the same weather/self-test infrastructure Phase 7 already built
      — do not build a second simulator). Plot the resulting time series with clearly marked charging
      (daytime), melt-plateau, and overnight-discharge regions. Verification block: confirm no
      unphysical jumps (max hour-to-hour |dT| below a sane threshold, e.g. 10K/hour) and that
      overnight Tw stays in the 58-62°C Avargani benchmark band for at least several hours, matching
      Phase 7's already-passing night-delivery test.

11. Summary figure — recommended PCM + key properties, per cluster
    → outputs/objective1_plots_rajasthan/07_recommendation_summary/summary_cards_rajasthan.png
    - Parse recommendation_cards_rajasthan.md (or, preferably, read the same underlying CSVs it was
      built from — mcdm_rankings_rajasthan.csv Top-1 per cluster, physics_validation_rajasthan.csv,
      cluster_profiles_rajasthan.csv) and render a 3-panel (one per cluster) card-style figure: Top-1
      PCM name, Tm, latent heat, MCDM confidence (Monte Carlo inclusion probability), and the Phase 7
      Spearman rho with an explicit "physics validation: NOT confirmed" flag where rho <= 0.4.
      Verification block: assert the Top-1 name/numbers shown match whichever source file was
      actually read, print them side by side with what recommendation_cards_rajasthan.md itself
      states, and WARN on any mismatch (this catches stale-file drift directly).

============================================================
PART B — comparison plots per pipeline step (separate folder: comparison_plots/)
============================================================

Build each of these as its own script in outputs/objective1_plots_rajasthan/comparison_plots/<name>/,
reading only files already produced by the pipeline (do not re-run simulations beyond what Part A's
item 10 already requires):

a. phase2_5_raw_vs_clean/ — 5-panel comparison (T_amb, RHum, W_spd, GHI, CSI), raw vs. clean,
   generalizing plot 1 above to all Hampel-relevant variables in one figure.
b. phase3_tier1_vs_tier2/ — scatter of diurnal_gradient (Tier 1) vs. DTR_true (Tier 2) from
   climate_signature_rajasthan.csv, with a y=x reference line, to visualize Tier 1's documented
   underestimate.
c. phase3_tmcap_old_vs_new/ — Tm_target_capped_C (current, worst-month basis) vs.
   Tm_target_capped_C_p05day (retained-for-audit old basis), both columns already present in
   climate_signature_rajasthan.csv per the Phase 3 audit — bar or scatter per point/cluster.
d. phase4_levelA_vs_levelB/ — contingency-style comparison of cluster_assignments_rajasthan_levelA.csv
   vs. cluster_assignments_rajasthan_levelB.csv (per point, per season) — visualize the regime-shift
   fraction already computed by 05_cluster_rajasthan.py.
e. phase5_lrequired_before_after/ — bar chart, survivor counts per cluster, if BOTH a pre-correction
   and post-correction feasibility_survivors file are found on disk (check timestamps/fingerprints);
   if only one exists, skip this script gracefully with a printed message rather than erroring.
f. phase6_vikor_bugfix_before_after/ — only buildable if a pre-fix intermediate CSV/backup still
   exists on disk; check for one (e.g. in a backups/ or archive/ folder) before writing this script;
   if none exists, skip gracefully and note in the script's docstring that this comparison is
   historical and not reproducible without the original buggy output.
g. phase7_pcm_vs_plaintank/ — bar or grouped comparison of solar fraction, PCM-tank vs. plain-tank,
   per cluster medoid, from whatever Phase 7 already persisted (check physics_validation_rajasthan.csv
   for a plain-tank comparator row/column; if not present, this needs the same save_timeseries hook
   as Part A item 10, run once with latent_heat=0).
h. phase8_penalty_k0_vs_k3/ — grouped bar, Spearman rho per cluster at k=0.0 vs. k=0.3, from
   phase8_supercooling_sweep_rajasthan.csv.

Before writing any script in Part B, list the exact column names you find in each source CSV (or
confirm a file's absence) so the schema is verified rather than assumed.
```
