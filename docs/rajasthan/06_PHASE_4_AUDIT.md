# 06 — Phase 4 Audit: Climate Regime Clustering

Scripts: `cluster_lib.py`, `05_cluster_rajasthan.py`, `05a_level_b_regime_shift_rajasthan.py`,
`11_seasonal_pcm_sensitivity.py`; `05b_cluster_interactive.py`, `05c_explore_interactive.py`,
`05d_plots_comprehensive.py` (visualization / interactive — audit stubs at the end of this file).

**Unified with Tamil Nadu (2026-09-08).** Three structural changes, no change to what the clustering
finds:

- The shared Phase-4 machinery — `bootstrap_ari_stability()`, the 3-tier `suggest_k()` cascade, the
  GMM+KMeans `fit_k_range()` scan, and `canonical_relabel_by_latitude()` — moved out of this script
  into **`cluster_lib.py`**, imported here and by `05a` and by both of Tamil Nadu's equivalents
  (one implementation, four callers). `GMM_COVARIANCE_TYPE='diag'`, `N_BOOTSTRAP=50`,
  `SILHOUETTE_LO/HI=[0.15,0.35]` now live there too.
- **Level B was extracted** from the bottom of this file into `05a_level_b_regime_shift_rajasthan.py`
  (see its own section below), so both states have the same file structure and the two analyses
  that were BOTH called "Level B" stop colliding.
- **`11_seasonal_pcm_sensitivity.py` is new** — the post-Phase-6 seasonal PCM re-ranking
  Tamil Nadu already had (there renamed from `11_level_b_seasonal_analysis.py`). Rajasthan now has
  the identical check.
- Two QC-plot improvements applied to BOTH states: the k-selection curve overlays **bootstrap-ARI**
  as a third series (it is the actual tiebreaker in `suggest_k()` and was previously not plotted
  anywhere), and the Level-A cluster map scales **marker size by population**.

## Purpose

Discover climate regimes empirically (Gaussian Mixture Model) rather than assume hand-drawn zones —
this is novelty claim N1 from the framework doc. Two levels: Level A (spatial — one signature vector
per point, whole 10-year record) and Level B (temporal — one vector per point per season, detects
whether a point's PCM-relevant regime shifts materially between seasons).

## Inputs

`climate_signature_rajasthan.csv` (Level A, direct read of the `*_z` columns) and
`climate_rajasthan_points.csv` + `suntimes.csv` (Level B, which **rebuilds** Tier-1 signatures
per-season directly via `signature_lib.build_tier1_signature(group_keys=["point_id","season"])`
rather than reading any saved Level-A file — Level B therefore has no Tier 2, PCA, or interaction
terms, only 19 raw Tier-1 columns, freshly standardized with its own independent `StandardScaler`).

## Processing

### Level A
- `GaussianMixture(covariance_type="diag", random_state=42, n_init=5)` fit for `k=2..12`.
- Per k: BIC, AIC, silhouette (guarded for `n_unique>1`), Davies-Bouldin, Calinski-Harabasz, and
  **bootstrap-ARI stability** (50 resamples: fit once on full data → `base_labels`; 50× fit a fresh
  GMM on a with-replacement resample of the same size → predict on the *original* data → Adjusted
  Rand Index against `base_labels`; report the mean).
- K-Means (`n_init=10`) fit in parallel purely as a **reported comparison baseline**, never the
  primary model — silhouette curves for both appear side-by-side in `bic_selection_rajasthan.csv`.
- **No population-weighting of the GMM fit** — confirmed by direct code inspection (no
  `sample_weight` argument anywhere) — by design, since the point *sampling* is already
  population-weighted by construction (Phase 1); weighting the fit again would double-count
  population. Population enters only later, in cluster-profile weighted means.
- k-selection (`suggest_k()`): a documented 3-tier cascade — (1) k in the expected single-state
  range [2,4] AND silhouette in the realistic band [0.15, 0.35], pick highest bootstrap-ARI among
  those; (2) any k in the silhouette band, highest bootstrap-ARI; (3) fallback to lowest-BIC k, with
  a printed warning. **Not a forced single "K_FINAL"** — the framework doc explicitly asks for
  k=2–4 for a single-state run (vs k=6–10 expected once all four states combine), and the code
  enforces exactly that expectation rather than letting BIC alone pick (BIC here monotonically
  decreases across the entire scanned range with no interior minimum — it would otherwise "select"
  k=12, the edge of the scan, which is not a meaningful answer).

### Level B — regime shift (now `05a_level_b_regime_shift_rajasthan.py`)
Same GMM/K-Means machinery (via `cluster_lib.py`), `k=2..8`, on a freshly-built
per-point-per-season Tier-1 matrix (`signature_lib.build_tier1_signature(group_keys=["point_id",
"season"])` — no Tier 2, PCA or interaction terms). Additional checks specific to Level B: a
**regime-shift analysis** (fraction of points whose cluster assignment differs across the 4 seasons)
and a **season-tautology check** (contingency table + ARI/NMI between cluster labels and season
labels, plus an ANOVA F-statistic feature-ranking to check whether temperature/GHI features
dominate the clustering — which would suggest it is just rediscovering the season labels rather
than finding independent structure). The `LEVEL_B_EXCLUDE_FEATURES` ablation switch (default empty,
inactive) is retained in the extracted script.

**Completed-run result (2026-09-08 re-run):** k=8, bootstrap-ARI 0.813 (50/50 resamples).
Regime-shift fraction **320/320 points (100%)**. Season-tautology ARI(cluster, season) = **0.691**,
NMI = 0.805 — *high* agreement, i.e. Rajasthan's Level B substantially rediscovers "season" itself
(report it that way, not as a novel regime-shift discovery). New plot:
`outputs/qc_level_b_regime_shift_sankey_rajasthan.html` (alluvial of cluster membership across the
4 seasons, shifting flows highlighted).

### Seasonal PCM sensitivity (`11_seasonal_pcm_sensitivity.py`, post-Phase-6)
Holds the Level-A clusters and each cluster's annual blended MCDM weight vector fixed, recomputes
only `L_required` per season (300 L/day draw, shared `SHARE_PCM`), re-ranks each cluster's Phase-5
survivor pool with TOPSIS, and counts #1-PCM flips vs the annual Phase-6 pick. **Completed-run
result (2026-09-08):** **7 of 9** resolved (cluster, season) cells flip — every cluster's Summer,
Monsoon and Retreat #1 differs from its annual pick in at least one case (Cluster 0 flips in all
three; Cluster 2 only in Retreat). **Winter is unresolved for all 3 clusters**: Winter `L_required`
is the year's highest (403–431 kJ/kg, lowest ambient → largest ΔT), which drops every cluster below
2 seasonal survivors — a real, reportable ceiling effect, not a bug. New plot:
`outputs/qc_seasonal_pcm_flip_heatmap_rajasthan.html` ((cluster × season) grid coloured by #1-PCM
identity, flipped cells red-outlined).

### External validation
**Köppen-Geiger is now wired in for real (updated 2026-08-11)** — Beck et al. (2018),
doi:10.1038/sdata.2018.214, 1-km raster, genuine per-point classification lookup (not a stub).
Rajasthan's 320 points classify as BSh=203, BWh=85, Aw=20, Cwa=12. Result: ARI(GMM cluster, Köppen
class)=0.19, NMI=0.32 — low-to-moderate agreement, read as "the GMM finds climate structure at a
finer resolution than Köppen's broad classes capture within Rajasthan" (a plausible, legitimate
finding in its own right, arguably the point of empirical clustering instead of applying Köppen
directly) rather than evidence the clustering failed to find anything real. NBC/ECBC climate-zone
validation remains stubbed (`nbc_ari = nbc_nmi = None`) — no local India-specific zone lookup exists
in this project tree, not fabricated. State-identity external validation is explicitly noted as "not
meaningful yet" for a single-state run.

## A documented, fixed methodology bug: GMM covariance type

Root-caused and fixed on **2026-08-10**: `full` covariance was changed to `diag`. Cause: at Level
A's dimensionality (35 standardized columns) and k=3 on 320 points (~106 points/cluster), `full`
covariance requires `d·(d+1)/2 = 630` parameters per cluster from ~106 samples — badly
underdetermined. Symptom: `max_membership_prob` was saturating to ~1.0 for essentially 100% of
points (zero genuinely ambiguous/soft cases) despite only a moderate silhouette (~0.31) — a
mismatch between a distance-based measure (silhouette, unaffected) and a probability-based measure
(GMM posterior, badly affected) that revealed the covariance estimate was numerically extreme rather
than reflecting real geometric separation. Fix verified empirically: `diag` restores a realistic
membership spread (min ~0.58, ~1.6% of points genuinely <0.90) while silhouette barely moves (0.3028
vs 0.3090) — confirming the fix changes *how confidently* the model reports its answer, not *what*
the answer is. Two alternative fixes (bumping `reg_covar`, PCA-reducing the feature set first) were
also verified to work but rejected as either a less-principled band-aid or a loss of the
per-named-index interpretability the framework doc requires for Level A.

## A second documented, fixed bug: GMM cluster-index instability across re-runs (2026-08-11)

**Distinct from the covariance-type fix above** — found while building Phase 7, not during this
phase's own original construction. sklearn's `GaussianMixture` gives no guarantee that cluster index
0 refers to the same physical climate group across separate re-runs of this script, even with the
same `random_state=42`, if anything about the fit changes between runs (the `full`→`diag` covariance
fix itself is one such change). Symptom: Phase 5's and Phase 6's outputs (both downstream of this
script) disagreed cluster-by-cluster on which PCMs belonged to which `cluster_id` — Phase 5's
"cluster 0" candidate set matched Phase 6's "cluster 2" set verbatim, and vice versa, because the two
phases had been run against different invocations of this script.

**Fix**: immediately after the final Level-A GMM fit, hard labels are canonically relabeled 0..k-1 by
sorting each raw cluster's MEAN LATITUDE ascending (south to north) — a simple, always-available,
fit-independent ordering key computed directly from the points themselves, not from anything the GMM
produces. "Cluster 0" now means the same physical (southernmost) climate regime regardless of which
run produced the underlying fit, as long as the underlying point PARTITION is equivalent. This does
**not** protect against Phase 5/6/7/8 being run against a genuinely DIFFERENT partition from a
different re-run (different data or parameters) — that risk is separately covered by a hard-fail
provenance-fingerprint check (`provenance_lib.py`) now run at every Phase 5→6→7→8 handoff, which
raises `SystemExit` (not a warning) if a downstream phase's input doesn't match the current on-disk
`cluster_profiles_rajasthan.csv`. See `09_PHASE_7_AUDIT.md` ("Completion Report" and "Code Quality &
Documented Design Decisions") for the full incident writeup, and `provenance_lib.py`'s own module
docstring for the provenance mechanism.

## ✅ VALIDATED (2026-08-31 re-run complete)

**L_required Methodology Correction (OPTION A) validated.** Phase 3's methodology was corrected to use SHARE_PCM=0.5 (literature-anchored fractional-share), halving all L_required values. Phase 4 clustering **remained stable** under this change, confirming the fix is robust.

## Actual Rajasthan result — Level A

**k=3, 2026-09-08 re-run (unified pipeline).** bootstrap-ARI at k=3 = **0.8200** (50/50 resamples).
Köppen ARI(GMM cluster, Köppen class) = **0.2787**, NMI = **0.3817** (meaningful). k=3 was selected
because k ∈ {2,3,4} all satisfy the silhouette-band + expected single-state range gate, and among
those k=3 has the highest bootstrap-ARI — the tier-1 selection rule working as designed, not a
default or a hand-pick. (An earlier 2026-08-31 run reported k=3 with bootstrap-ARI 0.8272 and
cluster sizes 114/103/103; the unified re-run's cluster sizes shift slightly because the Phase 3
signature schema changed — see `05_PHASE_3_AUDIT.md` — but k, the selection reason and the
qualitative regimes are unchanged.)

**Notable limitation, self-flagged by the code and confirmed empirically**: Clusters 0 and 2 receive
the *identical* auto-generated qualitative description string despite being numerically distinct
(e.g., HDD18 1100 vs 2237, monsoon_index 0.93 vs 1.03) — the 4-axis threshold-based description
generator is too coarse to distinguish them. The code's own docstring already calls this "a
first-pass label to hand-edit, not a final publication-ready caption" — treat it exactly that way in
any write-up; do not quote the auto-generated Cluster 0/2 descriptions as if they were independently
differentiated.

## Literature support

Silhouette expectation band [0.15, 0.35] is cited (not invented) from a *Building and Environment*
(2024) India climate-classification study reporting silhouette 0.21 vs −0.2 for the existing NBC
classification (peaking ~0.3 at k=6 in a 4-state design), and a 2026 thermal-comfort clustering
study independently reporting mean silhouette 0.235 — both citations appear in the code comments
with enough specificity to be traceable, though full BibTeX entries for both were not located in
`references.bib`/`references.md` during this audit and should be added before formal citation.
Beck et al. (2018) is the correctly-named, DOI-verified citation for the Köppen-Geiger external
validation, wired in for real (1-km raster, per-point lookup). **2026-09-08 unified re-run:**
ARI(GMM cluster, Köppen class) = **0.2787**, NMI = **0.3817** vs. the GMM clusters (the earlier
0.19 / 0.32 figures were from the pre-unification signature).

## Validation

Bootstrap-ARI stability (internal), silhouette/BIC/Davies-Bouldin/Calinski-Harabasz (internal),
season-tautology ANOVA check (Level B internal, in `05a_level_b_regime_shift_rajasthan.py`).
External classification validation: Köppen-Geiger real (2026-09-08: ARI=0.2787, NMI=0.3817);
NBC/ECBC still stubbed.

## Outputs

**Level A (`05_cluster_rajasthan.py`)**: `cluster_assignments_rajasthan_levelA.csv`,
`bic_selection_rajasthan.csv`, `cluster_profiles_rajasthan.csv`,
`outputs/cluster_profile_cards_rajasthan.md`, `outputs/qc_cluster_map_rajasthan.html` (now
population-scaled marker size), `koppen_validation_rajasthan.csv`,
`outputs/qc_k_selection_curve_rajasthan.html` (BIC + silhouette + **bootstrap-ARI**, chosen k
marked), `outputs/qc_cluster_profile_bars_rajasthan.html`,
`outputs/qc_cluster_population_share_rajasthan.html`.

**Level B — regime shift (`05a_level_b_regime_shift_rajasthan.py`)**:
`bic_selection_rajasthan_levelB.csv` (**now persisted**, previously console-only),
`cluster_assignments_rajasthan_levelB.csv`, `level_b_feature_importance_rajasthan.csv`,
`level_b_season_tautology_rajasthan.csv`, `level_b_season_contingency_rajasthan.csv`,
`outputs/qc_level_b_regime_shift_sankey_rajasthan.html` (new).

**Seasonal PCM sensitivity (`11_seasonal_pcm_sensitivity.py`)**:
`seasonal_pcm_sensitivity_rajasthan.csv`, `outputs/seasonal_pcm_sensitivity_rajasthan.md`,
`outputs/qc_seasonal_pcm_flip_heatmap_rajasthan.html` (new).

## Dependencies

Requires Phase 3's `climate_signature_rajasthan.csv`. Feeds Phase 5 directly — every feasibility
constraint is evaluated per cluster using `Tm_target_C`, `Tm_target_capped_C`,
`L_required_kJ_per_kg`, and `HSI_sunrise` from `cluster_profiles_rajasthan.csv`.

## Problems / risks

- ~~Level B's k-scan metric table is not persisted to disk~~ — **RESOLVED**: `05a_level_b_regime_
  shift_rajasthan.py` writes `bic_selection_rajasthan_levelB.csv` (same schema as the Level A
  table).
- ~~`bootstrap_ari_stability()` silently drops any bootstrap resample whose GMM fit raises~~ —
  **RESOLVED** in `cluster_lib.bootstrap_ari_stability()`: failed resamples are logged, and
  `bootstrap_effective_n_resamples` / `bootstrap_n_failed_resamples` are columns in the k-scan
  table, so a degraded resample count is always visible.
- `weighted_mean()` (used throughout cluster-profile generation) silently falls back to an
  unweighted mean if population weights are `None` or sum to zero — no warning printed; low
  practical risk given Rajasthan's actual weight distribution, but worth knowing if a future state's
  data is sparser.
- External validation is now partially wired in (Köppen) but NBC/ECBC remains stubbed — the
  clustering's "these are real climate regimes, not clustering artifacts" claim now rests on internal
  statistical measures PLUS one external classification (low-to-moderate agreement, itself a
  legitimate finding), not internal statistics alone.
- **GMM cluster-index labels are not stable across separate re-runs of this script** (see the second
  documented bug above) — anyone re-running this script and comparing against a previously-saved
  Phase 5/6/7/8 output MUST re-run the full downstream chain, not assume `cluster_id=0` still means
  the same climate regime. The canonical-relabeling fix mitigates but does not eliminate this risk
  for a genuinely different partition; the provenance hard-fail check is the actual safety net.

## Status

**COMPLETE (unified pipeline, re-run 2026-09-08).** Level A k=3 (bootstrap-ARI 0.8200, Köppen ARI
0.2787 / NMI 0.3817); Level B regime-shift k=8 (100% shift, season-tautology ARI 0.691 — recovers
"season"); seasonal PCM sensitivity 7/9 resolved cells flip (Winter unresolved — L_required
ceiling). Downstream Phase 5→6 re-run on the new `cluster_profiles_rajasthan.csv` (fingerprint
stamped through). Phase 7/8 not re-run in this pass — they hard-fail the provenance check until
re-run against the current clustering. The two previously-documented bugs (GMM covariance type; GMM
cluster-index instability across re-runs) remain fixed; the canonical-relabeling fix now lives in
`cluster_lib.canonical_relabel_by_latitude()` and is called by every state/level.

## Visualization / interactive scripts — audit stubs

None are in `run_all_rajasthan.py`'s core chain; all are read-only.

- **`05b_cluster_interactive.py`** (renamed 2026-09-08 from `05e_cluster_interactive.py` to match
  Tamil Nadu's numbering; content unchanged) — interactive (Folium + Plotly) explorer for
  `05_cluster_rajasthan.py`'s Level-A output: a cluster map with hoverable soft GMM membership
  probabilities (which the static PNG discards), an interactive cluster-profile comparison, and the
  K-selection curves. Reads `cluster_assignments_rajasthan_levelA.csv`,
  `cluster_profiles_rajasthan.csv`, `bic_selection_rajasthan.csv`; writes
  `PLOTSV2/clustering_interactive/*.html`.
- **`05c_explore_interactive.py`** (renamed from `05f_explore_interactive.py`; content unchanged) —
  a Streamlit app (run with `streamlit run 05c_explore_interactive.py`, not plain `python`). One row
  per `(point_id, date, event)`; every time-series chart plots three traces (sunrise/noon/sunset)
  against date. Deliberately excluded from `run_all_rajasthan.py`'s optional list for that reason.
- **`05d_plots_comprehensive.py`** (renamed from `05g_plots_comprehensive.py`; content unchanged) —
  a comprehensive static-visualization batch. Reads the processed backbone
  (`rajasthan_cleaned_physical.csv` from `04_preprocess_rajasthan.py`) by default — flip `USE_PROCESSED`
  to `False` for raw-data plots — with a `usecols` filter (the file is large). Map centre taken from
  the data; the second monsoon band relabelled "retreat / post-monsoon" for Rajasthan.
