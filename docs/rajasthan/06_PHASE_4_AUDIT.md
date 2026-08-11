# 06 — Phase 4 Audit: Climate Regime Clustering

Script: `05_cluster_rajasthan.py`.

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

### Level B
Same GMM/K-Means machinery, `k=2..8`, on the freshly-built per-point-per-season Tier-1 matrix.
Additional checks specific to Level B: a **regime-shift analysis** (fraction of points whose
cluster assignment differs across the 4 seasons) and a **season-tautology check** (contingency
table + Adjusted Rand Index/Normalized Mutual Information between cluster labels and season labels,
plus an ANOVA F-statistic feature-ranking to check whether temperature/GHI features dominate the
clustering — which would suggest the clustering is just rediscovering the season labels rather than
finding independent structure). An `LEVEL_B_EXCLUDE_FEATURES` ablation switch exists (default empty,
inactive) to drop deterministic-by-construction features like `daylength_mean`/`daylength_amplitude`
from the fit if needed.

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
`cluster_profiles_rajasthan.csv`. See `19_PHASE_7_ONWARD.md` for the full incident writeup and
`21_REPRODUCIBILITY.md` for the provenance mechanism.

## Actual Rajasthan result (k=3, ground-truthed from the real output files)

| Cluster | Points | Population | Medoid | Description | Tm_target_C | L_required (kJ/kg) |
|---|---|---|---|---|---|---|
| 0 | 114 | 22,568,150 | RJP_0132 (24.375, 74.125) | Cooler, arid/low-monsoon, erratic solar, short low-clearness runs | 57.0 | 627 |
| 1 | 103 | 17,959,813 | RJP_0202 (26.875, 73.625) | Hot, monsoon-influenced, steady solar, long low-clearness runs (high autonomy demand) | 57.0 | 609 |
| 2 | 103 | 29,775,240 | RJP_0055 (26.625, 76.375) | Cooler, arid/low-monsoon, erratic solar, short low-clearness runs | 57.0 | 641 |

k=3 was selected because k∈{2,3,4} all satisfy the silhouette-band + expected-range gate, and among
those k=3 has the highest bootstrap-ARI (0.8137, vs 0.6965 at k=2 and 0.5904 at k=4) — the tier-1
selection rule, working as designed, not a default or a hand-pick.

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
Beck et al. (2018) is the correctly-named, DOI-verified citation for the (not-yet-wired) Köppen
validation.

## Validation

Bootstrap-ARI stability (internal), silhouette/BIC/Davies-Bouldin/Calinski-Harabasz (internal),
season-tautology ANOVA check (Level B internal). External classification validation: Köppen-Geiger
now real (ARI=0.19, NMI=0.32 — see above); NBC/ECBC still stubbed.

## Outputs

`cluster_assignments_rajasthan_levelA.csv`, `cluster_assignments_rajasthan_levelB.csv`,
`bic_selection_rajasthan.csv` (Level A only — Level B's k-scan is console-printed, never persisted),
`cluster_profiles_rajasthan.csv`, `cluster_profile_cards_rajasthan.md`,
`outputs/qc_cluster_map_rajasthan.html`, `koppen_validation_rajasthan.csv` (cluster_id x Köppen-class
contingency counts), `level_b_feature_importance_rajasthan.csv`,
`level_b_season_tautology_rajasthan.csv`, `level_b_season_contingency_rajasthan.csv`. Plus, added
2026-08-11: `outputs/qc_k_selection_curve_rajasthan.html` (BIC + silhouette vs. k, chosen k marked),
`outputs/qc_cluster_profile_bars_rajasthan.html` (headline signature indices by cluster),
`outputs/qc_cluster_population_share_rajasthan.html` (population-share pie chart) — pure
visualization of data this script already computes.

## Dependencies

Requires Phase 3's `climate_signature_rajasthan.csv`. Feeds Phase 5 directly — every feasibility
constraint is evaluated per cluster using `Tm_target_C`, `Tm_target_capped_C`,
`L_required_kJ_per_kg`, and `HSI_sunrise` from `cluster_profiles_rajasthan.csv`.

## Problems / risks

- Level B's k-scan metric table is not persisted to disk — reproducibility gap (see
  `21_REPRODUCIBILITY.md`).
- `bootstrap_ari_stability()` silently drops any bootstrap resample whose GMM fit raises an
  exception (`except Exception: continue`), which could quietly reduce the effective resample count
  below 50 without this being visible anywhere in the output table.
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

**COMPLETE — with TWO caught-and-fixed bugs (GMM covariance type; GMM cluster-index instability
across re-runs) and Köppen-Geiger external validation now wired in (NBC/ECBC still stubbed).** The
internal clustering result (k=3) is statistically well-supported and now partially externally
corroborated; the cluster-index-instability fix and its accompanying provenance hard-fail check are
what make the downstream Phase 5→6→7→8 chain trustworthy across separate re-runs — see
`19_PHASE_7_ONWARD.md` for the real incident this was caught from.
