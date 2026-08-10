# 06 — Phase 4 Audit: Climate Regime Clustering

True scripts: `05_cluster_tamilnadu.py` (disk: `06_build_pcm_database (2).py` — the script actually
used), `05_cluster_regions.py` (disk: `05_cluster_tamilnadu (3).py` — multi-region, not runnable),
`05b_cluster_interactive.py` (disk: `05c_explore_interactive (2).py`).

## Status: code complete, never executed

## GMM setup — exact constants

```python
K_CANDIDATES = list(range(2, 11))          # 2..10
K_FINAL = 5                                 # hardcoded — "set after reviewing bic_selection..., re-run"
SILHOUETTE_ACCEPT_LO, SILHOUETTE_ACCEPT_HI = 0.15, 0.40
RANDOM_STATE = 42
```
`covariance_type="full"` is used **unconditionally, in both the K-scan loop and the final fit** —
confirmed by a full-file grep for `covariance_type`/`"diag"`: only two hits, both `"full"`. **There is
no diag-covariance issue or fix present anywhere in this file** — a genuinely different situation
from Rajasthan, where `full` covariance was found to cause membership-probability saturation at
Rajasthan's dimensionality (35 columns, k=3, ~106 points/cluster) and was fixed to `diag`. Tamil
Nadu's signature matrix and expected cluster sizes have not been run, so whether the same
underdetermination risk applies here (fewer points — ~133 vs 320 — spread across `K_FINAL=5` clusters
= ~27 points/cluster on average, a *smaller* per-cluster sample than Rajasthan had *before* its fix)
is an open, unverified question. **Given Rajasthan's own history, this is worth checking explicitly
the first time TN's clustering is actually run** — the smaller per-cluster point count here makes the
same covariance-underdetermination risk plausible, not less likely.

**`K_FINAL=5` is genuinely hardcoded**, not derived — confirmed: the only place it's used is a safety
clamp (`k_final_safe = min(K_FINAL, len(X)-1)`), not a selection procedure. The script computes BIC +
silhouette + Davies-Bouldin + Calinski-Harabasz for every k in `K_CANDIDATES`, writes them to
`bic_selection_tamilnadu.csv`, and expects a human to review that table and manually set `K_FINAL`
before the "final" run — this is an explicit, documented manual step (not a silent default), but it
does mean **no k=5 result exists yet to review**, since the pipeline has never run even once.

## Metrics — wider set than the multi-region script

The single-state script computes **BIC, silhouette, Davies-Bouldin, Calinski-Harabasz** (4 metrics);
the multi-region script (`05_cluster_regions.py`) computes only BIC + silhouette. This is a genuine,
minor asymmetry — not a bug, just a scope difference between the two scripts.

## Population weighting — confirmed absent from the fit, present in profiling (same as Rajasthan)

Grepped for `sample_weight`: zero hits in either clustering script. Population weighting is applied
only in the post-hoc `weighted_mean()` profile-aggregation step
(`np.average(g[col], weights=g["population"])`), identical design rationale to Rajasthan (the point
*sampling* is already population-weighted; weighting the fit again would double-count).

## Silhouette acceptance bands — TN differs from Rajasthan/multi-region, both bands independently justified in-code

**TN single-state: 0.15–0.40.** **Multi-region: 0.15–0.35** (matches Rajasthan's own band exactly).
The TN-only script's comment explains the wider band: *"single-state band is a bit wider than the
4-state 0.15-0.35 (no artificial between-state gaps inflating it here)."* Both bands are grounded in
the same cited source as Rajasthan uses (the *Building and Environment* 2024 India climate
classification study, silhouette 0.21 vs. −0.2 NBC, peaking 0.3 at k=6) — `FIXES.md` in this same
folder independently verified this citation and confirms the number is accurate, not fabricated.

## Multi-region clustering — confirmed genuinely not runnable, not just labeled as such

```python
REGION_FILES = {
    "TamilNadu": SIGNATURE_DIR / "climate_signature_tamilnadu.csv",
    "Rajasthan": SIGNATURE_DIR.parent.parent / "era5-rajasthan" / "data" / "processed" / "signatures" / "climate_signature_rajasthan.csv",
    # "Region3": ..., "Region4": ...
}
if len(frames) < 2:
    print("Fewer than 2 regions available yet ... Stopping here.")
    return
```
The Rajasthan path this script points at (`era5-rajasthan/data/processed/signatures/...`) does not
match Rajasthan's actual confirmed output path (`era5-rajasthan/data/processed/climate_signature_rajasthan.csv`,
no `signatures/` subfolder — confirmed in `docs/era5_rajasthan/06_PHASE_4_AUDIT.md`'s outputs list) —
so even once Tamil Nadu's own signature file exists, this specific hardcoded path would need
correcting before a genuine cross-region run could succeed, an additional small fix beyond simply
"needs ≥2 states' files to exist." One genuine algorithmic difference from the single-state script:
it **re-standardizes across the combined matrix** after concatenation (`StandardScaler` applied a
second time to the pooled data), with the explicit rationale that each region's own `_z` scores were
computed within that region's own ~130-point sample alone and shouldn't dominate cross-region
distances — a correctly-reasoned design choice for the eventual multi-state run.

## Interactive explorer

Pure read-only visualization consumer of the clustering script's outputs (Folium cluster map with
soft-membership popups and boundary-point styling based on `max_membership_prob < 1.5/n_clusters`,
a grouped bar chart of 9 signature indices per cluster, a population-share pie chart explicitly
labeled *"novelty claim N6"* in its title string, and — if `bic_selection_tamilnadu.csv` exists — a
dual-axis BIC/silhouette k-selection curve). No new computation; nothing to critique methodologically
beyond it being unable to run without upstream outputs that don't yet exist.

## Literature support

Same *Building and Environment* (2024) citation as Rajasthan, independently spot-verified accurate
by this project's own `FIXES.md` (not just asserted). GMM-vs-K-Means rationale (climate as a
continuous gradient, soft membership appropriate for boundary points) is stated identically to
Rajasthan's framing and is consistent with the framework doc's own §7.3.

## Validation

None possible yet — no `bic_selection_tamilnadu.csv` exists to review, so `K_FINAL=5`'s appropriateness
cannot currently be assessed against real TN data.

## Outputs (expected)

`bic_selection_tamilnadu.csv`, `kmeans_comparison_tamilnadu.csv`, `cluster_assignments_tamilnadu.csv`,
`cluster_profiles_tamilnadu.csv`, `cluster_map_tamilnadu.png`.

## Dependencies

Requires Phase 3's `climate_signature_tamilnadu.csv` (itself blocked on the `L_required` fix being
applied first, or at minimum being aware `L_required` values will need re-deriving after the fix).

## Problems / risks

- **`K_FINAL=5` is a placeholder pending a first real run** — not wrong, but not yet validated against
  actual TN clustering metrics either.
- **The `covariance_type="full"` choice has not been stress-tested against Tamil Nadu's actual
  dimensionality/sample-size ratio** — Rajasthan's own history is a direct precedent for why this is
  worth checking rather than assuming safe by default.
- **Multi-region path bug**: the Rajasthan signature-file path hardcoded in `05_cluster_regions.py`
  does not match Rajasthan's actual output location — a small, easily-fixed but currently-real defect.
- No external classification validation exists in this pipeline at all (not even a stub structure,
  unlike Rajasthan which has an explicit `None`-valued TODO section for Köppen-Geiger/NBC-ECBC).

## Status

**CODE COMPLETE, NEVER RUN.** Methodologically comparable to Rajasthan's clustering design (same GMM
rationale, same literature-grounded silhouette band, same population-weighting-in-profiling-only
approach) but carries more first-run uncertainty than Rajasthan given the untested covariance-type
choice and the hardcoded, unvalidated `K_FINAL`.
