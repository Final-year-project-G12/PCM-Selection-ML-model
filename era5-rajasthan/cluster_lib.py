"""
cluster_lib.py
=============================================================================
Shared Phase-4 clustering machinery, factored out so it has exactly ONE
implementation across every state and both clustering levels.

Callers (4 scripts, 2 states):
  era5-rajasthan/05_cluster_rajasthan.py               Level A (spatial)
  era5-rajasthan/05a_level_b_regime_shift_rajasthan.py Level B (seasonal)
  era5-tamilnadu/05_cluster_tamilnadu.py               Level A (spatial)
  era5-tamilnadu/05a_level_b_regime_shift_tamilnadu.py Level B (seasonal)

Same rationale as signature_lib.py: if the bootstrap-stability definition or
the k-selection cascade changes, it must change in one place for all four,
not be re-derived per state (which is exactly how Tamil Nadu ended up with a
hardcoded K_FINAL=5 while Rajasthan used a documented 3-tier rule).

Not state-specific: nothing here reads a state name, a config path, or a
fixed column list — it takes a numeric matrix and a list of k values.

HOW TO RUN: not runnable directly — import from here.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score,
)

# ── Defaults shared by every caller ──────────────────────────────────────
# GMM covariance type — 'diag', not 'full'. Root-caused empirically on
# 2026-08-10 (Rajasthan) and independently in Tamil Nadu's own v3.1 fix:
# 'full' covariance has d*(d+1)/2 parameters per cluster, which at Level A's
# dimensionality (~35 standardized columns) on a few hundred points is badly
# underdetermined. Symptom: max_membership_prob saturating at ~1.0 for
# essentially every point (zero genuinely ambiguous cases) despite only a
# moderate silhouette — a divergence between a distance-based measure
# (silhouette, unaffected) and a probability-based one (GMM posterior, badly
# affected) that reveals a numerically extreme covariance estimate rather
# than real geometric separation. 'diag' (d params/cluster) restores a
# realistic membership spread while silhouette barely moves — i.e. it changes
# how CONFIDENTLY the model reports its answer, not WHAT the answer is.
GMM_COVARIANCE_TYPE = "diag"
N_BOOTSTRAP = 50
RANDOM_STATE = 42

# Realistic silhouette expectation for genuine climate-zone clustering —
# NOT the "typical ML clustering demo" range. Cited (not invented) from a
# Building & Environment (2024) India climate-classification study reporting
# silhouette 0.21 vs -0.2 for the existing NBC classification and peaking
# ~0.3 at k=6 in a 4-state design, plus a 2026 thermal-comfort clustering
# study independently reporting mean silhouette 0.235. A silhouette well
# ABOVE this band on real climate data usually means the signature collapsed
# onto 1-2 dominant variables, not that the regimes are unusually crisp.
SILHOUETTE_LO, SILHOUETTE_HI = 0.15, 0.35


def bootstrap_ari_stability(X, k, n_boot=N_BOOTSTRAP, random_state=RANDOM_STATE,
                             covariance_type=GMM_COVARIANCE_TYPE):
    """Bootstrap clustering stability at a given k.

    Fit GMM(k) on the FULL data once (``base_labels``), then ``n_boot`` times
    fit a fresh GMM(k) on a with-replacement resample of the SAME SIZE and
    predict labels for the full ORIGINAL X — comparing each resampled fit's
    predictions on the original data against ``base_labels`` via Adjusted
    Rand Index. Mean ARI near 1.0 means the clustering at this k is stable to
    resampling; a low/unstable mean ARI is itself informative (report it,
    don't just pick the k with the best point-estimate silhouette).

    A resample whose GMM fit raises is dropped from the mean (one bad
    resample shouldn't kill the whole check) but is LOGGED, never silently
    swallowed — the return carries ``effective_n_resamples`` and the failure
    list, so a degraded resample count can never masquerade as a full
    n_boot-resample result.

    Returns ``(mean_ari, base_labels, effective_n_resamples, failed_resamples)``.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    base_gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                random_state=random_state, n_init=5)
    base_labels = base_gmm.fit_predict(X)

    aris = []
    failed_resamples = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                                        random_state=random_state + b + 1, n_init=1)
            boot_gmm.fit(X[idx])
            boot_labels_on_full = boot_gmm.predict(X)
            aris.append(adjusted_rand_score(base_labels, boot_labels_on_full))
        except Exception as e:
            failed_resamples.append({
                "resample_index": b, "exception_type": type(e).__name__,
                "exception_message": str(e),
            })
            continue

    effective_n_resamples = n_boot - len(failed_resamples)
    if effective_n_resamples < n_boot:
        print(f"    WARNING: bootstrap_ari_stability(k={k}) — {len(failed_resamples)}/{n_boot} "
              f"resamples raised and were dropped (effective_n_resamples={effective_n_resamples}). "
              f"Failures: {failed_resamples}")

    mean_ari = float(np.mean(aris)) if aris else float("nan")
    return mean_ari, base_labels, effective_n_resamples, failed_resamples


def suggest_k(k_table, expected_range=None,
              sil_lo=SILHOUETTE_LO, sil_hi=SILHOUETTE_HI):
    """Documented, non-forced 3-tier k-selection cascade — returned as a
    SUGGESTION with its reason, never silently applied by this function.

      1. k within ``expected_range`` (if given) AND silhouette inside the
         realistic band [sil_lo, sil_hi] -> highest ``bootstrap_ari_mean``
         among those.
      2. Any k with silhouette inside the band -> highest
         ``bootstrap_ari_mean`` among those.
      3. Fallback: lowest-BIC k overall, flagged in the returned reason so
         the caller can print a warning that no k landed in the band.

    Bootstrap-ARI is the actual tiebreaker in tiers 1 and 2 — which is why
    the k-selection QC curve overlays it alongside BIC and silhouette.

    ``k_table`` must carry columns: k, BIC, silhouette, bootstrap_ari_mean.
    Returns ``(k, reason)``.
    """
    band = k_table[(k_table["silhouette"] >= sil_lo) &
                    (k_table["silhouette"] <= sil_hi)]
    if expected_range is not None:
        in_range = band[(band["k"] >= expected_range[0]) & (band["k"] <= expected_range[1])]
        if len(in_range):
            row = in_range.loc[in_range["bootstrap_ari_mean"].idxmax()]
            return int(row["k"]), "in silhouette band AND expected single-state k range"
    if len(band):
        row = band.loc[band["bootstrap_ari_mean"].idxmax()]
        return int(row["k"]), "in silhouette band (outside expected single-state k range)"
    row = k_table.loc[k_table["BIC"].idxmin()]
    return int(row["k"]), (f"FALLBACK: no k landed in the {sil_lo}-{sil_hi} silhouette band "
                            f"— lowest BIC used")


def fit_k_range(X, k_values, n_boot=N_BOOTSTRAP, random_state=RANDOM_STATE,
                covariance_type=GMM_COVARIANCE_TYPE,
                sil_lo=SILHOUETTE_LO, sil_hi=SILHOUETTE_HI, verbose=True):
    """GMM (primary) + KMeans (reported comparison baseline) over ``k_values``.

    Returns the metric table (one row per k): BIC, AIC, silhouette,
    Davies-Bouldin, Calinski-Harabasz, bootstrap-ARI mean + effective/failed
    resample counts, KMeans silhouette, and an in-band flag.

    The GMM is NEVER population-weighted here — the point sampling is already
    population-weighted by construction (00a_build_population_grid.py), so
    weighting the fit again would double-count population. Population enters
    only later, in the cluster-profile weighted means.
    """
    rows = []
    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                               random_state=random_state, n_init=5)
        gmm_labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        n_unique = len(set(gmm_labels))
        sil = silhouette_score(X, gmm_labels) if n_unique > 1 else float("nan")
        db = davies_bouldin_score(X, gmm_labels) if n_unique > 1 else float("nan")
        ch = calinski_harabasz_score(X, gmm_labels) if n_unique > 1 else float("nan")

        boot_ari, _, eff_n, failed = bootstrap_ari_stability(
            X, k, n_boot=n_boot, random_state=random_state, covariance_type=covariance_type)

        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels) if len(set(km_labels)) > 1 else float("nan")

        in_band = sil_lo <= sil <= sil_hi if sil == sil else False
        rows.append({
            "k": k, "BIC": bic, "AIC": aic, "silhouette": sil,
            "davies_bouldin": db, "calinski_harabasz": ch,
            "bootstrap_ari_mean": boot_ari,
            "bootstrap_effective_n_resamples": eff_n,
            "bootstrap_n_failed_resamples": len(failed),
            "kmeans_silhouette": km_sil,
            "in_silhouette_band": in_band,
        })
        if verbose:
            flag = f"  <- in {sil_lo}-{sil_hi} band" if in_band else ""
            eff_flag = "" if eff_n == n_boot else f"  [only {eff_n}/{n_boot} resamples succeeded]"
            print(f"    K={k:2d}  BIC={bic:10.1f}  AIC={aic:10.1f}  GMM_sil={sil:.4f}  "
                  f"DB={db:.3f}  CH={ch:8.1f}  bootARI={boot_ari:.3f}  "
                  f"KMeans_sil={km_sil:.4f}{flag}{eff_flag}")

    return pd.DataFrame(rows)


def canonical_relabel_by_latitude(raw_labels, lats, soft_probs=None):
    """Relabel GMM cluster IDs 0..k-1 by ascending MEAN LATITUDE (south to
    north), so "cluster 0" means the same physical regime across separate
    re-runs of the clustering script.

    sklearn's GaussianMixture assigns cluster index 0..k-1 in an arbitrary,
    fit-order-dependent way with NO stability guarantee across re-runs — even
    with a fixed random_state, if anything about the fit changes between runs
    (a covariance_type change, different input data) the raw index-to-regime
    mapping can shift. Every downstream phase keys off cluster_id, so an
    unstable label is a silent correctness bug: on 2026-08-11 Rajasthan's
    Phase 5 and Phase 6 outputs were found disagreeing on which PCMs belonged
    to which cluster because the two phases had been run against different
    invocations of the clustering script.

    Mean latitude is a simple, always-available, fit-INDEPENDENT ordering key
    computed from the points themselves, not from anything the GMM produces
    (which is the unstable part). This protects against the arbitrary INDEX
    ordering of an EQUIVALENT partition; it does NOT protect against a
    genuinely different partition from a re-run on different data/parameters
    — that is what provenance_lib.py's hard-fail fingerprint check at each
    Phase 5->6->7->8 handoff is for.

    Returns ``(hard_labels, soft_probs_reordered, relabel_map)``; if
    ``soft_probs`` is None the second element is None.
    """
    raw_mean_lat = pd.Series(np.asarray(lats)).groupby(np.asarray(raw_labels)).mean()
    canonical_order = raw_mean_lat.sort_values().index.tolist()   # raw label, ascending mean lat
    relabel_map = {raw: canonical for canonical, raw in enumerate(canonical_order)}
    hard_labels = np.array([relabel_map[r] for r in raw_labels])
    reordered = soft_probs[:, canonical_order] if soft_probs is not None else None
    return hard_labels, reordered, relabel_map
