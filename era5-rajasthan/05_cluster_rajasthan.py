"""
05_cluster_rajasthan.py
=============================================================================
PHASE 4 — CLIMATE REGIME CLUSTERING, RAJASTHAN ONLY
(Objective1_PCM_Climate_Framework_Plan_v3, §6.2/§7)

Implements BOTH clustering levels from the plan doc:
  LEVEL A — spatial: one signature vector per point (whole 10-year record),
            reading climate_signature_rajasthan.csv's *_z columns directly.
  LEVEL B — temporal: one signature vector per point PER SEASON, built with
            the same Tier-1 index formulas as Level A (via signature_lib.py,
            shared with 04_climate_signature_rajasthan.py — not duplicated),
            to detect whether a point's climate regime shifts materially
            between seasons.

STATE-AGNOSTIC BY DESIGN: STATE_NAME below is the only place a state name
is hardcoded. Every output filename is built from it. Reusing this script
for Assam/Tamil Nadu/Uttarakhand means copying this file (and
signature_lib.py) into that state's own pipeline folder — same pattern as
era5-rajasthan itself — changing STATE_NAME, and pointing config.py's
COMBINED_POINTS_FILE/SUNTIMES_FILE/CLIMATE_SIGNATURE-equivalent paths at
that state's own files (config.py itself stays per-folder/per-state, it is
NOT made multi-state aware here). A later multi-state Level-A run
combines each state's climate_signature_{state}.csv the same way
05_cluster_regions.py already does for Tamil Nadu.

INPUTS:
  data/processed/climate_signature_rajasthan.csv  (04_climate_signature_
      rajasthan.py's output — Level A reads its *_z columns directly)
  data/processed/climate_rajasthan_points.csv      (Level B rebuilds a
      per-point-per-season Tier 1 signature from this directly, via
      signature_lib.build_tier1_signature())
  data/processed/suntimes.csv                       (Level B daylength)

OUTPUTS:
  data/processed/bic_selection_rajasthan.csv                  (Level A,
      full k=2..12 metric table: BIC/AIC/silhouette/DB/CH/bootstrap ARI +
      bootstrap_effective_n_resamples/KMeans silhouette)
  data/processed/bic_selection_rajasthan_levelB.csv           (Level B,
      k=2..8, same schema — FIX (2026-08-11): previously console-print-only)
  data/processed/level_b_feature_importance_rajasthan.csv     (Level B
      season-tautology ANOVA F-stats per Tier-1 feature — FIX, ditto)
  data/processed/level_b_season_tautology_rajasthan.csv       (Level B
      regime-shift fraction + season ARI/NMI, one row — FIX, ditto)
  data/processed/level_b_season_contingency_rajasthan.csv     (Level B
      cluster x season contingency table — FIX, ditto)
  data/processed/cluster_assignments_rajasthan_levelA.csv     (point_id,
      hard cluster label, soft membership probabilities, chosen k,
      chosen-k bootstrap-ARI mean + effective_n_resamples)
  data/processed/cluster_assignments_rajasthan_levelB.csv     (point_id,
      season, seasonal cluster label + probabilities, chosen-k bootstrap
      stats)
  data/processed/cluster_profiles_rajasthan.csv                (per-cluster
      profile, now also carrying chosen-k bootstrap-ARI stats and
      koppen_ari/koppen_nmi/koppen_validation_meaningful)
  data/processed/koppen_validation_rajasthan.csv               (cluster_id
      x koppen_class contingency counts — FIX, see EXTERNAL VALIDATION)
  outputs/qc_cluster_map_rajasthan.html                        (folium —
      points colored by Level-A hard cluster, opacity by max membership
      probability, so ambiguous/transition points are visible)
  cluster_profile_cards_rajasthan.md                           (one card
      per Level-A cluster)

EXTERNAL VALIDATION: Koppen-Geiger (Beck et al. 2018, doi:10.1038/
sdata.2018.214) is WIRED IN (FIX 2026-08-11) — see data/raw/koppen/ and
the EXTERNAL VALIDATION section below. NBC/ECBC still has no local lookup
in this project and remains stubbed, not fabricated/approximated.

REQUIRED LIBRARIES (install if missing):
  pip install pandas numpy scikit-learn folium branca plotly rasterio

HOW TO RUN:
  python 05_cluster_rajasthan.py

PLOTS ADDED 2026-08-11 (cluster-level QC, see "CLUSTER-LEVEL QC PLOTS"
section near the end of this file — pure visualization of data this
script already computes, no new computation): outputs/qc_k_selection_
curve_rajasthan.html (BIC + silhouette vs k, Level A), outputs/qc_cluster_
profile_bars_rajasthan.html (headline signature indices by cluster), and
outputs/qc_cluster_population_share_rajasthan.html (population-share pie
chart by cluster) — in addition to the pre-existing outputs/qc_cluster_
map_rajasthan.html (folium).
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re

import numpy as np
import pandas as pd
import rasterio
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
)
from sklearn.feature_selection import f_classif
import folium
import branca.colormap as bcm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (
    COMBINED_POINTS_FILE, SUNTIMES_FILE, PROCESSED_DIR, OUTPUTS_DIR,
    KOPPEN_RASTER_FILE, KOPPEN_LEGEND_FILE, ensure_data_dirs,
)
from signature_lib import EVENT_ORDER, SEASON_ORDER, attach_season, build_tier1_signature

ensure_data_dirs()

# ═══════════════════════════════════════════════════════════
# STATE NAME — the only hardcoded state reference in this file.
# Every output path is built from this.
# ═══════════════════════════════════════════════════════════
STATE_NAME = "rajasthan"

SIGNATURE_FILE = PROCESSED_DIR / f"climate_signature_{STATE_NAME}.csv"

BIC_TABLE_FILE = PROCESSED_DIR / f"bic_selection_{STATE_NAME}.csv"
ASSIGN_A_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelA.csv"

# Level B feature-set ablation control — a reusable, documented robustness
# check, not a one-off hack. Default ([], "full") reproduces the primary
# Level B run on all 19 Tier 1 columns and writes to the canonical
# cluster_assignments_{state}_levelB.csv. Set LEVEL_B_EXCLUDE_FEATURES to
# drop specific columns for a comparison run (e.g. daylength_mean/
# daylength_amplitude, which carry zero climatic content by construction —
# they're a deterministic function of latitude/day-of-year, fully
# determined before any weather happens — see the 2026-08-10
# season-tautology investigation this ablation follows up on) and set
# LEVEL_B_RUN_TAG to something descriptive; the output filename picks up
# that tag automatically so an ablation run never overwrites the primary
# one. Re-run the SAME season-tautology checks (contingency table, ARI/
# NMI vs season, ANOVA feature importance) on both for a clean before/
# after comparison — that comparison is itself worth reporting as a
# robustness check, not just the ablation's raw numbers.
LEVEL_B_EXCLUDE_FEATURES = []
LEVEL_B_RUN_TAG = "full"

_levelB_suffix = "" if not LEVEL_B_EXCLUDE_FEATURES else f"_ablation_{LEVEL_B_RUN_TAG}"
ASSIGN_B_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelB{_levelB_suffix}.csv"
# Level B k-scan metric table + season-tautology diagnostics — previously
# console-printed only (reproducibility gap, see docs/rajasthan/06_PHASE_4_
# AUDIT.md "Problems/risks"). Same column schema as BIC_TABLE_FILE (Level
# A) so the two are directly diffable.
BIC_TABLE_B_FILE = PROCESSED_DIR / f"bic_selection_{STATE_NAME}_levelB{_levelB_suffix}.csv"
LEVEL_B_FEATURE_IMPORTANCE_FILE = PROCESSED_DIR / f"level_b_feature_importance_{STATE_NAME}{_levelB_suffix}.csv"
LEVEL_B_SEASON_TAUTOLOGY_FILE = PROCESSED_DIR / f"level_b_season_tautology_{STATE_NAME}{_levelB_suffix}.csv"
LEVEL_B_CONTINGENCY_FILE = PROCESSED_DIR / f"level_b_season_contingency_{STATE_NAME}{_levelB_suffix}.csv"
MAP_FILE = OUTPUTS_DIR / f"qc_cluster_map_{STATE_NAME}.html"
CARDS_FILE = OUTPUTS_DIR / f"cluster_profile_cards_{STATE_NAME}.md"
# Numeric companion to CARDS_FILE — same population-weighted per-cluster
# aggregates (including Tm_target_C / Tm_target_capped_C / L_required_
# kJ_per_kg / HSI_sunrise), but as a clean CSV rather than markdown prose,
# so a feasibility-filter script can read cluster-level PCM targets
# without parsing a .md table. Mirrors the Tamil Nadu pipeline's
# cluster_profiles_tamilnadu.csv naming/role.
PROFILE_FILE = PROCESSED_DIR / f"cluster_profiles_{STATE_NAME}.csv"
# Koppen-Geiger external-validation output (Fix 4): cluster_id x koppen_class
# contingency counts — the "koppen_class_distribution" companion file
# referenced in the EXTERNAL VALIDATION section below.
KOPPEN_CONTINGENCY_FILE = PROCESSED_DIR / f"koppen_validation_{STATE_NAME}.csv"

K_RANGE_A = list(range(2, 13))   # 2..12
K_RANGE_B = list(range(2, 9))    # 2..8
N_BOOTSTRAP = 50
RANDOM_STATE = 42

# GMM covariance type — 'diag', not 'full'. FIXED 2026-08-10, root-caused
# empirically after a user-flagged mismatch: max_membership_prob was
# saturating to ~1.0 for 100% of Level A points (zero soft/ambiguous
# cases) despite only a moderate silhouette (~0.31), which shouldn't
# happen if the soft assignment were reflecting genuine geometric
# separation. Root cause: 'full' covariance has d*(d+1)/2 parameters per
# cluster — at Level A's d=35 standardized columns and k=3 on 320 points
# (~106 points/cluster), that's 630 covariance parameters estimated from
# ~106 samples, badly underdetermined. Near-singular covariance estimates
# push GMM posterior probabilities to numerically extreme values almost
# independent of true geometric separation — silhouette (a raw-distance
# measure) isn't fooled by this, which is exactly why the two diverged.
# Level B has the same problem at a smaller scale (19 dims x k=8 on 1280
# point-season rows = ~160 rows/cluster vs 190 full-covariance params).
# Verified empirically (see the 2026-08-10 GMM covariance diagnostic):
# switching to 'diag' (d params/cluster instead of d*(d+1)/2) restores a
# realistic membership-probability spread (min ~0.58, ~1.6% of points
# genuinely ambiguous at <0.90) while silhouette barely moves (0.3028 vs
# 0.3090) and cluster sizes stay essentially the same — confirming this is
# a covariance-estimation artifact being fixed, not a change to what the
# clustering actually finds. 'diag' was chosen over bumping reg_covar
# (also verified to work, but a less principled band-aid over the same
# underdetermined fit) and over PCA-reducing the feature set first (also
# verified to work, and even improves silhouette slightly at very low
# dimensionality, but trades away exactly the per-named-index
# interpretability the plan doc insists on for Level A — 'diag' keeps
# clustering directly on the same named *_z columns, just without
# modeling cross-feature covariance the sample size can't support anyway).
GMM_COVARIANCE_TYPE = "diag"

# Realistic silhouette expectation for genuine climate-zone clustering —
# NOT the "typical ML clustering demo" range. Cited from the same source
# used elsewhere in this pipeline: Building & Environment (2024) India
# climate-classification study reports silhouette 0.21 vs -0.2 for the
# existing NBC classification, peaking at 0.3 for k=6 (4-state design); a
# 2026 thermal-comfort clustering study independently reports mean
# silhouette 0.235. A silhouette well above this band on REAL climate data
# usually means the signature collapsed to 1-2 dominant variables, not
# that the regimes are unusually crisp.
SILHOUETTE_LO, SILHOUETTE_HI = 0.15, 0.35

# This run is Rajasthan ALONE, not yet the eventual 4-state combined
# design (Rajasthan + Assam + Tamil Nadu + Uttarakhand) this framework is
# ultimately built for. Expect INTRA-state splitting here (the framework
# doc names arid-west vs semi-arid-east as the specific Rajasthan split),
# realistically k=2-4 — NOT the k=6-10 expected once all four states are
# combined. Do not mistake this run's chosen k for the eventual
# multi-state k when that run happens.
EXPECTED_K_RANGE_SINGLE_STATE = (2, 4)

# Optional manual overrides — leave None to use this script's auto-
# suggested k (see suggest_k() below); set an int to force a specific k
# after reviewing bic_selection_rajasthan.csv / the Level B printout.
LEVEL_A_K_OVERRIDE = None
LEVEL_B_K_OVERRIDE = None


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ═══════════════════════════════════════════════════════════
# SHARED CLUSTER-METRIC HELPERS  (used by both Level A and Level B)
# ═══════════════════════════════════════════════════════════

def bootstrap_ari_stability(X, k, n_boot=N_BOOTSTRAP, random_state=RANDOM_STATE):
    """Bootstrap clustering stability: fit GMM(k) on the FULL data once
    (base_labels), then n_boot times fit GMM(k) on a with-replacement
    resample of the SAME SIZE and predict labels for the full original X
    — compare each resampled-fit's predictions on the original data
    against base_labels via Adjusted Rand Index. Mean ARI close to 1.0
    means the clustering at this k is stable to resampling; low/unstable
    mean ARI at a given k is itself informative (report it, don't just
    pick the k with the best point-estimate silhouette).

    A resample whose GMM fit raises is dropped from the mean (one bad
    resample shouldn't kill the whole stability check) but is now LOGGED
    rather than silently swallowed — returns effective_n_resamples and the
    list of failures alongside the mean, so a degraded resample count is
    always visible rather than masquerading as a full n_boot-resample
    result."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    base_gmm = GaussianMixture(n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
                                random_state=random_state, n_init=5)
    base_labels = base_gmm.fit_predict(X)

    aris = []
    failed_resamples = []
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot_gmm = GaussianMixture(n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
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


def suggest_k(k_table, expected_range=None):
    """Documented, non-forced k-selection heuristic — printed as a
    suggestion, not silently applied. Preference order:
      1. k within expected_range (if given) AND silhouette in the
         [SILHOUETTE_LO, SILHOUETTE_HI] realistic band, highest
         bootstrap_ari_mean among those.
      2. Any k with silhouette in the realistic band, highest
         bootstrap_ari_mean among those.
      3. Fallback: lowest-BIC k overall (printed with a warning that no k
         landed in the realistic silhouette band)."""
    band = k_table[(k_table["silhouette"] >= SILHOUETTE_LO) &
                    (k_table["silhouette"] <= SILHOUETTE_HI)]
    if expected_range is not None:
        in_range = band[(band["k"] >= expected_range[0]) & (band["k"] <= expected_range[1])]
        if len(in_range):
            row = in_range.loc[in_range["bootstrap_ari_mean"].idxmax()]
            return int(row["k"]), "in silhouette band AND expected single-state k range"
    if len(band):
        row = band.loc[band["bootstrap_ari_mean"].idxmax()]
        return int(row["k"]), "in silhouette band (outside expected single-state k range)"
    row = k_table.loc[k_table["BIC"].idxmin()]
    return int(row["k"]), "FALLBACK: no k landed in the 0.15-0.35 silhouette band — lowest BIC used"


def fit_k_range(X, k_values, label):
    """GMM (primary) + KMeans (comparison baseline) over k_values. Returns
    the metric table. GMM is never population-weighted here — the point
    sampling is already population-weighted by construction (see
    00a_build_population_grid.py); weighting the fit again would double
    count population. Population is reporting-only, applied later when
    building cluster profiles, never inside the fit."""
    rows = []
    for k in k_values:
        gmm = GaussianMixture(n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
                               random_state=RANDOM_STATE, n_init=5)
        gmm_labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        n_unique = len(set(gmm_labels))
        sil = silhouette_score(X, gmm_labels) if n_unique > 1 else float("nan")
        db = davies_bouldin_score(X, gmm_labels) if n_unique > 1 else float("nan")
        ch = calinski_harabasz_score(X, gmm_labels) if n_unique > 1 else float("nan")

        boot_ari, _, eff_n_resamples, failed_resamples = bootstrap_ari_stability(X, k)

        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels) if len(set(km_labels)) > 1 else float("nan")

        in_band = SILHOUETTE_LO <= sil <= SILHOUETTE_HI if sil == sil else False
        rows.append({
            "k": k, "BIC": bic, "AIC": aic, "silhouette": sil,
            "davies_bouldin": db, "calinski_harabasz": ch,
            "bootstrap_ari_mean": boot_ari,
            "bootstrap_effective_n_resamples": eff_n_resamples,
            "bootstrap_n_failed_resamples": len(failed_resamples),
            "kmeans_silhouette": km_sil,
            "in_silhouette_band": in_band,
        })
        flag = "  <- in 0.15-0.35 band" if in_band else ""
        eff_flag = "" if eff_n_resamples == N_BOOTSTRAP else f"  [only {eff_n_resamples}/{N_BOOTSTRAP} resamples succeeded]"
        print(f"    K={k:2d}  BIC={bic:10.1f}  AIC={aic:10.1f}  GMM_sil={sil:.4f}  "
              f"DB={db:.3f}  CH={ch:8.1f}  bootARI={boot_ari:.3f}  "
              f"KMeans_sil={km_sil:.4f}{flag}{eff_flag}")

    table = pd.DataFrame(rows)
    return table


# ═══════════════════════════════════════════════════════════
# LEVEL A — SPATIAL CLUSTERING
# ═══════════════════════════════════════════════════════════

log_header(f"PHASE 4 — LEVEL A (spatial) — {STATE_NAME.title()}")

if not SIGNATURE_FILE.exists():
    raise SystemExit(f"ERROR: {SIGNATURE_FILE} not found — run "
                      f"04_climate_signature_{STATE_NAME}.py first.")

sig = pd.read_csv(SIGNATURE_FILE)
sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)

z_cols_a = [c for c in sig.columns if c.endswith("_z")]
# lat/lon are never among the *_z columns by construction (04's script
# excludes them from the standardized matrix entirely — see its own
# NON_CLUSTERING_COLS) but assert it explicitly here too, since a future
# state's 04-equivalent script is the thing actually enforcing this, and a
# silent regression there would otherwise cluster geography, not climate.
assert not any(c in ("lat_z", "lon_z") for c in z_cols_a), \
    "lat/lon must never be standardized clustering columns — check the upstream 04 script"

print(f"\n  Points: {len(sig)}  |  Standardized (*_z) columns: {len(z_cols_a)}")
X_a = sig[z_cols_a].fillna(sig[z_cols_a].median()).values

print(f"\n[Level A 1/4] GMM (primary) + KMeans (baseline), "
      f"K={K_RANGE_A[0]}..{K_RANGE_A[-1]}, {N_BOOTSTRAP} bootstrap resamples/k ...")
bic_table_a = fit_k_range(X_a, K_RANGE_A, "Level A")
bic_table_a.to_csv(BIC_TABLE_FILE, index=False)
print(f"  Saved: {BIC_TABLE_FILE}")

suggested_k_a, reason_a = suggest_k(bic_table_a, expected_range=EXPECTED_K_RANGE_SINGLE_STATE)
print(f"\n  Suggested k (auto, NOT forced): {suggested_k_a}  [{reason_a}]")
print(f"  Reminder: this is a Rajasthan-ONLY run. Realistic k here is "
      f"{EXPECTED_K_RANGE_SINGLE_STATE[0]}-{EXPECTED_K_RANGE_SINGLE_STATE[1]} "
      f"(arid-west vs semi-arid-east intra-state split, per the framework doc) — "
      f"do NOT mistake this for the eventual 4-state k (expected 6-10). Review "
      f"{BIC_TABLE_FILE.name} yourself; override via LEVEL_A_K_OVERRIDE at the "
      f"top of this script if you disagree with the auto-suggestion.")

k_final_a = LEVEL_A_K_OVERRIDE if LEVEL_A_K_OVERRIDE is not None else suggested_k_a
if LEVEL_A_K_OVERRIDE is not None:
    print(f"  LEVEL_A_K_OVERRIDE set — using k={k_final_a} instead of the auto-suggestion.")

# Pull the chosen k's bootstrap-ARI stats straight from bic_table_a (already
# computed above, not recomputed) so the number quoted in cluster_profiles/
# cluster_profile_cards always carries its effective_n_resamples alongside
# it — a degraded resample count can never masquerade as a full-N result.
_chosen_row_a = bic_table_a.loc[bic_table_a["k"] == k_final_a].iloc[0]
bootstrap_ari_mean_a = float(_chosen_row_a["bootstrap_ari_mean"])
bootstrap_eff_n_a = int(_chosen_row_a["bootstrap_effective_n_resamples"])
if bootstrap_eff_n_a < N_BOOTSTRAP:
    print(f"  WARNING: chosen k={k_final_a}'s bootstrap-ARI ({bootstrap_ari_mean_a:.4f}) is based "
          f"on only {bootstrap_eff_n_a}/{N_BOOTSTRAP} resamples — restate with this caveat "
          f"wherever quoted.")

print(f"\n[Level A 2/4] Final GMM fit at k={k_final_a} ...")
gmm_final_a = GaussianMixture(n_components=k_final_a, covariance_type=GMM_COVARIANCE_TYPE,
                               random_state=RANDOM_STATE, n_init=10)
raw_labels_a = gmm_final_a.fit_predict(X_a)
raw_soft_probs_a = gmm_final_a.predict_proba(X_a)

# CANONICAL CLUSTER RELABELING — fixed 2026-08-11, root-caused after
# Phase 7 (09_physics_validation_rajasthan.py) caught Phase 5's and
# Phase 6's outputs disagreeing on which PCMs belonged to cluster 0 vs.
# cluster 2. sklearn's GaussianMixture assigns cluster index 0..k-1 in an
# arbitrary, fit-order-dependent way with NO guarantee of stability
# across separate re-runs of this script — even with the same
# random_state=42, if anything about the fit changes between runs (the
# covariance_type='diag' fix landing 2026-08-10, for instance), the raw
# index-to-physical-cluster mapping can shift. Every downstream phase
# (5/6/7) keys off cluster_id, so an unstable label is a silent
# correctness bug waiting to happen the moment two of those phases are
# run from different invocations of this script — which is exactly what
# happened. FIX: relabel 0..k-1 canonically by sorting each raw cluster's
# MEAN LATITUDE ascending (south to north) — a simple, always-available,
# fit-independent ordering key computed directly from the points
# themselves (not from anything the GMM itself produces, which is the
# unstable part). "Cluster 0" now means the same physical (southernmost)
# climate regime regardless of which run produced the underlying GMM fit,
# as long as the underlying PARTITION of points is the same (this does
# NOT protect against a genuinely different partition from a re-run with
# different data/parameters — only against the arbitrary INDEX ordering
# of an equivalent partition; see the fingerprint-based hard-fail checks
# added to 07/08/09 for the data-level guard against a genuinely
# different partition).
raw_mean_lat = pd.Series(sig["lat"].values).groupby(raw_labels_a).mean()
canonical_order_a = raw_mean_lat.sort_values().index.tolist()   # raw label, ascending mean lat
relabel_map_a = {raw: canonical for canonical, raw in enumerate(canonical_order_a)}
hard_labels_a = np.array([relabel_map_a[r] for r in raw_labels_a])
soft_probs_a = raw_soft_probs_a[:, canonical_order_a]   # reorder columns to match
print(f"  Canonical relabel applied (raw GMM label -> canonical, by ascending mean latitude): "
      f"{relabel_map_a}")

assign_a = sig[["point_id"]].copy()
for c in ("lat", "lon", "population"):
    if c in sig.columns:
        assign_a[c] = sig[c]
assign_a["cluster_id"] = hard_labels_a
assign_a["max_membership_prob"] = soft_probs_a.max(axis=1)
assign_a["chosen_k"] = k_final_a
assign_a["bootstrap_ari_mean_chosen_k"] = bootstrap_ari_mean_a
assign_a["bootstrap_effective_n_resamples_chosen_k"] = bootstrap_eff_n_a
for k in range(k_final_a):
    assign_a[f"prob_cluster{k}"] = soft_probs_a[:, k]
assign_a.to_csv(ASSIGN_A_FILE, index=False)
print(f"  Saved: {ASSIGN_A_FILE}")

sig["cluster_id"] = hard_labels_a
sig["max_membership_prob"] = soft_probs_a.max(axis=1)


# ═══════════════════════════════════════════════════════════
# LEVEL B — TEMPORAL (SEASONAL) CLUSTERING
# ═══════════════════════════════════════════════════════════

log_header(f"PHASE 4 — LEVEL B (temporal/seasonal) — {STATE_NAME.title()}")

print("\n[Level B 1/5] Loading raw sun-event data for per-point-per-season "
      "Tier 1 construction ...")
pts_cols_b = ["point_id", "date", "event", "season", "era5_T_amb", "era5_RHum",
              "era5_GHI", "era5_CSI", "era5_W_spd"]
events_df_b = pd.read_csv(COMBINED_POINTS_FILE, usecols=pts_cols_b, parse_dates=["date"])
events_df_b["event"] = pd.Categorical(events_df_b["event"], categories=EVENT_ORDER, ordered=True)
events_df_b["season"] = pd.Categorical(events_df_b["season"], categories=SEASON_ORDER, ordered=True)

sun_df_b = pd.read_csv(SUNTIMES_FILE, parse_dates=["date"])
sun_df_b["time_utc"] = pd.to_datetime(sun_df_b["time_utc"], utc=True)
# suntimes.csv has no season column — derive it from date's month (same
# SEASON_MAP as 02_combine_rajasthan.py; see signature_lib.attach_season).
# events_df_b's own "season" column (already in climate_rajasthan_points.csv)
# is used directly rather than re-derived, so both stay from one source of
# truth for the actual climate data; only suntimes.csv needs the derivation.
sun_df_b = attach_season(sun_df_b, date_col="date")
sun_df_b["season"] = pd.Categorical(sun_df_b["season"], categories=SEASON_ORDER, ordered=True)

print(f"  {len(events_df_b):,} event rows, {len(sun_df_b):,} suntime rows")

print("\n[Level B 2/5] Building per-point-per-season Tier 1 signature "
      "(signature_lib.build_tier1_signature, group_keys=[point_id, season]) ...")
tier1_b = build_tier1_signature(events_df_b, sun_df_b, group_keys=["point_id", "season"])
tier1_b = tier1_b.reset_index()
print(f"  {len(tier1_b)} point-season rows ({tier1_b['point_id'].nunique()} points "
      f"x {tier1_b['season'].nunique()} seasons), {tier1_b.shape[1] - 2} Tier 1 columns")

tier1_b_cols = [c for c in tier1_b.columns
                if c not in ("point_id", "season") and c not in LEVEL_B_EXCLUDE_FEATURES]
if LEVEL_B_EXCLUDE_FEATURES:
    print(f"  [ABLATION run: '{LEVEL_B_RUN_TAG}'] excluded from the clustering "
          f"feature set: {LEVEL_B_EXCLUDE_FEATURES}  "
          f"({len(tier1_b_cols)}/{tier1_b.shape[1] - 2} Tier 1 columns remain)")
std_scaler_b = StandardScaler()
X_b = std_scaler_b.fit_transform(tier1_b[tier1_b_cols].fillna(tier1_b[tier1_b_cols].median()))

print(f"\n[Level B 3/5] GMM (primary) + KMeans (baseline), "
      f"K={K_RANGE_B[0]}..{K_RANGE_B[-1]}, {N_BOOTSTRAP} bootstrap resamples/k ...")
# NOTE ON SILHOUETTE COMPARABILITY: expect Level B to run HIGHER than Level
# A's 0.15-0.35 band, and don't read that as "better clustering." A given
# point's summer-vs-winter swing is a stronger, more artificial signal than
# genuine spatial climate variation between points, so a high Level-B
# silhouette is uninformative either way — it neither confirms nor refutes
# real regime structure. The two levels are not on a comparable silhouette
# scale. See the season-tautology check below (step 5/5) for the actual
# test of whether Level B found something beyond "season", which is what
# should drive the k choice here, not the silhouette curve.
bic_table_b = fit_k_range(X_b, K_RANGE_B, "Level B")
bic_table_b.to_csv(BIC_TABLE_B_FILE, index=False)
print(f"  Saved: {BIC_TABLE_B_FILE}")

suggested_k_b, reason_b = suggest_k(bic_table_b, expected_range=None)
print(f"\n  Suggested k (auto, NOT forced): {suggested_k_b}  [{reason_b}]")
if reason_b.startswith("FALLBACK"):
    print("  (Falling back to lowest-BIC k is expected/fine here, per the silhouette-"
          "comparability note above — the 0.15-0.35 band was never validated for "
          "Level B's task, so failing to land in it is not itself informative.)")

k_final_b = LEVEL_B_K_OVERRIDE if LEVEL_B_K_OVERRIDE is not None else suggested_k_b
if LEVEL_B_K_OVERRIDE is not None:
    print(f"  LEVEL_B_K_OVERRIDE set — using k={k_final_b} instead of the auto-suggestion.")

_chosen_row_b = bic_table_b.loc[bic_table_b["k"] == k_final_b].iloc[0]
bootstrap_ari_mean_b = float(_chosen_row_b["bootstrap_ari_mean"])
bootstrap_eff_n_b = int(_chosen_row_b["bootstrap_effective_n_resamples"])
if bootstrap_eff_n_b < N_BOOTSTRAP:
    print(f"  WARNING: chosen k={k_final_b}'s bootstrap-ARI ({bootstrap_ari_mean_b:.4f}) is based "
          f"on only {bootstrap_eff_n_b}/{N_BOOTSTRAP} resamples — restate with this caveat "
          f"wherever quoted.")

gmm_final_b = GaussianMixture(n_components=k_final_b, covariance_type=GMM_COVARIANCE_TYPE,
                               random_state=RANDOM_STATE, n_init=10)
hard_labels_b = gmm_final_b.fit_predict(X_b)
soft_probs_b = gmm_final_b.predict_proba(X_b)

assign_b = tier1_b[["point_id", "season"]].copy()
assign_b["cluster_id"] = hard_labels_b
assign_b["max_membership_prob"] = soft_probs_b.max(axis=1)
assign_b["chosen_k"] = k_final_b
assign_b["bootstrap_ari_mean_chosen_k"] = bootstrap_ari_mean_b
assign_b["bootstrap_effective_n_resamples_chosen_k"] = bootstrap_eff_n_b
for k in range(k_final_b):
    assign_b[f"prob_cluster{k}"] = soft_probs_b[:, k]
assign_b.to_csv(ASSIGN_B_FILE, index=False)
print(f"  Saved: {ASSIGN_B_FILE}")

print("\n[Level B 4/5] Regime-shift analysis (does a point's seasonal cluster "
      "label change across the year?) ...")
shift_table = assign_b.pivot_table(index="point_id", columns="season",
                                    values="cluster_id", aggfunc="first")
n_shifting = int((shift_table.nunique(axis=1) > 1).sum())
n_total_pts = len(shift_table)
print(f"  {n_shifting}/{n_total_pts} points ({100*n_shifting/n_total_pts:.1f}%) "
      f"have a DIFFERENT seasonal cluster label in at least one season — "
      f"i.e. their PCM-relevant climate regime shifts materially within the year.")
print("  This is a SEPARATE result from Level A (spatial regimes) — report both, "
      "don't merge them. Read this number together with step 5/5 below before "
      "calling it a finding: if the 8 clusters map ~1:1 onto the 4 seasons, a "
      "high shifting fraction is expected/tautological (points trivially look "
      "different in different seasons), not evidence of a richer regime structure.")

print("\n[Level B 5/5] Season-tautology check — does Level B just rediscover "
      "\"season\", or find something beyond it? (plan doc §7.1 applies the same "
      "logic to state identity at Level A: \"recovering the [known] boundaries "
      "alone is not a finding\" — swap state for season here) ...")

# 1. Contingency table: cluster_id x season. A clean ~1:1 block-diagonal
# pattern (each cluster concentrated in one season) is the tautological
# signature; clusters spanning multiple seasons, or a season splitting
# across several clusters in a non-trivial way, is the more interesting
# outcome.
contingency = pd.crosstab(assign_b["cluster_id"], assign_b["season"])
print(f"\n  Cluster x season contingency table (k={k_final_b}):")
print(contingency.to_string())

# 2. ARI / NMI between cluster label and season label directly — same
# metrics already used for the (stubbed) external validation above, same
# interpretation: ARI near 1.0 means the clustering IS season, nothing
# more; a meaningfully lower ARI with visible cross-season merging in the
# table above is the actual finding worth reporting.
season_ari = adjusted_rand_score(assign_b["season"].astype(str), assign_b["cluster_id"])
season_nmi = normalized_mutual_info_score(assign_b["season"].astype(str), assign_b["cluster_id"])
print(f"\n  ARI(cluster, season) = {season_ari:.3f}   NMI(cluster, season) = {season_nmi:.3f}")
if season_ari >= 0.7:
    print("  -> HIGH agreement: Level B is substantially rediscovering \"season\" — "
          "report this honestly as \"the clustering recovers known seasonal "
          "structure\", not as a novel regime-shift discovery.")
elif season_ari >= 0.3:
    print("  -> MODERATE agreement: some season-tracking, but with real cross-season "
          "structure too (see the contingency table) — worth digging into which "
          "clusters/points don't follow the trivial season assignment.")
else:
    print("  -> LOW agreement: Level B is finding structure clearly beyond season — "
          "this is the genuinely interesting result.")

# 3. Which features actually drive the 8-cluster split? One-way ANOVA
# F-statistic per Tier-1 feature against hard_labels_b — high-F features
# are what's separating the clusters. If the top features are all
# temperature/GHI (the season-DEFINING variables), that supports the
# tautology reading; if humidity, wind, or daylength also rank highly,
# that's evidence Level B captures something beyond season even where the
# cluster-season correspondence looks strong.
f_stats, p_vals = f_classif(X_b, hard_labels_b)
feature_importance = pd.DataFrame({
    "feature": tier1_b_cols, "F_statistic": f_stats, "p_value": p_vals,
}).sort_values("F_statistic", ascending=False)
print("\n  Feature importance (ANOVA F-statistic across the k={} clusters, "
      "highest first):".format(k_final_b))
print(feature_importance.to_string(index=False))

temp_ghi_features = {"T_sunrise_mean", "T_noon_mean", "T_sunset_mean", "Ta_mean",
                      "Ta_p95", "Ta_p05", "diurnal_gradient", "GHI_noon_mean",
                      "GHI_sunset_mean", "kt_noon_mean", "kt_noon_std"}
top5 = set(feature_importance.head(5)["feature"])
non_temp_in_top5 = top5 - temp_ghi_features
if non_temp_in_top5:
    print(f"\n  Non-temperature/GHI feature(s) in the top 5 drivers: {sorted(non_temp_in_top5)} "
          f"— some evidence the split isn't purely season-defining-variable-driven.")
else:
    print("\n  All top-5 drivers are temperature/GHI (season-defining) variables — "
          "consistent with the tautology reading; RH/wind/daylength are not "
          "meaningfully separating the clusters here.")

# Persist the season-tautology diagnostics computed above — same
# reproducibility fix as the Level B k-scan table: these were previously
# console-printed only. Companion files to BIC_TABLE_B_FILE, linked by the
# same chosen_k_levelB value and the same STATE_NAME/_levelB_suffix naming.
feature_importance_out = feature_importance.copy()
feature_importance_out["chosen_k_levelB"] = k_final_b
feature_importance_out.to_csv(LEVEL_B_FEATURE_IMPORTANCE_FILE, index=False)
print(f"\n  Saved: {LEVEL_B_FEATURE_IMPORTANCE_FILE}")

regime_shift_fraction = n_shifting / n_total_pts
season_tautology_summary = pd.DataFrame([{
    "chosen_k_levelB": k_final_b,
    "n_shifting_points": n_shifting,
    "n_total_points": n_total_pts,
    "regime_shift_fraction": regime_shift_fraction,
    "season_ari": season_ari,
    "season_nmi": season_nmi,
}])
season_tautology_summary.to_csv(LEVEL_B_SEASON_TAUTOLOGY_FILE, index=False)
print(f"  Saved: {LEVEL_B_SEASON_TAUTOLOGY_FILE}")

contingency.to_csv(LEVEL_B_CONTINGENCY_FILE)
print(f"  Saved: {LEVEL_B_CONTINGENCY_FILE}")


# ═══════════════════════════════════════════════════════════
# EXTERNAL VALIDATION  (Level A especially)
# ═══════════════════════════════════════════════════════════

log_header("EXTERNAL VALIDATION")


def load_koppen_legend(path):
    """Parse legend.txt (shipped inside Beck et al.'s Beck_KG_V1.zip) into
    {numeric class code: short Koppen symbol} — parsed at runtime rather
    than hardcoded so the mapping stays traceable to the actual shipped
    file instead of a copy that could silently drift from it."""
    legend = {}
    pattern = re.compile(r"^\s*(\d+):\s+(\S+)\s+")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line)
        if m:
            legend[int(m.group(1))] = m.group(2)
    return legend


def lookup_koppen_classes(raster_path, lats, lons):
    """Point-sample the Koppen-Geiger raster at each (lat, lon) — one class
    code per point, nearest-pixel (the raster is 1-km resolution, i.e.
    ~0.0083 degrees, far finer than the spacing between this state's 320
    sampled points, so nearest-pixel sampling is not a meaningful source of
    error here)."""
    with rasterio.open(raster_path) as src:
        coords = list(zip(lons, lats))
        return np.array([v[0] for v in src.sample(coords)], dtype=int)


if not KOPPEN_RASTER_FILE.exists():
    print(f"""
  Koppen-Geiger classification: NOT WIRED IN — raster not found at
  {KOPPEN_RASTER_FILE}.
  TODO to enable: download Beck et al. 2018's Beck_KG_V1.zip
  (doi:10.1038/sdata.2018.214, figshare article 6396959,
  https://ndownloader.figshare.com/files/12407516), extract
  Beck_KG_V1_present_0p0083.tif and legend.txt into {KOPPEN_RASTER_FILE.parent},
  then re-run. Stubbed here rather than fabricated — the raster is not
  present in this project tree as of this run.
""")
    koppen_ari = koppen_nmi = None
    koppen_validation_meaningful = False
    koppen_class_counts = pd.Series(dtype=int)
else:
    print(f"\n  Koppen-Geiger present-climate classification (Beck et al. 2018, "
          f"doi:10.1038/sdata.2018.214), 1-km raster, wired in for real this run.")
    koppen_legend = load_koppen_legend(KOPPEN_LEGEND_FILE)
    koppen_class_codes = lookup_koppen_classes(KOPPEN_RASTER_FILE, sig["lat"].values, sig["lon"].values)
    koppen_class_labels = np.array([koppen_legend.get(c, f"UNKNOWN_{c}") for c in koppen_class_codes])

    koppen_class_counts = pd.Series(koppen_class_labels).value_counts()
    dominant_frac = float(koppen_class_counts.iloc[0] / len(koppen_class_labels))
    print(f"\n  Koppen class distribution across {len(sig)} sampled points:")
    print(koppen_class_counts.to_string())

    # Degenerate-variance guard: mirrors the existing "not meaningful yet"
    # handling of state-identity validation at Level A — an ARI/NMI against
    # a near-constant label set is not informative, so flag it rather than
    # report a spuriously low/high number as if it meant something.
    koppen_validation_meaningful = bool(dominant_frac < 0.95 and len(koppen_class_counts) > 1)
    if not koppen_validation_meaningful:
        print(f"\n  KOPPEN VALIDATION NOT MEANINGFUL AT STATE SCALE — "
              f"{dominant_frac * 100:.1f}% of sampled points fall into a single Koppen class "
              f"({koppen_class_counts.index[0]}); insufficient Koppen-class variance across "
              f"sampled points for ARI/NMI to be informative here. Mirrors the existing "
              f"'not meaningful yet' handling of the state-identity external check below.")

    koppen_ari = float(adjusted_rand_score(koppen_class_labels, hard_labels_a))
    koppen_nmi = float(normalized_mutual_info_score(koppen_class_labels, hard_labels_a))
    print(f"\n  ARI(GMM cluster, Koppen class) = {koppen_ari:.4f}   "
          f"NMI(GMM cluster, Koppen class) = {koppen_nmi:.4f}")
    if koppen_validation_meaningful:
        if koppen_ari >= 0.5:
            print("  -> Relatively HIGH agreement with Koppen: the GMM clusters substantially "
                  "rediscover Koppen's existing class boundaries within this state — report this "
                  "honestly as recovering known structure, not as a novel finding.")
        else:
            print("  -> LOW-to-moderate agreement with Koppen: the GMM is finding climate "
                  "structure at a finer resolution than Koppen's broad classes capture within "
                  "Rajasthan. This is a plausible and legitimate finding in its own right — it is "
                  "arguably the point of empirical clustering instead of applying Koppen directly "
                  "— NOT evidence the clustering failed to find anything real.")
    else:
        print("  (Interpretation above intentionally omitted — see 'NOT MEANINGFUL' note; the "
              "numeric ARI/NMI values are still recorded for the record, not treated as evidence "
              "either way.)")

    koppen_contingency = pd.crosstab(pd.Series(hard_labels_a, name="cluster_id"),
                                      pd.Series(koppen_class_labels, name="koppen_class"))
    koppen_contingency.to_csv(KOPPEN_CONTINGENCY_FILE)
    print(f"\n  Saved: {KOPPEN_CONTINGENCY_FILE} (cluster_id x koppen_class contingency counts)")

print("""
  NBC/ECBC Indian climate zone classification: NOT WIRED IN.
  TODO to enable: source an NBC (National Building Code of India) or ECBC
  climate-zone lookup (Rajasthan spans the Hot-Dry and Composite zones)
  as a shapefile/lookup table, join it to each point the same way as
  Koppen above, then compute ARI/NMI the same way. No local lookup exists
  in this project tree as of this run — stubbed, not fabricated (per the
  task brief: do not approximate a zone map to fill this in).

  State-identity external check: NOT MEANINGFUL YET. This is a
  single-state (Rajasthan-only) run — an ARI/NMI against "state identity"
  is only informative once >=2 states' Level-A results are combined (a
  clustering that just reproduces state boundaries at that point would
  mean k is too low / the clustering learned nothing beyond the sampling
  design). Revisit this check when Assam/Tamil Nadu/Uttarakhand are added.
""")

nbc_ari = nbc_nmi = None


# ═══════════════════════════════════════════════════════════
# FOLIUM MAP  (Level A hard cluster, shaded by max membership probability)
# ═══════════════════════════════════════════════════════════

log_header("Cluster map (Level A)")

center = [sig["lat"].mean(), sig["lon"].mean()]
m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=6)

palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324"]
cluster_colors = {cid: palette[cid % len(palette)] for cid in range(k_final_a)}

for row in sig.itertuples(index=False):
    color = cluster_colors[int(row.cluster_id)]
    # opacity scaled by max_membership_prob: confident points are solid,
    # ambiguous/transition points (low max probability, i.e. genuinely
    # split between regimes per the GMM's soft assignment) fade out —
    # visually distinguishes "clearly this regime" from "boundary point".
    opacity = 0.35 + 0.6 * float(row.max_membership_prob)
    popup = folium.Popup(
        f"<b>{row.point_id}</b><br>Cluster: {int(row.cluster_id)}<br>"
        f"Max membership prob: {row.max_membership_prob:.3f}<br>"
        f"lat/lon: {row.lat:.3f}, {row.lon:.3f}",
        max_width=220,
    )
    folium.CircleMarker(
        location=[row.lat, row.lon], radius=6, color=color, weight=1,
        fill=True, fill_color=color, fill_opacity=opacity, popup=popup,
    ).add_to(m)

legend_html = f"""
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
            background: white; padding: 10px 14px; border: 1px solid #999;
            border-radius: 4px; font-size: 13px; line-height: 1.6;">
  <b>Level A clusters (k={k_final_a})</b><br>
  {''.join(f'<span style="color:{cluster_colors[c]}">&#9679;</span> Cluster {c}<br>' for c in range(k_final_a))}
  <span style="font-size:11px;color:#666">Faded points = low max membership<br>
  probability (near a regime boundary)</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save(str(MAP_FILE))
print(f"  Saved: {MAP_FILE}")


# ═══════════════════════════════════════════════════════════
# CLUSTER PROFILE CARDS  (Level A)
# ═══════════════════════════════════════════════════════════

log_header("Cluster profile cards (Level A)")

SIGNATURE_DISPLAY = [c for c in [
    "Ta_mean", "Ta_p95", "Ta_p05", "T_sunrise_mean", "T_noon_mean", "T_sunset_mean",
    "diurnal_gradient", "GHI_noon_mean", "GHI_sunset_mean", "GHI_daily_kWh",
    "kt_noon_mean", "kt_noon_std", "kt_daily_mean", "kt_daily_std", "SAI",
    "cloudy_frac", "CCI", "HDD18", "CDD24", "DTR_true", "RH_sunrise_mean",
    "HSI_sunrise", "wind_noon_mean", "wind_sunset_mean", "daylength_mean",
    "daylength_amplitude", "seasonality", "monsoon_index",
] if c in sig.columns]


_weighted_mean_fallback_log = []


def weighted_mean(g, col, context=None):
    """Population-weighted mean, falling back to an unweighted mean if
    population weights are missing/zero. The fallback itself is a
    reasonable behavior (kept as-is) — what was previously silent is now
    logged: every fallback is recorded (with the caller-supplied `context`
    label, e.g. "cluster_profile:cluster=1:Ta_mean") so it's traceable back
    to exactly which cluster-profile field was affected, not just a generic
    "fell back somewhere"."""
    w = g["population"].fillna(g["population"].median()) if "population" in g.columns else None
    if w is None or w.sum() == 0:
        print(f"    WARNING: weighted_mean fallback to UNWEIGHTED mean — "
              f"context={context!r}, col={col!r} (population weights missing or sum to zero)")
        _weighted_mean_fallback_log.append({"context": context, "col": col})
        return g[col].mean()
    return float(np.average(g[col], weights=w))


def describe_cluster(row, medians):
    """Rule-based, threshold-driven one-line description generator — NOT a
    fabricated string. Compares this cluster's key indices against the
    across-cluster median on three axes (heat, aridity/monsoon influence,
    solar variability) and composes a short phrase from the comparison.
    Intentionally simple; treat as a first-pass label to hand-edit, not a
    final publication-ready caption."""
    parts = []
    ta = row.get("Ta_mean", np.nan)
    if ta == ta and "Ta_mean" in medians:
        parts.append("hot" if ta >= medians["Ta_mean"] else "cooler")
    mi = row.get("monsoon_index", np.nan)
    if mi == mi and "monsoon_index" in medians:
        parts.append("monsoon-influenced" if mi >= medians["monsoon_index"] else "arid/low-monsoon")
    kt_std = row.get("kt_daily_std", np.nan)
    if kt_std == kt_std and "kt_daily_std" in medians:
        parts.append("erratic solar resource" if kt_std >= medians["kt_daily_std"] else "steady solar resource")
    cci = row.get("CCI", np.nan)
    if cci == cci and "CCI" in medians:
        parts.append("long low-clearness runs (high autonomy demand)" if cci >= medians["CCI"] else "short low-clearness runs")
    return ", ".join(parts).capitalize() + "." if parts else "Insufficient data to describe."


cluster_medians = sig.groupby("cluster_id")[[c for c in SIGNATURE_DISPLAY if c in sig.columns]].mean().median()

# Numeric per-cluster profile CSV — built alongside the markdown cards
# below so both read from the exact same weighted_mean() calls (no risk
# of the two drifting apart). PCM-facing columns (Tm_target_C,
# Tm_target_capped_C, L_required_kJ_per_kg) are included explicitly since
# these are what a downstream feasibility-filter script needs per cluster.
PCM_FACING_COLS = [c for c in ["Tm_target_C", "Tm_target_capped_C", "L_required_kJ_per_kg"]
                    if c in sig.columns]
profile_rows = []
for cid in sorted(sig["cluster_id"].unique()):
    g = sig[sig["cluster_id"] == cid]
    row = {
        "cluster_id": int(cid), "n_points": len(g), "total_population": g["population"].sum(),
        # Chosen-k bootstrap stability (same value for every cluster row —
        # it's a property of the whole k-way solution, not per-cluster —
        # included here so it's always visible alongside the profile it
        # supports, per the Phase 4 audit fix for silent resample-count loss).
        "bootstrap_ari_mean_chosen_k": bootstrap_ari_mean_a,
        "bootstrap_effective_n_resamples_chosen_k": bootstrap_eff_n_a,
        # Koppen-Geiger external validation (Fix 4) — likewise one value for
        # the whole clustering solution, not per-cluster.
        "koppen_ari": koppen_ari,
        "koppen_nmi": koppen_nmi,
        "koppen_validation_meaningful": koppen_validation_meaningful,
    }
    for col in SIGNATURE_DISPLAY + PCM_FACING_COLS:
        row[col] = weighted_mean(g, col, context=f"cluster_profiles_csv:cluster={cid}:{col}")
    profile_rows.append(row)
cluster_profiles = pd.DataFrame(profile_rows)
PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
cluster_profiles.to_csv(PROFILE_FILE, index=False)
print(f"  Saved: {PROFILE_FILE}  ({len(cluster_profiles)} cluster rows, "
      f"{len(SIGNATURE_DISPLAY) + len(PCM_FACING_COLS)} numeric columns)")

_boot_caveat_a = "" if bootstrap_eff_n_a == N_BOOTSTRAP else \
    f" **(CAVEAT: only {bootstrap_eff_n_a}/{N_BOOTSTRAP} bootstrap resamples succeeded — see " \
    f"{BIC_TABLE_FILE.name} for the per-resample failure log.)**"

if koppen_ari is None:
    _koppen_para = ("Koppen-Geiger external validation: NOT WIRED IN this run (raster not found) "
                    "— see the printed EXTERNAL VALIDATION section for how to enable it.")
elif not koppen_validation_meaningful:
    _koppen_para = (f"Koppen-Geiger external validation: **NOT MEANINGFUL AT STATE SCALE** — "
                     f"the 320 sampled points are overwhelmingly one Koppen class, so ARI="
                     f"{koppen_ari:.4f}/NMI={koppen_nmi:.4f} against Koppen labels are recorded "
                     f"for the record but carry no interpretive weight (see "
                     f"{KOPPEN_CONTINGENCY_FILE.name} for the full class distribution).")
else:
    _koppen_para = (f"Koppen-Geiger external validation: ARI={koppen_ari:.4f}, NMI={koppen_nmi:.4f} "
                     f"against Beck et al. (2018) present-climate classes for these same 320 points "
                     f"(see {KOPPEN_CONTINGENCY_FILE.name} for the full cluster x Koppen-class "
                     f"contingency table). " +
                     ("This is a relatively high agreement — the GMM clusters substantially "
                      "rediscover Koppen's existing boundaries within this state."
                      if koppen_ari >= 0.5 else
                      "This is a low-to-moderate agreement — the GMM is finding climate structure "
                      "at a finer resolution than Koppen's broad classes capture within Rajasthan, "
                      "which is a legitimate finding in its own right, not a failure of the "
                      "clustering."))

lines = [f"# {STATE_NAME.title()} — Level A Cluster Profile Cards\n",
         f"Generated from k={k_final_a} GMM clusters "
         f"({len(sig)} population points). Auto-suggested k selection reason: "
         f"{reason_a}. Bootstrap-ARI stability at this k: {bootstrap_ari_mean_a:.4f} "
         f"(mean over {bootstrap_eff_n_a}/{N_BOOTSTRAP} resamples).{_boot_caveat_a}\n",
         f"{_koppen_para}\n",
         f"This is a single-state run — see the printed "
         f"EXTERNAL VALIDATION section for what's not yet wired in (NBC/ECBC, state identity).\n"]

for cid in sorted(sig["cluster_id"].unique()):
    g = sig[sig["cluster_id"] == cid]
    lines.append(f"\n## Cluster {int(cid)}\n")
    lines.append(f"- **Points in regime:** {len(g)}")
    lines.append(f"- **Total population covered:** {g['population'].sum():,.0f}")

    # Medoid: nearest point (in the standardized clustering feature space,
    # not lat/lon) to this cluster's mean — the point most representative
    # of the cluster's CLIMATE signature, not just its geographic center.
    g_idx = g.index
    dists = np.sqrt(((X_a[g_idx] - X_a[g_idx].mean(axis=0)) ** 2).sum(axis=1))
    medoid_row = g.loc[g_idx[int(np.argmin(dists))]]
    lines.append(f"- **Medoid point (climate-feature-space, no district lookup "
                  f"available — lat/lon only):** {medoid_row['point_id']} "
                  f"({medoid_row['lat']:.3f}, {medoid_row['lon']:.3f})")

    lines.append("\n**Two-tier climate signature (population-weighted mean +/- std):**\n")
    lines.append("| Index | Mean | Std |")
    lines.append("|---|---|---|")
    for col in SIGNATURE_DISPLAY:
        wm = weighted_mean(g, col, context=f"cluster_profile_cards_md:cluster={cid}:signature_table:{col}")
        sd = float(g[col].std())
        lines.append(f"| {col} | {wm:.3f} | {sd:.3f} |")

    cluster_means = {c: weighted_mean(g, c, context=f"cluster_profile_cards_md:cluster={cid}:describe_cluster:{c}")
                      for c in SIGNATURE_DISPLAY}
    lines.append(f"\n**Physical description (auto-generated, review before publishing):** "
                  f"{describe_cluster(cluster_means, cluster_medians)}\n")

    tm_target = g["Tm_target_C"].mean() if "Tm_target_C" in g.columns else float("nan")
    l_req = weighted_mean(g, "L_required_kJ_per_kg",
                           context=f"cluster_profile_cards_md:cluster={cid}:L_required_kJ_per_kg") \
        if "L_required_kJ_per_kg" in g.columns else float("nan")
    lines.append(f"**Derived PCM targets:** Tm_target_C = {tm_target:.1f} C, "
                 f"L_required_kJ_per_kg = {l_req:.0f} kJ/kg "
                 f"(CEILING, not an achievability bar — see 04_climate_signature_"
                 f"{STATE_NAME}.py's docstring)\n")

CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
CARDS_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"  Saved: {CARDS_FILE}  ({sig['cluster_id'].nunique()} cluster cards)")

if _weighted_mean_fallback_log:
    print(f"\n  weighted_mean() fell back to an UNWEIGHTED mean {len(_weighted_mean_fallback_log)} "
          f"time(s) — see WARNING lines above for exactly which cluster/column. This means "
          f"population weights were missing or summed to zero for at least one cluster's "
          f"population subset; treat the affected field(s) as population-UNweighted.")
else:
    print(f"\n  weighted_mean() fallback check: CONFIRMED CLEAN — 0 fallbacks to an unweighted "
          f"mean across all {len(sig['cluster_id'].unique())} clusters x "
          f"{len(SIGNATURE_DISPLAY) + len(PCM_FACING_COLS)} columns. Population weights were "
          f"present and non-zero everywhere for this Rajasthan run (low practical risk noted in "
          f"the Phase 4 audit; revisit for any future state with sparser population data).")


# ═══════════════════════════════════════════════════════════
# CLUSTER-LEVEL QC PLOTS (added 2026-08-11 — the k-selection table
# (bic_selection_rajasthan.csv) and per-cluster profile CSV
# (cluster_profiles_rajasthan.csv) already existed as data; these three
# plots are just visualizations of data this script already computed
# above — no new computation.)
# ═══════════════════════════════════════════════════════════

log_header("Cluster-level QC plots")

# H. K-selection curve — BIC (primary selection criterion) and silhouette
#    (secondary/interpretability check) vs k, Level A, with the
#    auto-suggested k marked. Lets the k choice be checked by eye against
#    the same table suggest_k() already used numerically.
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=bic_table_a["k"], y=bic_table_a["BIC"], mode="lines+markers",
                          name="BIC (lower is better)", line=dict(color="#4c72b0")),
              secondary_y=False)
fig.add_trace(go.Scatter(x=bic_table_a["k"], y=bic_table_a["silhouette"], mode="lines+markers",
                          name="GMM silhouette", line=dict(color="#dd8452")),
              secondary_y=True)
fig.add_vline(x=k_final_a, line_dash="dash", line_color="green",
              annotation_text=f"chosen k={k_final_a}")
fig.add_hrect(y0=SILHOUETTE_LO, y1=SILHOUETTE_HI, secondary_y=True,
              fillcolor="green", opacity=0.08, line_width=0,
              annotation_text="realistic climate-zone silhouette band", annotation_position="top left")
fig.update_yaxes(title_text="BIC", secondary_y=False)
fig.update_yaxes(title_text="Silhouette", secondary_y=True)
fig.update_xaxes(title_text="k (number of clusters)")
fig.update_layout(title=f"Level A — K-Selection Curve (BIC + Silhouette) — {STATE_NAME.title()}")
kcurve_path = OUTPUTS_DIR / f"qc_k_selection_curve_{STATE_NAME}.html"
fig.write_html(str(kcurve_path))
print(f"  Saved: {kcurve_path}")

# I. Cluster-profile bar chart — a handful of headline signature indices,
#    grouped by cluster, so the "hot/arid" vs "monsoon-influenced" vs
#    "erratic solar" distinctions the markdown cards describe in words are
#    also visible as bars.
PROFILE_BAR_COLS = [c for c in ["Ta_mean", "GHI_daily_kWh", "monsoon_index",
                                 "kt_daily_std", "CCI", "HSI_sunrise"]
                     if c in cluster_profiles.columns]
fig = make_subplots(rows=1, cols=len(PROFILE_BAR_COLS), subplot_titles=PROFILE_BAR_COLS)
for i, col in enumerate(PROFILE_BAR_COLS, start=1):
    fig.add_trace(go.Bar(x=cluster_profiles["cluster_id"].astype(str), y=cluster_profiles[col],
                          showlegend=False, marker_color="#4c72b0"), row=1, col=i)
fig.update_layout(title=f"Level A — Cluster Profile, Headline Indices (population-weighted means) — "
                         f"{STATE_NAME.title()}")
profile_bar_path = OUTPUTS_DIR / f"qc_cluster_profile_bars_{STATE_NAME}.html"
fig.write_html(str(profile_bar_path))
print(f"  Saved: {profile_bar_path}")

# J. Population-share pie chart — how much of the state's population each
#    climate regime actually covers, since a numerically small cluster
#    (few points) can still cover a large population share or vice versa.
fig = go.Figure(data=go.Pie(
    labels=[f"Cluster {int(c)}" for c in cluster_profiles["cluster_id"]],
    values=cluster_profiles["total_population"], hole=0.35,
))
fig.update_layout(title=f"Level A — Population Share by Climate Regime — {STATE_NAME.title()}")
pop_pie_path = OUTPUTS_DIR / f"qc_cluster_population_share_{STATE_NAME}.html"
fig.write_html(str(pop_pie_path))
print(f"  Saved: {pop_pie_path}")


log_header("PHASE 4 COMPLETE")
print(f"  Level A: k={k_final_a} clusters, {len(sig)} points")
print(f"  Level B: k={k_final_b} seasonal clusters, {n_shifting}/{n_total_pts} points "
      f"shift regime across seasons")
print(f"  Outputs: {BIC_TABLE_FILE.name}, {BIC_TABLE_B_FILE.name}, {ASSIGN_A_FILE.name}, "
      f"{ASSIGN_B_FILE.name}, {LEVEL_B_FEATURE_IMPORTANCE_FILE.name}, "
      f"{LEVEL_B_SEASON_TAUTOLOGY_FILE.name}, {LEVEL_B_CONTINGENCY_FILE.name}, "
      f"{MAP_FILE.name}, {CARDS_FILE.name}, {PROFILE_FILE.name}, {KOPPEN_CONTINGENCY_FILE.name}")
