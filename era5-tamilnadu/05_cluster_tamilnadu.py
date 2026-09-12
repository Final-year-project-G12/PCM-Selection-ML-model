"""
05_cluster_tamilnadu.py
=============================================================================
PHASE 4 — LEVEL A: CLIMATE REGIME CLUSTERING (spatial), TAMIL NADU
(Objective1_PCM_Climate_Framework_Plan_v3, §6.2/§7)

UNIFIED WITH RAJASTHAN (2026-09-08)
-----------------------------------
This script is now a state-parameterised mirror of Rajasthan's Level A in
`era5-rajasthan/05_cluster_rajasthan.py`. What changed from the previous
Tamil Nadu version:

  * K-SELECTION — the fixed `K_FINAL = 5` hand-pick is REPLACED by
    Rajasthan's documented 3-tier `suggest_k()` cascade:
      (1) k in the expected single-state range [2,4] AND silhouette in the
          realistic band [0.15, 0.35] -> highest bootstrap-ARI among those;
      (2) any k with silhouette in the band -> highest bootstrap-ARI;
      (3) fallback: lowest-BIC k, with a printed warning.
    The k-scan range is widened from 2..10 to 2..12 to match Rajasthan.
  * BOOTSTRAP-ARI — `bootstrap_ari_stability()` (50 resamples: fit once on
    the full data for `base_labels`, then 50x fit a fresh GMM on a
    with-replacement resample of the same size, predict on the ORIGINAL
    data, and take the Adjusted Rand Index against `base_labels`) now runs
    at EVERY scanned k and is the actual tiebreaker in the selection rule.
    Failed resamples are logged, never silently swallowed.
  * EXTERNAL VALIDATION — the real Köppen-Geiger per-point lookup (Beck et
    al. 2018, doi:10.1038/sdata.2018.214, 1-km raster) with ARI/NMI against
    the GMM clusters and a cluster_id x Köppen-class contingency table.
  * CANONICAL RELABELING — after the final GMM fit, hard cluster IDs are
    relabeled 0..k-1 by ascending MEAN LATITUDE (south to north) BEFORE any
    output file is written, so "cluster 0" means the same physical regime
    across separate re-runs. sklearn's GaussianMixture gives no such
    guarantee on its own; this is the same fix Rajasthan applied on
    2026-08-11 after Phase 5 and Phase 6 disagreed on cluster membership.
  * QC PLOTS — the four inline HTML outputs Rajasthan already emitted, plus
    two improvements applied to BOTH states: the k-selection curve now
    overlays bootstrap-ARI as a third series (it is the tiebreaker in the
    selection rule and was previously not visualised anywhere), and the
    cluster map scales marker SIZE by population, not just colour by
    cluster_id.

LEVEL B lives in its own script now — see `05a_level_b_regime_shift_
tamilnadu.py` (per-point-per-season re-clustering, regime-shift fraction,
season-tautology check). The separate post-Phase-6 seasonal PCM re-ranking
that used to be called "Level B" is now `11_seasonal_pcm_sensitivity.py`;
the two are distinct analyses and the name collision is resolved.

Still uses Gaussian Mixture (not K-Means) because climate is a continuous
gradient within a state: the boundary between "interior dry" and "coastal
humid" Tamil Nadu is not a hard line, and a point near it genuinely has
partial membership in both. Soft membership probabilities are kept and are
what Phase 5/6 should read for boundary points. K-Means is fit in parallel
purely as a REPORTED comparison baseline, never as the primary model.

INPUT  : data/processed/signatures/climate_signature_tamilnadu.csv
           (04b_climate_signature.py's output — Level A reads its *_z
            columns directly)
OUTPUT : data/processed/clustering/
           bic_selection_tamilnadu.csv          (k=2..12 metric table:
               BIC/AIC/silhouette/DB/CH/bootstrap-ARI + effective resample
               count + KMeans silhouette)
           kmeans_comparison_tamilnadu.csv      (KMeans silhouette per k)
           cluster_assignments_tamilnadu.csv    (canonical labels + soft
               membership probabilities + chosen-k bootstrap stats)
           cluster_profiles_tamilnadu.csv       (population-weighted profile
               per regime — Phase 5 reads Tm_target_C / Tm_target_capped_C /
               L_required_kJ_per_kg / HSI_sunrise from here)
           cluster_profile_cards_tamilnadu.md
           koppen_validation_tamilnadu.csv      (cluster_id x Köppen class)
           cluster_map_tamilnadu.png            (static, kept for continuity)
         outputs/
           qc_cluster_map_tamilnadu.html
           qc_k_selection_curve_tamilnadu.html
           qc_cluster_profile_bars_tamilnadu.html
           qc_cluster_population_share_tamilnadu.html

REQUIRED LIBRARIES:
  pip install pandas numpy scikit-learn folium branca plotly rasterio

HOW TO RUN:
  python 05_cluster_tamilnadu.py
"""

import warnings
warnings.filterwarnings("ignore")

import re

import numpy as np
import pandas as pd
import rasterio
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import folium
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROCESSED_DIR, OUTPUTS_DIR,
    KOPPEN_RASTER_FILE, KOPPEN_LEGEND_FILE, ensure_data_dirs,
)
# Shared Phase-4 machinery — ONE implementation across both states and both
# clustering levels (see cluster_lib.py's docstring).
from cluster_lib import (
    GMM_COVARIANCE_TYPE, N_BOOTSTRAP, RANDOM_STATE,
    SILHOUETTE_LO, SILHOUETTE_HI,
    fit_k_range, suggest_k, canonical_relabel_by_latitude,
)

ensure_data_dirs()

# ═══════════════════════════════════════════════════════════
# STATE NAME — the only hardcoded state reference in this file.
# Every output path is built from it.
# ═══════════════════════════════════════════════════════════
STATE_NAME = "tamilnadu"

SIGNATURE_FILE = PROCESSED_DIR / "signatures" / f"climate_signature_{STATE_NAME}.csv"
OUT_DIR = PROCESSED_DIR / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIC_TABLE_FILE = OUT_DIR / f"bic_selection_{STATE_NAME}.csv"
KMEANS_TABLE_FILE = OUT_DIR / f"kmeans_comparison_{STATE_NAME}.csv"
ASSIGN_A_FILE = OUT_DIR / f"cluster_assignments_{STATE_NAME}.csv"
PROFILE_FILE = OUT_DIR / f"cluster_profiles_{STATE_NAME}.csv"
CARDS_FILE = OUT_DIR / f"cluster_profile_cards_{STATE_NAME}.md"
KOPPEN_CONTINGENCY_FILE = OUT_DIR / f"koppen_validation_{STATE_NAME}.csv"
STATIC_MAP_FILE = OUT_DIR / f"cluster_map_{STATE_NAME}.png"
MAP_FILE = OUTPUTS_DIR / f"qc_cluster_map_{STATE_NAME}.html"

# k-scan range widened 2..10 -> 2..12 to match Rajasthan.
K_RANGE_A = list(range(2, 13))

# N_BOOTSTRAP, RANDOM_STATE, GMM_COVARIANCE_TYPE ('diag') and the realistic
# silhouette band SILHOUETTE_LO/HI ([0.15, 0.35]) are imported from
# cluster_lib.py — one definition shared by both states and both clustering
# levels. See that module for the full reasoning behind each (in particular
# why 'diag' rather than 'full' covariance, and why the silhouette band is a
# cited expectation rather than an invented one).

# This run is Tamil Nadu ALONE, not the eventual multi-state design. Expect
# INTRA-state splitting here (Nilgiris hills vs. Chennai/Coromandel coast vs.
# interior dry Coimbatore belt), realistically k=2-4 — NOT the k=6-10
# expected once all four states are combined.
EXPECTED_K_RANGE_SINGLE_STATE = (2, 4)

# Optional manual override — leave None to use the auto-suggested k from
# suggest_k(); set an int to force a specific k after reviewing
# bic_selection_tamilnadu.csv. (The former hardcoded K_FINAL = 5 is gone.)
LEVEL_A_K_OVERRIDE = None


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


# ═══════════════════════════════════════════════════════════
# LEVEL A — SPATIAL CLUSTERING
# ═══════════════════════════════════════════════════════════

log_header(f"PHASE 4 — LEVEL A (spatial) — {STATE_NAME.title()}")

if not SIGNATURE_FILE.exists():
    raise SystemExit(f"ERROR: {SIGNATURE_FILE} not found — run "
                      f"04b_climate_signature.py first.")

sig = pd.read_csv(SIGNATURE_FILE)
sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)

z_cols_a = [c for c in sig.columns if c.endswith("_z")]
# lat/lon are never among the *_z columns by construction (04b excludes them
# from the standardized matrix entirely — see its NON_CLUSTERING_COLS) but
# assert it explicitly here too: a silent regression upstream would
# otherwise cluster geography, not climate.
assert not any(c in ("lat_z", "lon_z") for c in z_cols_a), \
    "lat/lon must never be standardized clustering columns — check the upstream 04b script"

print(f"\n  Points: {len(sig)}  |  Standardized (*_z) columns: {len(z_cols_a)}")
X_a = sig[z_cols_a].fillna(sig[z_cols_a].median()).values

print(f"\n[Level A 1/4] GMM (primary) + KMeans (baseline), "
      f"K={K_RANGE_A[0]}..{K_RANGE_A[-1]}, {N_BOOTSTRAP} bootstrap resamples/k ...")
bic_table_a = fit_k_range(X_a, K_RANGE_A)
bic_table_a.to_csv(BIC_TABLE_FILE, index=False)
print(f"  Saved: {BIC_TABLE_FILE}")

# KMeans-only companion table, kept for continuity with the previous Tamil
# Nadu output contract (the same numbers also live in BIC_TABLE_FILE).
bic_table_a[["k", "kmeans_silhouette"]].to_csv(KMEANS_TABLE_FILE, index=False)
print(f"  Saved: {KMEANS_TABLE_FILE}")

suggested_k_a, reason_a = suggest_k(bic_table_a, expected_range=EXPECTED_K_RANGE_SINGLE_STATE)
print(f"\n  Suggested k (auto, NOT forced): {suggested_k_a}  [{reason_a}]")
print(f"  Reminder: this is a Tamil-Nadu-ONLY run. Realistic k here is "
      f"{EXPECTED_K_RANGE_SINGLE_STATE[0]}-{EXPECTED_K_RANGE_SINGLE_STATE[1]} "
      f"(intra-state split — hills vs. coast vs. interior dry belt) — do NOT mistake "
      f"this for the eventual 4-state k (expected 6-10). Review "
      f"{BIC_TABLE_FILE.name} yourself; override via LEVEL_A_K_OVERRIDE at the top of "
      f"this script if you disagree with the auto-suggestion.")

k_final_a = LEVEL_A_K_OVERRIDE if LEVEL_A_K_OVERRIDE is not None else suggested_k_a
if LEVEL_A_K_OVERRIDE is not None:
    print(f"  LEVEL_A_K_OVERRIDE set — using k={k_final_a} instead of the auto-suggestion.")

# Pull the chosen k's bootstrap-ARI stats straight from bic_table_a (already
# computed above, not recomputed) so the number quoted in cluster_profiles/
# cluster_profile_cards always carries its effective_n_resamples alongside it.
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

# CANONICAL CLUSTER RELABELING — ported from Rajasthan's 2026-08-11 fix,
# applied BEFORE any output file is written. Relabels 0..k-1 by ascending
# mean latitude so "cluster 0" is the same physical regime across re-runs;
# see cluster_lib.canonical_relabel_by_latitude() for the full reasoning and
# the incident it guards against.
hard_labels_a, soft_probs_a, relabel_map_a = canonical_relabel_by_latitude(
    raw_labels_a, sig["lat"].values, raw_soft_probs_a)
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
# EXTERNAL VALIDATION — Köppen-Geiger
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
    ~0.0083 degrees, far finer than the spacing between this state's
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
  then re-run. Stubbed here rather than fabricated.
""")
    koppen_ari = koppen_nmi = None
    koppen_validation_meaningful = False
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

    # Degenerate-variance guard: an ARI/NMI against a near-constant label
    # set is not informative, so flag it rather than report a spuriously
    # low/high number as if it meant something.
    koppen_validation_meaningful = bool(dominant_frac < 0.95 and len(koppen_class_counts) > 1)
    if not koppen_validation_meaningful:
        print(f"\n  KOPPEN VALIDATION NOT MEANINGFUL AT STATE SCALE — "
              f"{dominant_frac * 100:.1f}% of sampled points fall into a single Koppen class "
              f"({koppen_class_counts.index[0]}); insufficient Koppen-class variance across "
              f"sampled points for ARI/NMI to be informative here.")

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
                  "Tamil Nadu. This is a plausible and legitimate finding in its own right — it "
                  "is arguably the point of empirical clustering instead of applying Koppen "
                  "directly — NOT evidence the clustering failed to find anything real.")
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
  climate-zone lookup (Tamil Nadu spans the Warm-Humid and Hot-Dry zones)
  as a shapefile/lookup table, join it to each point the same way as
  Koppen above, then compute ARI/NMI the same way. No local lookup exists
  in this project tree as of this run — stubbed, not fabricated.

  State-identity external check: NOT MEANINGFUL YET. This is a
  single-state (Tamil-Nadu-only) run — an ARI/NMI against "state identity"
  is only informative once >=2 states' Level-A results are combined.
""")


# ═══════════════════════════════════════════════════════════
# CLUSTER PROFILES + PROFILE CARDS
# ═══════════════════════════════════════════════════════════

log_header("Cluster profiles + profile cards (Level A)")

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
    population weights are missing/zero. The fallback itself is reasonable
    behavior; what is NOT acceptable is it being silent — every fallback is
    logged with the caller-supplied `context` label so it's traceable back
    to exactly which cluster-profile field was affected."""
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
    across-cluster median on four axes and composes a short phrase.
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

# Numeric per-cluster profile CSV — built alongside the markdown cards below
# so both read from the exact same weighted_mean() calls. PCM-facing columns
# are included explicitly since these are what Phase 5 needs per cluster.
PCM_FACING_COLS = [c for c in ["Tm_target_C", "Tm_target_capped_C", "L_required_kJ_per_kg"]
                    if c in sig.columns]
profile_rows = []
for cid in sorted(sig["cluster_id"].unique()):
    g = sig[sig["cluster_id"] == cid]
    row = {
        "cluster_id": int(cid), "n_points": len(g), "total_population": g["population"].sum(),
        # Chosen-k bootstrap stability and Koppen validation are properties of
        # the whole k-way solution, not per-cluster — repeated on every row so
        # they are always visible alongside the profile they support.
        "bootstrap_ari_mean_chosen_k": bootstrap_ari_mean_a,
        "bootstrap_effective_n_resamples_chosen_k": bootstrap_eff_n_a,
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
                     f"the {len(sig)} sampled points are overwhelmingly one Koppen class, so ARI="
                     f"{koppen_ari:.4f}/NMI={koppen_nmi:.4f} against Koppen labels are recorded "
                     f"for the record but carry no interpretive weight (see "
                     f"{KOPPEN_CONTINGENCY_FILE.name} for the full class distribution).")
else:
    _koppen_para = (f"Koppen-Geiger external validation: ARI={koppen_ari:.4f}, NMI={koppen_nmi:.4f} "
                     f"against Beck et al. (2018) present-climate classes for these same {len(sig)} "
                     f"points (see {KOPPEN_CONTINGENCY_FILE.name} for the full cluster x "
                     f"Koppen-class contingency table). " +
                     ("This is a relatively high agreement — the GMM clusters substantially "
                      "rediscover Koppen's existing boundaries within this state."
                      if koppen_ari >= 0.5 else
                      "This is a low-to-moderate agreement — the GMM is finding climate structure "
                      "at a finer resolution than Koppen's broad classes capture within Tamil "
                      "Nadu, which is a legitimate finding in its own right, not a failure of the "
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
    # not lat/lon) to this cluster's mean — the point most representative of
    # the cluster's CLIMATE signature, not just its geographic center.
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
                 f"(CEILING, not an achievability bar — see 04b_climate_signature.py's docstring)\n")

CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
CARDS_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"  Saved: {CARDS_FILE}  ({sig['cluster_id'].nunique()} cluster cards)")

if _weighted_mean_fallback_log:
    print(f"\n  weighted_mean() fell back to an UNWEIGHTED mean {len(_weighted_mean_fallback_log)} "
          f"time(s) — see WARNING lines above for exactly which cluster/column.")
else:
    print(f"\n  weighted_mean() fallback check: CONFIRMED CLEAN — 0 fallbacks across all "
          f"{sig['cluster_id'].nunique()} clusters x "
          f"{len(SIGNATURE_DISPLAY) + len(PCM_FACING_COLS)} columns.")


# ═══════════════════════════════════════════════════════════
# CLUSTER-LEVEL QC PLOTS
# (ported from Rajasthan, inline — no separate script; plus the two
#  improvements applied to BOTH states: bootstrap-ARI on the k-selection
#  curve, and population-scaled marker size on the cluster map.)
# ═══════════════════════════════════════════════════════════

log_header("Cluster-level QC plots")

# --- Folium cluster map: colour by cluster, OPACITY by membership
#     confidence, and (improvement 7d) marker SIZE by population.
center = [sig["lat"].mean(), sig["lon"].mean()]
m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=7)

palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#fabebe", "#008080", "#e6beff", "#9a6324"]
cluster_colors = {cid: palette[cid % len(palette)] for cid in range(k_final_a)}

# Marker radius scaled by sqrt(population) so AREA is proportional to
# population (the perceptually correct encoding), normalized to a 4-18 px
# range across this state's points.
_pop = sig["population"].fillna(0.0).clip(lower=0.0)
_pop_sqrt = np.sqrt(_pop)
_lo, _hi = float(_pop_sqrt.min()), float(_pop_sqrt.max())
RADIUS_MIN, RADIUS_MAX = 4.0, 18.0
if _hi > _lo:
    _radii = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * (_pop_sqrt - _lo) / (_hi - _lo)
else:
    _radii = pd.Series(np.full(len(sig), 7.0), index=sig.index)
# NB: no leading underscore — pandas' itertuples() renames any column whose
# name starts with "_" to a positional placeholder (_12, _13, ...), which
# would break the row.marker_radius access in the loop below.
sig["marker_radius"] = _radii

for row in sig.itertuples(index=False):
    color = cluster_colors[int(row.cluster_id)]
    # opacity scaled by max_membership_prob: confident points are solid,
    # ambiguous/transition points (genuinely split between regimes per the
    # GMM's soft assignment) fade out.
    opacity = 0.35 + 0.6 * float(row.max_membership_prob)
    popup = folium.Popup(
        f"<b>{row.point_id}</b><br>Cluster: {int(row.cluster_id)}<br>"
        f"Max membership prob: {row.max_membership_prob:.3f}<br>"
        f"Population: {row.population:,.0f}<br>"
        f"lat/lon: {row.lat:.3f}, {row.lon:.3f}",
        max_width=240,
    )
    folium.CircleMarker(
        location=[row.lat, row.lon], radius=float(row.marker_radius),
        color=color, weight=1, fill=True, fill_color=color,
        fill_opacity=opacity, popup=popup,
    ).add_to(m)

legend_html = f"""
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
            background: white; padding: 10px 14px; border: 1px solid #999;
            border-radius: 4px; font-size: 13px; line-height: 1.6;">
  <b>Level A clusters (k={k_final_a})</b><br>
  {''.join(f'<span style="color:{cluster_colors[c]}">&#9679;</span> Cluster {c}<br>' for c in range(k_final_a))}
  <span style="font-size:11px;color:#666">Marker SIZE = population (sqrt-scaled)<br>
  Faded points = low max membership<br>probability (near a regime boundary)</span>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))
m.save(str(MAP_FILE))
print(f"  Saved: {MAP_FILE}")

# --- Static PNG map, kept for continuity with the previous Tamil Nadu
#     output contract. Marker size also scaled by population.
fig_s, ax = plt.subplots(figsize=(8, 9))
ax.scatter(sig["lon"], sig["lat"], c=hard_labels_a, cmap="tab10",
           s=(sig["marker_radius"] ** 2) * 0.8, alpha=0.85,
           edgecolors="white", linewidths=0.6)
for cid in sorted(sig["cluster_id"].unique()):
    sub = sig[sig["cluster_id"] == cid]
    ax.annotate(f"C{cid}", (sub["lon"].mean(), sub["lat"].mean()),
                 fontsize=11, fontweight="bold", ha="center")
ax.set_title(f"Tamil Nadu Climate Regimes — GMM, k={k_final_a} (marker size ~ population)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(STATIC_MAP_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {STATIC_MAP_FILE}")

# --- K-selection curve: BIC (primary) + silhouette + (improvement 7c)
#     bootstrap-ARI, which is the ACTUAL tiebreaker in suggest_k()'s rule
#     and was previously not visualised anywhere. Chosen k marked.
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=bic_table_a["k"], y=bic_table_a["BIC"], mode="lines+markers",
                          name="BIC (lower is better)", line=dict(color="#4c72b0")),
              secondary_y=False)
fig.add_trace(go.Scatter(x=bic_table_a["k"], y=bic_table_a["silhouette"], mode="lines+markers",
                          name="GMM silhouette", line=dict(color="#dd8452")),
              secondary_y=True)
fig.add_trace(go.Scatter(x=bic_table_a["k"], y=bic_table_a["bootstrap_ari_mean"],
                          mode="lines+markers",
                          name="bootstrap-ARI (selection tiebreaker)",
                          line=dict(color="#55a868", dash="dot")),
              secondary_y=True)
fig.add_vline(x=k_final_a, line_dash="dash", line_color="green",
              annotation_text=f"chosen k={k_final_a}")
fig.add_hrect(y0=SILHOUETTE_LO, y1=SILHOUETTE_HI, secondary_y=True,
              fillcolor="green", opacity=0.08, line_width=0,
              annotation_text="realistic climate-zone silhouette band", annotation_position="top left")
fig.update_yaxes(title_text="BIC", secondary_y=False)
fig.update_yaxes(title_text="Silhouette / bootstrap-ARI", secondary_y=True)
fig.update_xaxes(title_text="k (number of clusters)")
fig.update_layout(title=f"Level A — K-Selection Curve (BIC + Silhouette + bootstrap-ARI) — "
                         f"{STATE_NAME.title()}")
kcurve_path = OUTPUTS_DIR / f"qc_k_selection_curve_{STATE_NAME}.html"
fig.write_html(str(kcurve_path))
print(f"  Saved: {kcurve_path}")

# --- Cluster-profile bar chart
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

# --- Population-share pie chart
fig = go.Figure(data=go.Pie(
    labels=[f"Cluster {int(c)}" for c in cluster_profiles["cluster_id"]],
    values=cluster_profiles["total_population"], hole=0.35,
))
fig.update_layout(title=f"Level A — Population Share by Climate Regime — {STATE_NAME.title()}")
pop_pie_path = OUTPUTS_DIR / f"qc_cluster_population_share_{STATE_NAME}.html"
fig.write_html(str(pop_pie_path))
print(f"  Saved: {pop_pie_path}")


log_header("PHASE 4 LEVEL A COMPLETE")
print(f"  k={k_final_a} clusters, {len(sig)} points")
print(f"  bootstrap-ARI at chosen k: {bootstrap_ari_mean_a:.4f} "
      f"({bootstrap_eff_n_a}/{N_BOOTSTRAP} resamples)")
if koppen_ari is not None:
    print(f"  Koppen ARI={koppen_ari:.4f}  NMI={koppen_nmi:.4f}  "
          f"(meaningful={koppen_validation_meaningful})")
for _, row in cluster_profiles.iterrows():
    print(f"    Cluster {int(row['cluster_id'])}: {int(row['n_points'])} points  "
          f"GHI_daily_kWh~{row.get('GHI_daily_kWh', float('nan')):.2f}  "
          f"Ta_mean~{row.get('Ta_mean', float('nan')):.1f}C  "
          f"population~{row['total_population']:,.0f}")
print(f"\n  Outputs: {BIC_TABLE_FILE.name}, {ASSIGN_A_FILE.name}, {PROFILE_FILE.name}, "
      f"{CARDS_FILE.name}, {KOPPEN_CONTINGENCY_FILE.name}, {MAP_FILE.name}")
print("\nNext: 05a_level_b_regime_shift_tamilnadu.py (Level B — regime shift), "
      "then Phase 5 (07_feasibility_filter.py).")
