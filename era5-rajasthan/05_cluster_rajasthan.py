"""
05_cluster_rajasthan.py
=============================================================================
PHASE 4 — LEVEL A: CLIMATE REGIME CLUSTERING (spatial), RAJASTHAN
(Objective1_PCM_Climate_Framework_Plan_v3, §6.2/§7)

LEVEL A — spatial: one signature vector per point (whole 10-year record),
reading climate_signature_rajasthan.csv's *_z columns directly.

LEVEL B MOVED OUT (2026-09-08): the per-point-per-season re-clustering that
used to live inline at the bottom of this file is now
`05a_level_b_regime_shift_rajasthan.py`. It was extracted so that both
states have the same structure and so the two analyses that were BOTH
called "Level B" stop colliding:
  * 05a_level_b_regime_shift_rajasthan.py — "Level B — Regime Shift", a
    genuine Phase-4 clustering step (fresh Tier-1 build per point per
    season, fresh GMM, regime-shift fraction, season-tautology check).
  * 11_seasonal_pcm_sensitivity.py — "Seasonal PCM Sensitivity",
    a post-Phase-6 re-ranking of the PCM shortlist per (cluster, season).
    Not a clustering step; no longer carries the "level_b" name.

The shared Phase-4 machinery (bootstrap-ARI stability, the 3-tier
`suggest_k` cascade, the GMM+KMeans k-scan, canonical latitude relabeling)
now lives in `cluster_lib.py`, imported by this script, by 05a, and by both
of Tamil Nadu's equivalents — one implementation, four callers.

STATE-AGNOSTIC BY DESIGN: STATE_NAME below is the only place a state name
is hardcoded. Every output filename is built from it.

INPUTS:
  data/processed/climate_signature_rajasthan.csv  (04b_climate_signature.py's
      output — Level A reads its *_z columns directly)

OUTPUTS:
  data/processed/bic_selection_rajasthan.csv                  (Level A,
      full k=2..12 metric table: BIC/AIC/silhouette/DB/CH/bootstrap ARI +
      bootstrap_effective_n_resamples/KMeans silhouette)
  data/processed/cluster_assignments_rajasthan_levelA.csv     (point_id,
      canonically-relabeled hard cluster label, soft membership
      probabilities, chosen k, chosen-k bootstrap-ARI mean +
      effective_n_resamples)
  data/processed/cluster_profiles_rajasthan.csv                (per-cluster
      profile, also carrying chosen-k bootstrap-ARI stats and
      koppen_ari/koppen_nmi/koppen_validation_meaningful)
  data/processed/koppen_validation_rajasthan.csv               (cluster_id
      x koppen_class contingency counts)
  outputs/cluster_profile_cards_rajasthan.md                   (one card
      per Level-A cluster)
  outputs/qc_cluster_map_rajasthan.html                        (folium —
      colour by cluster, opacity by max membership probability, and — new
      2026-09-08 — marker SIZE by population)
  outputs/qc_k_selection_curve_rajasthan.html                  (BIC +
      silhouette + — new 2026-09-08 — bootstrap-ARI vs k, chosen k marked;
      bootstrap-ARI is the actual tiebreaker in suggest_k()'s rule and was
      previously not visualised anywhere)
  outputs/qc_cluster_profile_bars_rajasthan.html               (headline
      signature indices by cluster)
  outputs/qc_cluster_population_share_rajasthan.html           (population
      share by cluster)

EXTERNAL VALIDATION: Koppen-Geiger (Beck et al. 2018, doi:10.1038/
sdata.2018.214) is WIRED IN (FIX 2026-08-11) — see data/raw/koppen/ and
the EXTERNAL VALIDATION section below. NBC/ECBC still has no local lookup
in this project and remains stubbed, not fabricated/approximated.

REQUIRED LIBRARIES (install if missing):
  pip install pandas numpy scikit-learn folium plotly rasterio

HOW TO RUN:
  python 05_cluster_rajasthan.py
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
# Every output path is built from this.
# ═══════════════════════════════════════════════════════════
STATE_NAME = "rajasthan"

SIGNATURE_FILE = PROCESSED_DIR / f"climate_signature_{STATE_NAME}.csv"

BIC_TABLE_FILE = PROCESSED_DIR / f"bic_selection_{STATE_NAME}.csv"
ASSIGN_A_FILE = PROCESSED_DIR / f"cluster_assignments_{STATE_NAME}_levelA.csv"

# LEVEL B lives in its own script now — see
# 05a_level_b_regime_shift_rajasthan.py (extracted 2026-09-08). Its outputs
# (bic_selection_rajasthan_levelB.csv, cluster_assignments_rajasthan_levelB
# .csv, level_b_feature_importance/season_tautology/season_contingency
# _rajasthan.csv, plus the new regime-shift Sankey) are produced there, not
# here. Nothing in this script reads them.

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

# N_BOOTSTRAP, RANDOM_STATE, GMM_COVARIANCE_TYPE ('diag') and the realistic
# silhouette band SILHOUETTE_LO/HI ([0.15, 0.35]) are imported from
# cluster_lib.py — one definition shared by both states and both clustering
# levels. The long-form reasoning for each (why 'diag' rather than 'full'
# covariance after the 2026-08-10 membership-saturation incident; why the
# silhouette band is a cited expectation rather than an invented one) now
# lives in that module's docstrings.

# This run is Rajasthan ALONE, not yet the eventual 4-state combined
# design (Rajasthan + Assam + Tamil Nadu + Uttarakhand) this framework is
# ultimately built for. Expect INTRA-state splitting here (the framework
# doc names arid-west vs semi-arid-east as the specific Rajasthan split),
# realistically k=2-4 — NOT the k=6-10 expected once all four states are
# combined. Do not mistake this run's chosen k for the eventual
# multi-state k when that run happens.
EXPECTED_K_RANGE_SINGLE_STATE = (2, 4)

# Optional manual override — leave None to use the auto-suggested k from
# cluster_lib.suggest_k(); set an int to force a specific k after reviewing
# bic_selection_rajasthan.csv. (Level B has its own override in
# 05a_level_b_regime_shift_rajasthan.py.)
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
bic_table_a = fit_k_range(X_a, K_RANGE_A)
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
# Phase 7 (10_physics_validation.py) caught Phase 5's and
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

# Marker radius scaled by sqrt(population) so marker AREA is proportional to
# population (the perceptually correct encoding), normalized to a 4-18 px
# range across this state's points. Added 2026-09-08: a fixed radius=6 gave
# every point equal visual weight, which badly misrepresents a
# population-weighted sample — a cluster of a few very large cities and a
# cluster of many small towns looked identical.
_pop = sig["population"].fillna(0.0).clip(lower=0.0)
_pop_sqrt = np.sqrt(_pop)
_lo, _hi = float(_pop_sqrt.min()), float(_pop_sqrt.max())
RADIUS_MIN, RADIUS_MAX = 4.0, 18.0
if _hi > _lo:
    _radii = RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * (_pop_sqrt - _lo) / (_hi - _lo)
else:
    _radii = pd.Series(np.full(len(sig), 7.0), index=sig.index)
# NB: no leading underscore — pandas' itertuples() renames any column whose
# name starts with "_" to a positional placeholder, breaking row.marker_radius.
sig["marker_radius"] = _radii

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
        f"Population: {row.population:,.0f}<br>"
        f"lat/lon: {row.lat:.3f}, {row.lon:.3f}",
        max_width=240,
    )
    folium.CircleMarker(
        location=[row.lat, row.lon], radius=float(row.marker_radius),
        color=color, weight=1,
        fill=True, fill_color=color, fill_opacity=opacity, popup=popup,
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

# H. K-selection curve — BIC (primary selection criterion), silhouette
#    (secondary/interpretability check), and bootstrap-ARI vs k, Level A,
#    with the auto-suggested k marked. Lets the k choice be checked by eye
#    against the same table suggest_k() already used numerically.
#    bootstrap-ARI added 2026-09-08: it is the ACTUAL tiebreaker in
#    suggest_k()'s tier-1 and tier-2 rules (among the k values that clear
#    the silhouette band, the highest bootstrap-ARI wins), yet it was the
#    one input to that decision not plotted anywhere.
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


log_header("PHASE 4 LEVEL A COMPLETE")
print(f"  k={k_final_a} clusters, {len(sig)} points")
print(f"  bootstrap-ARI at chosen k: {bootstrap_ari_mean_a:.4f} "
      f"({bootstrap_eff_n_a}/{N_BOOTSTRAP} resamples)")
if koppen_ari is not None:
    print(f"  Koppen ARI={koppen_ari:.4f}  NMI={koppen_nmi:.4f}  "
          f"(meaningful={koppen_validation_meaningful})")
print(f"  Outputs: {BIC_TABLE_FILE.name}, {ASSIGN_A_FILE.name}, {PROFILE_FILE.name}, "
      f"{CARDS_FILE.name}, {KOPPEN_CONTINGENCY_FILE.name}, {MAP_FILE.name}")
print("\nNext: 05a_level_b_regime_shift_rajasthan.py (Level B — regime shift), "
      "then Phase 5 (07_feasibility_filter.py).")
