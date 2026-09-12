"""
05a_level_b_regime_shift_rajasthan.py
=============================================================================
PHASE 4 — LEVEL B: TEMPORAL (SEASONAL) REGIME-SHIFT CLUSTERING, RAJASTHAN
(Objective1_PCM_Climate_Framework_Plan_v3, §7.2)

EXTRACTED FROM 05_cluster_rajasthan.py (2026-09-08). This analysis used to
live inline at the bottom of the Level-A script; it is now its own script so
that both states have the same structure and so the two analyses that were
BOTH called "Level B" stop colliding:

  * THIS script — "Level B — Regime Shift". A genuine Phase-4 CLUSTERING
    step: rebuild a Tier-1 climate signature per point PER SEASON, fit a
    fresh GMM on it, and ask whether a point's PCM-relevant climate regime
    changes materially across the year.
  * 11_seasonal_pcm_sensitivity.py — "Seasonal PCM Sensitivity".
    A post-Phase-6 analysis: re-rank the PCM shortlist per (cluster, season)
    using Phase 6's MCDM weights and count how often the #1 PCM flips. It is
    NOT a clustering step and no longer carries the "level_b" name.

Level B rebuilds its input from scratch via
`signature_lib.build_tier1_signature(group_keys=["point_id", "season"])` —
the SAME index formulas Level A uses, just grouped differently — rather than
reading any saved Level-A file. It therefore has no Tier 2, no PCA and no
interaction terms, only the 19 raw Tier-1 columns, freshly standardized with
its own independent StandardScaler.

NOTE ON SILHOUETTE COMPARABILITY: expect Level B to run HIGHER than Level
A's 0.15-0.35 band, and do NOT read that as "better clustering". A given
point's summer-vs-winter swing is a stronger, more artificial signal than
genuine spatial climate variation between points, so a high Level-B
silhouette is uninformative either way. The two levels are not on a
comparable silhouette scale. The season-tautology check (step 5/5) is the
actual test of whether Level B found anything beyond "season" — that, not
the silhouette curve, is what should drive the k choice here.

INPUTS:
  data/processed/climate_rajasthan_points.csv  (per-point-per-season Tier 1
      is rebuilt from this directly — it already carries a `season` column)
  data/processed/suntimes.csv                   (daylength; no season column,
      so it is derived via signature_lib.attach_season)

OUTPUTS:
  data/processed/bic_selection_rajasthan_levelB.csv
  data/processed/cluster_assignments_rajasthan_levelB.csv
  data/processed/level_b_feature_importance_rajasthan.csv
  data/processed/level_b_season_tautology_rajasthan.csv
  data/processed/level_b_season_contingency_rajasthan.csv
  outputs/qc_level_b_regime_shift_sankey_rajasthan.html   (NEW 2026-09-08 —
      alluvial/Sankey of every point's cluster membership across the 4
      seasons, with shifting flows highlighted against non-shifting ones)

REQUIRED LIBRARIES:
  pip install pandas numpy scikit-learn plotly

HOW TO RUN:
  python 05a_level_b_regime_shift_rajasthan.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.feature_selection import f_classif
import plotly.graph_objects as go

from config import (
    COMBINED_POINTS_FILE, SUNTIMES_FILE, PROCESSED_DIR, OUTPUTS_DIR,
    ensure_data_dirs,
)
from signature_lib import EVENT_ORDER, SEASON_ORDER, attach_season, build_tier1_signature
from cluster_lib import (
    GMM_COVARIANCE_TYPE, N_BOOTSTRAP, RANDOM_STATE,
    fit_k_range, suggest_k,
)

ensure_data_dirs()

# ═══════════════════════════════════════════════════════════
# STATE NAME — the only hardcoded state reference in this file.
# Every output path is built from it.
# ═══════════════════════════════════════════════════════════
STATE_NAME = "rajasthan"
OUT_DIR = PROCESSED_DIR

K_RANGE_B = list(range(2, 9))    # 2..8

# Level B feature-set ablation control — a reusable, documented robustness
# check, not a one-off hack. Default ([], "full") reproduces the primary
# Level B run on all 19 Tier 1 columns and writes to the canonical
# cluster_assignments_{state}_levelB.csv. Set LEVEL_B_EXCLUDE_FEATURES to
# drop specific columns for a comparison run (e.g. daylength_mean/
# daylength_amplitude, which carry zero climatic content by construction —
# they are a deterministic function of latitude/day-of-year, fully
# determined before any weather happens) and set LEVEL_B_RUN_TAG to
# something descriptive; the output filename picks up that tag
# automatically so an ablation run never overwrites the primary one. Re-run
# the SAME season-tautology checks on both for a clean before/after
# comparison — that comparison is itself worth reporting as a robustness
# check, not just the ablation's raw numbers.
LEVEL_B_EXCLUDE_FEATURES = []
LEVEL_B_RUN_TAG = "full"

_suffix = "" if not LEVEL_B_EXCLUDE_FEATURES else f"_ablation_{LEVEL_B_RUN_TAG}"

BIC_TABLE_B_FILE = OUT_DIR / f"bic_selection_{STATE_NAME}_levelB{_suffix}.csv"
ASSIGN_B_FILE = OUT_DIR / f"cluster_assignments_{STATE_NAME}_levelB{_suffix}.csv"
FEATURE_IMPORTANCE_FILE = OUT_DIR / f"level_b_feature_importance_{STATE_NAME}{_suffix}.csv"
SEASON_TAUTOLOGY_FILE = OUT_DIR / f"level_b_season_tautology_{STATE_NAME}{_suffix}.csv"
CONTINGENCY_FILE = OUT_DIR / f"level_b_season_contingency_{STATE_NAME}{_suffix}.csv"
SANKEY_FILE = OUTPUTS_DIR / f"qc_level_b_regime_shift_sankey_{STATE_NAME}.html"

# Optional manual override — leave None to use suggest_k()'s auto-suggestion.
LEVEL_B_K_OVERRIDE = None


def log_header(title):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


log_header(f"PHASE 4 — LEVEL B (temporal/seasonal regime shift) — {STATE_NAME.title()}")

# ═══════════════════════════════════════════════════════════
# 1. LOAD + REBUILD PER-POINT-PER-SEASON TIER 1
# ═══════════════════════════════════════════════════════════

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
# SEASON_MAP as 02_combine; see signature_lib.attach_season). events_df_b's
# own "season" column (already in climate_*_points.csv) is used directly
# rather than re-derived, so the actual climate data stays from one source
# of truth; only suntimes.csv needs the derivation.
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


# ═══════════════════════════════════════════════════════════
# 2. K-SCAN + FINAL FIT
# ═══════════════════════════════════════════════════════════

print(f"\n[Level B 3/5] GMM (primary) + KMeans (baseline), "
      f"K={K_RANGE_B[0]}..{K_RANGE_B[-1]}, {N_BOOTSTRAP} bootstrap resamples/k ...")
bic_table_b = fit_k_range(X_b, K_RANGE_B)
bic_table_b.to_csv(BIC_TABLE_B_FILE, index=False)
print(f"  Saved: {BIC_TABLE_B_FILE}")

suggested_k_b, reason_b = suggest_k(bic_table_b, expected_range=None)
print(f"\n  Suggested k (auto, NOT forced): {suggested_k_b}  [{reason_b}]")
if reason_b.startswith("FALLBACK"):
    print("  (Falling back to lowest-BIC k is expected/fine here, per the silhouette-"
          "comparability note in this script's docstring — the 0.15-0.35 band was "
          "never validated for Level B's task, so failing to land in it is not "
          "itself informative.)")

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


# ═══════════════════════════════════════════════════════════
# 3. REGIME-SHIFT ANALYSIS
# ═══════════════════════════════════════════════════════════

print("\n[Level B 4/5] Regime-shift analysis (does a point's seasonal cluster "
      "label change across the year?) ...")
shift_table = assign_b.pivot_table(index="point_id", columns="season",
                                    values="cluster_id", aggfunc="first")
n_shifting = int((shift_table.nunique(axis=1) > 1).sum())
n_total_pts = len(shift_table)
regime_shift_fraction = n_shifting / n_total_pts
print(f"  {n_shifting}/{n_total_pts} points ({100*regime_shift_fraction:.1f}%) "
      f"have a DIFFERENT seasonal cluster label in at least one season — "
      f"i.e. their PCM-relevant climate regime shifts materially within the year.")
print("  This is a SEPARATE result from Level A (spatial regimes) — report both, "
      "don't merge them. Read this number together with step 5/5 below before "
      "calling it a finding: if the clusters map ~1:1 onto the 4 seasons, a "
      "high shifting fraction is expected/tautological (points trivially look "
      "different in different seasons), not evidence of a richer regime structure.")


# ═══════════════════════════════════════════════════════════
# 4. SEASON-TAUTOLOGY CHECK
# ═══════════════════════════════════════════════════════════

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

# 2. ARI / NMI between cluster label and season label directly. ARI near
# 1.0 means the clustering IS season, nothing more; a meaningfully lower
# ARI with visible cross-season merging in the table above is the actual
# finding worth reporting.
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

# 3. Which features actually drive the split? One-way ANOVA F-statistic per
# Tier-1 feature against hard_labels_b — high-F features are what's
# separating the clusters. If the top features are all temperature/GHI (the
# season-DEFINING variables), that supports the tautology reading; if
# humidity, wind, or daylength also rank highly, that's evidence Level B
# captures something beyond season even where the cluster-season
# correspondence looks strong.
f_stats, p_vals = f_classif(X_b, hard_labels_b)
feature_importance = pd.DataFrame({
    "feature": tier1_b_cols, "F_statistic": f_stats, "p_value": p_vals,
}).sort_values("F_statistic", ascending=False)
print(f"\n  Feature importance (ANOVA F-statistic across the k={k_final_b} clusters, "
      f"highest first):")
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

feature_importance_out = feature_importance.copy()
feature_importance_out["chosen_k_levelB"] = k_final_b
feature_importance_out.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
print(f"\n  Saved: {FEATURE_IMPORTANCE_FILE}")

season_tautology_summary = pd.DataFrame([{
    "chosen_k_levelB": k_final_b,
    "n_shifting_points": n_shifting,
    "n_total_points": n_total_pts,
    "regime_shift_fraction": regime_shift_fraction,
    "season_ari": season_ari,
    "season_nmi": season_nmi,
    "bootstrap_ari_mean_chosen_k": bootstrap_ari_mean_b,
    "bootstrap_effective_n_resamples_chosen_k": bootstrap_eff_n_b,
}])
season_tautology_summary.to_csv(SEASON_TAUTOLOGY_FILE, index=False)
print(f"  Saved: {SEASON_TAUTOLOGY_FILE}")

contingency.to_csv(CONTINGENCY_FILE)
print(f"  Saved: {CONTINGENCY_FILE}")


# ═══════════════════════════════════════════════════════════
# 5. REGIME-SHIFT SANKEY / ALLUVIAL  (added 2026-09-08)
# ═══════════════════════════════════════════════════════════
# The regime-shift FRACTION above is a single scalar; it says how many
# points move but nothing about WHERE they move between. This alluvial
# shows every point's cluster membership across the four seasons in
# order, so the actual structure of the shift is visible: which regime
# pairs exchange points, whether the movement is one big two-way swap or a
# broad reshuffle, and which flows stay put. Non-shifting flows are drawn
# grey and shifting flows orange, so the shifting fraction is legible as
# coloured ribbon volume rather than only as a number.

print("\nRegime-shift Sankey (cluster membership across the 4 seasons) ...")

seasons_present = [s for s in SEASON_ORDER if s in shift_table.columns]

# Node index: one node per (season, cluster) pair, laid out left-to-right
# in calendar order.
node_labels, node_x, node_colors = [], [], []
node_index = {}
palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c"]
for si, season in enumerate(seasons_present):
    for cid in range(k_final_b):
        node_index[(season, cid)] = len(node_labels)
        node_labels.append(f"{season} · C{cid}")
        node_x.append(si / max(1, len(seasons_present) - 1))
        node_colors.append(palette[cid % len(palette)])

link_source, link_target, link_value, link_color, link_label = [], [], [], [], []
for si in range(len(seasons_present) - 1):
    s_from, s_to = seasons_present[si], seasons_present[si + 1]
    pair = shift_table[[s_from, s_to]].dropna()
    flows = pair.groupby([s_from, s_to]).size()
    for (c_from, c_to), n in flows.items():
        link_source.append(node_index[(s_from, int(c_from))])
        link_target.append(node_index[(s_to, int(c_to))])
        link_value.append(int(n))
        shifted = int(c_from) != int(c_to)
        # orange = this point's regime CHANGED between these two seasons;
        # grey = stayed in the same regime.
        link_color.append("rgba(245,130,49,0.55)" if shifted else "rgba(160,160,160,0.28)")
        link_label.append(f"{'SHIFT' if shifted else 'stable'}: "
                          f"C{int(c_from)} -> C{int(c_to)} ({n} points)")

fig = go.Figure(data=[go.Sankey(
    arrangement="snap",
    node=dict(label=node_labels, x=node_x, color=node_colors,
              pad=14, thickness=16,
              line=dict(color="white", width=0.5)),
    link=dict(source=link_source, target=link_target, value=link_value,
              color=link_color, label=link_label),
)])
fig.update_layout(
    title=(f"Level B — Seasonal Regime Shift, {STATE_NAME.title()} "
           f"(k={k_final_b}; {n_shifting}/{n_total_pts} points = "
           f"{100*regime_shift_fraction:.1f}% shift regime in at least one season)"
           f"<br><sub>Orange ribbons = the point's regime changed between those two "
           f"seasons; grey = stayed put. ARI(cluster, season)={season_ari:.3f}</sub>"),
    font=dict(size=12), height=680,
)
fig.write_html(str(SANKEY_FILE))
print(f"  Saved: {SANKEY_FILE}")


log_header(f"PHASE 4 LEVEL B COMPLETE — {STATE_NAME.title()}")
print(f"  k={k_final_b} seasonal clusters over {len(tier1_b)} point-season rows")
print(f"  bootstrap-ARI at chosen k: {bootstrap_ari_mean_b:.4f} "
      f"({bootstrap_eff_n_b}/{N_BOOTSTRAP} resamples)")
print(f"  Regime shift: {n_shifting}/{n_total_pts} points "
      f"({100*regime_shift_fraction:.1f}%)")
print(f"  Season tautology: ARI={season_ari:.3f}  NMI={season_nmi:.3f}")
print(f"  Outputs: {BIC_TABLE_B_FILE.name}, {ASSIGN_B_FILE.name}, "
      f"{FEATURE_IMPORTANCE_FILE.name}, {SEASON_TAUTOLOGY_FILE.name}, "
      f"{CONTINGENCY_FILE.name}, {SANKEY_FILE.name}")
