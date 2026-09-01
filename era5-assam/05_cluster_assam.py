"""
05_cluster_assam.py
=========================
PHASE 4 — CLIMATE REGIME CLUSTERING, ASSAM (Level A — Spatial)

WHY LEVEL A FIRST
------------------
Level A clusters the population-grid sites spatially: one 18-index
climate signature vector per site → GMM → k climate regimes.
This answers the primary objective: "which PCM for a system installed
at location X in Assam?"

Assam spans a remarkable climate gradient despite being a single state:
  • Brahmaputra valley floodplain  — hot-humid, extreme monsoon (>2500 mm/yr)
  • Barak valley (south)           — slightly drier, warmer winters
  • Hill districts (Karbi Anglong, Dima Hasao) — cooler, higher elevation
  • Char islands (riverine)        — high flood risk, distinct thermal profile

A data-driven k of 3–5 that partially reproduces these geography-driven
differences is a strong quotable result. This script uses Gaussian Mixture
(full covariance) as the primary algorithm — climate boundaries in Assam
are gradients, not hard lines, and soft membership matters most for the
Brahmaputra valley fringe where two regimes genuinely overlap.

WHY GMM OVER K-MEANS (§7.2)
-----------------------------
1. Soft membership — boundary sites get a weighted PCM ranking.
2. Full covariance — Assam's indices are correlated (monsoon_index with
   RH_mean, GHI with CCI). K-Means would split elongated clusters.
3. BIC gives a principled k — no subjective elbow interpretation.

K-Means is still run as a robustness check (§7.2 spec).

BOOTSTRAP STABILITY (§7.3)
----------------------------
500 resamplings with replacement; reports the adjusted Rand index (ARI)
against the full-data GMM solution. ARI > 0.75 = stable partition.
This is the criterion most papers omit and reviewers ask for (§7.3).

REPRODUCIBILITY (§7.5)
------------------------
random_state is set on every fit. The fitted StandardScaler and GMM are
saved with joblib. The scikit-learn version is recorded in the output CSV.
"Unsupervised results that cannot be regenerated exactly are not results."

INPUT  : data/processed/climate_signatures_matrix.csv
         (04b_climate_signature.py output — already standardised;
          all numeric columns except point_id/lat/lon/population
          are the feature set)
OUTPUT : data/processed/clustering/
           bic_selection_assam.csv
           kmeans_comparison_assam.csv
           bootstrap_stability_assam.csv
           cluster_assignments_assam.csv   ← soft membership per point
           cluster_profiles_assam.csv      ← population-weighted profiles
                                              → feed into Phase 5 PCM filter
           scaler_assam.joblib             ← reproducibility
           gmm_model_assam.joblib          ← reproducibility
         data/plots/
           cluster_map_assam.png

HOW TO RUN:
  python 05_cluster_assam.py

  After the first run, review bic_selection_assam.csv, set K_FINAL
  below, then re-run to produce the final cluster assignments.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import joblib

from config import PROCESSED_DIR, PLOTS_DIR

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
SIGNATURE_FILE = PROCESSED_DIR / "climate_signatures_matrix.csv"
OUT_DIR        = PROCESSED_DIR / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# TUNING PARAMETERS  (review bic_selection_assam.csv, then set K_FINAL)
# ─────────────────────────────────────────────────────────────
K_CANDIDATES = list(range(2, 11))
K_FINAL      = 4          # ← set after reviewing bic_selection_assam.csv, re-run

# Assam single-state band — similar to TN spec but Assam's monsoon
# dominance makes within-cluster distances shorter → can be tighter.
SILHOUETTE_LO  = 0.15
SILHOUETTE_HI  = 0.45

N_BOOTSTRAP    = 500      # §7.3 bootstrap stability resamplings
RANDOM_STATE   = 42       # fixed for reproducibility §7.5

# ─────────────────────────────────────────────────────────────
# NON-FEATURE COLUMNS (excluded from the clustering feature matrix)
# ─────────────────────────────────────────────────────────────
NON_FEATURE_COLS = {
    "point_id", "lat", "lon", "population",
    "Tm_target", "L_required_kWh", "T_mains_est",
    "Ta_mean", "Ta_p95", "Ta_p05",
    "HDD18", "CDD24", "RH_mean", "elev_proxy",
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def detect_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are part of the clustering feature set."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(weights.median())
    if w.sum() == 0:
        return series.mean()
    return np.average(series, weights=w)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  Phase 4 — Climate Regime Clustering (GMM, Level A) — Assam")
    print(f"  scikit-learn {sklearn.__version__}  |  random_state={RANDOM_STATE}")
    print("=" * 68)

    if not SIGNATURE_FILE.exists():
        print(f"\n  ERROR: {SIGNATURE_FILE} not found.")
        print("  Run 04b_climate_signature.py first.")
        return

    # ── Load ─────────────────────────────────────────────────
    sig = pd.read_csv(SIGNATURE_FILE)
    # Normalise the first column name to point_id
    sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)

    # Load lat/lon/population metadata and merge it in
    grid_file = PROCESSED_DIR / "population_grid_points.csv"
    if grid_file.exists():
        grid = pd.read_csv(grid_file)
        if "point_id" in grid.columns:
            sig = sig.merge(grid[["point_id", "lat", "lon", "population"]], on="point_id", how="left")
    
    feat_cols = detect_feature_cols(sig)
    print(f"\n  Sites loaded    : {len(sig)}")
    print(f"  Feature columns : {len(feat_cols)}")
    print(f"  Features        : {feat_cols}")
    print("  (lat/lon excluded — never cluster on geography)")

    # The matrix from 04b is already standardised. We keep it as-is
    # for clustering, but also save/reload the scaler for reproducibility
    # in downstream scripts that need to transform new points.
    X = sig[feat_cols].fillna(sig[feat_cols].median()).values

    # Save an identity-scaler placeholder (already scaled by 04b) so
    # downstream scripts can call scaler.transform() uniformly.
    scaler = StandardScaler()
    scaler.fit(X)           # fit on already-standardised data (mean≈0, std≈1)
    joblib.dump(scaler, OUT_DIR / "scaler_assam.joblib")
    print(f"\n  Scaler saved: {OUT_DIR / 'scaler_assam.joblib'}")

    k_safe = [k for k in K_CANDIDATES if k < len(X)]

    # ── Step 1/5 — BIC + three metrics across k ───────────────
    print(f"\n[1/5] BIC + Silhouette + Davies-Bouldin + Calinski-Harabasz, "
          f"K={k_safe[0]}..{k_safe[-1]} ...")

    rows = []
    for k in k_safe:
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            random_state=RANDOM_STATE, n_init=5
        )
        labels = gmm.fit_predict(X)
        bic    = gmm.bic(X)
        n_unique = len(set(labels))
        if n_unique > 1:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            ch  = calinski_harabasz_score(X, labels)
        else:
            sil = db = ch = float("nan")

        in_band = (SILHOUETTE_LO <= sil <= SILHOUETTE_HI) if sil == sil else False
        rows.append({
            "k": k, "BIC": bic,
            "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch,
            "in_accept_band": in_band,
            "sklearn_version": sklearn.__version__,
        })
        flag = "  ← in accept band" if in_band else ""
        print(f"    K={k:2d}  BIC={bic:11.1f}  sil={sil:.4f}  "
              f"DB={db:.3f}  CH={ch:8.1f}{flag}")

    k_table = pd.DataFrame(rows)
    bic_path = OUT_DIR / "bic_selection_assam.csv"
    k_table.to_csv(bic_path, index=False)
    best_bic_k = int(k_table.loc[k_table["BIC"].idxmin(), "k"])
    in_band_ks = k_table[k_table["in_accept_band"]]["k"].tolist()
    print(f"\n  Saved: {bic_path}")
    print(f"  Lowest-BIC K   : {best_bic_k}")
    print(f"  In silhouette band [{SILHOUETTE_LO}–{SILHOUETTE_HI}]: {in_band_ks}")
    print(f"  K_FINAL is currently {K_FINAL} — update after reviewing the table, then re-run.")

    # ── Step 2/5 — K-Means comparison ────────────────────────
    print("\n[2/5] K-Means comparison (reported only; GMM remains primary) ...")
    km_rows = []
    for k in k_safe:
        km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels) if len(set(labels)) > 1 else float("nan")
        km_rows.append({"k": k, "kmeans_silhouette": sil})
    km_table = pd.DataFrame(km_rows)
    km_path  = OUT_DIR / "kmeans_comparison_assam.csv"
    km_table.to_csv(km_path, index=False)
    print(km_table.to_string(index=False))
    print(f"  Saved: {km_path}")
    print("  (K-Means silhouette vs GMM silhouette quantifies the benefit of "
          "soft/full-covariance clustering for Assam's correlated index space.)")

    # ── Step 3/5 — Bootstrap stability (§7.3) ─────────────────
    k_final_safe = min(K_FINAL, len(X) - 1)
    print(f"\n[3/5] Bootstrap stability — {N_BOOTSTRAP} resamplings at K={k_final_safe} ...")

    # Reference partition on full data
    gmm_ref = GaussianMixture(
        n_components=k_final_safe, covariance_type="full",
        random_state=RANDOM_STATE, n_init=10
    )
    labels_ref = gmm_ref.fit_predict(X)

    rng  = np.random.default_rng(RANDOM_STATE)
    aris = []
    for b in range(N_BOOTSTRAP):
        idx  = rng.choice(len(X), size=len(X), replace=True)
        X_b  = X[idx]
        gmm_b = GaussianMixture(
            n_components=k_final_safe, covariance_type="full",
            random_state=int(rng.integers(0, 100_000)), n_init=3
        )
        labs_b_boot = gmm_b.fit_predict(X_b)
        # ARI against the labels the full model would assign to the same subset
        labs_ref_subset = labels_ref[idx]
        aris.append(adjusted_rand_score(labs_ref_subset, labs_b_boot))
        if (b + 1) % 100 == 0:
            print(f"    {b+1}/{N_BOOTSTRAP}  running ARI mean={np.mean(aris):.4f}")

    ari_mean, ari_std = np.mean(aris), np.std(aris)
    stable = ari_mean >= 0.75
    boot_df = pd.DataFrame({
        "k_final": [k_final_safe],
        "n_bootstrap": [N_BOOTSTRAP],
        "ARI_mean": [ari_mean],
        "ARI_std": [ari_std],
        "stable": [stable],
        "sklearn_version": [sklearn.__version__],
    })
    boot_path = OUT_DIR / "bootstrap_stability_assam.csv"
    boot_df.to_csv(boot_path, index=False)
    verdict = "STABLE (ARI ≥ 0.75)" if stable else "WEAK (ARI < 0.75) — see §7.3 guidance"
    print(f"\n  Bootstrap ARI: {ari_mean:.4f} ± {ari_std:.4f}  →  {verdict}")
    print(f"  Saved: {boot_path}")

    # ── Step 4/5 — Final GMM fit + cluster assignments ────────
    print(f"\n[4/5] Final Gaussian Mixture fit at K={k_final_safe} ...")
    gmm_final = GaussianMixture(
        n_components=k_final_safe, covariance_type="full",
        random_state=RANDOM_STATE, n_init=10
    )
    hard_labels = gmm_final.fit_predict(X)
    soft_probs  = gmm_final.predict_proba(X)

    # Save model for reproducibility (§7.5)
    gmm_path = OUT_DIR / "gmm_model_assam.joblib"
    joblib.dump(gmm_final, gmm_path)
    print(f"  GMM model saved: {gmm_path}")

    assign = sig[["point_id"]].copy()
    for col in ("lat", "lon", "population"):
        if col in sig.columns:
            assign[col] = sig[col]
    assign["cluster_id"]           = hard_labels
    assign["max_membership_prob"]  = soft_probs.max(axis=1)
    for k in range(k_final_safe):
        assign[f"prob_cluster{k}"] = soft_probs[:, k]

    assign_path = OUT_DIR / "cluster_assignments_assam.csv"
    assign.to_csv(assign_path, index=False)
    print(f"  Assignments saved: {assign_path}")

    # ── Step 5/5 — Population-weighted profiles + map ─────────
    print("\n[5/5] Population-weighted cluster profile cards + map ...")

    sig["cluster_id"] = hard_labels
    # Profile on original (pre-standardised) columns from the raw signature
    raw_sig_path = PROCESSED_DIR / "climate_signatures_raw.csv"
    if raw_sig_path.exists():
        raw_sig = pd.read_csv(raw_sig_path)
        raw_sig.rename(columns={raw_sig.columns[0]: "point_id"}, inplace=True)
        raw_sig["cluster_id"] = hard_labels
        profile_source = raw_sig
    else:
        profile_source = sig

    # Columns to profile (exclude metadata and cluster_id)
    if grid_file.exists():
        grid = pd.read_csv(grid_file)
        if "point_id" in grid.columns:
            profile_source = profile_source.merge(grid[["point_id", "lat", "lon", "population"]], on="point_id", how="left")
            
    exclude_profile = {"point_id", "cluster_id", "lat", "lon"}
    numeric_profile_cols = [
        c for c in profile_source.columns
        if c not in exclude_profile
        and pd.api.types.is_numeric_dtype(profile_source[c])
    ]

    profile_rows = []
    for cid, g in profile_source.groupby("cluster_id"):
        w = g["population"].fillna(g["population"].median()) if "population" in g.columns else None
        row = {
            "cluster_id": cid,
            "n_points":   len(g),
            "total_population": g["population"].sum() if "population" in g.columns else None,
        }
        for col in numeric_profile_cols:
            if col == "population":
                continue
            vals = g[col].dropna()
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"]  = np.nan
                continue
            if w is not None and w.sum() > 0:
                row[f"{col}_mean"] = np.average(vals, weights=w.loc[vals.index])
            else:
                row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = vals.std()
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).sort_values("cluster_id").reset_index(drop=True)
    profile_path = OUT_DIR / "cluster_profiles_assam.csv"
    profiles.to_csv(profile_path, index=False)
    print(f"  Profiles saved: {profile_path}")

    # Map
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = cm.tab10(np.linspace(0, 0.9, k_final_safe))
    for cid in sorted(sig["cluster_id"].unique()):
        sub = sig[sig["cluster_id"] == cid]
        ax.scatter(
            sub["lon"], sub["lat"],
            color=colours[cid], s=60,
            alpha=0.85, edgecolors="white", linewidths=0.5,
            label=f"Cluster {cid}  (n={len(sub)})"
        )
        ax.annotate(
            f"C{cid}", (sub["lon"].mean(), sub["lat"].mean()),
            fontsize=12, fontweight="bold", ha="center",
            color=colours[cid]
        )
    ax.set_title(f"Assam Climate Regimes — GMM, K={k_final_safe}", fontsize=14)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    plt.tight_layout()
    map_path = PLOTS_DIR / "cluster_map_assam.png"
    plt.savefig(map_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Map saved: {map_path}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  DONE — Phase 4, Level A")
    print(f"  Bootstrap ARI  : {ari_mean:.4f} ± {ari_std:.4f}  {'✓ STABLE' if stable else '⚠ WEAK'}")
    print()
    for _, row in profiles.iterrows():
        ghi  = row.get("GHI_daily_kWh_mean", row.get("GHI_daily_kWh_mean", float("nan")))
        ta   = row.get("Ta_mean_mean", float("nan"))
        rh   = row.get("RH_mean_mean", float("nan"))
        mi   = row.get("monsoon_index_mean", float("nan"))
        pop  = row.get("total_population", float("nan"))
        cid  = int(row["cluster_id"])
        n    = int(row["n_points"])
        print(f"    Cluster {cid}: {n:3d} sites  "
              f"GHI={ghi:.2f} kWh/day  Ta={ta:.1f}°C  "
              f"RH={rh:.1f}%  monsoon_idx={mi:.2f}  pop≈{pop:,.0f}")
    print()
    print("  Outputs:")
    print(f"    {bic_path}")
    print(f"    {km_path}")
    print(f"    {boot_path}")
    print(f"    {assign_path}")
    print(f"    {profile_path}")
    print(f"    {gmm_path}")
    print(f"    {OUT_DIR / 'scaler_assam.joblib'}")
    print(f"    {map_path}")
    print()
    print("  External validation (§7.4) — MANUAL STEP:")
    print("    Compare cluster_assignments_assam.csv against Köppen-Geiger")
    print("    (Aw/Am/Cwa for valley/hill/upland) and NBC/ECBC zones using")
    print("    adjusted_rand_score. Add to paper §7.4.")
    print()
    print("  Next: run 05b_level_b_seasonal_assam.py (after Phase 5/8 are done)")
    print("=" * 68)


if __name__ == "__main__":
    main()
