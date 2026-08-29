"""
05_cluster_tamilnadu.py
=========================
PHASE 4 — CLIMATE REGIME CLUSTERING, TAMIL NADU ONLY

WHY THIS EXISTS INSTEAD OF 05_cluster_regions.py
---------------------------------------------------
05_cluster_regions.py was written for the ORIGINAL v3.0 scope: combine
signature matrices from FOUR states (Rajasthan, Assam, Tamil Nadu,
Uttarakhand) and cluster across all of them together, so a discovered
regime can cross state boundaries.

You are working on Tamil Nadu only right now, with a 4-day deadline. That
cross-state comparison isn't available yet (no other state has run Phase
3), and it isn't required for Objective 1 to stand on its own: the
objective is "cluster meteorological data and identify Top-2/Top-3 PCM
candidates per climatic regime" — nothing in the objective statement
requires those regimes to span multiple states. Clustering the 133 TN
population points into their own regimes still gives you exactly what you
need: different PCM recommendations for different LOCATIONS within Tamil
Nadu (e.g. Nilgiris hills vs. Chennai coast vs. interior dry Coimbatore
belt are very plausibly different regimes).

If you add another state's Phase 3 output later, 05_cluster_regions.py
(unchanged, already state-parameterised) is what you switch to — this
script does NOT need to be thrown away, its output format is compatible,
it's just scoped to n=1 state for now.

Still uses Gaussian Mixture (not K-Means) for the same reason as the
4-state version: climate is a continuous gradient within a state too —
the boundary between "interior dry" and "coastal humid" Tamil Nadu is not
a hard line, and a point near that boundary genuinely has partial
membership in both. Soft membership probabilities are kept and are what
Phase 5/6 should read for boundary points.

INPUT  : data/processed/signatures/climate_signature_tamilnadu.csv
         (04b_climate_signature.py's output, Tier1+Tier2 merged)
OUTPUT : data/processed/clustering/
           bic_selection_tamilnadu.csv
           cluster_assignments_tamilnadu.csv   <- soft membership probabilities
           cluster_profiles_tamilnadu.csv      <- population-weighted profile
                                                   per regime — feed this into
                                                   Phase 5 PCM feasibility
                                                   filtering
           cluster_map_tamilnadu.png

HOW TO RUN:
  python 05_cluster_tamilnadu.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PROCESSED_DIR

SIGNATURE_FILE = PROCESSED_DIR / "signatures" / "climate_signature_tamilnadu.csv"
OUT_DIR = PROCESSED_DIR / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_CANDIDATES = list(range(2, 11))     # smaller state -> smaller upper bound than the 4-state script
K_FINAL = 5                            # <-- set after reviewing bic_selection_tamilnadu.csv, re-run
SILHOUETTE_ACCEPT_LO, SILHOUETTE_ACCEPT_HI = 0.15, 0.40   # single-state band is a bit wider than
                                                            # the 4-state 0.15-0.35 (no artificial
                                                            # between-state gaps inflating it here)
RANDOM_STATE = 42


def main():
    print("=" * 68)
    print("  Phase 4 — Climate Regime Clustering (Gaussian Mixture) — Tamil Nadu only")
    print("=" * 68)

    if not SIGNATURE_FILE.exists():
        print(f"\n  ERROR: {SIGNATURE_FILE} not found — run 04b_climate_signature.py first.")
        return

    sig = pd.read_csv(SIGNATURE_FILE)
    sig.rename(columns={sig.columns[0]: "point_id"}, inplace=True)
    z_cols = [c for c in sig.columns if c.endswith("_z")]
    print(f"\n  Points: {len(sig)}  |  Standardized signature columns: {len(z_cols)}")
    print(f"  (lat/lon are NOT among these — never cluster on geography, plan v3.0 Section 6.2)")

    X = sig[z_cols].fillna(sig[z_cols].median()).values

    k_candidates_safe = [k for k in K_CANDIDATES if k < len(X)]
    print(f"\n[1/4] BIC + silhouette + Davies-Bouldin + Calinski-Harabasz, "
          f"K={k_candidates_safe[0]}..{k_candidates_safe[-1]} ...")

    rows = []
    for k in k_candidates_safe:
        # covariance_type="diag" (diagonal) — NOT "full" — is the correct choice
        # for 133 samples with 27+ feature dimensions.  A full covariance matrix
        # requires (D*(D+1)/2) = 378 parameters PER component; with K=5 that is
        # 1,890 parameters fit on only 133 data points, severely overdetermining
        # the model and causing membership probability saturation (probs ≈ 1.0
        # at boundaries).  Diagonal covariance assumes feature independence after
        # PCA and requires only D=27 parameters per component, consistent with
        # standard practice for high-dimensional, low-sample-count GMM.
        # (See docs/era5_tamilnadu/20_IMPLEMENTATION_ISSUES.md, issue #4 and
        #  docs/era5_tamilnadu/METHODS.md, §05 for full justification.)
        gmm = GaussianMixture(n_components=k, covariance_type="diag",
                               random_state=RANDOM_STATE, n_init=5)
        labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        if len(set(labels)) > 1:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
        else:
            sil = db = ch = float("nan")
        in_band = SILHOUETTE_ACCEPT_LO <= sil <= SILHOUETTE_ACCEPT_HI if sil == sil else False
        rows.append({"k": k, "BIC": bic, "silhouette": sil,
                     "davies_bouldin": db, "calinski_harabasz": ch,
                     "in_accept_band": in_band})
        flag = "  <- in accept band" if in_band else ""
        print(f"    K={k:2d}   BIC={bic:10.1f}   silhouette={sil:.4f}   "
              f"DB={db:.3f}   CH={ch:8.1f}{flag}")

    k_table = pd.DataFrame(rows)
    k_path = OUT_DIR / "bic_selection_tamilnadu.csv"
    k_table.to_csv(k_path, index=False)
    best_bic_k = int(k_table.loc[k_table["BIC"].idxmin(), "k"])
    in_band = k_table[k_table["in_accept_band"]]
    print(f"\n  Saved: {k_path}")
    print(f"  Lowest-BIC K: {best_bic_k}")
    if len(in_band):
        print(f"  K values in the {SILHOUETTE_ACCEPT_LO}-{SILHOUETTE_ACCEPT_HI} silhouette band: "
              f"{in_band['k'].tolist()}")
    print(f"  K_FINAL is currently {K_FINAL} at the top of this script — "
          f"update after reviewing this table, then re-run.")

    # Reported comparison against K-Means, same purpose as the 4-state
    # script: answers the "why not K-Means" question with a number instead
    # of an assertion.
    print("\n[2/4] K-Means comparison (reported only, GMM remains primary) ...")
    km_rows = []
    for k in k_candidates_safe:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km_labels = km.fit_predict(X)
        km_sil = silhouette_score(X, km_labels) if len(set(km_labels)) > 1 else float("nan")
        km_rows.append({"k": k, "kmeans_silhouette": km_sil})
    km_table = pd.DataFrame(km_rows)
    km_table.to_csv(OUT_DIR / "kmeans_comparison_tamilnadu.csv", index=False)
    print(km_table.to_string(index=False))
    print(f"  Saved: {OUT_DIR / 'kmeans_comparison_tamilnadu.csv'}")

    k_final_safe = min(K_FINAL, len(X) - 1)
    print(f"\n[3/4] Final Gaussian Mixture fit at K={k_final_safe} "
          f"(covariance_type=diag, see METHODS.md §05) ...")
    gmm_final = GaussianMixture(n_components=k_final_safe, covariance_type="diag",
                                 random_state=RANDOM_STATE, n_init=10)
    hard_labels = gmm_final.fit_predict(X)
    soft_probs = gmm_final.predict_proba(X)

    assign = sig[["point_id"]].copy()
    for c in ["lat", "lon", "population"]:
        if c in sig.columns:
            assign[c] = sig[c]
    assign["cluster_id"] = hard_labels
    assign["max_membership_prob"] = soft_probs.max(axis=1)
    for k in range(k_final_safe):
        assign[f"prob_cluster{k}"] = soft_probs[:, k]

    assign_path = OUT_DIR / "cluster_assignments_tamilnadu.csv"
    assign.to_csv(assign_path, index=False)
    print(f"  Saved: {assign_path}")

    print("\n[4/4] Population-weighted cluster profiles + map ...")
    sig["cluster_id"] = hard_labels
    profile_cols = [c for c in sig.columns
                     if c not in ("point_id", "cluster_id") and not c.endswith("_z")]

    def weighted_mean(g, col):
        w = g["population"].fillna(g["population"].median()) if "population" in g.columns else None
        if w is None or w.sum() == 0:
            return g[col].mean()
        return np.average(g[col], weights=w)

    profile_rows = []
    for cid, g in sig.groupby("cluster_id"):
        row = {"cluster_id": cid, "n_points": len(g),
               "total_population_covered": g["population"].sum() if "population" in g.columns else None}
        for col in profile_cols:
            if pd.api.types.is_numeric_dtype(g[col]):
                row[col] = weighted_mean(g, col)
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).sort_values("cluster_id")
    profile_path = OUT_DIR / "cluster_profiles_tamilnadu.csv"
    profiles.to_csv(profile_path, index=False)
    print(f"  Saved: {profile_path}")

    fig, ax = plt.subplots(figsize=(8, 9))
    sc = ax.scatter(sig["lon"], sig["lat"], c=hard_labels, cmap="tab10", s=60,
                     alpha=0.9, edgecolors="white", linewidths=0.6)
    for cid in sorted(sig["cluster_id"].unique()):
        sub = sig[sig["cluster_id"] == cid]
        ax.annotate(f"C{cid}", (sub["lon"].mean(), sub["lat"].mean()),
                     fontsize=11, fontweight="bold", ha="center")
    ax.set_title(f"Tamil Nadu Climate Regimes — GMM, K={k_final_safe}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_map_tamilnadu.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_DIR / 'cluster_map_tamilnadu.png'}")

    print("\n" + "=" * 68)
    print("  DONE")
    for _, row in profiles.iterrows():
        ghi = row.get("GHI_daily_kWh", float("nan"))
        ta = row.get("Ta_mean", float("nan"))
        print(f"    Cluster {int(row['cluster_id'])}: {row['n_points']} points  "
              f"GHI_daily_kWh~{ghi:.2f}  Ta_mean~{ta:.1f}C  "
              f"population~{row['total_population_covered']:,.0f}"
              if row['total_population_covered'] == row['total_population_covered'] else
              f"    Cluster {int(row['cluster_id'])}: {row['n_points']} points")
    print("=" * 68)
    print("\nNext (Phase 5): build the PCM property database (40+ candidates, 42-70C "
          "band), hard-filter it against each cluster's Tm_target / L_required "
          "from cluster_profiles_tamilnadu.csv, then rank with TOPSIS + GRA "
          "(minimum viable MCDM stack) to get a Top-2/3 per cluster.")


if __name__ == "__main__":
    main()
