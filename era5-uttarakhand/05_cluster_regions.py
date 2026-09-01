"""
05_cluster_regions.py
=======================
PHASE 4 — CLIMATE REGIME CLUSTERING (Objective 1 plan v2.0, Section 7)

NOT run yet for a reason: needs climate_signature_{region}.csv from AT
LEAST 2 regions (ideally all 4) — produced by 04b_climate_signature.py,
run once per region. Do not run this until Uttarakhand + at least one more
region have both finished 02 -> 03 -> 04 -> 04b.

WHAT CHANGED FROM THE EARLIER K-MEANS VERSION OF THIS SCRIPT
----------------------------------------------------------------
The plan doc's v2.0 explicitly rejected a reviewer's suggestion to switch
from Gaussian Mixture to K-Means: climate is a continuous gradient, not
hard-boundaried, so soft (probabilistic) cluster membership is the
correct model — a point near a regime boundary genuinely IS partially
both regimes, and GMM represents that; K-Means forces a hard, arbitrary
line through it. This version uses GaussianMixture, selected by BIC, with
an explicit realistic silhouette acceptance band of 0.15-0.35 (the doc
rejects >0.75 as unattainable and methodologically suspicious for
climate-signature clustering — a very high silhouette here usually means
the signature matrix collapsed to 1-2 dominant variables, not that the
climate regimes are unusually crisp).

INPUT  : data/processed/signatures/climate_signature_{region}.csv
         (one file per region, from that region's 04b_climate_signature.py)
OUTPUT : data/processed/clustering/
           point_fingerprints.csv     <- combined signature matrix, all regions
           bic_selection.csv          <- BIC + silhouette per candidate K
           cluster_assignments.csv    <- point_id, region, cluster_id, and the
                                          full posterior probability vector
                                          (soft membership — this is the
                                          object Phase 5/6 actually consumes,
                                          not just the hard argmax label)
           cluster_profiles.csv       <- population-weighted mean climate
                                          signature per cluster (feed this
                                          into Phase 5/6 PCM ranking)

HOW TO RUN
----------
  python 05_cluster_regions.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from config import PROCESSED_DIR

SIGNATURE_DIR = PROCESSED_DIR / "signatures"
OUT_DIR = PROCESSED_DIR / "clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CONFIG — point this at each region's Phase-3 signature output.
# Add region 3/4 paths here once their 04b_climate_signature.py has run.
# ═══════════════════════════════════════════════════════════

REGION_FILES = {
    "Uttarakhand": SIGNATURE_DIR / "climate_signature_uttarakhand.csv",
    "Rajasthan": SIGNATURE_DIR.parent.parent / "era5-rajasthan" / "data" / "processed" / "signatures" / "climate_signature_rajasthan.csv",
    # "Region3": SIGNATURE_DIR.parent.parent / "..." / "climate_signature_region3.csv",
    # "Region4": SIGNATURE_DIR.parent.parent / "..." / "climate_signature_region4.csv",
}

K_CANDIDATES = list(range(3, 13))
K_FINAL = 6              # <-- set after reviewing bic_selection.csv, then re-run
SILHOUETTE_ACCEPT_LO, SILHOUETTE_ACCEPT_HI = 0.15, 0.35   # per v2.0 §7 — NOT >0.75
RANDOM_STATE = 42

# Signature columns to cluster on — the *_z (standardized) columns from
# 04b, EXCLUDING PCA components derived FROM the temperature block (kept
# here to avoid double-counting) is a judgment call; the plan doc says
# cluster on "roughly 20-26 columns after PCA replacement" — i.e. USE the
# PC*_z columns (they replace the raw correlated block) and the
# non-PCA'd solar/variability index z-columns together. That's exactly
# what 04b's `_z` columns already are.


def load_region(name, path):
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: file not found — {path}")
        return None
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "point_id"}, inplace=True)  # index was point_id
    df["region"] = name
    print(f"  [OK]   {name}: {len(df)} points  ({path})")
    return df


def select_k(X, k_candidates):
    rows = []
    for k in k_candidates:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               random_state=RANDOM_STATE, n_init=5)
        labels = gmm.fit_predict(X)
        bic = gmm.bic(X)
        sil = silhouette_score(X, labels) if k > 1 and len(set(labels)) > 1 else float("nan")
        rows.append({"k": k, "BIC": bic, "silhouette": sil,
                      "in_accept_band": SILHOUETTE_ACCEPT_LO <= sil <= SILHOUETTE_ACCEPT_HI
                      if sil == sil else False})
        print(f"    K={k:2d}   BIC={bic:10.1f}   silhouette={sil:.4f}"
              f"{'  <- in 0.15-0.35 accept band' if SILHOUETTE_ACCEPT_LO <= sil <= SILHOUETTE_ACCEPT_HI else ''}")
    return pd.DataFrame(rows)


def main():
    print("=" * 68)
    print("  Phase 4 — Climate Regime Clustering (Gaussian Mixture, v2.0)")
    print("=" * 68)

    print("\n[1/5] Loading region signature files ...")
    frames = [f for f in (load_region(n, p) for n, p in REGION_FILES.items()) if f is not None]
    if len(frames) < 2:
        print("\n  Fewer than 2 regions available yet — clustering across regions "
              "needs at least 2 to be meaningful. Run 04b_climate_signature.py "
              "for another region first, or check REGION_FILES paths above. "
              "Stopping here.")
        return

    all_sig = pd.concat(frames, ignore_index=True)
    z_cols = [c for c in all_sig.columns if c.endswith("_z")]
    print(f"\n  Combined: {len(all_sig)} points across {len(frames)} region(s), "
          f"{len(z_cols)} standardized signature columns")

    fp_path = OUT_DIR / "point_fingerprints.csv"
    all_sig.to_csv(fp_path, index=False)
    print(f"  Saved: {fp_path}")

    # Columns are already standardized per-region by 04b. Re-standardize
    # across the COMBINED multi-region matrix so no single region's raw
    # scale dominates distances (04b's z-scores were computed within one
    # region's ~130 points only).
    print("\n[2/5] Re-standardizing across combined regions ...")
    scaler = StandardScaler()
    X = scaler.fit_transform(all_sig[z_cols].fillna(all_sig[z_cols].median()))

    k_candidates_safe = [k for k in K_CANDIDATES if k < len(X)]
    if len(k_candidates_safe) < len(K_CANDIDATES):
        print(f"\n  [NOTE] Only {len(X)} total points available — capping K candidates to "
              f"{k_candidates_safe} (GMM needs n_components < n_samples). This will stop "
              f"mattering once all 4 regions' full point sets are loaded.")

    print(f"\n[3/5] BIC + silhouette selection (K={k_candidates_safe[0]}..{k_candidates_safe[-1]}) ...")
    k_table = select_k(X, k_candidates_safe)
    k_path = OUT_DIR / "bic_selection.csv"
    k_table.to_csv(k_path, index=False)
    best_bic_k = int(k_table.loc[k_table["BIC"].idxmin(), "k"])
    in_band = k_table[k_table["in_accept_band"]]
    print(f"\n  Saved: {k_path}")
    print(f"  Lowest-BIC K: {best_bic_k}")
    if len(in_band):
        print(f"  K values landing in the 0.15-0.35 accept band: {in_band['k'].tolist()}")
    else:
        print("  [NOTE] No K landed in the 0.15-0.35 band with only 2 regions loaded — "
              "expected until more regions/points are in the matrix. Re-check once "
              "all 4 regions are combined.")
    print(f"  K_FINAL is currently {K_FINAL} at the top of this script — "
          f"update it once you've looked at this table, then re-run.")

    k_final_safe = min(K_FINAL, len(X) - 1) if len(X) - 1 < K_FINAL else K_FINAL
    if k_final_safe != K_FINAL:
        print(f"\n  [NOTE] K_FINAL={K_FINAL} exceeds available points ({len(X)}) — "
              f"using K={k_final_safe} instead for this run.")
    print(f"\n[4/5] Final Gaussian Mixture fit at K={k_final_safe} ...")
    gmm_final = GaussianMixture(n_components=k_final_safe, covariance_type="full",
                                 random_state=RANDOM_STATE, n_init=10)
    hard_labels = gmm_final.fit_predict(X)
    soft_probs = gmm_final.predict_proba(X)

    assign = all_sig[["point_id", "region"]].copy()
    if "lat" in all_sig.columns:
        assign["lat"] = all_sig["lat"]
    if "lon" in all_sig.columns:
        assign["lon"] = all_sig["lon"]
    if "population" in all_sig.columns:
        assign["population"] = all_sig["population"]
    assign["cluster_id"] = hard_labels
    for k in range(k_final_safe):
        assign[f"prob_cluster{k}"] = soft_probs[:, k]

    assign_path = OUT_DIR / "cluster_assignments.csv"
    assign.to_csv(assign_path, index=False)
    print(f"  Saved: {assign_path}  (includes SOFT membership probabilities per cluster — "
          f"Phase 5/6 should use these, not just the hard label, for boundary points)")

    print("\n[5/5] Population-weighted cluster profiles ...")
    profile_cols = [c for c in all_sig.columns if c not in
                     ("point_id", "region", "lat", "lon", "population") and not c.endswith("_z")]

    def weighted_mean(g, col):
        w = g["population"].fillna(g["population"].median()) if "population" in g.columns else None
        if w is None or w.sum() == 0:
            return g[col].mean()
        return np.average(g[col], weights=w)

    all_sig["cluster_id"] = hard_labels
    profile_rows = []
    for cid, g in all_sig.groupby("cluster_id"):
        row = {"cluster_id": cid, "n_points": len(g),
               "regions_present": ", ".join(sorted(g["region"].unique())),
               "total_population_covered": g["population"].sum() if "population" in g.columns else None}
        for col in profile_cols:
            if col in g.columns:
                row[col] = weighted_mean(g, col)
        profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).sort_values("cluster_id")
    profile_path = OUT_DIR / "cluster_profiles.csv"
    profiles.to_csv(profile_path, index=False)
    print(f"  Saved: {profile_path}")

    print("\n" + "=" * 68)
    print("  DONE")
    for _, row in profiles.iterrows():
        ghi = row.get("GHI_mean", float("nan"))
        ta = row.get("Ta_mean", float("nan"))
        print(f"    Cluster {int(row['cluster_id'])}: {row['n_points']} points "
              f"[{row['regions_present']}]  GHI_mean≈{ghi:.0f} W/m²  Ta_mean≈{ta:.1f}°C")
    print("=" * 68)
    print("\nNext (Phase 5): filter the PCM database (45-65°C core window, "
          "42-70°C with documented relaxation, per v2.0) against each cluster's "
          "Tm_target / L_required from Phase 3, then rank with the MCDM engine "
          "(Phase 6). See NEXT_STEPS.md.")


if __name__ == "__main__":
    main()
