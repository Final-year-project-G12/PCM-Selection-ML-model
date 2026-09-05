"""
comparison_plots_uttarakhand.py
================================
Cross-step comparison plots to help verify Objective 1's results make
sense together, not just individually. Output: data/plots/comparison/

Uses config.py for all paths (this pipeline's convention) rather than a
relative-to-script BASE guess, since every script in era5-uttarakhand/
sits directly in the project root next to data/, not in a subfolder.

Plots:
  1. Cluster GHI profiles: mean GHI by cluster
  2. PCM temperature target vs cluster mean temperature
  3. TOPSIS / GRA / consensus rankings side-by-side per cluster (top 5)
  4. TOPSIS vs GRA direct agreement (see note below — NOT the original
     Tamil Nadu "Monte Carlo stability" plot; that method isn't
     implemented in this pipeline, so this replaces it with something
     that uses data you actually have)
  5. Latent heat distribution: feasible survivors vs all candidates
  6. Physics validation: hours_target_met vs MCDM rank (only renders if
     you've run 10_physics_validation.py — optional in this pipeline)
  7. Cross-cluster summary: key properties of top PCM per cluster
  8. Sensitivity: how rank changes if weight shifts between TOPSIS and GRA

NOTE ON PLOT 4 — READ BEFORE ASSUMING THIS MATCHES THE OLD PLOT
--------------------------------------------------------------------------
Your Objective 1 audit confirmed this pipeline implements TOPSIS + GRA
only (no PROMETHEE II, VIKOR, or Monte Carlo stability analysis) — and
that the two implemented methods are strongly ANTI-correlated:
Spearman rho = -0.930 pooled across clusters. The original Tamil Nadu
version of this script plotted "Monte Carlo top-3 inclusion probability
vs consensus rank," which has no equivalent in your data at all (that
file/column simply doesn't exist here). Rather than ship a plot that
always silently renders nothing, this version plots TOPSIS rank against
GRA rank directly, per cluster — which is both real data you have AND a
more diagnostically useful plot for your specific situation, since a
near-perfect anti-correlation between your two methods is one of your
project's most important open findings to visualize and discuss.

HOW TO RUN:
  python comparison_plots_uttarakhand.py
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from config import PROCESSED_DIR, PLOTS_DIR

CLUSTERS = PROCESSED_DIR / "clustering" / "cluster_assignments_uttarakhand.csv"
CPROF = PROCESSED_DIR / "clustering" / "cluster_profiles_uttarakhand.csv"
SIG_CSV = PROCESSED_DIR / "signatures" / "climate_signature_uttarakhand.csv"
FEAS = PROCESSED_DIR / "pcm" / "feasibility_survivors_by_cluster.csv"
PCM_DB = PROCESSED_DIR / "pcm" / "pcm_database_uttarakhand.csv"
TOPK = PROCESSED_DIR / "pcm" / "mcdm_topk_by_cluster.csv"
PHYS = PROCESSED_DIR / "pcm" / "physics_validation_results.csv"
OUT = PLOTS_DIR / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
       "#42d4f4", "#f032e6", "#bfef45"]


def load(p, label=""):
    if not p.exists():
        print(f"  skip {label}: not found ({p})")
        return None
    return pd.read_csv(p)


def sfig(n):
    plt.savefig(OUT / n, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {n}")


def ensure_ranks(df):
    # PROMETHEE/VIKOR entries are safe no-ops here — those score columns
    # don't exist in this pipeline's output, so nothing gets added for
    # them. Left in so this function needs no changes if those methods
    # are ever added later.
    for sc, rc, asc in [("topsis_score", "topsis_rank", False),
                         ("gra_grade", "gra_rank", False),
                         ("promethee_flow", "promethee_rank", False),
                         ("vikor_Q", "vikor_rank", True),
                         ("borda_score", "consensus_rank", False)]:
        if rc not in df.columns and sc in df.columns:
            df[rc] = df.groupby("cluster_id")[sc].rank(ascending=asc, method="min").astype(int)
    return df


# ── Comparison 1: Cluster Mean GHI from Signature ────────────────────────
print("[1/8] Cluster GHI profiles from signature")
sig = load(SIG_CSV, "signature")
clu = load(CLUSTERS, "clusters")
if sig is not None and clu is not None:
    mg = sig.merge(clu[["point_id", "cluster_id"]], on="point_id", how="inner")
    # GHI_daily_kWh is this pipeline's canonical Tier-2 solar column;
    # GHI_mean was the Tamil Nadu original's name — check both, then fall
    # back to any GHI-named column.
    ghi_col = next((c for c in ["GHI_daily_kWh", "GHI_mean"] if c in mg.columns), None)
    if ghi_col is None:
        matches = [c for c in mg.columns if "GHI" in c.upper()]
        ghi_col = matches[0] if matches else None
    if ghi_col:
        fig, ax = plt.subplots(figsize=(9, 5))
        for cid, g in mg.groupby("cluster_id"):
            ax.bar(str(cid), g[ghi_col].mean(), color=PAL[int(cid) % len(PAL)],
                   edgecolor="white", lw=1, alpha=0.9, label=f"Cluster {cid}")
            ax.errorbar(str(cid), g[ghi_col].mean(), yerr=g[ghi_col].std(),
                        fmt="none", color="black", capsize=5, lw=1.5)
        ax.set(xlabel="Cluster", ylabel=f"{ghi_col} (mean +/- std)",
               title="Comparison 1: Mean GHI by Climate Regime (Uttarakhand)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        sfig("01_comparison_cluster_ghi.png")
    else:
        print("  No GHI column found in signature file")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    mg2 = sig.merge(clu[["point_id", "cluster_id"]], on="point_id", how="inner")
    # Ta_mean is this pipeline's canonical signature column name (not
    # Ta_mean_proxy, which was specific to the Tamil Nadu original).
    t_col = next((c for c in ["Ta_mean", "Ta_mean_proxy"] if c in mg2.columns), None)
    if t_col is None:
        matches = [c for c in mg2.columns if "T_" in c or "temp" in c.lower() or c.startswith("Ta_")]
        t_col = matches[0] if matches else None

    feas = load(FEAS, "feasibility")
    cprof = load(CPROF, "cluster_profiles")
    # Tm_target_C is confirmed to live in cluster_profiles_uttarakhand.csv
    # (09_recommendation_cards.py reads it from there); the feasibility
    # file may or may not also carry it. Prefer feasibility if present
    # (matches the original script's behaviour), fall back to profiles.
    tm_target = None
    if feas is not None and "Tm_target_C" in feas.columns:
        tm_target = feas.groupby("cluster_id")["Tm_target_C"].first()
    elif cprof is not None and "Tm_target_C" in cprof.columns:
        tm_target = cprof.set_index("cluster_id")["Tm_target_C"]
    elif cprof is not None and "Tm_target_C_regime_capped" in cprof.columns:
        tm_target = cprof.set_index("cluster_id")["Tm_target_C_regime_capped"]

    if t_col and tm_target is not None:
        clust_T = mg2.groupby("cluster_id")[t_col].mean()
        comp = pd.DataFrame({"ClusterMeanT_C": clust_T, "PCM_Tm_target": tm_target}).dropna()
        fig, ax = plt.subplots(figsize=(9, 6))
        for cid, row in comp.iterrows():
            ax.scatter(row["ClusterMeanT_C"], row["PCM_Tm_target"],
                       color=PAL[int(cid) % len(PAL)], s=150, zorder=3, label=f"Cluster {cid}")
            ax.annotate(f"C{cid}", (row["ClusterMeanT_C"], row["PCM_Tm_target"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        xlim = ax.get_xlim()
        ax.plot(xlim, [x + 25 for x in xlim], "r--", lw=1, label="+25C offset")
        ax.plot(xlim, [x + 35 for x in xlim], "g--", lw=1, label="+35C offset")
        ax.set(xlabel=f"Cluster Mean Temperature ({t_col}) (C)",
               ylabel="PCM Target Melting Point (C)",
               title="Comparison 2: Cluster Temperature vs PCM Tm Target (Uttarakhand)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        sfig("02_comparison_temp_vs_tm_target.png")
        # Your audit found Tm_target is held CONSTANT at 57C for every
        # point by design — if every point in `comp` lands on the same
        # horizontal line here, that's not a plotting bug, it's exactly
        # that finding made visible.
        if comp["PCM_Tm_target"].nunique() == 1:
            print(f"    NOTE: all clusters show the same Tm_target "
                  f"({comp['PCM_Tm_target'].iloc[0]:.1f}C) — this matches your "
                  f"audit's finding that Tm_target is constant by design, not a bug here.")
    else:
        print("  Could not find both a temperature column and a Tm_target source")

# ── Comparison 3: TOPSIS / GRA / Consensus Rankings per Cluster ─────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk = load(TOPK, "topk")
if topk is not None:
    topk = ensure_ranks(topk)
    methods = ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"]
    methods = [m for m in methods if m in topk.columns]
    if len(methods) >= 2:
        clus_ids = sorted(topk["cluster_id"].unique())
        fig, axes = plt.subplots(len(clus_ids), 1, figsize=(13, 4.5 * len(clus_ids)), squeeze=False)
        for idx, cid in enumerate(clus_ids):
            sort_col = "consensus_rank" if "consensus_rank" in methods else methods[0]
            sub = topk[topk["cluster_id"] == cid].sort_values(sort_col).head(5)
            x = np.arange(len(sub))
            w = 0.15
            ax = axes[idx, 0]
            for mi, m in enumerate(methods):
                if m in sub.columns:
                    ax.bar(x + mi * w, sub[m].values, width=w, label=m.replace("_rank", "").upper(),
                           color=sns.color_palette("Set2", len(methods))[mi], edgecolor="white")
            ax.set_xticks(x + (len(methods) - 1) * w / 2)
            ax.set_xticklabels(sub["name"].tolist(), rotation=20, ha="right", fontsize=9)
            ax.set(title=f"Cluster {cid} - Top 5 PCM Ranks by Method", ylabel="Rank (lower=better)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25, axis="y")
        plt.suptitle("Comparison 3: MCDM Methods Side-by-Side (Uttarakhand)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        sfig("03_comparison_mcdm_methods.png")
    else:
        print(f"  Only {len(methods)} ranking method(s) found ({methods}) — need >=2 to compare")

# ── Comparison 4: TOPSIS vs GRA Direct Agreement (replaces Monte Carlo) ──
print("[4/8] TOPSIS vs GRA rank agreement (Monte Carlo not implemented in this pipeline)")
if topk is not None:
    topk = ensure_ranks(topk)
    if {"topsis_rank", "gra_rank"}.issubset(topk.columns):
        fig, ax = plt.subplots(figsize=(10, 8))
        for cid, g in topk.groupby("cluster_id"):
            ax.scatter(g["topsis_rank"], g["gra_rank"], color=PAL[int(cid) % len(PAL)],
                       s=90, alpha=0.8, edgecolors="black", linewidth=0.5, label=f"Cluster {cid}")
            for _, row in g.iterrows():
                if "name" in row:
                    ax.annotate(str(row["name"])[:14], (row["topsis_rank"], row["gra_rank"]),
                                fontsize=6, alpha=0.7)
        max_rank = max(topk["topsis_rank"].max(), topk["gra_rank"].max())
        ax.plot([1, max_rank], [1, max_rank], "k--", lw=1.5, alpha=0.6, label="Perfect agreement")
        ax.set(xlabel="TOPSIS Rank", ylabel="GRA Rank",
               title="Comparison 4: TOPSIS vs GRA Direct Agreement (Uttarakhand)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        sfig("04_comparison_topsis_vs_gra_agreement.png")

        valid = topk[["topsis_rank", "gra_rank"]].dropna()
        if len(valid) >= 3:
            rho, pval = spearmanr(valid["topsis_rank"], valid["gra_rank"])
            print(f"    Pooled TOPSIS-vs-GRA Spearman rho = {rho:.3f} (p={pval:.3g})")
            if rho < 0:
                print("    NOTE: negative rho means the methods actively DISAGREE, not just")
                print("    weakly agree — this matches your audit's -0.930 finding. Points")
                print("    scattered far from the diagonal above are exactly where that shows.")
    else:
        print("  Need both topsis_rank and gra_rank to plot this comparison")

# ── Comparison 5: Latent heat distribution - feasible vs all ─────────────
print("[5/8] Latent heat distribution comparison")
feas = load(FEAS, "feasibility")
db = load(PCM_DB, "pcm_db")
if feas is not None and "latent_heat_kJ_kg" in feas.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    if db is not None and "latent_heat_kJ_kg" in db.columns:
        ax.hist(db["latent_heat_kJ_kg"].dropna(), bins=40, alpha=0.5, color="gray",
                label=f"All candidates (n={len(db)})", density=True)
    ax.hist(feas["latent_heat_kJ_kg"].dropna(), bins=30, alpha=0.8, color="#3b7dd8",
            label=f"Feasible survivors (n={len(feas)})", density=True)
    ax.axvline(feas["latent_heat_kJ_kg"].median(), color="#3b7dd8", ls="--", lw=2,
               label=f"Feasible median: {feas['latent_heat_kJ_kg'].median():.0f} kJ/kg")
    ax.set(xlabel="Latent Heat (kJ/kg)", ylabel="Density",
           title="Comparison 5: Latent Heat Distribution - All vs Feasible Survivors (Uttarakhand)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    sfig("05_comparison_latent_heat_distribution.png")

# ── Comparison 6: Physics validation - hours_met vs MCDM rank ────────────
print("[6/8] Physics validation vs MCDM rank")
phys = load(PHYS, "physics_val")
if phys is None:
    print("  (This is expected if you haven't run 10_physics_validation.py yet — "
          "it's an optional phase in this pipeline, not a required one.)")
if phys is not None and topk is not None and "hours_target_met_per_year" in phys.columns:
    merge_cols = ["cluster_id", "name", "hours_target_met_per_year"]
    if "complete_cycles_per_year" in phys.columns:
        merge_cols.append("complete_cycles_per_year")
    mg6 = topk.merge(phys[merge_cols].drop_duplicates(subset=["cluster_id", "name"]),
                      on=["cluster_id", "name"], how="inner")
    if "consensus_rank" in mg6.columns and not mg6.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for cid, g in mg6.groupby("cluster_id"):
            v = g[["consensus_rank", "hours_target_met_per_year"]].notna().all(axis=1)
            axes[0].scatter(g.loc[v, "consensus_rank"], g.loc[v, "hours_target_met_per_year"],
                            color=PAL[int(cid) % len(PAL)], s=80, alpha=0.8, label=f"Cluster {cid}")
        axes[0].set(xlabel="Consensus Rank", ylabel="Hours Target Met per Year",
                   title="MCDM Rank vs Hours Target Met")
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.25)
        if "complete_cycles_per_year" in mg6.columns:
            for cid, g in mg6.groupby("cluster_id"):
                v = g[["consensus_rank", "complete_cycles_per_year"]].notna().all(axis=1)
                axes[1].scatter(g.loc[v, "consensus_rank"], g.loc[v, "complete_cycles_per_year"],
                                color=PAL[int(cid) % len(PAL)], s=80, alpha=0.8, label=f"Cluster {cid}")
            axes[1].set(xlabel="Consensus Rank", ylabel="Complete Cycles per Year",
                       title="MCDM Rank vs Complete Cycles")
            axes[1].legend(fontsize=9)
            axes[1].grid(alpha=0.25)
        plt.suptitle("Comparison 6: Physics Validation vs MCDM Ranking (Uttarakhand)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        sfig("06_comparison_physics_vs_rank.png")

# ── Comparison 7: Cross-cluster top PCM key properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None and "consensus_rank" in topk.columns:
    top1 = topk[topk["consensus_rank"] == 1].drop_duplicates(subset="cluster_id").copy()
    props = [c for c in ["Tm_C", "latent_heat_kJ_kg", "rho_H_MJ_m3", "TC_W_mK", "cycles_tested"]
             if c in top1.columns]
    if len(props) >= 2:
        fig, axes = plt.subplots(1, len(props), figsize=(4 * len(props), 5))
        if len(props) == 1:
            axes = [axes]
        for i, p in enumerate(props):
            axes[i].bar(top1["cluster_id"].astype(str), top1[p],
                       color=[PAL[int(c) % len(PAL)] for c in top1["cluster_id"]], edgecolor="white")
            axes[i].set(title=p, xlabel="Cluster")
            axes[i].grid(alpha=0.3, axis="y")
            for j, (_, row) in enumerate(top1.iterrows()):
                if pd.notna(row.get(p)):
                    axes[i].text(j, row[p] * 1.01, str(row.get("name", ""))[:10],
                                ha="center", fontsize=7, rotation=25)
        plt.suptitle("Comparison 7: Top PCM Properties per Cluster (Uttarakhand) - Rank #1",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        sfig("07_comparison_cross_cluster_top_pcm.png")
        # Your audit found RT60 is consensus rank 1 in ALL 5 clusters — if
        # every bar above is identical across clusters, that's this
        # finding made visible, not a bug in this plot.
        if top1["name"].nunique() == 1:
            print(f"    NOTE: the same PCM ({top1['name'].iloc[0]}) is rank 1 in every "
                  f"cluster shown — matches your audit's finding, not a plotting error.")
    else:
        print(f"  Only {len(props)} property column(s) available in top-k file "
              f"({props}) — need >=2 to compare")

# ── Comparison 8: Weight sensitivity (TOPSIS vs GRA blend) ───────────────
print("[8/8] Rank sensitivity to weight perturbation")
if topk is not None:
    topk = ensure_ranks(topk)
    score_cols = [c for c in ["topsis_score", "gra_grade", "promethee_flow"] if c in topk.columns]
    if len(score_cols) >= 2:
        c1, c2 = score_cols[0], score_cols[1]
        results = []
        for w1 in [0.3, 0.5, 0.7]:
            w2 = 1 - w1
            for cid, g in topk.groupby("cluster_id"):
                v = g[[c1, c2]].notna().all(axis=1)
                if v.sum() > 0:
                    comb = (w1 * g.loc[v, c1] / max(1, g.loc[v, c1].max())
                            + w2 * g.loc[v, c2] / max(1, g.loc[v, c2].max()))
                    rk = comb.rank(ascending=False, method="min")
                    for idx2 in g.loc[v].index:
                        results.append({"w1": w1, "Cluster": cid,
                                        "Name": g.loc[idx2, "name"] if "name" in g.columns else str(idx2),
                                        "ComboRank": rk.loc[idx2]})
        rdf = pd.DataFrame(results)
        if not rdf.empty:
            fig, ax = plt.subplots(figsize=(12, 6))
            if "consensus_rank" in topk.columns and "name" in topk.columns:
                top_names = topk[topk["consensus_rank"] <= 3]["name"].unique()
            else:
                top_names = rdf["Name"].value_counts().head(6).index
            for nm in top_names[:6]:
                sub = rdf[rdf["Name"] == nm]
                if not sub.empty:
                    ax.plot(sub["w1"], sub["ComboRank"], "-o", label=str(nm)[:20], lw=1.5, ms=6)
            c1_label = c1.replace("_score", "").replace("_grade", "").upper()
            c2_label = c2.replace("_score", "").replace("_grade", "").upper()
            ax.set(xlabel=f"Weight on {c1_label} (remainder on {c2_label})",
                   ylabel="Combined Rank",
                   title="Comparison 8: Rank Sensitivity to Weight Perturbation (Uttarakhand)")
            ax.invert_yaxis()
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
            sfig("08_comparison_rank_sensitivity.png")
            print(f"    Blending between {c1_label} and {c2_label} weight — given your "
                  f"pipeline's -0.930 anti-correlation between these two, expect this plot "
                  f"to show real rank volatility, not near-flat lines.")
    else:
        print("  Insufficient score columns for sensitivity (need >=2)")

print("\nAll comparison plots saved to:", OUT)