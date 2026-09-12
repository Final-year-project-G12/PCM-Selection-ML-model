"""
Comparison Plots - Assam PCM Pipeline
======================================
Generates cross-step comparison plots to help verify results make sense.
Output: data/plots/comparison/

Plots:
  1. Cluster GHI profiles: mean GHI by cluster over months
  2. PCM temperature target vs cluster mean temperature
  3. All MCDM method rankings side-by-side per cluster (top 5)
  4. Monte Carlo stability: top3 prob vs consensus rank scatter
  5. Latent heat distribution: feasible survivors vs all candidates
  6. Physics validation: hours_target_met vs MCDM rank
  7. Cross-cluster summary: key properties of top PCM per cluster
  8. Sensitivity: how rank changes if weights shift +/- 20% per criterion
"""

import os, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

BASE     = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
CLUSTERS = os.path.join(BASE, "data", "processed", "clustering", "cluster_assignments_assam.csv")
SIG_CSV  = os.path.join(BASE, "data", "processed", "climate_signatures_raw.csv")
FEAS     = os.path.join(BASE, "data", "processed", "pcm", "feasibility_survivors_assam.csv")
PCM_DB   = os.path.join(BASE, "data", "processed", "pcm", "pcm_database_assam.csv")
TOPK     = os.path.join(BASE, "data", "processed", "pcm", "mcdm_topk_assam.csv")
MC_CSV   = os.path.join(BASE, "data", "processed", "pcm", "monte_carlo_stability_assam.csv")
PHYS     = os.path.join(BASE, "data", "processed", "pcm", "physics_validation_assam.csv")
CMP_PHYS = os.path.join(BASE, "data", "processed", "pcm", "mcdm_vs_physics_comparison.csv")
CPROF    = os.path.join(BASE, "data", "processed", "clustering", "cluster_profiles_assam.csv")
OUT      = os.path.join(BASE, "data", "plots", "comparison")
os.makedirs(OUT, exist_ok=True)

PAL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]
MEDOID_MAP = {0: "ASP_0012", 1: "ASP_0092", 2: "ASP_0028"}

ALT_OUT = os.path.join(BASE, "plots", "comparison")
PORTAL_OUT = os.path.abspath(os.path.join(BASE, "..", "..", "Documentation-Portal", "public", "plots", "comparison"))
os.makedirs(ALT_OUT, exist_ok=True)

def load(p, label=""):
    if not os.path.exists(p):
        print(f"  skip {label}: not found ({p})")
        return None
    return pd.read_csv(p)

def sfig(n):
    for d in [OUT, ALT_OUT, PORTAL_OUT]:
        if os.path.exists(d):
            plt.savefig(os.path.join(d, n), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {n}")

def ensure_ranks(df):
    for sc, rc, asc in [
        ("topsis_score", "topsis_rank", False),
        ("gra_grade", "gra_rank", False),
        ("promethee_flow", "promethee_rank", False),
        ("vikor_Q", "vikor_rank", True),
        ("borda_score", "consensus_rank", False)
    ]:
        if rc not in df.columns and sc in df.columns and "cluster_id" in df.columns:
            df[rc] = df.groupby("cluster_id")[sc].rank(ascending=asc, method="min").astype(int)
    return df

# ── Comparison 1: Cluster Mean GHI from Signature ────────────────────────
print("[1/8] Cluster GHI profiles from signature")
sig = load(SIG_CSV, "signature")
clu = load(CLUSTERS, "clusters")
if clu is not None and "cluster_id" not in clu.columns and "cluster" in clu.columns:
    clu["cluster_id"] = clu["cluster"]
VALID_CLUSTERS = sorted(clu["cluster_id"].unique()) if clu is not None else [0, 1, 2]

if sig is not None and clu is not None:
    merge_col = "point_id" if "point_id" in sig.columns and "point_id" in clu.columns else None
    if merge_col:
        mg = sig.merge(clu[[merge_col, "cluster_id"]], on=merge_col, how="inner")
    elif len(sig) == len(clu):
        mg = sig.copy()
        mg["cluster_id"] = clu["cluster_id"].values
    else:
        mg = None

    if mg is not None:
        ghi_col = "GHI_mean" if "GHI_mean" in mg.columns else ([c for c in mg.columns if "GHI" in c.upper()] or [None])[0]
        if ghi_col:
            fig, ax = plt.subplots(figsize=(9, 5))
            for cid in sorted(mg["cluster_id"].unique()):
                g = mg[mg["cluster_id"] == cid]
                ax.bar(str(cid), g[ghi_col].mean(), color=PAL[int(cid) % len(PAL)], edgecolor="white", lw=1, alpha=0.9, label=f"Cluster {cid}")
                ax.errorbar(str(cid), g[ghi_col].mean(), yerr=g[ghi_col].std(), fmt="none", color="black", capsize=5, lw=1.5)
            ax.set(xlabel="Cluster", ylabel=f"{ghi_col} (mean +/- std)", title="Comparison 1: Mean GHI by Climate Regime (Assam)")
            ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y"); sfig("01_comparison_cluster_ghi.png")
        else:
            print("  No GHI column in signature")

# ── Comparison 2: PCM Tm_target vs Cluster Mean Temp ────────────────────
print("[2/8] PCM Tm_target vs cluster mean temperature")
if sig is not None and clu is not None:
    if "point_id" in sig.columns and "point_id" in clu.columns:
        mg2 = sig.merge(clu[["point_id", "cluster_id"]], on="point_id", how="inner")
    elif len(sig) == len(clu):
        mg2 = sig.copy()
        mg2["cluster_id"] = clu["cluster_id"].values
    else:
        mg2 = None

    feas = load(FEAS, "feasibility")
    if mg2 is not None and feas is not None:
        t_col = "Ta_mean" if "Ta_mean" in mg2.columns else ("Ta_mean_proxy" if "Ta_mean_proxy" in mg2.columns else ([c for c in mg2.columns if "T_" in c or "temp" in c.lower()] or [None])[0])
        tm_col = "Tm_target_C" if "Tm_target_C" in feas.columns else ([c for c in feas.columns if "target" in c.lower() or "Tm" in c] or [None])[0]
        if t_col and tm_col:
            clust_T = mg2.groupby("cluster_id")[t_col].mean()
            tm_target = feas.groupby("cluster_id")[tm_col].first()
            comp = pd.DataFrame({"ClusterMeanT_C": clust_T, "PCM_Tm_target": tm_target}).dropna()
            if not comp.empty:
                fig, ax = plt.subplots(figsize=(11.5, 7.5))
                
                # Feasibility screening window [38°C, 54°C]
                ax.axhspan(38.0, 54.0, color="#2ca02c", alpha=0.10, label="Phase 5 Feasibility Acceptance Window [38°C – 54°C] (Tm ± screening offset)")
                ax.axhline(54.0, color="#2ca02c", ls="--", lw=1.2, alpha=0.7, label="Upper Screening Boundary: 54.0°C (Tm + 10°C)")
                ax.axhline(38.0, color="#2ca02c", ls=":", lw=1.2, alpha=0.7, label="Lower Screening Boundary: 38.0°C (Tm − 6°C)")
                
                # Fixed Physical SWH Design Target: 44.0°C
                ax.axhline(44.0, color="#1f77b4", lw=2.2, label="SWH Design Target: Tm = 44.0°C (T_delivery 50°C − ΔT_approach 6 K)")
                
                regime_names = {
                    0: "Lower Brahmaputra Valley",
                    1: "Upper Assam Tea Belt",
                    2: "Barak Valley & Southern Hills"
                }
                
                xlim_arr = np.linspace(20, 28, 100)
                
                # Actual ambient offsets from cluster mean temperatures:
                # C0 (Ta=25.89°C): offset = +18.11°C
                # C1 (Ta=25.10°C): offset = +18.90°C
                # C2 (Ta=22.59°C): offset = +21.41°C
                ax.plot(xlim_arr, xlim_arr + 18.11, ls="-.", color="#e07b39", lw=1.3, alpha=0.75, label="Regime C0 Offset: Ta + 18.1°C")
                ax.plot(xlim_arr, xlim_arr + 21.41, ls="-.", color="#9467bd", lw=1.3, alpha=0.75, label="Regime C2 Offset: Ta + 21.4°C")
                
                for cid, row in comp.iterrows():
                    cid_int = int(cid)
                    ta_val = row["ClusterMeanT_C"]
                    tm_val = row["PCM_Tm_target"]
                    offset_val = tm_val - ta_val
                    r_name = regime_names.get(cid_int, f"Cluster {cid_int}")
                    
                    ax.scatter(ta_val, tm_val, color=PAL[cid_int % len(PAL)], s=200, zorder=5, edgecolor="black", lw=1.3,
                               label=f"Cluster {cid_int}: {r_name}")
                    
                    # Annotate point with exact values and lift
                    xytext = (12, 16) if cid_int != 1 else (-20, -42)
                    ax.annotate(
                        f"Cluster {cid_int} ({r_name})\nTa = {ta_val:.2f}°C, Tm = {tm_val:.1f}°C\nLift ΔT = +{offset_val:.2f}°C",
                        (ta_val, tm_val),
                        textcoords="offset points",
                        xytext=xytext,
                        fontsize=8.5,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PAL[cid_int % len(PAL)], alpha=0.9),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color=PAL[cid_int % len(PAL)], lw=1.2)
                    )
                
                ax.set_xlim(20.5, 27.5)
                ax.set_ylim(32.0, 58.0)
                ax.set_xlabel(f"Cluster Mean Ambient Temperature ({t_col}) (°C)", fontsize=11, fontweight="bold")
                ax.set_ylabel("PCM Target Melting Point (°C)", fontsize=11, fontweight="bold")
                ax.set_title("Comparison 2: Cluster Ambient Temperature vs. PCM Tm Target (Assam)\nThermodynamic Derivation: Fixed Delivery Target (50°C) with Ambient Thermal Lift (+18.1°C to +21.4°C)",
                             fontsize=12, fontweight="bold", pad=12)
                
                # Design note textbox
                design_box = (
                    "Thermodynamic Formulation (§05b SWH Specification):\n"
                    "• Hot Water Delivery: T_delivery = 50.0°C (Domestic sanitary requirement)\n"
                    "• Heat Exchanger Approach: ΔT_approach = 6.0 K\n"
                    "• System PCM Target: Tm_target = 50.0°C − 6.0 K = 44.0°C (Uniform across Assam)\n"
                    "• Feasibility Acceptance Window: [38.0°C, 54.0°C] (Tm − 6°C to Tm + 10°C)\n"
                    "• Required Climate Lift (Tm − Ta): +18.11°C (C0), +18.90°C (C1), +21.41°C (C2)\n"
                    "* Note: Legacy +25°C/+35°C lines from Rajasthan do not apply to Assam's climate."
                )
                ax.text(0.03, 0.04, design_box, transform=ax.transAxes, fontsize=8,
                        verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#cccccc', alpha=0.92))
                
                ax.legend(fontsize=8, loc="upper right", framealpha=0.92)
                ax.grid(alpha=0.25, linestyle="--")
                plt.tight_layout()
                sfig("02_comparison_temp_vs_tm_target.png")

# ── Comparison 3: All MCDM Rankings Side-by-Side per Cluster ───────────
print("[3/8] MCDM method comparison: top 5 per cluster")
topk = load(TOPK, "topk")
if topk is not None:
    if "cluster_id" in topk.columns:
        topk = topk[topk["cluster_id"].isin(VALID_CLUSTERS)]
    topk = ensure_ranks(topk)
    methods = ["topsis_rank", "gra_rank", "promethee_rank", "vikor_rank", "consensus_rank"]
    methods = [m for m in methods if m in topk.columns]
    name_col = "name" if "name" in topk.columns else ("PCM_Name" if "PCM_Name" in topk.columns else "name")
    if len(methods) >= 2 and "cluster_id" in topk.columns:
        clus_ids = sorted(topk["cluster_id"].unique())
        fig, axes = plt.subplots(len(clus_ids), 1, figsize=(13, 4.5 * len(clus_ids)), squeeze=False)
        for idx, cid in enumerate(clus_ids):
            sub = topk[topk["cluster_id"] == cid].sort_values("consensus_rank" if "consensus_rank" in methods else methods[0]).head(5)
            x = np.arange(len(sub)); w = 0.15; ax = axes[idx, 0]
            for mi, m in enumerate(methods):
                if m in sub.columns:
                    ax.bar(x + mi * w, sub[m].values, width=w, label=m.replace("_rank", "").upper(), color=sns.color_palette("Set2", len(methods))[mi], edgecolor="white")
            ax.set_xticks(x + (len(methods) - 1) * w / 2)
            ax.set_xticklabels(sub[name_col].tolist() if name_col in sub.columns else sub.index.astype(str), rotation=20, ha="right", fontsize=9)
            med = MEDOID_MAP.get(int(cid), f"C{cid}")
            ax.set(title=f"Cluster {cid} ({med}) - Top 5 Ranked PCMs (K=3)", ylabel="Rank (lower=better)")
            ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
        plt.suptitle("Comparison 3: MCDM Method Consistency Across Assam Climate Regimes (K=3)\nTop 5 Ranked Candidates per Cluster (Borda Consensus of TOPSIS, GRA, PROMETHEE II, VIKOR)", fontsize=11, fontweight="bold")
        plt.tight_layout(); sfig("03_comparison_mcdm_methods.png")

# Marker mapping per cluster: Cluster 0 -> circle, Cluster 1 -> square, Cluster 2 -> triangle
MARKERS = {0: 'o', 1: 's', 2: '^'}

# ── Comparison 4: Monte Carlo Top3-Prob vs Consensus Rank ────────────────
print("[4/8] Monte Carlo stability vs consensus rank")
mc = load(MC_CSV, "monte_carlo")
if mc is not None and "cluster_id" in mc.columns:
    mc = mc[mc["cluster_id"].isin(VALID_CLUSTERS)]
if mc is not None and "product_name" in mc.columns and "name" not in mc.columns:
    mc = mc.rename(columns={"product_name": "name"})
if mc is not None and topk is not None and "top3_inclusion_probability" in mc.columns and mc["top3_inclusion_probability"].notna().any():
    name_col = "name" if "name" in topk.columns else "PCM_Name"
    mc[name_col] = mc[name_col].astype(str)
    topk[name_col] = topk[name_col].astype(str)
    mg4 = mc.merge(topk[["cluster_id", name_col, "consensus_rank"]].drop_duplicates(subset=["cluster_id", name_col]), on=["cluster_id", name_col], how="inner").dropna(subset=["consensus_rank"])
    if not mg4.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        scale = 100 if mg4["top3_inclusion_probability"].max() <= 1.0 else 1
        for cid, g in mg4.groupby("cluster_id"):
            m = MARKERS.get(int(cid) % len(MARKERS), 'o')
            ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
            for _, row in g.iterrows():
                ax.annotate(str(row[name_col]), (row["consensus_rank"], row["top3_inclusion_probability"] * scale), fontsize=6, alpha=0.7)
        ax.set(xlabel="MCDM Consensus Rank (1 = Best)", ylabel="Top-3 Inclusion Probability (%)",
               title="Comparison 4: Monte Carlo Ranking Stability vs. Consensus Rank (Assam K=3)\n(5,000 Perturbation Draws per Climate Regime)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")
    elif topk is not None and "top3_inclusion_probability" in topk.columns:
        fig, ax = plt.subplots(figsize=(10, 7))
        scale = 100 if topk["top3_inclusion_probability"].max() <= 1.0 else 1
        for cid, g in topk.groupby("cluster_id"):
            m = MARKERS.get(int(cid) % len(MARKERS), 'o')
            ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
        ax.set(xlabel="MCDM Consensus Rank (1 = Best)", ylabel="Top-3 Prob (%)",
               title="Comparison 4: Monte Carlo Ranking Stability vs. Consensus Rank (Assam K=3)\n(5,000 Perturbation Draws per Climate Regime)")
        ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")
    else:
        print("  skip MC comparison: no active Monte Carlo draws")
elif topk is not None and "top3_inclusion_probability" in topk.columns:
    fig, ax = plt.subplots(figsize=(10, 7))
    scale = 100 if topk["top3_inclusion_probability"].max() <= 1.0 else 1
    for cid, g in topk.groupby("cluster_id"):
        m = MARKERS.get(int(cid) % len(MARKERS), 'o')
        ax.scatter(g["consensus_rank"], g["top3_inclusion_probability"] * scale, color=PAL[int(cid) % len(PAL)], marker=m, s=100, alpha=0.85, edgecolors="black", lw=0.6, label=f"Cluster {cid}")
    ax.set(xlabel="MCDM Consensus Rank (1 = Best)", ylabel="Top-3 Prob (%)",
           title="Comparison 4: Monte Carlo Ranking Stability vs. Consensus Rank (Assam K=3)\n(5,000 Perturbation Draws per Climate Regime)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("04_comparison_mc_vs_rank.png")

# ── Comparison 5: Latent Heat Distribution - Feasible vs All ─────────────
print("[5/8] Latent heat distribution comparison")
feas = load(FEAS, "feasibility")
db = load(PCM_DB, "pcm_db")
if feas is not None and "latent_heat_kJ_kg" in feas.columns:
    fig, ax = plt.subplots(figsize=(10, 6))
    if db is not None and "latent_heat_kJ_kg" in db.columns:
        ax.hist(db["latent_heat_kJ_kg"].dropna(), bins=40, alpha=0.5, color="gray", label=f"All candidates (n={len(db)})", density=True)
    ax.hist(feas["latent_heat_kJ_kg"].dropna(), bins=30, alpha=0.8, color="#3b7dd8", label=f"Feasible survivors (n={len(feas)})", density=True)
    ax.axvline(feas["latent_heat_kJ_kg"].median(), color="#3b7dd8", ls="--", lw=2, label=f"Feasible median: {feas['latent_heat_kJ_kg'].median():.0f} kJ/kg")
    ax.set(xlabel="Latent Heat (kJ/kg)", ylabel="Density", title="Comparison 5: Latent Heat Distribution - All Database Candidates vs. Feasible Survivors (Assam)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25); sfig("05_comparison_latent_heat_distribution.png")

# ── Comparison 6: Physics Validation - Solar Fraction & Hours Target Met vs MCDM Rank ────
print("[6/8] Physics validation vs MCDM rank")
phys = load(PHYS, "physics_val")
cmp_phys = load(CMP_PHYS, "mcdm_vs_physics")

if phys is not None:
    if "cluster_id" in phys.columns:
        phys = phys[phys["cluster_id"].isin(VALID_CLUSTERS)]
    
    h_col = "hours_Tw_ge_50C_per_year" if "hours_Tw_ge_50C_per_year" in phys.columns else ("hours_target_met_per_year" if "hours_target_met_per_year" in phys.columns else "hours_target_met")
    sf_col = "solar_fraction" if "solar_fraction" in phys.columns else ("annual_solar_fraction" if "annual_solar_fraction" in phys.columns else None)
    name_col = "pcm_name" if "pcm_name" in phys.columns else ("name" if "name" in phys.columns else "PCM_Name")

    if cmp_phys is not None and "historical_mcdm_rank" in cmp_phys.columns:
        cmp_name = "pcm_name" if "pcm_name" in cmp_phys.columns else "name"
        mg6 = phys.merge(
            cmp_phys[["cluster_id", cmp_name, "historical_mcdm_rank"]].drop_duplicates(),
            left_on=["cluster_id", name_col],
            right_on=["cluster_id", cmp_name],
            how="inner"
        )
        mg6["consensus_rank"] = mg6["historical_mcdm_rank"]
    elif topk is not None:
        topk_name = "name" if "name" in topk.columns else "PCM_Name"
        cols_to_merge = ["cluster_id", topk_name, "consensus_rank"]
        mg6 = phys.merge(
            topk[cols_to_merge].drop_duplicates(subset=["cluster_id", topk_name]),
            left_on=["cluster_id", name_col],
            right_on=["cluster_id", topk_name],
            how="inner"
        )
    else:
        mg6 = None

    if mg6 is not None and "consensus_rank" in mg6.columns and not mg6.empty:
        mg6 = mg6[mg6["cluster_id"].isin(VALID_CLUSTERS)]
        
        fig, axes = plt.subplots(1, 2 if sf_col else 1, figsize=(14 if sf_col else 9, 6))
        ax1 = axes[0] if sf_col else axes
        for cid in sorted(mg6["cluster_id"].unique()):
            g = mg6[mg6["cluster_id"] == cid]
            m = MARKERS.get(int(cid), 'o')
            med = MEDOID_MAP.get(int(cid), f"C{cid}")
            lbl = f"Cluster {cid} ({med})"
            v = g[["consensus_rank", h_col]].notna().all(axis=1)
            ax1.scatter(g.loc[v, "consensus_rank"], g.loc[v, h_col],
                        color=PAL[int(cid) % len(PAL)], marker=m, s=110, alpha=0.85,
                        edgecolors="black", lw=0.7, label=lbl)
            
        ax1.set_xticks(sorted(mg6["consensus_rank"].unique()))
        ax1.set(xlabel="MCDM Consensus Rank (1 = Best)",
                ylabel="Hours Target Met per Year (Tw >= 50C)",
                title="Physics Validation: Annual Hot Water Delivery Hours vs. MCDM Rank")
        ax1.legend(fontsize=9, loc="upper left")
        ax1.grid(alpha=0.25)
        
        # Annotate key materials for physical interpretability
        rank1_row = mg6[mg6["consensus_rank"] == 1]
        if not rank1_row.empty:
            r1_name = str(rank1_row.iloc[0][name_col]).split()[0]
            ax1.annotate(f"{r1_name} (MCDM #1)", (1, rank1_row[h_col].mean()),
                         textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8, fontweight="bold")
        
        if sf_col:
            ax2 = axes[1]
            scale = 100 if mg6[sf_col].max() <= 1.0 else 1
            for cid in sorted(mg6["cluster_id"].unique()):
                g = mg6[mg6["cluster_id"] == cid]
                m = MARKERS.get(int(cid), 'o')
                med = MEDOID_MAP.get(int(cid), f"C{cid}")
                lbl = f"Cluster {cid} ({med})"
                v = g[["consensus_rank", sf_col]].notna().all(axis=1)
                ax2.scatter(g.loc[v, "consensus_rank"], g.loc[v, sf_col] * scale,
                            color=PAL[int(cid) % len(PAL)], marker=m, s=110, alpha=0.85,
                            edgecolors="black", lw=0.7, label=lbl)
            ax2.set_xticks(sorted(mg6["consensus_rank"].unique()))
            ax2.set(xlabel="MCDM Consensus Rank (1 = Best)",
                    ylabel="Annual Solar Thermal Fraction (%)",
                    title="Physics Validation: Annual Solar Fraction (%) vs. MCDM Rank")
            ax2.legend(fontsize=9, loc="upper left")
            ax2.grid(alpha=0.25)
            
            if not rank1_row.empty:
                r1_name = str(rank1_row.iloc[0][name_col]).split()[0]
                ax2.annotate(f"{r1_name} (MCDM #1)", (1, rank1_row[sf_col].mean() * scale),
                             textcoords="offset points", xytext=(0, -15), ha="center", fontsize=8, fontweight="bold")
        
        plt.suptitle("Comparison 6: Grey-Box Physics Validation vs. MCDM Consensus Rank (Assam K=3)\n(Medoid Weather Data Over 10-Year ERA5 Climatology)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        sfig("06_comparison_physics_vs_rank.png")

# ── Comparison 7: Cross-Cluster Top PCM Key Properties ───────────────────
print("[7/8] Cross-cluster summary: top PCM properties")
if topk is not None and "consensus_rank" in topk.columns:
    top1 = topk[topk["consensus_rank"] == 1].drop_duplicates(subset="cluster_id").copy()
    name_col = "name" if "name" in top1.columns else "PCM_Name"
    props = [c for c in ["Tm_C", "latent_heat_kJ_kg", "rho_H_MJ_m3", "TC_W_mK", "cycles_tested"] if c in top1.columns]
    if len(props) >= 2:
        fig, axes = plt.subplots(1, len(props), figsize=(4 * len(props), 5))
        if len(props) == 1: axes = [axes]
        for i, p in enumerate(props):
            axes[i].bar(top1["cluster_id"].astype(str), top1[p], color=[PAL[int(c) % len(PAL)] for c in top1["cluster_id"]], edgecolor="white")
            axes[i].set(title=p, xlabel="Cluster"); axes[i].grid(alpha=0.3, axis="y")
            for j, (_, row) in enumerate(top1.iterrows()):
                if pd.notna(row.get(p)) and name_col in top1.columns:
                    axes[i].text(j, row[p] * 1.01, str(row.get(name_col, ""))[:10], ha="center", fontsize=7, rotation=25)
        plt.suptitle("Comparison 7: Consensus Rank #1 Candidate Properties Across Regimes (Assam K=3)\n(Unanimous Selection: RT44HC across Clusters 0, 1, and 2)", fontsize=11, fontweight="bold")
        plt.tight_layout(); sfig("07_comparison_cross_cluster_top_pcm.png")

# ── Comparison 8: Weight Sensitivity ─────────────────────────────────────
print("[8/8] Rank sensitivity to weight perturbation")
full_df = load(os.path.join(BASE, "data", "processed", "pcm", "mcdm_full_scores_assam.csv"), "mcdm_full")
if full_df is None:
    full_df = topk

if full_df is not None:
    full_df = ensure_ranks(full_df)
    score_cols = [c for c in ["topsis_score", "gra_grade", "promethee_flow"] if c in full_df.columns]
    name_col = "name" if "name" in full_df.columns else "PCM_Name"
    
    if len(score_cols) >= 2 and "cluster_id" in full_df.columns:
        c1, c2 = score_cols[0], score_cols[1]
        name1 = c1.replace("_score", "").replace("_grade", "").upper()
        name2 = c2.replace("_score", "").replace("_grade", "").upper()
        
        regimes = {
            0: ("Cluster 0: Lower Brahmaputra Valley", "ASP_0012"),
            1: ("Cluster 1: Upper Assam Tea Belt", "ASP_0092"),
            2: ("Cluster 2: Barak Valley & Southern Hills", "ASP_0028")
        }
        
        cids = sorted(full_df["cluster_id"].unique())
        nc = len(cids)
        
        w_vals = np.linspace(0.0, 1.0, 21)
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        markers = ["o", "s", "^", "D", "v", "p", "h"]
        
        fig, axes = plt.subplots(1, nc, figsize=(5.5 * nc, 5.5), sharey=True)
        if nc == 1:
            axes = [axes]
            
        for idx, cid in enumerate(cids):
            ax = axes[idx]
            sub = full_df[full_df["cluster_id"] == cid].copy()
            r_name, med = regimes.get(int(cid), (f"Cluster {cid}", f"C{cid}"))
            
            top_max = sub[c1].max()
            gra_max = sub[c2].max()
            sub["s1"] = sub[c1] / max(1e-6, top_max)
            sub["s2"] = sub[c2] / max(1e-6, gra_max)
            
            cand_trajectories = {nm: [] for nm in sub[name_col]}
            
            for w in w_vals:
                comb = w * sub["s1"] + (1 - w) * sub["s2"]
                ranks = comb.rank(ascending=False, method="min").astype(int)
                for nm, r in zip(sub[name_col], ranks):
                    cand_trajectories[nm].append(r)
            
            sort_col = "consensus_rank" if "consensus_rank" in sub.columns else ("topsis_rank" if "topsis_rank" in sub.columns else name_col)
            sorted_names = sub.sort_values(sort_col)[name_col].tolist()
            
            for ci, nm in enumerate(sorted_names):
                ranks = cand_trajectories[nm]
                c = palette[ci % len(palette)]
                m = markers[ci % len(markers)]
                clean_name = str(nm).replace(" (docosane-class paraffin)", "").replace("savE®", "savE").replace("savE", "savE®")
                ax.plot(w_vals, ranks, marker=m, markersize=5, lw=1.8, color=c, label=clean_name, alpha=0.9)
            
            ax.set_title(f"{r_name}\n(Medoid: {med})", fontsize=11, fontweight="bold", pad=8)
            ax.set_xlabel(f"Weight on {name1} ($w$)\n[Remainder $1-w$ on {name2}]", fontsize=10, fontweight="bold")
            ax.set_xticks(np.arange(0.0, 1.1, 0.2))
            ax.set_yticks(range(1, len(sub) + 1))
            ax.invert_yaxis()
            ax.grid(alpha=0.25, linestyle="--")
            ax.legend(fontsize=8, loc="lower left", framealpha=0.92)
            
        axes[0].set_ylabel("MCDM Rank (1 = Best)", fontsize=11, fontweight="bold")
        plt.suptitle(f"Comparison 8: MCDM Rank Sensitivity to Decision Method Weighting (Assam K=3)\nContinuous Blending: Composite Score = $w \\cdot \\mathrm{{{name1}}} + (1-w) \\cdot \\mathrm{{{name2}}}$ Across Feasible Survivors",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        sfig("08_comparison_rank_sensitivity.png")

print("\nAll comparison plots saved to:", OUT)
