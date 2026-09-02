"""
Verification Script: PCM Feasibility Filtering
============================================

Purpose:
  Validate the actual feasibility filtering output produced by the pipeline:
  - plot actual survivor counts per cluster
  - show the realistic property space the surviving PCM candidates occupy
  - use the real saved target window (`window_lo`, `window_hi`) rather than a generic band

Output folder: data/plots/verify_feasibility/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_ALL_CANDIDATES = "data/processed/pcm/pcm_database_uttarakhand.csv"
INPUT_SURVIVORS = "data/processed/pcm/feasibility_survivors_by_cluster.csv"
OUTPUT_DIR = "data/plots/verify_feasibility"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
try:
    survivors = pd.read_csv(INPUT_SURVIVORS)
    print(f"  Survivors: {survivors.shape}")
except FileNotFoundError:
    print(f"ERROR: {INPUT_SURVIVORS} not found")
    print("Make sure you have run 07_feasibility_filter.py first.")
    raise SystemExit(1)

try:
    all_candidates = pd.read_csv(INPUT_ALL_CANDIDATES)
    has_full = True
    print(f"  Full database: {all_candidates.shape}")
except FileNotFoundError:
    has_full = False
    print("  ⚠ Full candidate database missing; some summary ratios will be skipped")

print("\n[1/6] Generating survival rates...")
fig, ax = plt.subplots(figsize=(10, 6))
cluster_counts = survivors['cluster_id'].value_counts().sort_index()
clusters_in_data = sorted(survivors['cluster_id'].unique())
if has_full and 'cluster_id' in all_candidates.columns:
    survival_pcts = []
    for cid in clusters_in_data:
        total = len(all_candidates[all_candidates['cluster_id'] == cid])
        surv = len(survivors[survivors['cluster_id'] == cid])
        survival_pcts.append(100 * surv / total if total > 0 else 0)
    bars = ax.bar(clusters_in_data, survival_pcts, color='steelblue', edgecolor='black')
    ax.set_ylabel('Survival rate (%)')
    ax.axhline(10, color='green', linestyle='--', linewidth=2, label='Lower threshold')
    ax.axhline(50, color='green', linestyle='--', linewidth=2, label='Upper threshold')
else:
    bars = ax.bar(clusters_in_data, cluster_counts.values, color='steelblue', edgecolor='black')
    ax.set_ylabel('Number of survivors')
ax.set_xlabel('Cluster ID')
ax.set_title('PCM survival rate per climate regime')
ax.grid(alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_survival_rate_by_cluster.png'), dpi=150)
plt.close()
print('  ✓ Saved: 01_survival_rate_by_cluster.png')

print('[2/6] Generating feasible property space...')
fig, ax = plt.subplots(figsize=(10, 8))
if {'Tm_C', 'latent_heat_kJ_kg'}.issubset(survivors.columns):
    scatter = ax.scatter(survivors['Tm_C'], survivors['latent_heat_kJ_kg'],
                         c=survivors['cluster_id'], cmap='tab10', s=100, alpha=0.6,
                         edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Melting Point (°C)')
    ax.set_ylabel('Latent Heat (kJ/kg)')
    ax.set_title('Feasible PCM property space (survivors only)')
    ax.grid(alpha=0.3)
    if {'window_lo', 'window_hi'}.issubset(survivors.columns):
        lo = survivors['window_lo'].dropna().min()
        hi = survivors['window_hi'].dropna().max()
        ax.axvspan(lo, hi, alpha=0.12, color='green', label=f'Accepted Tm band: {lo:.0f}–{hi:.0f}°C')
    ax.axhline(100, color='gray', linestyle=':', alpha=0.6, label='Typical latent heat floor')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_feasible_property_space.png'), dpi=150)
plt.close()
print('  ✓ Saved: 02_feasible_property_space.png')

print('[3/6] Generating top candidates per cluster...')
fig, ax = plt.subplots(figsize=(12, 6))
if 'latent_heat_kJ_kg' in survivors.columns and 'name' in survivors.columns:
    top_per_cluster = survivors.groupby('cluster_id').apply(
        lambda x: x.nlargest(3, 'latent_heat_kJ_kg') if len(x) >= 3 else x
    ).reset_index(drop=True)
    top_candidates = top_per_cluster['name'].value_counts().head(12)
    bars = ax.barh(range(len(top_candidates)), top_candidates.values, color='coral', edgecolor='black')
    ax.set_yticks(range(len(top_candidates)))
    ax.set_yticklabels(top_candidates.index, fontsize=10)
    ax.set_xlabel('Frequency in top-3 by cluster')
    ax.set_title('Highest-latent-heat survivors per cluster')
    ax.grid(alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_top_candidates_per_cluster.png'), dpi=150)
plt.close()
print('  ✓ Saved: 03_top_candidates_per_cluster.png')

print('[4/6] Generating constraint analysis...')
constraint_cols = [c for c in survivors.columns if 'pass_' in c.lower() or 'constraint' in c.lower()]
if constraint_cols:
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = []
    for c in constraint_cols:
        if c in survivors.columns:
            pass_count = (survivors[c] == True).sum() if pd.api.types.is_bool_dtype(survivors[c]) else survivors[c].sum()
            summary.append({'Constraint': c, 'Pass': pass_count, 'Fail': len(survivors) - pass_count})
    if summary:
        constraint_df = pd.DataFrame(summary).set_index('Constraint')
        constraint_df[['Pass', 'Fail']].plot(kind='barh', stacked=True, ax=ax, color=['green', 'red'], edgecolor='black')
        ax.set_xlabel('Number of candidates')
        ax.set_title('Constraint pass/fail distribution')
        ax.grid(alpha=0.3, axis='x')
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No constraint columns available', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_constraint_analysis.png'), dpi=150)
plt.close()
print('  ✓ Saved: 04_constraint_analysis.png')

print('[5/6] Generating property distributions...')
prop_cols = ['Tm_C', 'latent_heat_kJ_kg', 'density_liquid_kg_m3', 'Cp_liquid_kJ_kgK']
available_props = [p for p in prop_cols if p in survivors.columns][:3]
if available_props:
    fig, axes = plt.subplots(1, len(available_props), figsize=(5 * len(available_props), 5))
    if len(available_props) == 1:
        axes = [axes]
    for i, prop in enumerate(available_props):
        sns.boxplot(data=survivors, x='cluster_id', y=prop, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{prop} by Cluster')
        axes[i].set_ylabel(prop)
        axes[i].grid(alpha=0.3, axis='y')
    plt.suptitle('PCM property distributions (survivors)', fontsize=14, y=1.02)
    plt.tight_layout()
else:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No property columns available', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='gray')
plt.savefig(os.path.join(OUTPUT_DIR, '05_property_distributions.png'), dpi=150)
plt.close()
print('  ✓ Saved: 05_property_distributions.png')

print('[6/6] Generating survivor statistics...')
fig, ax = plt.subplots(figsize=(10, 6))
total_survivors = len(survivors)
unique_clusters = survivors['cluster_id'].nunique()
summary = f"""
PCM FEASIBILITY FILTERING SUMMARY
{'=' * 50}

Total Survivors: {total_survivors}
Number of Clusters: {unique_clusters}
Avg Survivors per Cluster: {total_survivors / unique_clusters:.1f}

Survivors by Cluster:
"""
for cid in sorted(survivors['cluster_id'].unique()):
    summary += f"\n  Cluster {cid}: {len(survivors[survivors['cluster_id'] == cid])} PCMs"
if has_full and 'cluster_id' in all_candidates.columns:
    total_all = len(all_candidates)
    overall_survival = 100 * total_survivors / total_all
    summary += f"\n\nOverall survival rate: {overall_survival:.1f}%"
    if 10 <= overall_survival <= 50:
        summary += ' ✓ (GOOD)'
    elif overall_survival < 10:
        summary += ' ⚠ (TOO STRICT)'
    else:
        summary += ' ⚠ (TOO LOOSE)'
ax.text(0.05, 0.95, summary, transform=ax.transAxes, va='top', ha='left',
        fontsize=12, family='monospace', color='black')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_summary.png'), dpi=150)
plt.close()
print('  ✓ Saved: 06_summary.png')

print('\n' + '=' * 70)
print('FEASIBILITY VALIDATION SUMMARY')
print('=' * 70)
print(f'Total survivors: {total_survivors}')
print(f'Clusters: {unique_clusters}')
print(f'Avg survivors/cluster: {total_survivors / unique_clusters:.1f}')
if has_full and 'cluster_id' in all_candidates.columns:
    print(f'Overall survival rate: {100 * total_survivors / len(all_candidates):.1f}%')
print('=' * 70)
