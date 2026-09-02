"""
Verification Script: MCDM Ranking Verification
============================================

Purpose:
  Validate the actual ranking outputs the project writes:
  - use the saved TOPSIS / GRA / consensus ranks from mcdm_topk_by_cluster.csv
  - compare ranking agreement using the real columns available in the dataset
  - summarize which PCMs dominate the final ranking

Output folder: data/plots/verify_ranking/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

INPUT_TOPK = "data/processed/pcm/mcdm_topk_by_cluster.csv"
INPUT_FULL = "data/processed/pcm/mcdm_full_scores_by_cluster.csv"
OUTPUT_DIR = "data/plots/verify_ranking"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
try:
    topk = pd.read_csv(INPUT_TOPK)
    full = pd.read_csv(INPUT_FULL)
    print(f"  Top-k results: {topk.shape}")
    print(f"  Full scores: {full.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure you have run 08_mcdm_ranking.py first.")
    raise SystemExit(1)

if 'topsis_rank' not in topk.columns and 'topsis_score' in topk.columns:
    topk['topsis_rank'] = topk.groupby('cluster_id')['topsis_score'].rank(ascending=False, method='min').astype(int)
if 'gra_rank' not in topk.columns and 'gra_grade' in topk.columns:
    topk['gra_rank'] = topk.groupby('cluster_id')['gra_grade'].rank(ascending=False, method='min').astype(int)
if 'consensus_rank' not in topk.columns and 'borda_score' in topk.columns:
    topk['consensus_rank'] = topk.groupby('cluster_id')['borda_score'].rank(ascending=False, method='min').astype(int)

rank_cols = [c for c in ['topsis_rank', 'gra_rank', 'consensus_rank'] if c in topk.columns]
score_cols = [c for c in full.columns if c.endswith('_score') or c.endswith('_grade')]
print(f"Ranking columns used: {rank_cols}")
print(f"Score columns available: {score_cols[:10]}")

print("\n[1/6] Computing rank correlations...")
fig, ax = plt.subplots(figsize=(10, 8))
if len(rank_cols) >= 2:
    matrix = np.eye(len(rank_cols))
    for i, c1 in enumerate(rank_cols):
        for j, c2 in enumerate(rank_cols):
            if i < j:
                valid = topk[c1].notna() & topk[c2].notna()
                if valid.sum() > 1:
                    corr, _ = spearmanr(topk.loc[valid, c1], topk.loc[valid, c2])
                    matrix[i, j] = corr
                    matrix[j, i] = corr
    labels = [c.replace('_rank', '').upper() for c in rank_cols]
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0.7,
                xticklabels=labels, yticklabels=labels, ax=ax,
                vmin=-1, vmax=1, cbar_kws={'label': 'Spearman ρ'})
    ax.set_title('MCDM method rank correlation')
else:
    ax.text(0.5, 0.5, 'Insufficient ranking methods available', ha='center', va='center',
            transform=ax.transAxes, fontsize=12, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_method_correlation.png'), dpi=150)
plt.close()
print('  ✓ Saved: 01_method_correlation.png')

print('[2/6] Computing top-3 inclusion probability...')
fig, ax = plt.subplots(figsize=(12, 6))
if 'name' in topk.columns and rank_cols:
    inclusion = {}
    for _, row in topk.iterrows():
        candidate = row['name']
        in_top3 = sum(1 for c in rank_cols if pd.notna(row.get(c)) and row[c] <= 3)
        inclusion[candidate] = inclusion.get(candidate, 0) + in_top3
    if inclusion:
        probs = {k: 100 * v / len(rank_cols) for k, v in inclusion.items()}
        top_candidates = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:15]
        names, vals = zip(*top_candidates)
        bars = ax.barh(range(len(names)), vals, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.set_xlabel('Top-3 inclusion probability (%)')
        ax.set_title('PCM frequency in top-3 across ranking methods')
        ax.axvline(80, color='green', linestyle='--', linewidth=2, label='Good (≥80%)')
        ax.axvline(50, color='orange', linestyle='--', linewidth=2, label='Fair (≥50%)')
        ax.legend()
        ax.grid(alpha=0.3, axis='x')
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.0f}%', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_top3_inclusion_probability.png'), dpi=150)
plt.close()
print('  ✓ Saved: 02_top3_inclusion_probability.png')

print('[3/6] Generating rank distributions...')
fig, ax = plt.subplots(figsize=(10, 6))
if len(rank_cols) >= 2:
    rows = []
    for c in rank_cols:
        for r in topk[c].dropna().tolist():
            rows.append({'Method': c.replace('_rank', '').upper(), 'Rank': float(r)})
    rank_df = pd.DataFrame(rows)
    sns.boxplot(data=rank_df, x='Method', y='Rank', ax=ax, palette='Set2')
    ax.set_ylabel('Rank position')
    ax.set_title('Rank distribution across methods')
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_rank_distributions.png'), dpi=150)
plt.close()
print('  ✓ Saved: 03_rank_distributions.png')

print('[4/6] Computing rank reversal frequency...')
fig, ax = plt.subplots(figsize=(10, 6))
if 'name' in topk.columns and len(rank_cols) >= 2:
    data = []
    for _, row in topk.iterrows():
        ranks = [row[c] for c in rank_cols if pd.notna(row.get(c))]
        if len(ranks) > 1:
            spread = max(ranks) - min(ranks)
            data.append({'Candidate': row['name'], 'Spread': spread})
    reversal_df = pd.DataFrame(data).sort_values('Spread', ascending=False).head(15)
    bars = ax.barh(range(len(reversal_df)), reversal_df['Spread'], color='coral', edgecolor='black')
    ax.set_yticks(range(len(reversal_df)))
    ax.set_yticklabels(reversal_df['Candidate'].tolist(), fontsize=10)
    ax.set_xlabel('Absolute rank spread')
    ax.set_title('Rank instability across ranking methods')
    ax.grid(alpha=0.3, axis='x')
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.0f}', ha='left', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_rank_reversal_frequency.png'), dpi=150)
plt.close()
print('  ✓ Saved: 04_rank_reversal_frequency.png')

print('[5/6] Generating agreement analysis...')
fig, ax = plt.subplots(figsize=(10, 8))
if 'consensus_rank' in topk.columns and 'topsis_rank' in topk.columns:
    valid = topk[['topsis_rank', 'consensus_rank']].notna().all(axis=1)
    if valid.sum() > 0:
        x = topk.loc[valid, 'topsis_rank']
        y = topk.loc[valid, 'consensus_rank']
        scatter = ax.scatter(x, y, c=topk.loc[valid, 'cluster_id'], cmap='tab10', s=90, alpha=0.7, edgecolors='black')
        max_rank = max(x.max(), y.max())
        ax.plot([1, max_rank], [1, max_rank], 'r--', label='Perfect agreement')
        ax.set_xlabel('TOPSIS rank')
        ax.set_ylabel('Consensus rank')
        ax.set_title('Consensus rank vs. TOPSIS rank')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster ID')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_method_agreement.png'), dpi=150)
plt.close()
print('  ✓ Saved: 05_method_agreement.png')

print('[6/6] Generating ranking summary...')
fig, ax = plt.subplots(figsize=(12, 8))
summary = f"""
MCDM RANKING VALIDATION SUMMARY
{'=' * 60}

Number of methods: {len(rank_cols)}
Methods: {', '.join([c.replace('_rank', '').upper() for c in rank_cols])}
Number of ranked candidates: {len(topk)}
Number of clusters: {topk['cluster_id'].nunique()}

Method agreement (Spearman rho):
"""
if 'topsis_rank' in topk.columns and 'gra_rank' in topk.columns:
    corr, _ = spearmanr(topk['topsis_rank'].dropna(), topk['gra_rank'].dropna())
    summary += f"  TOPSIS vs GRA: {corr:.3f}\n"
if 'topsis_rank' in topk.columns and 'consensus_rank' in topk.columns:
    corr, _ = spearmanr(topk['topsis_rank'].dropna(), topk['consensus_rank'].dropna())
    summary += f"  TOPSIS vs CONSENSUS: {corr:.3f}\n"
if 'gra_rank' in topk.columns and 'consensus_rank' in topk.columns:
    corr, _ = spearmanr(topk['gra_rank'].dropna(), topk['consensus_rank'].dropna())
    summary += f"  GRA vs CONSENSUS: {corr:.3f}\n"
summary += f"\nTop-3 consensus candidates:\n"
if 'name' in topk.columns and 'consensus_rank' in topk.columns:
    top3 = topk[topk['consensus_rank'] <= 3][['name', 'consensus_rank']].drop_duplicates().sort_values('consensus_rank').head(3)
    for _, row in top3.iterrows():
        summary += f"  {row['consensus_rank']}. {row['name']}\n"
summary += f"\nData completeness: {100 * topk.notna().mean().mean():.1f}%"
ax.text(0.05, 0.95, summary, transform=ax.transAxes, va='top', ha='left',
        fontsize=11, family='monospace', color='black')
ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_summary.png'), dpi=150)
plt.close()
print('  ✓ Saved: 06_summary.png')

print('\n' + '=' * 70)
print('MCDM VALIDATION SUMMARY')
print('=' * 70)
print(f'Methods: {rank_cols}')
print(f'Cluster count: {topk["cluster_id"].nunique()}')
print(f'Rows: {len(topk)}')
print(f'Overall completeness: {100 * topk.notna().mean().mean():.1f}%')
print('=' * 70)
