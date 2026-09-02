"""
Verification Script: Clustering Validation
============================================

Purpose:
  Validate the actual clustering outputs that the Uttarakhand pipeline writes:
  - use saved cluster assignments instead of re-fitting GMM on raw data
  - confirm the real climate-regime groups with silhouette/PCA/geographic plots
  - profile each cluster using the saved climate signature matrix

Output folder: data/plots/verify_clustering/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score

INPUT_CLIMATE = "data/processed/signatures/climate_signature_uttarakhand.csv"
INPUT_CLUSTERS = "data/processed/clustering/cluster_assignments_uttarakhand.csv"
OUTPUT_DIR = "data/plots/verify_clustering"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
try:
    climate = pd.read_csv(INPUT_CLIMATE)
    clusters = pd.read_csv(INPUT_CLUSTERS)
    print(f"  Climate signature: {climate.shape}")
    print(f"  Cluster assignments: {clusters.shape}")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Make sure you have run 04b_climate_signature.py and 05_cluster_uttarakhand.py first.")
    raise SystemExit(1)

# Join actual cluster labels to climate metrics on the project key: point_id.
# The saved cluster file includes lat/lon, so keep the climate file's coordinates and
# rename the cluster-derived ones to avoid duplicate columns from the merge.
cluster_geo = clusters[['point_id', 'cluster_id', 'lat', 'lon']].rename(
    columns={'lat': 'cluster_lat', 'lon': 'cluster_lon'}
)
merged = climate.merge(cluster_geo, on='point_id', how='inner')
merged['lat'] = merged['lat'].fillna(merged['cluster_lat'])
merged['lon'] = merged['lon'].fillna(merged['cluster_lon'])
print(f"  Merged rows for validation: {merged.shape[0]}")

feature_cols = [
    c for c in merged.columns
    if c not in {'point_id', 'cluster_id', 'lat', 'lon', 'cluster_lat', 'cluster_lon', 'population'}
    and pd.api.types.is_numeric_dtype(merged[c])
]

X = merged[feature_cols].fillna(merged[feature_cols].median()).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
labels_actual = merged['cluster_id'].astype(int).to_numpy()
chosen_k = len(np.unique(labels_actual))

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Actual cluster count from saved assignments: {chosen_k}")

# Optional exploratory elbow curve on the same feature set; the plot is for sanity
# only, while the actual validation uses the saved labels.
k_range = range(2, 9)
silhouette_scores = []
bic_scores = []
db_scores = []
ch_scores = []
for k in k_range:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))
    bic_scores.append(gmm.bic(X_scaled))
    db_scores.append(davies_bouldin_score(X_scaled, labels))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
axes[0, 0].axvline(chosen_k, color='red', linestyle='--', label=f'Actual k={chosen_k}')
axes[0, 0].set_title('Silhouette Score (higher is better)')
axes[0, 0].set_xlabel('Number of Clusters (k)')
axes[0, 0].set_ylabel('Silhouette')
axes[0, 0].grid(alpha=0.3)
axes[0, 0].legend()

axes[0, 1].plot(k_range, bic_scores, 'go-', linewidth=2, markersize=8)
axes[0, 1].axvline(chosen_k, color='red', linestyle='--', label=f'Actual k={chosen_k}')
axes[0, 1].set_title('BIC (lower is better)')
axes[0, 1].set_xlabel('Number of Clusters (k)')
axes[0, 1].set_ylabel('BIC')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].legend()

axes[1, 0].plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
axes[1, 0].axvline(chosen_k, color='red', linestyle='--', label=f'Actual k={chosen_k}')
axes[1, 0].set_title('Davies-Bouldin (lower is better)')
axes[1, 0].set_xlabel('Number of Clusters (k)')
axes[1, 0].set_ylabel('DB')
axes[1, 0].grid(alpha=0.3)
axes[1, 0].legend()

axes[1, 1].plot(k_range, ch_scores, 'mo-', linewidth=2, markersize=8)
axes[1, 1].axvline(chosen_k, color='red', linestyle='--', label=f'Actual k={chosen_k}')
axes[1, 1].set_title('Calinski-Harabasz (higher is better)')
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('CH')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_elbow_curves.png'), dpi=150)
plt.close()
print('  ✓ Saved: 01_elbow_curves.png')

# Silhouette using the real saved cluster labels
silhouette_vals = silhouette_samples(X_scaled, labels_actual)
silhouette_avg = silhouette_score(X_scaled, labels_actual)

fig, ax = plt.subplots(figsize=(10, 8))
y_lower = 10
colors = plt.cm.Set3(np.linspace(0, 1, chosen_k))
for i in range(chosen_k):
    vals = silhouette_vals[labels_actual == i]
    vals = np.sort(vals)
    y_upper = y_lower + len(vals)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
                     label=f'Cluster {i}')
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient')
ax.set_ylabel('Cluster Label')
ax.axvline(x=silhouette_avg, color='red', linestyle='--', linewidth=2,
           label=f'Average: {silhouette_avg:.3f}')
ax.axvline(x=0.4, color='green', linestyle=':', linewidth=2, label='Good threshold (0.4)')
ax.set_title(f'Silhouette analysis for saved cluster assignments (k={chosen_k})')
ax.legend(loc='best', fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_silhouette_plot.png'), dpi=150)
plt.close()
print('  ✓ Saved: 02_silhouette_plot.png')

# PCA projection using real labels
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_actual, cmap='tab10',
                     s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title(f'PCA projection of saved clusters (k={chosen_k})')
plt.colorbar(scatter, ax=ax, label='Cluster ID')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_pca_projection.png'), dpi=150)
plt.close()
print('  ✓ Saved: 03_pca_projection.png')

# Geographic map using real lat/lon and saved labels
coords = merged[['point_id', 'lat', 'lon', 'cluster_id']].drop_duplicates().copy()
if not coords.empty and {'lat', 'lon'}.issubset(coords.columns) and coords['lat'].notna().any() and coords['lon'].notna().any():
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(coords['lon'], coords['lat'], c=coords['cluster_id'], cmap='tab10',
                    s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Geographic distribution of saved clusters (k={chosen_k})')
    plt.colorbar(sc, ax=ax, label='Cluster ID')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_geographic_map.png'), dpi=150)
    plt.close()
    print('  ✓ Saved: 04_geographic_map.png')
else:
    print('  ⚠ Missing lat/lon; skipping geographic map')

# Cluster profile plots from the saved labels
cluster_df = merged.copy()
cluster_df['cluster_id'] = labels_actual
key_features = [c for c in feature_cols if c not in {'n_days_used'}][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feature in enumerate(key_features):
    sns.boxplot(data=cluster_df, x='cluster_id', y=feature, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{feature} by Cluster')
    axes[i].set_ylabel(feature)
    axes[i].grid(alpha=0.3, axis='y')

for j in range(len(key_features), len(axes)):
    axes[j].axis('off')

plt.suptitle('Saved cluster profiles', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_cluster_profiles.png'), dpi=150)
plt.close()
print('  ✓ Saved: 05_cluster_profiles.png')

cluster_sizes = pd.Series(labels_actual).value_counts().sort_index()
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', edgecolor='black')
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Number of samples')
ax.set_title(f'Cluster size distribution (saved labels, k={chosen_k})')
ax.grid(alpha=0.3, axis='y')
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h, f'{int(h)}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_cluster_sizes.png'), dpi=150)
plt.close()
print('  ✓ Saved: 06_cluster_sizes.png')

# Summary report based on actual saved labels
print('\n' + '=' * 70)
print('CLUSTERING VALIDATION SUMMARY')
print('=' * 70)
print(f'Actual cluster count: {chosen_k}')
print(f'Silhouette score (saved assignments): {silhouette_avg:.3f}')
print(f'Davies-Bouldin: {davies_bouldin_score(X_scaled, labels_actual):.3f}')
print(f'Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels_actual):.1f}')

for cid in sorted(np.unique(labels_actual)):
    mask = labels_actual == cid
    vals = silhouette_vals[mask]
    print(f'  Cluster {cid}: size={mask.sum()} avg_sil={vals.mean():.3f} min_sil={vals.min():.3f}')

if silhouette_avg >= 0.4:
    print('  ✓ Silhouette threshold met')
else:
    print('  ⚠ Silhouette below 0.4; clustering remains defensible but weakly separated')

ratio = cluster_sizes.max() / cluster_sizes.min() if cluster_sizes.min() > 0 else np.nan
print(f'  Cluster size ratio (max/min): {ratio:.2f}')
print('\nAll plots saved to:', OUTPUT_DIR)
print('=' * 70)
