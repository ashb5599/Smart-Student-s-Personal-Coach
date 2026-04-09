# ============================================================
# PHASE 4 — CLUSTERING ALGORITHMS
# K-Means, Hierarchical (Agglomerative), DBSCAN
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# ── LOAD ────────────────────────────────────────────────────
X_scaled, y_reg, y_clf, X_raw, df = joblib.load('processed_data.pkl')
print(f"Loaded: X={X_scaled.shape}")

# Use PCA for visualization (reduce to 2D)
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_scaled)
print(f"PCA 2D explained variance: {pca.explained_variance_ratio_.sum():.3f}")

COLORS_3 = ['#ef4444', '#f59e0b', '#22c55e']
LABELS_3 = ['At-Risk 🔴', 'Average 🟡', 'High Performer 🟢']

# ============================================================
# ALGORITHM 1 — K-MEANS CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("ALGORITHM 1 — K-MEANS CLUSTERING")
print("=" * 60)

# 1a. Elbow Method + Silhouette to find optimal k
inertias, silhouettes = [], []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means: Choosing Optimal k', color='white', fontsize=13)
axes[0].plot(k_range, inertias, 'o-', color='#818cf8', lw=2, markersize=7)
axes[0].set_title('Elbow Method (Inertia)', color='white')
axes[0].set_xlabel('Number of Clusters k'); axes[0].set_ylabel('Inertia')
axes[1].plot(k_range, silhouettes, 's-', color='#34d399', lw=2, markersize=7)
axes[1].set_title('Silhouette Score', color='white')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette')
plt.tight_layout()
plt.savefig('plot_10_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()

optimal_k = k_range[np.argmax(silhouettes)]
print(f"  Optimal k = {optimal_k} (Silhouette = {max(silhouettes):.4f})")

# 1b. Final K-Means with k=3 (for Low/Mid/High interpretation)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, km_labels)
dbi = davies_bouldin_score(X_scaled, km_labels)
chi = calinski_harabasz_score(X_scaled, km_labels)

print(f"\n  K-Means (k=3) Metrics:")
print(f"    Silhouette Score     : {sil:.4f}  (higher = better)")
print(f"    Davies-Bouldin Index : {dbi:.4f}  (lower = better)")
print(f"    Calinski-Harabasz    : {chi:.4f}  (higher = better)")

# 1c. 2D Visualization
fig, ax = plt.subplots(figsize=(9, 6))
for i in range(3):
    mask = km_labels == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               color=COLORS_3[i], label=LABELS_3[i], alpha=0.6, s=18)
centers_2d = pca.transform(kmeans.cluster_centers_)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='white', marker='X', s=200, zorder=5, label='Centroids')
ax.set_title('K-Means Clusters (PCA 2D)', color='white', fontsize=12)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(facecolor='#1e293b', labelcolor='white')
plt.tight_layout()
plt.savefig('plot_11_kmeans_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# Add cluster labels to df
km_cluster_df = pd.DataFrame({'KMeans_Cluster': km_labels})

# ============================================================
# ALGORITHM 2 — HIERARCHICAL (AGGLOMERATIVE) CLUSTERING
# ============================================================
print("\n" + "=" * 60)
print("ALGORITHM 2 — HIERARCHICAL CLUSTERING")
print("=" * 60)

# 2a. Dendrogram (use subset for speed)
sample_idx = np.random.choice(len(X_scaled), size=min(300, len(X_scaled)), replace=False)
Z = linkage(X_scaled[sample_idx], method='ward')

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode='lastp', p=30,
           color_threshold=Z[-3, 2], above_threshold_color='#64748b')
ax.set_title('Hierarchical Clustering Dendrogram (Ward Linkage)', color='white', fontsize=12)
ax.set_xlabel('Sample Index (truncated)')
ax.set_ylabel('Distance')
plt.tight_layout()
plt.savefig('plot_12_dendrogram.png', dpi=150, bbox_inches='tight')
plt.show()

# 2b. Fit Agglomerative
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_labels = agg.fit_predict(X_scaled)

sil_a = silhouette_score(X_scaled, agg_labels)
dbi_a = davies_bouldin_score(X_scaled, agg_labels)
chi_a = calinski_harabasz_score(X_scaled, agg_labels)

print(f"  Hierarchical (k=3) Metrics:")
print(f"    Silhouette Score     : {sil_a:.4f}")
print(f"    Davies-Bouldin Index : {dbi_a:.4f}")
print(f"    Calinski-Harabasz    : {chi_a:.4f}")

# 2c. Visualize
fig, ax = plt.subplots(figsize=(9, 6))
for i in range(3):
    mask = agg_labels == i
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               color=COLORS_3[i], label=LABELS_3[i], alpha=0.6, s=18)
ax.set_title('Hierarchical Clustering (PCA 2D)', color='white', fontsize=12)
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(facecolor='#1e293b', labelcolor='white')
plt.tight_layout()
plt.savefig('plot_13_hierarchical_clusters.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# ALGORITHM 3 — DBSCAN
# ============================================================
print("\n" + "=" * 60)
print("ALGORITHM 3 — DBSCAN CLUSTERING")
print("=" * 60)

# Test multiple eps values
print(f"  {'eps':>5} {'min_samples':>12} {'Clusters':>9} {'Noise':>7} {'Silhouette':>12}")
print("  " + "-" * 50)

best_dbscan = None
best_dbscan_sil = -1

for eps in [0.5, 1.0, 1.5, 2.0]:
    for min_s in [5, 10, 15]:
        db = DBSCAN(eps=eps, min_samples=min_s)
        db_labels = db.fit_predict(X_scaled)
        n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()
        if n_clusters >= 2 and n_noise < len(X_scaled) * 0.3:
            sil_d = silhouette_score(X_scaled[db_labels != -1], db_labels[db_labels != -1]) if n_clusters >= 2 else -1
            print(f"  {eps:>5} {min_s:>12} {n_clusters:>9} {n_noise:>7} {sil_d:>12.4f}")
            if sil_d > best_dbscan_sil:
                best_dbscan_sil = sil_d
                best_dbscan = (eps, min_s, db_labels)

if best_dbscan:
    eps_b, ms_b, db_labels_best = best_dbscan
    print(f"\n  Best DBSCAN: eps={eps_b}, min_samples={ms_b}, Silhouette={best_dbscan_sil:.4f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    unique_labels = sorted(set(db_labels_best))
    dbscan_colors = ['#64748b'] + ['#818cf8','#34d399','#f472b6','#fbbf24','#38bdf8'][:len(unique_labels)-1]
    for i, label in enumerate(unique_labels):
        mask = db_labels_best == label
        lname = 'Noise (Outliers)' if label == -1 else f'Cluster {label}'
        color = '#64748b' if label == -1 else dbscan_colors[i % len(dbscan_colors)]
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=color, label=lname, alpha=0.6, s=18)
    ax.set_title('DBSCAN Clusters (PCA 2D)', color='white', fontsize=12)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(facecolor='#1e293b', labelcolor='white')
    plt.tight_layout()
    plt.savefig('plot_14_dbscan_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("  ⚠️ DBSCAN: No good configuration found. Try on 2D PCA data instead.")
    db = DBSCAN(eps=1.5, min_samples=10)
    db_labels_best = db.fit_predict(X_2d)

# ============================================================
# CLUSTERING COMPARISON TABLE
# ============================================================
print("\n" + "=" * 60)
print("CLUSTERING ALGORITHM COMPARISON")
print("=" * 60)

cluster_comparison = pd.DataFrame([
    {'Algorithm': 'K-Means (k=3)',       'Silhouette': sil,   'Davies-Bouldin': dbi,   'Calinski-Harabasz': chi},
    {'Algorithm': 'Hierarchical (k=3)',   'Silhouette': sil_a, 'Davies-Bouldin': dbi_a, 'Calinski-Harabasz': chi_a},
    {'Algorithm': f'DBSCAN (best)',       'Silhouette': best_dbscan_sil if best_dbscan else np.nan, 'Davies-Bouldin': np.nan, 'Calinski-Harabasz': np.nan},
])
print(cluster_comparison.to_string(index=False))

# ── SAVE ────────────────────────────────────────────────────
cluster_comparison.to_csv('results_clustering.csv', index=False)
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump({'kmeans': km_labels, 'hierarchical': agg_labels}, 'cluster_labels.pkl')

print("\n✅ Phase 4 Complete — Clustering Done!")
print("   Saved: results_clustering.csv, kmeans_model.pkl")
print("   Next: Run phase5_deep_learning.py")
