"""
ΕΡΩΤΗΜΑ 3: Ομαδοποίηση (K-Means, Hierarchical, DBSCAN)
Προϋπόθεση: να έχει τρέξει το question1_eda.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                             adjusted_rand_score, normalized_mutual_info_score)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(_HERE, "output")
CLEANED_CSV = os.path.join(_HERE, "cleaned_dataset.csv")


# 1. Φόρτωση και στρωματοποιημένο δείγμα 5000
# Μικρό δείγμα επειδή το Hierarchical Clustering είναι O(n²) σε μνήμη/χρόνο.
df_full = pd.read_csv(CLEANED_CSV)
_, df = train_test_split(df_full, test_size=5000,
                          stratify=df_full["Label"], random_state=42)
df = df.reset_index(drop=True)
print(f"Δείγμα: {df.shape}")


# 2. Προεπεξεργασία
# Το Label ΔΕΝ χρησιμοποιείται στην ομαδοποίηση (unsupervised) -
# μόνο εκ των υστέρων για ερμηνεία.
X = df.drop("Label", axis=1)
y_true = df["Label"].values
y_bin  = (y_true != "BENIGN").astype(int)   # 0=BENIGN, 1=ATTACK (για viz)

# Τυποποίηση: οι αλγόριθμοι ομαδοποίησης βασίζονται σε αποστάσεις -
# χωρίς scaling, οι στήλες μεγάλης κλίμακας θα κυριαρχούσαν.
X_scaled = StandardScaler().fit_transform(X)


def metrics(labels):
    # Εξαιρούμε τον θόρυβο (-1 του DBSCAN) από τους υπολογισμούς.
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2:
        return n_clusters, int((~mask).sum()), np.nan, np.nan
    return (n_clusters, int((~mask).sum()),
            silhouette_score(X_scaled[mask], labels[mask]),
            davies_bouldin_score(X_scaled[mask], labels[mask]))


# 3. K-Means: διερεύνηση του k
print("\n--- K-Means ---")
k_range = list(range(2, 11))
km_labels, km_sil, km_db = [], [], []
for k in k_range:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_scaled)
    _, _, sil, db = metrics(labels)
    km_labels.append(labels); km_sil.append(sil); km_db.append(db)
    print(f"  k={k:2d} | Silhouette={sil:.3f} | Davies-Bouldin={db:.3f}")

best_k_idx = int(np.argmax(km_sil))
best_k = k_range[best_k_idx]
kmeans_labels = km_labels[best_k_idx]
print(f"  -> Επιλεγμένο k = {best_k}")

# Γράφημα Silhouette & Davies-Bouldin ανά k.
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(k_range, km_sil, "o-", color="steelblue")
axes[0].axvline(best_k, color="red", linestyle="--")
axes[0].set_title("Silhouette ανά k (μεγαλύτερο = καλύτερο)")
axes[0].set_xlabel("k")
axes[1].plot(k_range, km_db, "o-", color="darkorange")
axes[1].axvline(best_k, color="red", linestyle="--")
axes[1].set_title("Davies-Bouldin ανά k (μικρότερο = καλύτερο)")
axes[1].set_xlabel("k")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "10_kmeans_k_selection.png"), dpi=100)
plt.close()


# 4. Hierarchical Clustering: σύγκριση 3 linkages
print("\n--- Hierarchical Clustering ---")
hc_results = {}
for link in ["ward", "complete", "average"]:
    labels = AgglomerativeClustering(n_clusters=best_k, linkage=link).fit_predict(X_scaled)
    _, _, sil, db = metrics(labels)
    smallest = int(pd.Series(labels).value_counts().min())
    hc_results[link] = (labels, sil, db, smallest)
    print(f"  linkage={link:9s} | Silhouette={sil:.3f} | Davies-Bouldin={db:.3f} | μικρότερη ομάδα={smallest}")

# Φιλτράρισμα εκφυλισμένων λύσεων: αν η μικρότερη ομάδα είναι <1% του δείγματος,
# το Silhouette μπορεί να είναι τεχνητά υψηλό χωρίς ουσιαστική δομή.
min_size = int(0.01 * len(X_scaled))
meaningful = {l: r for l, r in hc_results.items() if r[3] >= min_size}
best_link = max(meaningful or hc_results, key=lambda l: hc_results[l][1])
hier_labels = hc_results[best_link][0]
print(f"  -> Επιλεγμένο linkage: {best_link} (φιλτράρονται λύσεις με ομάδα <{min_size} σημείων)")

# Δενδρόγραμμα (ward): δείχνει την ιεραρχία των συγχωνεύσεων.
# truncate_mode="lastp" + p=20: μόνο οι τελευταίες 20 συγχωνεύσεις είναι διαβάσιμες.
plt.figure(figsize=(11, 5))
dendrogram(linkage(X_scaled, method="ward"), truncate_mode="lastp", p=20, leaf_rotation=45)
plt.title("Δενδρόγραμμα (linkage=ward, τελευταίες 20 συγχωνεύσεις)")
plt.ylabel("Απόσταση συγχώνευσης")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "11_hierarchical_dendrogram.png"), dpi=100)
plt.close()


# 5. DBSCAN: επιλογή eps και min_samples
print("\n--- DBSCAN ---")
# k-distance plot: για κάθε σημείο η απόσταση από τον 5ο γείτονα, ταξινομημένη.
# Η "γωνία" της καμπύλης υποδεικνύει καλή τιμή eps.
nn = NearestNeighbors(n_neighbors=5).fit(X_scaled)
kdist = np.sort(nn.kneighbors(X_scaled)[0][:, -1])

plt.figure(figsize=(8, 4))
plt.plot(kdist, color="steelblue")
plt.yscale("log")
plt.title("k-distance plot (5ος γείτονας, log-y)")
plt.xlabel("Σημεία (ταξινομημένα)")
plt.ylabel("Απόσταση")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "12_dbscan_kdistance.png"), dpi=100)
plt.close()

# Πλέγμα eps × min_samples. Τα eps από εκατοστημόρια του k-distance
# (αυτόματα στην κλίμακα των δεδομένων).
eps_grid = np.round(np.percentile(kdist, [80, 90, 95]), 2)
dbscan_results = []
for eps in eps_grid:
    for ms in [5, 10]:
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_scaled)
        n_cl, n_noise, sil, _ = metrics(labels)
        dbscan_results.append({"eps": eps, "min_samples": ms, "labels": labels,
                               "n_clusters": n_cl, "sil": sil})
        sil_str = "—" if np.isnan(sil) else f"{sil:.3f}"
        print(f"  eps={eps:5.2f} ms={ms:2d} | ομάδες={n_cl} θόρυβος={n_noise:4d} | Sil={sil_str}")

# Επιλογή με μέγιστο Silhouette (αν υπάρχει έγκυρη).
best_db = max((r for r in dbscan_results if r["n_clusters"] >= 2 and not np.isnan(r["sil"])),
              key=lambda r: r["sil"])
dbscan_labels = best_db["labels"]
print(f"  -> Επιλεγμένο: eps={best_db['eps']}, min_samples={best_db['min_samples']}")


# 6. Οπτικοποίηση με PCA (μείωση σε 2D)
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
pc1, pc2 = pca.explained_variance_ratio_ * 100
print(f"\nPCA: PC1+PC2 εξηγούν το {pc1+pc2:.1f}% της διακύμανσης")


# Ευθυγράμμιση cluster IDs με το ground truth (μέσω Hungarian matching)
# ώστε το ίδιο χρώμα να σημαίνει σημασιολογικά το ίδιο σε όλα τα panels.
from scipy.optimize import linear_sum_assignment

def align_labels(pred_labels, true_labels):
    pred_unique = np.unique(pred_labels[pred_labels != -1])
    true_unique = np.unique(true_labels)
    n = max(len(pred_unique), len(true_unique))
    cost = np.zeros((n, n))
    for i, p in enumerate(pred_unique):
        for j, t in enumerate(true_unique):
            cost[i, j] = -np.sum((pred_labels == p) & (true_labels == t))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = {pred_unique[r]: (true_unique[c] if c < len(true_unique) else c)
               for r, c in zip(row_ind, col_ind) if r < len(pred_unique)}
    return np.array([mapping.get(l, l) for l in pred_labels])

kmeans_aligned = align_labels(kmeans_labels, y_bin)
hier_aligned   = align_labels(hier_labels,   y_bin)


def scatter(ax, labels, title, cmap="tab10"):
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")

fig, axes = plt.subplots(2, 2, figsize=(13, 11))
scatter(axes[0, 0], kmeans_aligned, f"K-Means (k={best_k})")
scatter(axes[0, 1], hier_aligned,   f"Hierarchical ({best_link})")
scatter(axes[1, 0], dbscan_labels,  f"DBSCAN (eps={best_db['eps']}, ms={best_db['min_samples']})",
        cmap="tab20")
scatter(axes[1, 1], y_bin,          "Πραγματικό Label (BENIGN/ATTACK)")
plt.suptitle("Σύγκριση clusters με πραγματικές κλάσεις (PCA 2D)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "13_pca_clusters.png"), dpi=100)
plt.close()


# 7. Συγκριτικός πίνακας + γράφημα ARI/NMI
# ARI & NMI: συμφωνία clusters με ground truth (1 = τέλεια, ~0 = τυχαία).
summary = []
for name, labels, params in [
    ("K-Means",      kmeans_labels, f"k={best_k}"),
    ("Hierarchical", hier_labels,   f"linkage={best_link}"),
    ("DBSCAN",       dbscan_labels, f"eps={best_db['eps']}, ms={best_db['min_samples']}"),
]:
    n_cl, _, sil, db = metrics(labels)
    summary.append({
        "algorithm": name, "params": params, "n_clusters": n_cl,
        "silhouette": sil, "davies_bouldin": db,
        "ARI": adjusted_rand_score(y_true, labels),
        "NMI": normalized_mutual_info_score(y_true, labels),
    })

res = pd.DataFrame(summary)
print(f"\n--- ΣΥΓΚΡΙΣΗ ΑΛΓΟΡΙΘΜΩΝ ---\n{res.to_string(index=False)}")
res.to_csv(os.path.join(OUTPUT_DIR, "clustering_comparison.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 4))
res.plot(x="algorithm", y=["ARI", "NMI"], kind="bar", ax=ax, color=["#2196F3", "#FF5722"])
ax.set_title("Συμφωνία ομάδων με το πραγματικό Label")
ax.set_ylabel("Τιμή (0=τυχαίο, 1=τέλειο)")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "14_clustering_label_comparison.png"), dpi=100)
plt.close()


# 8. Πίνακας διασταύρωσης για τον καλύτερο αλγόριθμο
best_algo = res.loc[res["silhouette"].idxmax(), "algorithm"]
best_labels = {"K-Means": kmeans_labels, "Hierarchical": hier_labels,
               "DBSCAN": dbscan_labels}[best_algo]
print(f"\nΠίνακας διασταύρωσης (Label × ομάδες) - {best_algo}:")
print(pd.crosstab(y_true, best_labels))
