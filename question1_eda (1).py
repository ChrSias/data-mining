"""
ΕΡΩΤΗΜΑ 1: Διερευνητική Ανάλυση Δεδομένων (EDA)
Παράγει: cleaned_dataset.csv + γραφήματα στο output/
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)

_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, "data")
OUTPUT_DIR  = os.path.join(_HERE, "output")
CLEANED_CSV = os.path.join(_HERE, "cleaned_dataset.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. Φόρτωση και ένωση των 8 CSV
df = pd.concat(
    [pd.read_csv(f, low_memory=False) for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))],
    ignore_index=True,
)
# Καθαρισμός κενών στην αρχή των ονομάτων στηλών (γνωστό issue του CIC-IDS-2017).
df.columns = df.columns.str.strip()

print(f"Αρχικό dataset: {df.shape[0]:,} γραμμές x {df.shape[1]} στήλες")
print(f"\nΤύποι δεδομένων:\n{df.dtypes.value_counts()}")


# 2. Καθαρισμός
# Άπειρες τιμές (από διαιρέσεις με 0) -> NaN.
df.replace([np.inf, -np.inf], np.nan, inplace=True)

print(f"\nΕλλιπείς τιμές: {df.isna().sum().sum():,}")
print(f"Διπλότυπες γραμμές: {df.duplicated().sum():,}")

# Αρνητικές τιμές -> 0 (τα δικτυακά features είναι φυσικά μη-αρνητικά).
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].clip(lower=0)

# Αφαίρεση στηλών με >50% NaN, μετά γραμμών με NaN, μετά διπλότυπων.
df.dropna(thresh=int(len(df) * 0.5), axis=1, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Sanitize labels (αφαίρεση κενών + διόρθωση του χαλασμένου en-dash στα Web Attack).
df["Label"] = df["Label"].str.strip().str.replace("�", "-", regex=False)

print(f"\nΜετά καθαρισμό: {df.shape[0]:,} γραμμές x {df.shape[1]} στήλες")


# 3. Κατανομή κλάσεων
counts = df["Label"].value_counts()
print(f"\nΚατανομή κλάσεων:\n{counts}")
print(f"\nΛόγος ανισορροπίας (max/min): {counts.max() / counts.min():,.0f} : 1")

# Γράφημα: γραμμική + λογαριθμική κλίμακα (η log αναδεικνύει τις σπάνιες κλάσεις).
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
counts.plot(kind="bar", ax=axes[0], color="steelblue")
axes[0].set_title("Κατανομή κλάσεων (γραμμική)")
axes[0].tick_params(axis="x", rotation=45)
counts.plot(kind="bar", ax=axes[1], color="steelblue")
axes[1].set_yscale("log")
axes[1].set_title("Κατανομή κλάσεων (log)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_label_distribution.png"), dpi=100)
plt.close()


# 4. Στατιστική περιγραφή
numeric = df.select_dtypes(include=[np.number])
desc = numeric.describe().T
print(f"\n--- Στατιστική περιγραφή ---\n{desc}")
desc.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))


# 5. Οπτικοποιήσεις - ιστογράμματα + boxplots
features = [
    "Destination Port", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Packet Length Mean",
    "Average Packet Size", "SYN Flag Count",
    "ACK Flag Count", "Down/Up Ratio",
]

# Ιστογράμματα: clipping στο 1%-99% για να μη χαλάνε οι ακραίες τιμές το γράφημα.
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax, feat in zip(axes.flatten(), features):
    s = numeric[feat]
    ax.hist(s.clip(s.quantile(0.01), s.quantile(0.99)), bins=40,
            color="steelblue", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title(feat, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_histograms.png"), dpi=100)
plt.close()

# Boxplots: εντοπισμός outliers. symlog για να δουλεύει και με 0.
fig, axes = plt.subplots(3, 4, figsize=(16, 11))
for ax, feat in zip(axes.flatten(), features):
    ax.boxplot(numeric[feat], vert=True, widths=0.6, patch_artist=True,
               boxprops=dict(facecolor="lightblue", color="steelblue"),
               medianprops=dict(color="red"),
               flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_yscale("symlog")
    ax.set_title(feat, fontsize=9)
    ax.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_boxplots.png"), dpi=100)
plt.close()

# Heatmap συσχετίσεων στα επιλεγμένα features.
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(numeric[features].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_heatmap_subset.png"), dpi=100)
plt.close()


# 6. Feature Selection με τεκμηρίωση
print("\n--- ΣΤΗΛΕΣ ΠΟΥ ΑΦΑΙΡΕΘΗΚΑΝ ---")

# Κριτήριο 1: σχεδόν μηδενική διακύμανση (στήλη πάντα ίδια τιμή = μηδέν πληροφορία).
variances = numeric.var()
low_var = variances[variances < 1e-5].index.tolist()
print(f"\nΛόγω χαμηλής διακύμανσης ({len(low_var)} στήλες):")
for c in low_var:
    print(f"  - {c} (var={variances[c]:.2e})")

X = numeric.loc[:, VarianceThreshold(threshold=1e-5).fit(numeric).get_support()].copy()

# Κριτήριο 2: υψηλή συσχέτιση (>0.95). Κρατάμε μία από κάθε ζευγάρι.
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = []
print(f"\nΛόγω υψηλής συσχέτισης (>0.95):")
for col in upper.columns:
    partners = upper[col][upper[col] > 0.95]
    if not partners.empty:
        high_corr.append(col)
        for p, v in partners.items():
            print(f"  - {col}  <->  {p}  ({v:.3f})")
X.drop(columns=high_corr, inplace=True)

print(f"\nΤελικά features: {X.shape[1]} (από {numeric.shape[1]})")


# 7. Πλήρες heatmap στα features που έμειναν (επιβεβαίωση ότι δεν υπάρχουν πια >0.95).
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(X.corr(), cmap="coolwarm", center=0, ax=ax,
            xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.7})
plt.xticks(rotation=90, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_heatmap_full.png"), dpi=100)
plt.close()


# 8. Αποθήκευση καθαρού dataset
# Δεύτερο dedup: μετά την αφαίρεση στηλών, γραμμές που πριν διέφεραν σε αυτές
# έγιναν ταυτόσημες. Το κάνουμε με τη Label μέσα ώστε να μη χάσουμε ground truth.
X["Label"] = df["Label"].values
X.drop_duplicates(inplace=True)
X.to_csv(CLEANED_CSV, index=False)
print(f"\nΑποθηκεύτηκε: cleaned_dataset.csv ({X.shape[0]:,} γραμμές x {X.shape[1]} στήλες)")
